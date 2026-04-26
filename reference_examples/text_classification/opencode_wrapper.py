"""
Drop-in wrapper around `opencode run` for programmatic usage with logging.

This mirrors the public API of `claude_wrapper.py` closely enough for
`meta_harness.py` to swap wrappers without changing the proposer contract:
run a coding agent, capture text/tool activity, log the session, and return a
SessionResult.

Set OPENCODE_WRAPPER_MODEL (or OPENCODE_MODEL) to choose the OpenCode model,
for example:

    OPENCODE_WRAPPER_MODEL=ollama/qwen3-coder uv run python meta_harness.py ...

If the caller passes Claude aliases such as "opus" or "sonnet", this wrapper
omits `--model` and lets OpenCode use its configured default unless an override
environment variable is set.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any as _Any

DEFAULT_LOG_DIR = os.environ.get(
    "OPENCODE_WRAPPER_LOG_DIR",
    os.environ.get("CLAUDE_WRAPPER_LOG_DIR", "experience"),
)

# Common tool sets; names intentionally match claude_wrapper.py.
TOOLS_READ = ["Read", "Glob", "Grep"]
TOOLS_WRITE = ["Read", "Glob", "Grep", "Edit", "Write"]
TOOLS_BASH = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
TOOLS_ALL = TOOLS_BASH + ["Agent", "WebSearch", "WebFetch"]

_CLAUDE_MODEL_ALIASES = {
    "claude",
    "claude-3",
    "claude-3-5",
    "claude-3-7",
    "claude-4",
    "haiku",
    "opus",
    "sonnet",
}

_TOOL_NAME_MAP = {
    "agent": "Agent",
    "applypatch": "Edit",
    "apply_patch": "Edit",
    "bash": "Bash",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list": "Glob",
    "ls": "Glob",
    "multiedit": "Edit",
    "patch": "Edit",
    "read": "Read",
    "task": "Agent",
    "todoread": "Tool",
    "todowrite": "Tool",
    "webfetch": "WebFetch",
    "web_fetch": "WebFetch",
    "websearch": "WebSearch",
    "web_search": "WebSearch",
    "write": "Write",
}

_TOOL_PERMISSION_MAP = {
    "Agent": {"task"},
    "Bash": {"bash"},
    "Edit": {"edit"},
    "Glob": {"glob"},
    "Grep": {"grep"},
    "Read": {"read"},
    "WebFetch": {"webfetch"},
    "WebSearch": {"websearch"},
    "Write": {"edit"},
}

_KNOWN_OPENCODE_PERMISSIONS = {
    "bash",
    "codesearch",
    "edit",
    "glob",
    "grep",
    "lsp",
    "question",
    "read",
    "skill",
    "task",
    "webfetch",
    "websearch",
}


def _slugify(text, max_words=4):
    """Create a short slug from text for directory names."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "run"


def _clean_read_output(output):
    """Strip line number prefixes (e.g. '     1→') from Read tool output."""
    lines = []
    for line in output.split("\n"):
        m = re.match(r"\s*\d+\u2192(.*)", line)
        lines.append(m.group(1) if m else line)
    return "\n".join(lines)


def _count_read_lines(output):
    """Count Read output lines, accepting Claude-numbered or plain OpenCode text."""
    numbered = sum(1 for line in output.split("\n") if re.match(r"\s*\d+\u2192", line))
    if numbered:
        return numbered
    return len(output.splitlines()) if output else 0


@dataclass
class ToolCall:
    name: str
    tool_id: str
    input: dict
    output: str = ""
    is_error: bool = False


@dataclass
class SessionResult:
    prompt: str
    text: str
    tool_calls: list
    files_read: dict  # {path: {"reads": N, "lines": M}}
    files_written: dict  # {path: {"lines_written": M}}
    token_usage: dict
    duration_seconds: float
    model: str
    session_id: str
    exit_code: int
    cost_usd: float
    raw_events: list
    command: list = None
    cwd: str = None
    stderr: str = ""
    skill: dict = None
    name: str = None
    log_dir: str = None

    def show(self):
        """Print compact one-line-per-event summary."""
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code})")
            print(f"  {(self.stderr or 'No stderr.')[:300]}")
            return
        for tc in self.tool_calls:
            inp = tc.input
            arg = inp.get("file_path") or inp.get("pattern") or ""
            if not arg and "command" in inp:
                arg = inp["command"][:120]
            if not arg and "description" in inp:
                arg = inp["description"][:120]
            if not arg and "prompt" in inp:
                arg = inp["prompt"][:120]
            err = " ERR" if tc.is_error else ""
            print(f"  tool: {tc.name}({arg}){err}")
        text = self.text.strip().replace("\n", " ")
        if text:
            print(f"  text: {text[:200]}")
        if self.files_read:
            items = ", ".join(
                f"{p}({v['reads']}x, {v['lines']}L)" for p, v in self.files_read.items()
            )
            print(f"  read: {items}")
        if self.files_written:
            items = ", ".join(
                f"{p}({v['lines_written']}L)" for p, v in self.files_written.items()
            )
            print(f"  wrote: {items}")
        print(
            f"  {self.token_usage['input_tokens']}in/"
            f"{self.token_usage['output_tokens']}out  "
            f"${self.cost_usd:.4f}  {self.duration_seconds:.1f}s"
        )


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_model(model: str | None) -> str | None:
    override = os.environ.get("OPENCODE_WRAPPER_MODEL") or os.environ.get(
        "OPENCODE_MODEL"
    )
    if override:
        return override
    if not model:
        return None

    normalized = str(model).strip()
    lowered = normalized.lower()
    if lowered in _CLAUDE_MODEL_ALIASES:
        return None
    if "/" not in normalized:
        # OpenCode models are usually provider/model. A bare Claude alias from
        # existing harness code should defer to the user's OpenCode config.
        return None
    return normalized


def _compose_prompt(prompt: str, system_prompt: str | None) -> str:
    if not system_prompt:
        return prompt
    return (
        "Follow these proposer instructions before handling the task.\n\n"
        f"{system_prompt.strip()}\n\n"
        "Task:\n"
        f"{prompt}"
    )


def build_command(
    prompt,
    model="sonnet",
    allowed_tools=None,
    system_prompt=None,
    tools=None,
    disallowed_tools=None,
    disable_skills=True,
    disable_mcp=True,
    effort=None,
):
    """Build the opencode CLI command list.

    The signature matches claude_wrapper.build_command. Tool restrictions and
    disable_* flags are applied through environment variables in run().
    """
    del allowed_tools, tools, disallowed_tools, disable_skills, disable_mcp, effort

    cmd = ["opencode", "run", "--format", "json"]

    agent = os.environ.get("OPENCODE_WRAPPER_AGENT") or os.environ.get(
        "OPENCODE_AGENT"
    )
    if agent:
        cmd.extend(["--agent", agent])

    resolved_model = _resolve_model(model)
    if resolved_model:
        cmd.extend(["--model", resolved_model])

    if _truthy(os.environ.get("OPENCODE_WRAPPER_SKIP_PERMISSIONS", "1")):
        cmd.append("--dangerously-skip-permissions")

    cmd.append(_compose_prompt(prompt, system_prompt))
    return cmd


def _make_relative(filepath, cwd):
    """Convert absolute path to relative if it's under cwd."""
    if not cwd or not filepath:
        return filepath
    try:
        return os.path.relpath(filepath, cwd)
    except ValueError:
        return filepath


def _camel_to_snake(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return key.replace("-", "_").lower()


def _normalize_tool_name(name: str | None) -> str:
    if not name:
        return "Tool"
    key = re.sub(r"[^a-z0-9_]", "", str(name).replace("-", "_").lower())
    return _TOOL_NAME_MAP.get(key, name[:1].upper() + name[1:])


def _normalize_tool_input(tool_name: str, data: _Any) -> dict:
    if not isinstance(data, dict):
        return {"value": data}

    normalized = {}
    for key, value in data.items():
        nkey = _camel_to_snake(str(key))
        if nkey in {"filepath", "file"}:
            nkey = "file_path"
        if nkey == "oldstring":
            nkey = "old_string"
        if nkey == "newstring":
            nkey = "new_string"
        normalized[nkey] = value

    if (
        tool_name in {"Read", "Write", "Edit"}
        and "file_path" not in normalized
        and "path" in normalized
    ):
        normalized["file_path"] = normalized["path"]

    return normalized


def _stringify_output(value: _Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _dig(obj: _Any, *keys: str) -> _Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _event_part(event: dict) -> dict:
    part = event.get("part")
    return part if isinstance(part, dict) else {}


def _event_type(event: dict) -> str:
    return str(event.get("type") or event.get("event") or "").lower()


def _extract_text(event: dict) -> str:
    part = _event_part(event)
    candidates = [
        part.get("text"),
        event.get("text"),
        _dig(event, "message", "content"),
        event.get("content"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, list):
            pieces = []
            for item in candidate:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            if pieces:
                return "".join(pieces)
    return ""


def _extract_error_text(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        data = error.get("data")
        pieces = [error.get("name")]
        if isinstance(data, dict):
            pieces.append(data.get("message"))
        pieces.append(error.get("message"))
        return ": ".join(str(p) for p in pieces if p)
    if isinstance(error, str):
        return error
    return ""


def _extract_tool_call(event: dict) -> ToolCall | None:
    part = _event_part(event)
    state = part.get("state") if isinstance(part.get("state"), dict) else {}

    raw_name = (
        part.get("tool")
        or part.get("name")
        or event.get("tool")
        or event.get("name")
        or _dig(event, "tool", "name")
    )
    if not raw_name:
        return None

    name = _normalize_tool_name(raw_name)
    input_data = (
        state.get("input")
        or part.get("input")
        or event.get("input")
        or event.get("arguments")
        or {}
    )
    normalized_input = _normalize_tool_input(name, input_data)

    output = (
        state.get("output")
        or state.get("result")
        or part.get("output")
        or part.get("result")
        or event.get("output")
        or event.get("result")
    )
    status = str(state.get("status") or part.get("status") or event.get("status") or "")
    error = state.get("error") or part.get("error") or event.get("error")

    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    exit_code = metadata.get("exit") or metadata.get("exit_code")

    return ToolCall(
        name=name,
        tool_id=str(
            part.get("callID")
            or part.get("call_id")
            or part.get("id")
            or event.get("callID")
            or event.get("call_id")
            or event.get("id")
            or ""
        ),
        input=normalized_input,
        output=_stringify_output(output or error),
        is_error=bool(error)
        or status.lower() in {"error", "failed", "failure"}
        or (exit_code not in (None, 0, "0")),
    )


def _extract_step_usage(event: dict) -> tuple[dict, float, str]:
    part = _event_part(event)
    tokens = (
        part.get("tokens")
        or event.get("tokens")
        or part.get("usage")
        or event.get("usage")
        or {}
    )
    if not isinstance(tokens, dict):
        tokens = {}

    usage = {"input_tokens": 0, "output_tokens": 0}
    usage["input_tokens"] += int(
        tokens.get("input", 0)
        or tokens.get("input_tokens", 0)
        or tokens.get("prompt", 0)
        or tokens.get("prompt_tokens", 0)
    )
    usage["output_tokens"] += int(
        tokens.get("output", 0)
        or tokens.get("output_tokens", 0)
        or tokens.get("completion", 0)
        or tokens.get("completion_tokens", 0)
    )

    if tokens.get("cache", {}).get("read") is not None:
        usage["cache_read_input_tokens"] = int(tokens["cache"]["read"])
    if tokens.get("cache", {}).get("write") is not None:
        usage["cache_creation_input_tokens"] = int(tokens["cache"]["write"])
    if tokens.get("cache_read_input_tokens") is not None:
        usage["cache_read_input_tokens"] = int(tokens["cache_read_input_tokens"])
    if tokens.get("cache_creation_input_tokens") is not None:
        usage["cache_creation_input_tokens"] = int(tokens["cache_creation_input_tokens"])

    cost = part.get("cost") or event.get("cost") or 0.0
    session_id = (
        str(part.get("sessionID") or part.get("session_id") or "")
        or str(event.get("sessionID") or event.get("session_id") or "")
    )
    return usage, float(cost or 0.0), session_id


def parse_stream_events(stdout, prompt, model, duration, exit_code, cwd=None):
    """Parse newline-delimited JSON from `opencode run --format json` output."""
    events = []
    text_parts = []
    tool_calls = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    session_id = ""
    cost_usd = 0.0

    for line in stdout.strip().split("\n") if stdout.strip() else []:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        events.append(event)
        etype = _event_type(event)

        if etype in {"text", "assistant", "message", "content"}:
            text = _extract_text(event)
            if text:
                text_parts.append(text)

        if etype == "error":
            text = _extract_error_text(event)
            if text:
                text_parts.append(text)

        if etype in {"tool_use", "tool", "tool_call"}:
            tc = _extract_tool_call(event)
            if tc:
                tool_calls.append(tc)

        if etype in {"step_finish", "result", "finish", "done"}:
            usage, cost, sid = _extract_step_usage(event)
            for key, val in usage.items():
                token_usage[key] = token_usage.get(key, 0) + val
            cost_usd += cost
            session_id = sid or session_id

        session_id = str(
            event.get("sessionID")
            or event.get("session_id")
            or _dig(event, "session", "id")
            or session_id
        )

    files_read = {}
    files_written = {}
    for tc in tool_calls:
        if tc.name == "Read" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            lines = _count_read_lines(tc.output)
            if path in files_read:
                files_read[path]["reads"] += 1
                files_read[path]["lines"] += lines
            else:
                files_read[path] = {"reads": 1, "lines": lines}
        elif tc.name == "Write" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            content = str(tc.input.get("content", ""))
            lines = content.count("\n") + (1 if content else 0)
            files_written[path] = {"lines_written": lines}
        elif tc.name == "Edit" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            new_str = str(
                tc.input.get("new_string")
                or tc.input.get("content")
                or tc.input.get("patch")
                or ""
            )
            lines = new_str.count("\n") + (1 if new_str else 0)
            if path in files_written:
                files_written[path]["lines_written"] += lines
            else:
                files_written[path] = {"lines_written": lines}

    return SessionResult(
        prompt=prompt,
        text="".join(text_parts),
        tool_calls=tool_calls,
        files_read=files_read,
        files_written=files_written,
        token_usage=token_usage,
        duration_seconds=duration,
        model=model,
        session_id=session_id,
        exit_code=exit_code,
        cost_usd=cost_usd,
        raw_events=events,
    )


def _extract_json_blocks(text):
    """Extract named JSON code blocks from response text."""
    results = []
    pattern = re.compile(
        r"(?:\*\*`?([^`*\n]+\.json)`?\*\*[: \t]*\n)?"
        r"```json\s*\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        name_hint = m.group(1)
        body = m.group(2).strip()
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        filename = Path(name_hint).name if name_hint else None
        results.append((filename, parsed))
    return results


def _summary_arg(tc: ToolCall) -> str:
    return (
        str(tc.input.get("command", ""))[:120]
        or str(tc.input.get("description", ""))[:120]
    )


def log_session(result, log_dir):
    """Write session to a directory. Returns the directory path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = result.name or _slugify(result.prompt)
    run_dir = Path(log_dir) / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": result.prompt,
        "model": result.model,
        "session_id": result.session_id,
        "exit_code": result.exit_code,
        "duration_seconds": round(result.duration_seconds, 2),
        "cost_usd": result.cost_usd,
        "token_usage": result.token_usage,
        "command": result.command,
        "cwd": result.cwd,
        "skill": result.skill,
        "files_read": result.files_read,
        "files_written": result.files_written,
        "tool_summary": [
            f"{tc.name}({'ERR ' if tc.is_error else ''}"
            f"{tc.input.get('file_path') or tc.input.get('pattern') or _summary_arg(tc)})"
            for tc in result.tool_calls
        ],
    }
    if result.stderr:
        meta["stderr"] = result.stderr
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if result.text:
        (run_dir / "response.md").write_text(result.text)

    if result.text:
        json_blocks = _extract_json_blocks(result.text)
        if json_blocks:
            art_dir = run_dir / "artifacts"
            art_dir.mkdir(exist_ok=True)
            for i, (name, data) in enumerate(json_blocks, 1):
                fname = name or f"{i:03d}.json"
                (art_dir / fname).write_text(json.dumps(data, indent=2) + "\n")

    if result.raw_events:
        lines = [json.dumps(e, default=str) for e in result.raw_events]
        (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n")

    if result.tool_calls:
        tools_dir = run_dir / "tools"
        tools_dir.mkdir(exist_ok=True)
        for i, tc in enumerate(result.tool_calls, 1):
            parts = []
            file_path = tc.input.get("file_path", "")
            if file_path:
                file_path = _make_relative(file_path, result.cwd)
            header = f"{tc.name}: {file_path}" if file_path else tc.name
            if tc.is_error:
                header += " [ERROR]"
            parts.append(header)
            parts.append("")

            for k, v in tc.input.items():
                if k == "file_path":
                    continue
                val = str(v)
                if "\n" in val or len(val) > 80:
                    parts.append(f"{k}:")
                    parts.append(val)
                    parts.append("")
                else:
                    parts.append(f"{k}: {v}")

            if tc.output:
                output = (
                    _clean_read_output(tc.output) if tc.name == "Read" else tc.output
                )
                parts.append("")
                parts.append("--- output ---")
                parts.append(output)

            (tools_dir / f"{i:03d}_{tc.name}.txt").write_text("\n".join(parts))

    result.log_dir = str(run_dir)
    return str(run_dir)


def load_skill(skill_path):
    """Load a skill markdown file. Returns content string or None if not found."""
    path = Path(skill_path)
    if path.exists():
        return path.read_text()
    return None


def load_skills(skills, skill_dir=None):
    """Load one or more skills by path, name, or from a directory."""
    if skill_dir is None:
        skill_dir = ".claude/skills"
    skill_dir = Path(skill_dir)
    loaded = []

    for s in skills:
        p = Path(s)
        if p.is_dir() and (p / "SKILL.md").is_file():
            skill_file = p / "SKILL.md"
            loaded.append(
                {
                    "path": str(skill_file),
                    "name": p.name,
                    "content": skill_file.read_text(),
                }
            )
        elif p.is_dir():
            for md in sorted(p.glob("*.md")):
                loaded.append(
                    {"path": str(md), "name": md.stem, "content": md.read_text()}
                )
        elif p.is_file():
            loaded.append({"path": str(p), "name": p.stem, "content": p.read_text()})
        else:
            candidates = [
                skill_dir / s / "SKILL.md",
                skill_dir / s,
                skill_dir / f"{s}.md",
            ]
            for c in candidates:
                if c.is_file():
                    name = c.parent.name if c.name == "SKILL.md" else c.stem
                    loaded.append(
                        {"path": str(c), "name": name, "content": c.read_text()}
                    )
                    break

    return loaded


def _default_progress(event, tool_calls):
    """Default progress callback: print one line per tool call to stderr."""
    etype = _event_type(event)
    if etype not in {"tool_use", "tool", "tool_call"}:
        return

    tc = _extract_tool_call(event)
    if not tc:
        return

    inp = tc.input
    arg = inp.get("file_path") or inp.get("pattern") or ""
    if not arg and "command" in inp:
        arg = str(inp["command"])[:120]
    if not arg and "description" in inp:
        arg = str(inp["description"])[:120]
    if not arg and "prompt" in inp:
        arg = str(inp["prompt"])[:120]
    arg = str(arg).replace("\n", " ").strip()
    n = len(tool_calls)
    print(f"  [{n}] {tc.name}({arg[:120]})", flush=True)


def _enqueue_lines(pipe, q, stream_name):
    """Read lines from a pipe in a background thread and push them into a queue."""
    try:
        for line in iter(pipe.readline, ""):
            q.put((stream_name, line))
    finally:
        pipe.close()


def _permission_names(tool_names: list[str] | None) -> set[str]:
    permissions = set()
    for tool in tool_names or []:
        normalized = _normalize_tool_name(tool)
        permissions.update(_TOOL_PERMISSION_MAP.get(normalized, set()))
    return permissions


def _build_permissions(allowed_tools=None, tools=None, disallowed_tools=None) -> dict:
    effective_tools = tools if tools is not None else allowed_tools
    permissions = {}

    if effective_tools is not None:
        allowed = _permission_names(effective_tools)
        for name in _KNOWN_OPENCODE_PERMISSIONS:
            permissions[name] = "allow" if name in allowed else "deny"

    for name in _permission_names(disallowed_tools):
        permissions[name] = "deny"

    return permissions


def _merge_json_env(env: dict, key: str, overlay: dict) -> None:
    if not overlay:
        return
    existing = env.get(key)
    base = {}
    if existing:
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                base = loaded
        except (json.JSONDecodeError, ValueError):
            pass
    base.update(overlay)
    env[key] = json.dumps(base)


def run(
    prompt,
    model="sonnet",
    allowed_tools=None,
    tools=None,
    disallowed_tools=None,
    cwd=None,
    log_dir=None,
    name=None,
    system_prompt=None,
    skill_path=None,
    skills=None,
    skill_dir=None,
    timeout_seconds=None,
    disable_skills=True,
    disable_mcp=True,
    progress=True,
    effort=None,
):
    """Run `opencode run` and return parsed SessionResult. Logs to log_dir."""
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    if allowed_tools is None:
        allowed_tools = list(TOOLS_BASH)
    if disallowed_tools is None:
        disallowed_tools = []

    all_skills = []
    if skill_path:
        content = load_skill(skill_path)
        if content:
            all_skills.append(
                {"path": skill_path, "name": Path(skill_path).stem, "content": content}
            )
    if skills:
        all_skills.extend(load_skills(skills, skill_dir))

    skill_info = all_skills if all_skills else None
    if all_skills:
        skill_text = "\n\n".join(
            f"## Skill: {s['name']}\n{s['content']}" for s in all_skills
        )
        prefix = f"Follow these skill instructions:\n\n{skill_text}\n\n"
        system_prompt = prefix + (system_prompt or "")

    cmd = build_command(
        prompt,
        model,
        allowed_tools,
        system_prompt,
        tools=tools,
        disallowed_tools=disallowed_tools,
        disable_skills=disable_skills,
        disable_mcp=disable_mcp,
        effort=effort,
    )

    effective_cwd = cwd or os.getcwd()
    display_model = _resolve_model(model) or "opencode-config-default"

    env = os.environ.copy()
    permissions = _build_permissions(
        allowed_tools=allowed_tools,
        tools=tools,
        disallowed_tools=disallowed_tools,
    )
    _merge_json_env(env, "OPENCODE_PERMISSION", permissions)

    if disable_skills:
        env.setdefault("OPENCODE_DISABLE_CLAUDE_CODE_SKILLS", "true")
        env.setdefault("OPENCODE_DISABLE_CLAUDE_CODE_PROMPT", "true")
    if disable_mcp:
        env.setdefault("OPENCODE_DISABLE_DEFAULT_PLUGINS", "true")

    if permissions.get("websearch") == "allow":
        env.setdefault("OPENCODE_ENABLE_EXA", "true")

    if progress is True:
        on_event = _default_progress
    elif callable(progress):
        on_event = progress
    else:
        on_event = None

    start = time.time()
    stdout_lines = []
    stderr_lines = []
    exit_code = 0
    live_tool_calls = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
            env=env,
        )
        deadline = start + timeout_seconds if timeout_seconds else None
        q = queue.Queue()
        stdout_thread = threading.Thread(
            target=_enqueue_lines,
            args=(proc.stdout, q, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_enqueue_lines,
            args=(proc.stderr, q, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        while True:
            if deadline and time.time() > deadline:
                proc.kill()
                stderr_lines.append(
                    f"\nProcess timed out after {timeout_seconds} seconds."
                )
                exit_code = 124
                break

            try:
                stream_name, line = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if stream_name == "stdout":
                stdout_lines.append(line)
                if on_event:
                    try:
                        event = json.loads(line)
                        if _event_type(event) in {"tool_use", "tool", "tool_call"}:
                            live_tool_calls.append(event)
                        on_event(event, live_tool_calls)
                    except (json.JSONDecodeError, ValueError):
                        pass
            else:
                stderr_lines.append(line)

        proc.wait()
        if exit_code == 0:
            exit_code = proc.returncode
    except FileNotFoundError as e:
        stderr_lines = [str(e)]
        exit_code = 127

    duration = time.time() - start
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    result = parse_stream_events(
        stdout, prompt, display_model, duration, exit_code, cwd=effective_cwd
    )
    result.command = cmd
    result.cwd = effective_cwd
    result.stderr = stderr
    result.skill = skill_info
    result.name = name
    log_session(result, log_dir)
    return result


if __name__ == "__main__":
    run(
        "Read through the important files in this directory and summarize them.",
        allowed_tools=TOOLS_READ,
        name="summarize-repo",
    ).show()
