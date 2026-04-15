# Terminal-Bench 2

Paper reference experiment for Meta-Harness on Terminal-Bench 2.0. The default search config uses Harbor on the full 89-task TB2 dataset with 2 trials per task on Opus 4.6, matching the intended paper-style search setup in this release.

## Quick Start

Install:

```bash
cd reference_examples/terminal_bench_2
uv sync
```

Read setup details first:

```bash
sed -n '1,200p' SETUP.md
```

Start with a cheap smoke check:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

When trying a new idea, validate it on the cheaper 30-task `hard` subset before paying for a full default search:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness hard 1 50
```

Run one evolve iteration with the default full-dataset search config:

```bash
uv run python meta_harness.py --iterations 1
```

Add `--full-eval` if you also want the optional 5-trial winner pass on the full dataset.

## Key Files

- `.claude/skills/meta-harness-terminal-bench-2/SKILL.md`: proposer prior used by `meta_harness.py`.
- `agents/`: baseline agents and the write target for generated candidates.
- `prompt-templates/terminus-kira.txt`: prompt template used by `baseline_kira.py`.

## Runtime And Cost

With Opus 4.6 and a high-tier API key, the default 89x2 search run at concurrency `50` takes about 4-6 hours and costs roughly $500 _for each iteration_. The recommended bring-up path is to first smoke-test a candidate on the `extract-elf` task, then run the cheaper 30-task `hard` subset (which still contains plenty of signal for evaluation), and finally move to the full default run with all 89 tasks.

This setup is sensitive to concurrency and Anthropic API throughput. API tier matters, and sharing the same API key with other active projects can make runs much slower. Many task failures under an otherwise reasonable scaffold are just timeout failures caused by insufficient API throughput at the chosen concurrency.

This is a cleaned-up version of the code we used for the paper. Please let us know if anything goes wrong.
