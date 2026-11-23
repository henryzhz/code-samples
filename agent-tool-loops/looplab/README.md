# LoopLab

LoopLab runs two bounded variants of a synthetic agent task from the same saved state. The 40 fixtures use Deep Agents, a LangGraph controller, and SQLite checkpoints. Public reports that motivated the fixtures are listed in `looplab/sources.py`. A fixture's intervention and limitations are printed by `list`.

## Install and list cases

From this directory, use Python 3.12 and uv. The lock file pins the library versions.

```sh
uv sync --frozen --no-dev --python 3.12
uv run --frozen --no-dev python -m looplab.runner list
```

## Try a pair without an API key

The smoke command uses the catalog-refresh fixture (C30). Only the model messages are scripted. The tool, controller, independent SQLite branches, and result checks run locally. Supply a fresh output directory on each run.

```sh
uv run --frozen --no-dev python -m looplab.runner smoke --out runs/smoke
```

Inspect `fork-point.json` for the shared seed, `faulty/state.json` and `fixed/state.json` for the two catalog states, and `comparison.json` for the measured differences. Each variant also records its individual turns, tool calls, and stopping condition.

## Resume a saved decision

For the offline demo, choose an outer checkpoint that is about to start an agent turn. This command picks one after the original variant has already attempted a refresh.

```sh
checkpoint=$(python3 -c 'import json; print(next(row["checkpoint_id"] for row in json.load(open("runs/smoke/faulty/history.json")) if row["cycle"] == 1 and row["next"] == ["act"]))')
uv run --frozen --no-dev python -m looplab.runner replay C30 runs/smoke faulty "$checkpoint" --offline --apply-fix --out runs/replay
```

Replay copies the selected variant's database into a new directory. It retains the recorded history, applies the requested change, and executes the continuation. Inspect `replay-summary.json` for activity after the checkpoint. Tool environments are saved at completed outer steps. A checkpoint inside a tool call is insufficient to reconstruct its surrounding state.

## Optional live run

Set `DEEPSEEK_API_KEY` in your environment to use your own DeepSeek account. The default model alias is `deepseek-flash`. The paired command below shares its request budget between both variants. It writes API request and response bodies for these synthetic tasks locally. Authentication headers are excluded. Keep outputs private when adapting fixtures to your own data.

```sh
uv run --frozen --no-dev python -m looplab.runner run --ids C03 --out runs/live --concurrency 1 --model-call-limit 12
```

Use comma-separated case IDs to run several cases. For a live replay, select a checkpoint in `runs/live/C03/faulty/history.json`, pass `runs/live/C03` as the pair path and omit `--offline`. Replay consumes new model requests. Use runs you created with the same code version. The assigned behavior in a synthetic fixture and the behavior of a deployed agent can differ.
