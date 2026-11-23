# Agent tool loops

Run the commands from this directory. The lookup, file-read, and tool-choice examples use the standard runtime libraries.

## LoopLab

For the checkpoint runner, dependencies, and offline smoke check, follow the [LoopLab setup](looplab/README.md).

## Missing test log

Use Python 3.12. The script prints the stored records and the text delivered to the caller.

```sh
python3 missing-log/lookup.py
```

## File-read offload

Use Node.js 24.19.0. The driver loads the bundled DeepAgentsJS source snapshots, checks their hashes, and prints JSON. An optional argument names a new output file.

```sh
node offload/eviction_poc.mjs
```

## Tool choice

Use Python 3.12. The default command reads the recorded request configuration and archived continuation summaries.

```sh
python3 tool-choice/compare.py
```

For an optional live comparison, set `DEEPSEEK_API_KEY` in your environment. The script submits the same saved conversation once with `required` and once with `auto` to the DeepSeek Chat Completions API. This mode makes up to two billable requests, prints the first decision for each setting, and executes zero proposed tools. The model alias and tool schemas are in `tool-choice/request.json`.

```sh
python3 tool-choice/compare.py --live
```
