"""From agent-tool-loops/, run: python3 missing-log/lookup.py (Python 3.12).

A small offline version of the lookup in the missing-log case. The dictionary
stands in for storage. This demonstrates the tool response, not model behavior.
"""
import json


def lookup(run_id, catalog, files):
    path = catalog.get(run_id)
    if path is None or path not in files:
        # The studied configuration returns the same empty text for both misses.
        return ""
    return json.dumps({"status": "ok", "path": path, "content": files[path]})


if __name__ == "__main__":
    catalog = {"run-001": "/logs/run-001.log"}
    files = {}
    for run_id in ("run-000", "run-001", "run-002"):
        print(f"{run_id}: registered={run_id in catalog}, reply={lookup(run_id, catalog, files)!r}")

    # Supply the requested log in the control. A neighboring log would not establish this result.
    files[catalog["run-001"]] = "exit_code=0"
    print(f"With the log present: {lookup('run-001', catalog, files)}")
