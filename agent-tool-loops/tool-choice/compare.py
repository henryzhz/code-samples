"""From agent-tool-loops/: python3 tool-choice/compare.py (Python 3.12).

Add --live and set DEEPSEEK_API_KEY to compare the next provider decisions.
"""
import argparse
import http.client
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_fixture(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def live_decision(request, api_key):
    connection = http.client.HTTPSConnection("api.deepseek.com", timeout=30)
    try:
        connection.request(
            "POST",
            "/chat/completions",
            body=json.dumps(request).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise SystemExit(f"DeepSeek returned HTTP {response.status} for {request['tool_choice']}.")
        choice = json.load(response)["choices"][0]
    finally:
        connection.close()
    message = choice["message"]
    return {
        "tool_choice": request["tool_choice"],
        "finish_reason": choice["finish_reason"],
        "answer": message.get("content"),
        "proposed_tools": [call["function"]["name"] for call in message.get("tool_calls") or []],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send up to two provider requests.")
    args = parser.parse_args()
    request = read_fixture("request.json")
    if args.live:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            parser.error("--live requires DEEPSEEK_API_KEY")
        # Reuse the completed search and change only the allowed next action.
        for policy in ("required", "auto"):
            decision = live_decision({**request, "tool_choice": policy}, key)
            print(json.dumps(decision, indent=2))
        return

    # In the saved comparison, present-auto reports hello-world with no further tool calls.
    settings = {k: v for k, v in request.items() if k not in ("tools", "messages", "tool_choice")}
    print(json.dumps({
        "common_settings": settings,
        "last_tool_result": request["messages"][-1]["content"],
        "compared_tool_choices": ["required", "auto"],
        "archived_continuations": read_fixture("archived-branches.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
