"""Settings and local event records shared by the public LoopLab runner."""
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain_core.callbacks import BaseCallbackHandler

PROJECT = Path(__file__).resolve().parent.parent
API_BASE = "https://api.deepseek.com"
PACKAGES = ("deepagents", "langchain", "langchain-core", "langchain-deepseek",
            "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite")

# The fixtures model delegation explicitly. Keep built-in subagents off so all
# model activity passes through the runner's request budget.
register_harness_profile("deepseek", HarnessProfile(
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
))


@dataclass
class Settings:
    api_key: str = field(default="", repr=False)
    model: str = "deepseek-flash"
    api_base: str = API_BASE
    max_output_tokens: int = 1024
    timeout_seconds: float = 180

    def __post_init__(self):
        if self.api_base != API_BASE:
            raise ValueError("Live requests must use the DeepSeek API endpoint.")
        if self.timeout_seconds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("Timeout and output limit must be positive.")

    @classmethod
    def load(cls):
        return cls(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                   model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))


def json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def plain(value):
    return json.loads(json.dumps(value, default=json_default, ensure_ascii=False))


def write_json(path, value, secret=""):
    text = json.dumps(value, default=json_default, ensure_ascii=False, indent=2)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    path.write_text(text + "\n", encoding="utf-8")


def fingerprint():
    return {"python": sys.version, "packages": {p: version(p) for p in PACKAGES}}


def disable_external_tracing():
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class ReproTrace(BaseCallbackHandler):
    """Record model inputs, outputs, and tool completions in the run directory."""

    def __init__(self, run_dir, api_key):
        self.path = run_dir / "events.jsonl"
        self.api_key = api_key
        self.lock = threading.Lock()
        self.model_calls = 0

    def write(self, event, **data):
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        line = json.dumps(payload, default=json_default, ensure_ascii=False)
        if self.api_key:
            line = line.replace(self.api_key, "[REDACTED]")
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.model_calls += 1
        params = kwargs.get("invocation_params", {})
        allowed = ("model", "temperature", "max_tokens", "tools", "tool_choice", "extra_body")
        self.write("model_start", run_id=run_id, parent_run_id=parent_run_id,
                   graph_step=kwargs.get("metadata", {}).get("langgraph_step"),
                   messages=messages, parameters={k: params[k] for k in allowed if k in params})

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.write("model_end", run_id=run_id, response=response)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.write("model_error", run_id=run_id, error_type=type(error).__name__, error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self.write("tool_start", run_id=run_id, parent_run_id=parent_run_id,
                   graph_step=kwargs.get("metadata", {}).get("langgraph_step"),
                   name=serialized.get("name"), arguments=kwargs.get("inputs") or input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.write("tool_end", run_id=run_id, output=output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.write("tool_error", run_id=run_id, error_type=type(error).__name__, error=str(error))
