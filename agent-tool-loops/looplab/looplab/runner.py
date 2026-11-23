"""Paired checkpoint experiments with real DeepSeek inside Deep Agents.

The outer graph persists fixture/controller state between agent turns. The child
agent persists its internal messages separately. Only outer boundaries are valid
fork points: tool closure state is committed there, not at every child checkpoint.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import TypedDict

import httpx
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .sources import SOURCE_URLS
from .support import PROJECT, Settings, ReproTrace, plain, write_json, fingerprint, disable_external_tracing


class LabState(TypedDict):
    world: dict
    turns: list[dict]
    cycle: int
    fixed: bool


class BudgetExceeded(RuntimeError):
    pass


class BatchBudgetExceeded(BudgetExceeded):
    pass


class Budget:
    def __init__(self, limit, shared=False):
        self.limit, self.used, self.shared = limit, 0, shared

    def take(self):
        if self.used >= self.limit:
            error = BatchBudgetExceeded if self.shared else BudgetExceeded
            raise error(f"{'Batch' if self.shared else 'Turn'} model-call budget exhausted ({self.limit})")
        self.used += 1


class RequestPolicy(AgentMiddleware):
    def __init__(self, options, budget, turn_budget):
        self.options, self.budget, self.turn_budget = options, budget, turn_budget

    async def awrap_model_call(self, request, handler):
        # Independent of invoke/resume counters; HITL resumes cannot reset these.
        self.turn_budget.take()
        self.budget.take()
        overrides = {}
        opts = dict(self.options)
        if "tool_choice" in opts:
            overrides["tool_choice"] = opts.pop("tool_choice")
        overrides["model_settings"] = {**request.model_settings, **opts}
        return await handler(request.override(**overrides))


def registry():
    from looplab.cases_tools import build_cases as tools
    from looplab.cases_context import build_cases as context
    from looplab.cases_lifecycle import build_cases as lifecycle
    cases = tools() + context() + lifecycle()
    result = {c.id: c for c in cases}
    if len(result) != 40 or set(result) != {f"C{i:02}" for i in range(1, 41)}:
        raise ValueError("Case registry must contain exactly C01..C40 once")
    return result


def source_files():
    return [PROJECT / "pyproject.toml", PROJECT / "uv.lock", *sorted((PROJECT / "looplab").glob("*.py"))]


def code_hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


class LabTrace(ReproTrace):
    def __init__(self, run_dir, api_key):
        super().__init__(run_dir, api_key)
        self.executed_tools = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.executed_tools.append({"name": serialized.get("name"), "args": kwargs.get("inputs") or input_str})
        super().on_tool_start(serialized, input_str, **kwargs)


async def agent_turn(case, state, settings, saver, run_dir, budget, model_factory=None):
    world = copy.deepcopy(state["world"])
    fixed = state["fixed"]
    cycle = state["cycle"]
    turn_dir = run_dir / f"turn-{cycle:02}"
    turn_dir.mkdir()
    trace = LabTrace(turn_dir, settings.api_key)
    options = case.model_options(world, fixed)
    tools = case.tools(world, fixed)
    thread_id = f"{run_dir.name}-{cycle}-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [trace], "recursion_limit": 150}
    rejects = []
    status, error, result = "completed", None, {}

    async def request_hook(request):
        # No headers: Authorization and transport credentials must never reach artifacts.
        trace.write("provider_request", method=request.method, url=str(request.url), body=json.loads(request.content))

    async def response_hook(response):
        await response.aread()
        try:
            body = response.json()
        except ValueError:
            body = response.text[:2000]
        trace.write("provider_response", status_code=response.status_code, body=body)

    async with httpx.AsyncClient(event_hooks={"request": [request_hook], "response": [response_hook]}) as client:
        model = model_factory() if model_factory else ChatDeepSeek(
            model=settings.model, api_key=settings.api_key, base_url=settings.api_base,
            temperature=0, max_tokens=settings.max_output_tokens, timeout=45, max_retries=0,
            reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}},
            http_async_client=client,
        )
        graph = create_deep_agent(
            model=model, name=f"lab-{case.id}", tools=tools, backend=StateBackend(),
            system_prompt=case.system_prompt,
            checkpointer=saver,
            middleware=[RequestPolicy(options, budget, Budget(case.max_model_calls)),
                        ModelCallLimitMiddleware(thread_limit=case.max_model_calls, exit_behavior="error"),
                        ToolCallLimitMiddleware(thread_limit=case.max_tool_calls, exit_behavior="error")],
            **case.graph_options,
        )
        user_task = case.task + "\n" + case.prompt(world, fixed)
        trace.write("turn_start", case=case.id, cycle=cycle, fixed=fixed, model_source="offline_test" if model_factory else "deepseek", world=world)
        async def invoke():
            nonlocal result, status
            result = await graph.ainvoke({"messages": [{"role": "user", "content": user_task}]}, config=config, durability="sync")
            for _ in range(2):
                interrupts = result.get("__interrupt__", ())
                if not interrupts:
                    break
                value = interrupts[0].value
                actions = value.get("action_requests", [])
                feedback = ("The user has permanently declined this operation. Do not request or execute it again; report that it was not authorized."
                            if fixed else "User rejected the tool call.")
                msg_index = max((i for i, m in enumerate(result.get("messages", [])) if isinstance(m, AIMessage)), default=-1)
                rejects.append({"actions": plain(actions), "feedback": feedback, "message_index": msg_index})
                trace.write("synthetic_approval_rejection", actions=actions, feedback=feedback)
                result = await graph.ainvoke(Command(resume={"decisions": [{"type": "reject", "message": feedback} for _ in actions]}), config=config, durability="sync")
            if result.get("__interrupt__"):
                status = "approval_limit"
        try:
            await asyncio.wait_for(invoke(), timeout=min(90, settings.timeout_seconds))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "batch_budget_exhausted" if isinstance(exc, BatchBudgetExceeded) else ("budget_exhausted" if isinstance(exc, BudgetExceeded) or "LimitExceeded" in type(exc).__name__ else ("timeout" if isinstance(exc, TimeoutError) else "error"))
        snap = await graph.aget_state(config)
        messages = snap.values.get("messages", [])
    proposed = [{**c, "message_index": index} for index, m in enumerate(messages) if isinstance(m, AIMessage) for c in m.tool_calls]
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    turn = {"cycle": cycle, "status": status, "error": error,
            "answer": last.content if last else "", "tool_calls": proposed,
            "executed_tools": trace.executed_tools, "model_calls": trace.model_calls,
            "approval_rejections": rejects, "finish_reason": (last.response_metadata.get("finish_reason") if last else None),
            "usage": [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata],
            "agent_thread_id": thread_id,
            "agent_checkpoint_id": snap.config.get("configurable", {}).get("checkpoint_id")}
    write_json(turn_dir / "turn.json", turn, settings.api_key)
    trace.write("turn_end", turn=turn, world=world)
    return {"world": world, "turns": state["turns"] + [turn], "cycle": cycle + 1}


def build_graph(case, settings, saver, run_dir, budget, model_factory=None):
    def seed(state):
        return {"world": copy.deepcopy(case.initial), "turns": [], "cycle": 0, "fixed": False}

    async def act(state):
        return await agent_turn(case, state, settings, saver, run_dir, budget, model_factory)

    def control(state):
        before = copy.deepcopy(state["world"])
        world = case.transition(copy.deepcopy(before), state["turns"][-1], state["fixed"])
        # Opaque API/setup errors must not be reported as experiment evidence.
        if state["turns"][-1]["status"] in ("error", "timeout", "batch_budget_exhausted"):
            world["done"] = True
        write_json(run_dir / f"control-{state['cycle']:02}.json", {"before": before, "after": world}, settings.api_key)
        return {"world": world}

    def route(state):
        return END if state["world"].get("done") or state["cycle"] >= case.max_cycles else "act"

    return (StateGraph(LabState).add_node("seed", seed).add_node("act", act).add_node("control", control)
            .add_edge(START, "seed").add_edge("seed", "act").add_edge("act", "control")
            .add_conditional_edges("control", route, ["act", END]).compile(checkpointer=saver))


def copy_db(source, target):
    # Caller closes source's writer before a copy, making independent fork bundles.
    wal = Path(str(source) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Close the source checkpoint writer before cloning; active WAL detected")
    with sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
            # DeltaChannel stores historical messages in writes. Keep them; the
            # subsequent update_state creates a new checkpoint, whose pending
            # task writes cannot reuse the recorded future branch.


async def run_branch(case, settings, branch, config, budget, fixed, model_factory=None):
    async with AsyncSqliteSaver.from_conn_string(str(branch / "checkpoints.sqlite")) as saver:
        graph = build_graph(case, settings, saver, branch, budget, model_factory)
        start_config = await graph.aupdate_state(config, {"fixed": fixed}, as_node="seed")
        begin = time.monotonic()
        outer_error = None
        try:
            await asyncio.wait_for(graph.ainvoke(None, start_config, durability="sync"), settings.timeout_seconds)
        except Exception as exc:
            outer_error = f"{type(exc).__name__}: {exc}"
        state = await graph.aget_state({"configurable": {"thread_id": config["configurable"]["thread_id"]}})
        history = [s async for s in graph.aget_state_history({"configurable": {"thread_id": config["configurable"]["thread_id"]}})]
        write_json(branch / "history.json", [{"checkpoint_id": s.config["configurable"]["checkpoint_id"], "next": s.next, "cycle": s.values.get("cycle"), "step": s.metadata.get("step")} for s in reversed(history)], settings.api_key)
        write_json(branch / "state.json", state.values, settings.api_key)
    turns = state.values.get("turns", [])
    evaluation = case.observe(copy.deepcopy(state.values.get("world", {})), copy.deepcopy(turns)) if turns else {"symptom_observed": False, "measurements": {"no_turns": True}}
    invalid = outer_error or any(t["status"] in ("error", "timeout") for t in turns)
    insufficient_budget = not turns or not sum(t["model_calls"] for t in turns) or any(t["status"] == "batch_budget_exhausted" for t in turns)
    summary = {"case_id": case.id, "variant": "fixed" if fixed else "faulty", "fidelity": case.fidelity,
               "status": "inconclusive_runtime_error" if invalid else ("inconclusive_budget" if insufficient_budget else ("symptom_observed" if evaluation["symptom_observed"] else "target_not_observed")),
               "oracle": evaluation, "cycles": len(turns), "model_calls": sum(t["model_calls"] for t in turns),
               "tool_calls": sum(len(t["executed_tools"]) for t in turns), "outer_error": outer_error,
               "errors": [t["error"] for t in turns if t["error"]], "elapsed_seconds": round(time.monotonic() - begin, 2),
               "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
               "next": state.next, "model_source": "offline_test" if model_factory else "deepseek"}
    summary["agent_statuses"] = [t["status"] for t in turns]
    summary["controller_done"] = bool(state.values.get("world", {}).get("done"))
    summary["budget_truncated"] = any(t["status"] in ("budget_exhausted", "batch_budget_exhausted", "approval_limit") for t in turns) or (not summary["controller_done"] and len(turns) >= case.max_cycles)
    write_json(branch / "summary.json", summary, settings.api_key)
    return summary


async def run_pair(case, settings, out, budget, model_factory=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "looplab-v1", "case_id": case.id, "title": case.title, "created_at": datetime.now(timezone.utc).isoformat(),
                "fidelity": case.fidelity, "limitation": case.limitation, "intervention": case.intervention,
                "task": case.task, "system_prompt": case.system_prompt, "initial_world": case.initial,
                "source": {"primary_url": SOURCE_URLS[case.id]}, "environment": fingerprint(), "code_hashes": code_hashes(),
                "settings": {k: v for k, v in asdict(settings).items() if k != "api_key"},
                "limits": {"cycles": case.max_cycles, "model_calls_per_cycle": case.max_model_calls, "tools_per_cycle": case.max_tool_calls},
                "fork_boundary": "outer graph after seed, before first agent turn; fixture state included",
                "claim_scope": "source-inspired injected mechanism; no original product regression claim",
                "model_source": "offline_test" if model_factory else "deepseek"}
    manifest["model_resolution"] = {
        "api_id": settings.model if model_factory is None else None,
        "default_request": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        "case_overrides": "Inspect each case's model_options and the recorded provider requests."}
    write_json(out / "manifest.json", manifest, settings.api_key)
    for p in source_files():
        dest = out / "definition" / p.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    config = {"configurable": {"thread_id": str(uuid4())}}
    async with AsyncSqliteSaver.from_conn_string(str(out / "seed.sqlite")) as saver:
        graph = build_graph(case, settings, saver, out, budget, model_factory)
        await graph.ainvoke({}, config, interrupt_after=["seed"], durability="sync")
        seed = await graph.aget_state(config)
        config = seed.config
    write_json(out / "fork-point.json", {"config": config, "world": seed.values, "next": seed.next})
    summaries = {}
    # Both variants start from the same closed SQLite snapshot. Only the intervention flag changes.
    for fixed in (False, True):
        name = "fixed" if fixed else "faulty"
        branch = out / name
        branch.mkdir()
        copy_db(out / "seed.sqlite", branch / "checkpoints.sqlite")
        summaries[name] = await run_branch(case, settings, branch, config, budget, fixed, model_factory)
    faulty, fixed = summaries["faulty"], summaries["fixed"]
    comparison = {"case_id": case.id, "historical_reproduction": False, "fidelity": case.fidelity,
                  "faulty": faulty, "fixed": fixed,
                  "paired_effect_observed": faulty["status"] == "symptom_observed" and fixed["status"] == "target_not_observed" and not fixed["budget_truncated"],
                  "interpretation": "Single paired trial of an injected/ported mechanism. Not a reliability estimate or historical-product reproduction."}
    write_json(out / "comparison.json", comparison, settings.api_key)
    return comparison


async def batch(ids, out, settings, concurrency=2, model_call_limit=400):
    cases = registry()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(model_call_limit, shared=True)
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(cid):
        async with sem:
            print(json.dumps({"event": "case_start", "id": cid}), flush=True)
            try:
                result = await run_pair(cases[cid], settings, out / cid, budget)
            except Exception as exc:
                result = {"case_id": cid, "setup_error": f"{type(exc).__name__}: {exc}", "paired_effect_observed": False}
            results.append(result)
            write_json(out / "results.json", {"complete": False, "model_calls_reserved": budget.used, "cases": sorted(results, key=lambda x: x["case_id"])}, settings.api_key)
            print(json.dumps({"event": "case_end", "id": cid, "faulty": result.get("faulty", {}).get("status"), "fixed": result.get("fixed", {}).get("status"), "setup_error": result.get("setup_error")}), flush=True)
    await asyncio.gather(*(one(cid) for cid in ids))
    final = {"complete": True, "requested": ids, "model_calls_reserved": budget.used,
             "cases": sorted(results, key=lambda x: x["case_id"])}
    write_json(out / "results.json", final, settings.api_key)
    report(out, final, cases)
    return final


def report(out, results, cases=None):
    cases = cases or registry()
    lines = ["# LoopLab paired runs", "",
             "| Case | Task | Original behavior | Candidate change | Target symptom disappeared |",
             "|---|---|---|---|---|"]
    for item in results["cases"]:
        cid = item["case_id"]
        original = item.get("faulty", {}).get("status", "incomplete")
        changed = item.get("fixed", {}).get("status", "incomplete")
        link = f"[{cid}]({cid}/comparison.json)" if "faulty" in item else cid
        lines.append(f"| {link} | {cases[cid].title} | {original} | {changed} | {item['paired_effect_observed']} |")
    (Path(out) / "REPORT.md").write_text("\n".join(lines) + "\n")


async def replay(case_id, pair, variant, checkpoint_id, out, settings, apply_fix=False, model_factory=None):
    """Resume a *completed outer controller* boundary using an independent DB copy."""
    case = registry()[case_id]
    pair, out = Path(pair).resolve(), Path(out).resolve()
    manifest = json.loads((pair / "manifest.json").read_text())
    if manifest["case_id"] != case_id or manifest["code_hashes"] != code_hashes():
        raise ValueError("Case/code differ from recorded definition")
    expected_source = "offline_test" if model_factory is not None else "deepseek"
    if manifest["model_source"] != expected_source:
        raise ValueError("The replay must use the same model source as the saved run.")
    if model_factory is not None and case_id != "C30":
        raise ValueError("The scripted offline model is specific to the C30 catalog fixture.")
    # Only authentication is loaded afresh; experimental model/limits are frozen.
    settings = Settings(**manifest["settings"], api_key=settings.api_key)
    state_json = json.loads((pair / "fork-point.json").read_text())
    tid = state_json["config"]["configurable"]["thread_id"]
    out.mkdir(parents=True, exist_ok=False)
    copy_db(pair / variant / "checkpoints.sqlite", out / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": checkpoint_id}}
    async with AsyncSqliteSaver.from_conn_string(str(out / "checkpoints.sqlite")) as saver:
        graph = build_graph(case, settings, saver, out, Budget(20), model_factory)
        snap = await graph.aget_state(config)
        if snap.next != ("act",):
            raise ValueError("Replay requires an outer seed/control boundary with next=['act']")
        prefix_turns = len(snap.values["turns"])
        before_observation = case.observe(copy.deepcopy(snap.values["world"]), copy.deepcopy(snap.values["turns"]))
        # New branch checkpoint makes previous future task writes unreachable.
        updates = {"world": copy.deepcopy(snap.values["world"])}
        if apply_fix:
            updates["fixed"] = True
        newconfig = await graph.aupdate_state(config, updates, as_node="seed" if snap.values["cycle"] == 0 else "control")
        await asyncio.wait_for(graph.ainvoke(None, newconfig, durability="sync"), settings.timeout_seconds)
        final = await graph.aget_state({"configurable": {"thread_id": tid}})
        write_json(out / "state.json", final.values, settings.api_key)
        write_json(out / "provenance.json", {"source": str(pair), "variant": variant, "checkpoint_id": checkpoint_id, "apply_fix": apply_fix})
    suffix = final.values["turns"][prefix_turns:]
    continuation_status = "inconclusive_runtime_error" if any(t["status"] in ("error", "timeout") for t in suffix) else ("inconclusive_budget" if not suffix or any(t["status"] == "batch_budget_exhausted" for t in suffix) else "executed")
    result = {"out": str(out), "cycles": final.values["cycle"],
              "before_oracle": before_observation,
              "oracle": case.observe(final.values["world"], final.values["turns"]),
              "continuation": {"turns": len(suffix), "model_calls": sum(t["model_calls"] for t in suffix),
                               "tool_calls": sum(len(t["executed_tools"]) for t in suffix),
                               "status": continuation_status, "errors": [t["error"] for t in suffix if t["error"]]},
              "interpretation": "Oracles are cumulative; compare new control events for recovery after an existing failure."}
    write_json(out / "replay-summary.json", result, settings.api_key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    smoke = sub.add_parser("smoke", help="Exercise the C30 checkpoint pair with a scripted offline model.")
    smoke.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--ids", default="all")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--concurrency", type=int, default=2, choices=(1, 2, 3))
    run.add_argument("--model-call-limit", type=int, default=400)
    again = sub.add_parser("replay")
    again.add_argument("case_id")
    again.add_argument("pair", type=Path)
    again.add_argument("variant", choices=("faulty", "fixed"))
    again.add_argument("checkpoint_id")
    again.add_argument("--out", type=Path, required=True)
    again.add_argument("--apply-fix", action="store_true")
    again.add_argument("--offline", action="store_true", help="Replay a saved C30 offline smoke checkpoint.")
    args = parser.parse_args()
    cases = registry()
    if args.command == "list":
        print(json.dumps([{k: getattr(c, k) for k in ("id", "title", "fidelity", "limitation", "intervention")} for c in sorted(cases.values(), key=lambda x: x.id)], ensure_ascii=False, indent=2))
        return
    disable_external_tracing()
    if args.command == "smoke" or (args.command == "replay" and args.offline):
        from .smoke import catalog_model
        settings = Settings(model="deepseek:offline")
        if args.command == "smoke":
            pair = asyncio.run(run_pair(cases["C30"], settings, args.out, Budget(20), catalog_model))
            # Both branches use the same scripted model decisions. The repaired
            # refresh commits revision 1; the original leaves revision 0 cached.
            for name in ("faulty", "fixed"):
                print(name, json.dumps(pair[name]["oracle"], sort_keys=True))
            if not pair["paired_effect_observed"]:
                raise SystemExit("Offline C30 comparison did not meet the expected criterion.")
        else:
            result = asyncio.run(replay(args.case_id, args.pair, args.variant, args.checkpoint_id,
                                        args.out, settings, args.apply_fix, model_factory=catalog_model))
            print(json.dumps(result, indent=2))
        return

    settings = Settings.load()
    if not settings.api_key:
        parser.error("Set DEEPSEEK_API_KEY in the environment for live runs.")
    if args.command == "run":
        ids = sorted(cases) if args.ids == "all" else args.ids.split(",")
        if not ids or set(ids) - cases.keys() or len(ids) != len(set(ids)) or args.model_call_limit <= 0:
            parser.error("Provide unique valid case IDs and positive model call limit")
        result = asyncio.run(batch(ids, args.out, settings, args.concurrency, args.model_call_limit))
        print(json.dumps({"out": str(args.out), "cases": len(result["cases"]), "model_calls_reserved": result["model_calls_reserved"]}))
    else:
        print(json.dumps(asyncio.run(replay(args.case_id, args.pair, args.variant, args.checkpoint_id, args.out, settings, args.apply_fix))))


if __name__ == "__main__":
    main()
