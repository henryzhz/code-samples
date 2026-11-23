"""Checkpointable lifecycle fixtures for message routing and execution.

These are reduced, deliberately injected mechanisms. They are not executions of
AutoGen, CrewAI, Claude Code, Codex, pi, LlamaIndex, Mastra or smolagents. Some
source records establish only a symptom: for those cases the injected mechanism
is a hypothesis, not a claim about proprietary/historical implementations.
Every tool operates on the runner's per-turn world copy. No network, subprocess,
real credential, live worker, wall-clock wait or external memory store is used.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from langchain_core.tools import tool

from .spec import Case, verdict

CASE_IDS = ("C19", "C20", "C21", "C22", "C23", "C30", "C31", "C34", "C36", "C37", "C38", "C39")


def _result(**data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _next(world):
    result = deepcopy(world)
    result["cycles"] = result.get("cycles", 0) + 1
    return result


def _snapshot(world, _fixed):
    return "Local fixture state: " + json.dumps(world, ensure_ascii=False, sort_keys=True)


def _title_tools(world, fixed):
    @tool
    def render_title() -> str:
        """Normalize the local document title and mark this finite edit complete."""
        world["rendered_title"] = "Checkpoint Debugging"
        world["task_completed"] = True
        return _result(title=world["rendered_title"], edit_status="complete")
    return [render_title]


def _title_transition(world, turn, fixed):
    w = _next(world)
    if w["task_completed"]:
        if w.get("completion_cycle") is None:
            w["completion_cycle"] = w["cycles"]
        # Fault hypothesis: a keepalive subscription is joined as finite work.
        w["joined_handles"] = [] if fixed else list(w["keepalive_handles"])
        w["cli_exit"] = not w["joined_handles"]
    w["done"] = w["cli_exit"]
    return w


def _title_observe(world, turns):
    complete = world["task_completed"]
    return verdict(
        complete and not world["cli_exit"],
        task_completed=complete, cli_exit=world["cli_exit"],
        joined_handles=world.get("joined_handles", []),
        additional_agent_turns=max(0, world.get("cycles", 0) - (world.get("completion_cycle") or world.get("cycles", 0))),
        empty_answer_turns=sum(not bool(t.get("answer")) for t in turns),
    )


def _handoff_tools(world, fixed):
    @tool
    def inspect_refund(order_id: str = "") -> str:
        """Inspect a synthetic refund request; a customer order ID is required."""
        world["lookup_attempts"] += 1
        if not order_id or order_id not in world["known_order_ids"]:
            return _result(status="needs_user_input", missing="customer order ID")
        world["needs_user_input"] = False
        world["refund_answer"] = "eligible"
        return _result(status="eligible")

    @tool
    def request_user_input(question: str) -> str:
        """Hand control to the user with a clarification question and pause the workflow."""
        if not question.strip():
            return _result(status="invalid", error="A question is required")
        world["handoff_question"] = question
        world["control_owner"] = "user"
        return _result(status="waiting_for_user")

    return [inspect_refund, request_user_input] if fixed else [inspect_refund]


def _handoff_prompt(world, fixed):
    common = "The request has no order ID. Do not invent one. "
    if fixed:
        common += "Use request_user_input if a required identifier is missing. "
    return common + _snapshot(world, fixed)


def _handoff_transition(world, turn, fixed):
    w = _next(world)
    answer = str(turn.get("answer", ""))
    mentions_identifier = bool(re.search(r"order.{0,30}(?:id|number)|订单.{0,12}(?:号|编号)", answer, re.I))
    requests_input = bool(re.search(r"[?？]|\b(?:provide|share|send|tell|need)\b|请(?:提供|告诉)|需要", answer, re.I))
    if answer and mentions_identifier and requests_input and w["needs_user_input"]:
        w["text_replies_waiting_for_identifier"].append(answer)
    w["done"] = w["control_owner"] == "user" or not w["needs_user_input"]
    return w


def _handoff_observe(world, turns):
    return verdict(
        bool(world["text_replies_waiting_for_identifier"]) and world["needs_user_input"] and world["control_owner"] != "user",
        control_owner=world["control_owner"], needs_user_input=world["needs_user_input"],
        text_reply_count=len(world["text_replies_waiting_for_identifier"]),
        handoff_question=world.get("handoff_question"),
        textual_question_detection="lexical order-ID request heuristic; inspect the saved answer",
    )


def _topic_tools(world, fixed):
    @tool
    def process_countdown() -> str:
        """Process one queued countdown message in the local topic-routing fixture."""
        if not world["queue"]:
            return _result(status="queue_empty")
        message = world["queue"].pop(0)
        world["processed"].append(deepcopy(message))
        remaining = message["value"] - 1
        if remaining >= 0:
            subscribers = [message["flow"]] if fixed else ["alpha", "beta"]
            for subscriber in subscribers:
                delivered = {"flow": subscriber, "value": remaining, "origin": message["origin"]}
                world["queue"].append(delivered)
                if subscriber != message["flow"]:
                    world["cross_flow_deliveries"] += 1
        return _result(processed=message, queue_length=len(world["queue"]))
    return [process_countdown]


def _topic_transition(world, turn, fixed):
    w = _next(world)
    w["done"] = not w["queue"]
    return w


def _topic_observe(world, turns):
    counts = {}
    for message in world["processed"]:
        key = (message["origin"], message["flow"], message["value"])
        counts[key] = counts.get(key, 0) + 1
    return verdict(
        world["cross_flow_deliveries"] > 0,
        cross_flow_deliveries=world["cross_flow_deliveries"],
        duplicate_deliveries_processed=sum(max(0, n - 1) for n in counts.values()),
        queue_remaining=len(world["queue"]), processed_count=len(world["processed"]),
    )


def _manager_tools(world, fixed):
    @tool
    def count_local_orders() -> str:
        """Return the exact order count from the immutable local fixture."""
        world["query_completed"] = True
        world["query_result"] = 42
        return _result(order_count=42, query_status="complete")
    return [count_local_orders]


def _manager_transition(world, turn, fixed):
    w = _next(world)
    answered = w["query_completed"] and "42" in str(turn.get("answer", ""))
    if answered and w["answer_cycle"] is None:
        w["answer_cycle"] = w["cycles"]
    if fixed and answered:
        w["group_completed"] = True
    if not w["group_completed"]:
        w["selected_agents"].append("reviewer" if w["cycles"] % 2 else "analyst")
    w["done"] = w["group_completed"]
    return w


def _manager_observe(world, turns):
    extra = max(0, world.get("cycles", 0) - (world["answer_cycle"] or world.get("cycles", 0)))
    return verdict(
        world["answer_cycle"] is not None and extra > 0,
        answer_cycle=world["answer_cycle"], extra_agent_turns=extra,
        group_completed=world["group_completed"], selected_agents=world["selected_agents"],
    )


def _delegation_tools(world, fixed):
    @tool
    def delegate_draft() -> str:
        """Assign the supplied workshop facts to the local writer role for one sentence."""
        if world["role"] != "researcher":
            return _result(status="invalid_role", required="researcher")
        assignment = {"request_id": "workshop-note", "role": "writer", "kind": "assignment"}
        world["pending_assignments"].append(assignment)
        world["routing_events"].append(deepcopy(assignment))
        return _result(status="queued", recipient="writer")

    @tool
    def return_draft(text: str) -> str:
        """Return the writer's completed sentence to the owner of the workshop request."""
        if world["role"] != "writer" or not text.strip():
            return _result(status="invalid", error="Only the writer can return a nonempty draft")
        world["draft"] = text
        if fixed:
            world["consumed_results"].append("workshop-note")
            return _result(status="result_consumed", draft=text)
        # Injected protocol fault: return is reissued as a fresh parent assignment.
        assignment = {"request_id": "workshop-note", "role": "researcher", "kind": "assignment"}
        world["pending_assignments"].append(assignment)
        world["routing_events"].append(deepcopy(assignment))
        world["returns_reissued_as_work"] += 1
        return _result(status="queued", recipient="researcher")
    return [delegate_draft, return_draft]


def _delegation_prompt(world, fixed):
    if world["role"] == "researcher":
        instruction = "You own the workshop note. Ask the writer to draft the supplied facts using delegate_draft."
    else:
        instruction = "You are the writer. Write one sentence and return it with return_draft."
    return instruction + " Fact: The workshop has 12 seats. " + _snapshot(world, fixed)


def _delegation_transition(world, turn, fixed):
    w = _next(world)
    if w["consumed_results"]:
        w["done"] = True
    elif w["pending_assignments"]:
        assignment = w["pending_assignments"].pop(0)
        w["role"] = assignment["role"]
        w["assignment_visits"].append(deepcopy(assignment))
        w["done"] = False
    else:
        w["done"] = False
    return w


def _delegation_observe(world, turns):
    seen = set()
    repeated = 0
    for assignment in world["assignment_visits"]:
        key = (assignment["request_id"], assignment["role"])
        repeated += key in seen
        seen.add(key)
    return verdict(
        world["returns_reissued_as_work"] > 0 and repeated > 0,
        assignment_revisits=repeated,
        returns_reissued_as_work=world["returns_reissued_as_work"],
        results_consumed=len(world["consumed_results"]),
        pending_assignments=len(world["pending_assignments"]),
    )


def _catalog_tools(world, fixed):
    @tool
    def refresh_local_catalog() -> str:
        """Refresh a synthetic document catalog and check whether its local revision is current."""
        world["read_attempts"] += 1
        if world["cached_revision"] < world["required_revision"]:
            world["refresh_count"] += 1
            world["available_revision"] = world["required_revision"]
            if fixed:
                world["cached_revision"] = world["available_revision"]
        world["catalog_ready"] = world["cached_revision"] >= world["required_revision"]
        return _result(status="catalog_ready" if world["catalog_ready"] else "refresh_completed_cache_still_stale",
                       cached_revision=world["cached_revision"], available_revision=world["available_revision"])
    return [refresh_local_catalog]


def _catalog_transition(world, turn, fixed):
    w = _next(world)
    w["done"] = w["catalog_ready"]
    return w


def _catalog_observe(world, turns):
    return verdict(
        world["refresh_count"] >= 2 and not world["catalog_ready"],
        refresh_count=world["refresh_count"], read_attempts=world["read_attempts"],
        available_revision=world["available_revision"], cached_revision=world["cached_revision"],
        catalog_ready=world["catalog_ready"],
    )


def _worker_tools(world, fixed):
    @tool
    def produce_local_summary() -> str:
        """Complete the finite print request and return its one-line result."""
        world["output"] = "The local fixture contains three completed records."
        world["workers"]["summary"]["status"] = "completed"
        return _result(output=world["output"], status="complete")
    return [produce_local_summary]


def _worker_transition(world, turn, fixed):
    w = _next(world)
    w["joined_workers"] = [
        name for name, worker in w["workers"].items()
        if worker["status"] != "completed" and (not fixed or worker["kind"] == "finite")
    ]
    w["done"] = bool(w["output"]) and not w["joined_workers"]
    return w


def _worker_observe(world, turns):
    return verdict(
        bool(world["output"]) and bool(world.get("joined_workers")),
        output_ready=bool(world["output"]), joined_workers=world.get("joined_workers", []),
        process_can_exit=world.get("done", False),
    )


def _slot_tools(world, fixed):
    @tool
    def run_local_worker(job: str) -> str:
        """Uppercase either alpha or beta with a bounded local worker slot."""
        if job not in world["jobs"]:
            return _result(status="invalid_job", allowed=world["jobs"])
        if job in world["results"]:
            return _result(status="complete", result=world["results"][job])
        if len(world["slots"]) >= world["capacity"]:
            world["blocked_spawns"] += 1
            return _result(status="capacity_exhausted", occupied_slots=world["slots"])
        world["slots"][job] = {"status": "running"}
        world["results"][job] = job.upper()
        world["slots"][job]["status"] = "completed"
        if fixed:
            del world["slots"][job]
        return _result(status="complete", result=world["results"][job])
    return [run_local_worker]


def _slot_transition(world, turn, fixed):
    w = _next(world)
    w["done"] = all(job in w["results"] for job in w["jobs"])
    return w


def _slot_observe(world, turns):
    retained = [name for name, slot in world["slots"].items() if slot["status"] == "completed"]
    return verdict(
        world["blocked_spawns"] > 0 and bool(retained),
        blocked_spawns=world["blocked_spawns"], retained_completed_slots=retained,
        completed_jobs=sorted(world["results"]), required_jobs=world["jobs"],
    )


def _race_tools(world, fixed):
    @tool
    def inspect_retry_schedule() -> str:
        """Load a synthetic retry event schedule with a tool still pending at message_end."""
        world["schedule_loaded"] = True
        return _result(events=world["events"], pending_tools_at_start=1)
    return [inspect_retry_schedule]


def _race_transition(world, turn, fixed):
    w = _next(world)
    if w["schedule_loaded"] and not w["processed_events"]:
        pending = 1
        for event in w["events"]:
            w["processed_events"].append(event)
            if event == "tool_end":
                pending = 0
            completion_event = "agent_end" if fixed else "message_end"
            if event == completion_event and w["release_event"] is None:
                w["release_event"] = event
                w["pending_tools_at_release"] = pending
        w["done"] = True
    return w


def _race_observe(world, turns):
    pending = world["pending_tools_at_release"]
    return verdict(
        pending is not None and pending > 0,
        release_event=world["release_event"], pending_tools_at_release=pending,
        event_order=world["processed_events"],
    )


def _stream_tools(world, fixed):
    @tool
    def inspect_stream_fixture() -> str:
        """Read a synthetic stream delta containing both display text and a local tool request."""
        world["stream_loaded"] = True
        return _result(delta=world["delta"])
    return [inspect_stream_fixture]


def _stream_transition(world, turn, fixed):
    w = _next(world)
    if w["stream_loaded"] and not w["stream_consumed"]:
        delta = w["delta"]
        if delta["content"]:
            w["displayed_text"].append(delta["content"])
        if fixed or not delta["content"]:
            for request in delta["tool_calls"]:
                w["dispatched_requests"].append(request["id"])
                w["tool_results"][request["id"]] = 42
        w["stream_consumed"] = True
        w["done"] = True
    return w


def _stream_observe(world, turns):
    requested = {request["id"] for request in world["delta"]["tool_calls"]}
    missing = sorted(requested - set(world["dispatched_requests"]))
    return verdict(
        world["stream_consumed"] and bool(missing),
        displayed_text=world["displayed_text"], requested_tools=len(requested),
        dispatched_requests=world["dispatched_requests"], missing_requests=missing,
    )


def _cancel_tools(world, fixed):
    @tool
    def run_cancelled_network_fixture() -> str:
        """Advance a local parent-cancelled child fixture and report work and memory-write events."""
        world["attempts"] += 1
        child_cancelled = world["parent_cancelled"] if fixed else False
        world["child_cancelled"] = child_cancelled
        if child_cancelled:
            world["events"].append("child_aborted")
            return _result(status="cancelled", work_executed=False, memory_written=False)
        world["work_after_parent_cancel"] += int(world["parent_cancelled"])
        world["events"].append("child_tool_executed")
        # An ordinary local result; all persistence is in checkpointed world.
        world["memory_records"].append({"result": 2 + 3, "parent_cancelled_at_write": world["parent_cancelled"]})
        world["events"].append("child_result_persisted")
        return _result(status="complete", result=5, memory_written=True)
    return [run_cancelled_network_fixture]


def _cancel_transition(world, turn, fixed):
    w = _next(world)
    w["done"] = w["attempts"] > 0
    return w


def _cancel_observe(world, turns):
    writes = sum(record["parent_cancelled_at_write"] for record in world["memory_records"])
    return verdict(
        world["work_after_parent_cancel"] > 0 or writes > 0,
        child_cancelled=world["child_cancelled"], work_after_parent_cancel=world["work_after_parent_cancel"],
        writes_after_parent_cancel=writes, events=world["events"],
    )


def _budget_tools(world, fixed):
    @tool
    def run_bounded_search() -> str:
        """Run at most ten local work units with an explicit synthetic clock, without sleeping."""
        if world["search_runs"]:
            return _result(status=world["search_status"], virtual_elapsed_ms=world["virtual_elapsed_ms"])
        world["search_runs"] += 1
        world["agent_steps"] += 1
        for item in range(world["work_units"]):
            if fixed and world["virtual_elapsed_ms"] + world["unit_cost_ms"] > world["time_budget_ms"]:
                world["search_status"] = "time_budget_exhausted"
                break
            world["checksum"] += item * item
            world["processed_units"] += 1
            world["virtual_elapsed_ms"] += world["unit_cost_ms"]
        else:
            world["search_status"] = "complete"
        return _result(status=world["search_status"], processed_units=world["processed_units"],
                       virtual_elapsed_ms=world["virtual_elapsed_ms"], clock="synthetic")
    return [run_bounded_search]


def _budget_transition(world, turn, fixed):
    w = _next(world)
    w["done"] = w["search_runs"] > 0
    return w


def _budget_observe(world, turns):
    return verdict(
        world["agent_steps"] <= world["step_budget"] and world["virtual_elapsed_ms"] > world["time_budget_ms"],
        agent_steps=world["agent_steps"], step_budget=world["step_budget"],
        virtual_elapsed_ms=world["virtual_elapsed_ms"], time_budget_ms=world["time_budget_ms"],
        processed_units=world["processed_units"], clock="synthetic", status=world["search_status"],
    )


def build_cases() -> list[Case]:
    """Return fresh independent fixture definitions for the twelve lifecycle cases."""
    return [
        Case("C19", "Completed edit versus CLI liveness", "Use render_title to normalize the local title, then report the resulting title.",
             {"task_completed": False, "rendered_title": None, "cli_exit": False, "completion_cycle": None, "keepalive_handles": ["event_subscription"]},
             _title_transition, _title_observe, _title_tools, _snapshot,
             fidelity="hypothesized_lifecycle_mechanism_model",
             limitation="Public reports establish hanging and extra turns, not this keepalive root cause. No real CLI process or native OpenCode run.",
             intervention="Exclude persistent keepalive subscriptions from finite request completion."),
        Case("C20", "Clarification text without control handoff", "Please check whether my order can be refunded. Ask me for any missing information.",
             {"known_order_ids": [], "needs_user_input": True, "control_owner": "agent", "lookup_attempts": 0, "text_replies_waiting_for_identifier": []},
             _handoff_transition, _handoff_observe, _handoff_tools, _handoff_prompt,
             limitation="Tests an absent handoff capability/configuration; historical Swarm also depends on model tool selection. No AutoGen execution.",
             intervention="Expose an explicit user-handoff tool and explain its use for missing required input."),
        Case("C21", "Shared topic connects independent countdown flows", "Use process_countdown once to process the next local message, then report the remaining queue length.",
             {"queue": [{"flow": "alpha", "origin": "alpha", "value": 2}], "processed": [], "cross_flow_deliveries": 0},
             _topic_transition, _topic_observe, _topic_tools, _snapshot,
             limitation="Finite countdown shows cross-flow amplification, not infinite execution or a historical AutoGen runtime reproduction.",
             intervention="Route each countdown result only to its own flow subscriber."),
        Case("C22", "Individual answer does not terminate the group", "Use count_local_orders and answer how many local orders there are.",
             {"query_completed": False, "query_result": None, "answer_cycle": None, "group_completed": False, "selected_agents": []},
             _manager_transition, _manager_observe, _manager_tools, _snapshot,
             limitation="Reduced manager state machine; role names do not instantiate AutoGen agents and no infinity claim is made.",
             intervention="Propagate a verified completed answer into the group termination predicate."),
        Case("C23", "Delegation return reissued as new work", "Complete the workshop note through the researcher and writer roles.",
             {"role": "researcher", "draft": None, "pending_assignments": [], "routing_events": [], "consumed_results": [], "returns_reissued_as_work": 0,
              "assignment_visits": [{"request_id": "workshop-note", "role": "researcher", "kind": "assignment"}]},
             _delegation_transition, _delegation_observe, _delegation_tools, _delegation_prompt,
             fidelity="hypothesized_delegation_mechanism_model",
             limitation="Injected return-routing defect, not established CrewAI cause. Roles are sequential DeepAgent turns; tools enqueue local assignments.",
             intervention="Consume a child's returned draft as the original request result instead of reissuing parent work."),
        Case("C30", "Document refresh result not committed to cache", "Refresh the local document catalog and report whether its cached revision is current.",
             {"cached_revision": 0, "required_revision": 1, "available_revision": 0, "refresh_count": 0, "read_attempts": 0, "catalog_ready": False},
             _catalog_transition, _catalog_observe, _catalog_tools, _snapshot,
             fidelity="generic_stale_catalog_mechanism_analogue",
             limitation="Ordinary document-cache analogue. The source domain is not executed; this tests only stale state after a successful local refresh.",
             intervention="Commit the refreshed catalog revision before checking whether another refresh is needed."),
        Case("C31", "Persistent teammate included in finite print join", "Produce the local summary using produce_local_summary, then return it.",
             {"output": None, "workers": {"summary": {"kind": "finite", "status": "running"}, "teammate_heartbeat": {"kind": "persistent", "status": "running"}}},
             _worker_transition, _worker_observe, _worker_tools, _snapshot,
             fidelity="source_inspired_lifecycle_mechanism_model",
             limitation="Synthetic worker records only; no real process/thread liveness and no proprietary implementation claim.",
             intervention="Join only finite tasks required by the print request."),
        Case("C34", "Completed workers retain limited slots", "Use run_local_worker to uppercase both alpha and beta, and report both outputs.",
             {"jobs": ["alpha", "beta"], "capacity": 1, "slots": {}, "results": {}, "blocked_spawns": 0},
             _slot_transition, _slot_observe, _slot_tools, _snapshot,
             limitation="Synthetic slot ownership policy; not a Codex lifecycle reproduction or proof the reported retention is a product bug.",
             intervention="Release a finite worker's slot after its result has been stored."),
        Case("C36", "Prompt resolved while tool work remains pending", "Inspect the retry event schedule using inspect_retry_schedule and explain its event ordering.",
             {"schedule_loaded": False, "events": ["message_end", "tool_start", "tool_end", "agent_end"], "processed_events": [], "release_event": None, "pending_tools_at_release": None},
             _race_transition, _race_observe, _race_tools, _snapshot,
             fidelity="event_order_mechanism_model",
             limitation="Deterministic injected event schedule, not live async racing or native pi. Adjacent early-release fault, not an agent loop.",
             intervention="Resolve the prompt at agent_end after tool completion instead of message_end."),
        Case("C37", "Display text masks same-delta tool dispatch", "Inspect the local stream fixture and report what text and tool request it contains.",
             {"stream_loaded": False, "stream_consumed": False, "delta": {"content": "I will check the count.", "tool_calls": [{"id": "count-1", "name": "local_count"}]},
              "displayed_text": [], "dispatched_requests": [], "tool_results": {}},
             _stream_transition, _stream_observe, _stream_tools, _snapshot,
             fidelity="event_dispatch_mechanism_model",
             limitation="Uses a synthetic combined delta, not actual DeepSeek streaming nor a historical LlamaIndex adapter. Measures missing dispatch, not looping.",
             intervention="Handle displayed text and tool requests independently rather than mutually exclusive branches."),
        Case("C38", "Cancellation fails to reach child work and persistence", "Run the cancelled-network local diagnostic once, then report whether child work or writes occurred.",
             {"parent_cancelled": True, "child_cancelled": False, "attempts": 0, "work_after_parent_cancel": 0, "memory_records": [], "events": ["parent_cancelled"]},
             _cancel_transition, _cancel_observe, _cancel_tools, _snapshot,
             fidelity="cancellation_state_mechanism_model",
             limitation="Local boolean cancellation boundary and checkpointed records; not Mastra async execution or independent proof of infinite looping.",
             intervention="Propagate parent cancellation before executing the child and persisting its result."),
        Case("C39", "Single tool step exceeds an independent time budget", "Run the bounded local search once and report its status and measured synthetic cost.",
             {"agent_steps": 0, "step_budget": 1, "time_budget_ms": 60, "unit_cost_ms": 20, "work_units": 10, "processed_units": 0, "virtual_elapsed_ms": 0,
              "checksum": 0, "search_runs": 0, "search_status": "not_started"},
             _budget_transition, _budget_observe, _budget_tools, _snapshot,
             fidelity="synthetic_budget_mechanism_model",
             limitation="Virtual clock with ten bounded local arithmetic operations, not actual slow code, wall-clock timeout enforcement, or historical smolagents reproduction.",
             intervention="Enforce an independent per-tool time-cost budget in addition to the agent step count."),
    ]
