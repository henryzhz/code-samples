"""Bounded context/controller models inspired by public reports.

These are deliberately small synthetic controllers, not ports of the historical
products. Deep Agents supplies real turns; injected usage, errors, queues and
boundaries below are fixture inputs and are never represented as provider facts.
Every changing value is returned in ``world`` for the outer graph to checkpoint.
"""

from copy import deepcopy
import hashlib
import json

from .spec import Case, verdict


def _next(world, turn):
    result = deepcopy(world)
    result["cycles"] = result.get("cycles", 0) + 1
    result["last_turn"] = {
        "status": turn.get("status"),
        "finish_reason": turn.get("finish_reason"),
        "answer_chars": len(turn.get("answer") or ""),
    }
    return result


def _result(observed, **measurements):
    return verdict(observed, evidence_scope="synthetic_controller", **measurements)


def _cache(world, turn, fixed):
    result = _next(world, turn)
    if result["phase"] == "warm_cache":
        # Actual aliasing is exercised here, on a checkpoint-local copy. Only
        # the warmed request gets a reduced token budget when isolation works.
        parameters = (dict(result["main_parameters"]) if fixed
                      else result["main_parameters"])
        parameters["max_tokens"] = 1
        result["warming_parameters"] = dict(parameters)
        result["phase"] = "main_request"
        result["parameter_history"].append(dict(result["main_parameters"]))
        return result
    finish = turn.get("finish_reason")
    result["continuations"].append({
        "max_tokens": result["main_parameters"]["max_tokens"],
        "finish_reason": finish,
        "answer_chars": len(turn.get("answer") or ""),
    })
    result["done"] = finish != "length"
    result["outcome"] = "length_continuation" if not result["done"] else "response_ended"
    return result


def _cache_observe(world, turns):
    truncated = [item for item in world["continuations"]
                 if item["max_tokens"] == 1 and item["finish_reason"] == "length"]
    return _result(len(truncated) >= 2,
                   shared_parameters_changed=world["main_parameters"]["max_tokens"] != 512,
                   observed_length_responses=len(truncated), continuations=world["continuations"],
                   historical_prefill_not_implemented=True)


def _stale_usage(world, turn, fixed):
    result = _next(world, turn)
    assistant = result["last_assistant"]
    stale = assistant["timestamp"] <= result["latest_compaction"]
    if fixed and stale:
        result.update(done=True, outcome="stale_usage_ignored")
        return result
    if assistant["usage"] >= result["threshold"]:
        result["compactions"].append({
            "usage": assistant["usage"], "usage_timestamp": assistant["timestamp"],
            "boundary": result["latest_compaction"],
            "actual_context": result["current_context"], "stale": stale,
        })
        result["current_context"] = max(1, result["current_context"] // 2)
        result["latest_compaction"] += 1
        result["outcome"] = "compacted"
    else:
        result.update(done=True, outcome="below_threshold")
    return result


def _stale_observe(world, turns):
    unnecessary = [item for item in world["compactions"]
                   if item["stale"] and item["actual_context"] < world["threshold"]]
    return _result(len(unnecessary) >= 2, stale_compactions=len(unnecessary),
                   checks=world["compactions"], usage_is_injected=True)


def _error_path(world, turn, fixed):
    result = _next(world, turn)
    estimate = result["last_successful_usage"] + result["tail_units"]
    if result["injected_api_error"] and not fixed:
        result["skipped_checks"].append({"estimate": estimate, "reason": "error_early_return"})
        result["outcome"] = "retry_without_maintenance"
        return result
    if estimate >= result["threshold"]:
        after = estimate // 3
        result["compactions"].append({"before": estimate, "after": after})
        result["last_successful_usage"] = after
        result["tail_units"] = 0
    result.update(done=True, outcome="blocked_api_error" if result["injected_api_error"]
                  else "maintenance_complete")
    return result


def _error_observe(world, turns):
    high = [item for item in world["skipped_checks"] if item["estimate"] >= world["threshold"]]
    return _result(len(high) >= 2, skipped_high_context_checks=len(high),
                   compactions=world["compactions"], api_error_is_injected=True,
                   provider_recovery_verified=False)


def _summary_input(world, turn, fixed):
    result = _next(world, turn)
    outputs = result["tool_results"]
    if fixed:
        outputs = [text[:result["tool_result_cap"]] for text in outputs]
    size = result["summary_overhead"] + sum(len(text) for text in outputs)
    accepted = size <= result["summary_input_limit"]
    result["summary_attempts"].append({"input_chars": size,
                                        "result": "accepted" if accepted else "synthetic_overflow"})
    result["done"] = accepted
    result["outcome"] = "summary_input_accepted" if accepted else "summary_input_overflow"
    return result


def _summary_observe(world, turns):
    failures = [item for item in world["summary_attempts"] if item["result"] == "synthetic_overflow"]
    return _result(bool(failures), summary_attempts=world["summary_attempts"],
                   summary_input_limit=world["summary_input_limit"],
                   overflow_is_local_character_gate=True, infinite_loop_established=False)


def _reopen_terminal(world, turn, fixed):
    result = _next(world, turn)
    completed = turn.get("status") == "completed" and turn.get("finish_reason") not in ("length", "tool_calls")
    over_configured_window = result["reported_usage"] >= result["configured_window"]
    if over_configured_window:
        result["compactions"].append({"reported_usage": result["reported_usage"],
                                       "response_completed": completed})
        if not (fixed and completed):
            if completed:
                result["terminal_reopenings"] += 1
            result["outcome"] = "retry_after_compaction"
            return result
    result["done"] = True
    result["outcome"] = "completed" if completed else "blocked_incomplete_response"
    if completed:
        result["terminal_events"].append("turn_end")
    return result


def _terminal_observe(world, turns):
    return _result(world["terminal_reopenings"] > 0,
                   terminal_reopenings=world["terminal_reopenings"],
                   terminal_events=world["terminal_events"], reported_usage_is_injected=True)


def _collapse(world, turn, fixed):
    result = _next(world, turn)
    before = len(result["text"])
    after_text = turn.get("answer") or ""
    after = len(after_text)
    # An errored/empty response is not a successful zero-length summary.
    valid = turn.get("status") == "completed" and bool(after_text.strip())
    if not valid:
        decision = "blocked_model_response"
        result["done"] = True
    elif after <= result["max_chars"]:
        decision = "within_limit"
        result["done"] = True
    elif fixed and after >= before:
        decision = "blocked_nonreducing_summary"
        result["done"] = True
    else:
        decision = "retry"
    result["reductions"].append({"before": before, "after": after, "decision": decision})
    result["text"] = after_text
    result["outcome"] = decision
    return result


def _collapse_observe(world, turns):
    stagnant = [item for item in world["reductions"]
                if item["decision"] == "retry" and item["after"] >= item["before"]]
    return _result(len(stagnant) >= 2, nonreducing_retries=len(stagnant),
                   reductions=world["reductions"], length_measured_from_actual_answer=True,
                   semantic_summary_quality_assessed=False)


def _headroom(world, turn, fixed):
    result = _next(world, turn)
    free = result["window"] - result["fixed_overhead"] - result["summary_units"]
    if free >= result["required_headroom"]:
        result.update(done=True, outcome="sufficient_headroom")
        return result
    possible = result["window"] - result["fixed_overhead"] - result["minimum_summary"]
    if fixed and possible < result["required_headroom"]:
        result.update(done=True, outcome="blocked_insufficient_headroom")
        return result
    result["summary_units"] = max(result["minimum_summary"], result["summary_units"] // 2)
    after = result["window"] - result["fixed_overhead"] - result["summary_units"]
    result["compactions"].append({"free_before": free, "free_after": after})
    result["done"] = after >= result["required_headroom"]
    result["outcome"] = "sufficient_headroom" if result["done"] else "retry_compaction"
    return result


def _headroom_observe(world, turns):
    insufficient = [item for item in world["compactions"]
                    if item["free_after"] < world["required_headroom"]]
    return _result(len(insufficient) >= 2, insufficient_compactions=len(insufficient),
                   compactions=world["compactions"], units_are_synthetic=True)


def _completion_summary(world, turn, fixed):
    result = _next(world, turn)
    if result["phase"] == "compact":
        # Exercise a real lossy serialization boundary, not a verdict toggle.
        encoded = [{"id": item["id"], "title": item["title"],
                    **({"completed": item["completed"]} if fixed else {})}
                   for item in result["task_ledger"]]
        result["serialized_summary"] = json.dumps(encoded, sort_keys=True)
        result["restored_tasks"] = [dict(item, completed=item.get("completed", False))
                                    for item in json.loads(result["serialized_summary"])]
        completed_ids = {item["id"] for item in result["task_ledger"] if item["completed"]}
        result["reactivated_tasks"] = [item["id"] for item in result["restored_tasks"]
                                        if item["id"] in completed_ids and not item["completed"]]
        result["phase"] = "resume"
        result["done"] = not any(not item["completed"] for item in result["restored_tasks"])
        result["outcome"] = "restored_pending_tasks" if not result["done"] else "completion_preserved"
    else:
        # This fixture stops after exposing the reopened task to one real turn.
        result.update(done=True, outcome="restored_ledger_inspected")
    return result


def _completion_observe(world, turns):
    return _result(bool(world["reactivated_tasks"]), reactivated_tasks=world["reactivated_tasks"],
                   restored_tasks=world["restored_tasks"], agent_reexecution_not_established=True)


def _refill(world, turn, fixed):
    result = _next(world, turn)
    if result["context_units"] < result["threshold"]:
        result.update(done=True, outcome="below_threshold")
        return result
    before = result["context_units"]
    result["context_units"] = result["compacted_units"]
    for event in result["pending_restorations"]:
        result["context_units"] += event["units"]
        result["restoration_applications"].append(event["id"])
    if fixed:
        result["pending_restorations"] = []
    result["compactions"].append({"before": before, "after": result["context_units"]})
    result["done"] = result["context_units"] < result["threshold"]
    result["outcome"] = "below_threshold" if result["done"] else "refilled_after_compaction"
    return result


def _refill_observe(world, turns):
    applications = world["restoration_applications"]
    repeated = len(applications) - len(set(applications))
    return _result(repeated > 0, duplicate_restoration_applications=repeated,
                   restoration_applications=applications, compactions=world["compactions"],
                   queue_replay_is_proposed_fixture_mechanism=True)


def _failure_hook(world, turn, fixed):
    result = _next(world, turn)
    failure = bool(result["injected_api_error"])
    hook = "StopFailure" if fixed and failure else "Stop"
    blocking = hook == "Stop" and result["hook_blocks_completion"]
    result["hook_events"].append({"terminal_event": "api_error" if failure else "completed",
                                  "hook": hook, "blocking": blocking})
    if blocking:
        result["feedback_queue"].append("Synthetic Stop hook requests further review.")
        result["outcome"] = "restart_from_stop_feedback"
    else:
        result.update(done=True, outcome="blocked_api_error" if failure else "completed")
    return result


def _failure_hook_observe(world, turns):
    wrong_route = [event for event in world["hook_events"]
                   if event["terminal_event"] == "api_error" and event["hook"] == "Stop" and event["blocking"]]
    return _result(len(wrong_route) >= 2, error_stop_feedback_cycles=len(wrong_route),
                   hook_events=world["hook_events"], api_error_is_injected=True)


def _blocking_hook(world, turn, fixed):
    result = _next(world, turn)
    if not result["hook_blocks_completion"]:
        result.update(done=True, outcome="completed")
        return result
    result["hook_blocks"] += 1
    bounded = fixed and result["hook_blocks"] >= result["max_hook_blocks"]
    result["hook_events"].append({"block_number": result["hook_blocks"], "rescheduled": not bounded})
    result["done"] = bounded
    result["outcome"] = "blocked_stop_hook" if bounded else "restart_from_stop_feedback"
    return result


def _blocking_observe(world, turns):
    beyond = [event for event in world["hook_events"]
              if event["block_number"] > world["max_hook_blocks"] and event["rescheduled"]]
    return _result(bool(beyond), hook_blocks=world["hook_blocks"],
                   max_hook_blocks=world["max_hook_blocks"], hook_events=world["hook_events"])


def _skill_fork(world, turn, fixed):
    result = _next(world, turn)
    requested = result["requested_skill"]
    recursive = requested in result["active_skills"]
    if fixed and recursive:
        result.update(done=True, outcome="blocked_recursive_skill")
        return result
    result["fork_edges"].append({"parent": result["active_skills"][-1],
                                 "child": requested, "recursive": recursive})
    result["active_skills"].append(requested)
    result["requested_skill"] = result["child_entrypoint"]
    result["outcome"] = "child_entrypoint_scheduled"
    return result


def _fork_observe(world, turns):
    recursive = [edge for edge in world["fork_edges"] if edge["recursive"]]
    return _result(bool(recursive), recursive_edges=len(recursive),
                   fork_edges=world["fork_edges"], actual_subagents_spawned=0)


def _failed_request(world, turn, fixed):
    result = _next(world, turn)
    request = result["request"]
    encoded = json.dumps(request, sort_keys=True).encode()
    size = request["overhead"] + len(request["content"])
    overflow = size > result["input_limit"]
    result["request_attempts"].append({
        "fingerprint": hashlib.sha256(encoded).hexdigest(), "input_chars": size,
        "result": "synthetic_overflow" if overflow else "accepted",
    })
    if overflow and fixed:
        available = max(0, result["input_limit"] - request["overhead"])
        request["content"] = request["content"][:min(result["retry_content_cap"], available)]
    result["done"] = not overflow
    result["outcome"] = "retry_request" if overflow else "request_input_accepted"
    return result


def _request_observe(world, turns):
    failed = [item["fingerprint"] for item in world["request_attempts"]
              if item["result"] == "synthetic_overflow"]
    repeats = len(failed) - len(set(failed))
    return _result(repeats > 0, duplicate_failed_requests=repeats,
                   request_attempts=world["request_attempts"],
                   overflow_is_local_character_gate=True)


def _retention(world, turn, fixed):
    result = _next(world, turn)
    visible = result["visible_messages"]
    boundary = result["keep_local_index"]
    # The defective fixture mixes an old global index with a new local slice.
    # This is one explicit candidate mechanism, not a claim about pi internals.
    start = 0 if fixed else result["previous_keep_global_index"]
    result["summary_covers"] = list(result["previous_summary_covers"]) + [
        item["id"] for item in visible[start:boundary]]
    result["retained_ids"] = [item["id"] for item in visible[boundary:]]
    result.update(done=True, outcome="second_compaction_applied")
    return result


def _retention_observe(world, turns):
    if world.get("outcome") != "second_compaction_applied":
        # Output slots are empty at the seed boundary, not evidence that the
        # still-visible input messages have already disappeared.
        return _result(False, evaluation_status="not_yet_compacted",
                       missing_required_ids=[], subsequent_agent_loop_not_established=True)
    represented = set(world["summary_covers"] + world["retained_ids"])
    missing = sorted(set(world["required_ids"]) - represented)
    return _result(bool(missing), evaluation_status="compaction_evaluated", missing_required_ids=missing,
                   summary_covers=world["summary_covers"], retained_ids=world["retained_ids"],
                   subsequent_agent_loop_not_established=True)


def _state_prompt(world, fixed):
    visible = {key: value for key, value in world.items()
               if key not in {"synthetic_preconditions", "last_turn"}}
    return ("This is a local synthetic controller fixture. Describe the current state briefly; "
            "do not claim its injected measurements came from your API call. State: "
            + json.dumps(visible, sort_keys=True))


def build_cases():
    cases = []

    def add(case_id, title, transition, observe, initial, precondition, intervention,
            *, prompt=_state_prompt, model_options=None, task=None, limitation=""):
        cases.append(Case(
            id=case_id, title=title,
            task=task or "Inspect the synthetic log-controller state and report a concise status.",
            initial={"done": False, "cycles": 0, "outcome": "not_started",
                     "synthetic_preconditions": precondition, **initial},
            transition=transition, observe=observe, prompt=prompt,
            model_options=model_options or (lambda world, fixed: {}),
            intervention=intervention,
            limitation=("Source-inspired local mechanism model, not execution of the historical product/version. "
                        "Fixture controller state is synthetic; real model turns do not establish the source report. "
                        + limitation),
        ))

    add("C09", "Cache warming aliases main request parameters", _cache, _cache_observe,
        {"phase": "warm_cache", "main_parameters": {"max_tokens": 512},
         "warming_parameters": {}, "parameter_history": [], "continuations": []},
        "A cache-warmer mutates max_tokens on a shared dict before subsequent real model calls.",
        "Copy the parameter dictionary before changing the warming request.",
        model_options=lambda world, fixed: dict(world["main_parameters"]),
        task="Write one short complete sentence confirming that the synthetic log contains status=ok.",
        limitation="Historical Anthropic prefill is not emulated. Observe actual finish_reason before claiming repeated truncation.")
    add("C10", "Compaction reuses pre-boundary usage", _stale_usage, _stale_observe,
        {"latest_compaction": 10, "last_assistant": {"timestamp": 5, "usage": 950},
         "recovery_entry_timestamp": 11, "current_context": 120, "threshold": 800, "compactions": []},
        "Compaction and a recovery entry already exist; no newer assistant usage is available. Usage units are injected.",
        "Reject assistant usage at or before the current compaction boundary.")
    add("C11", "Error branch skips context maintenance", _error_path, _error_observe,
        {"injected_api_error": "synthetic_529", "last_successful_usage": 900,
         "tail_units": 100, "threshold": 800, "skipped_checks": [], "compactions": []},
        "A persistent API error and last successful usage are preloaded; no live API failure is induced.",
        "Run the available usage estimate on the error path before reporting the persistent blocker.")
    add("C12", "Recovery summary input also exceeds its limit", _summary_input, _summary_observe,
        {"tool_results": ["log record; " * 20], "summary_overhead": 20,
         "summary_input_limit": 100, "tool_result_cap": 32, "summary_attempts": []},
        "A local character-budget gate models summary request rejection; it is not a provider token limit.",
        "Cap tool-result content before constructing the summary request.")
    add("C13", "Postprocessing reopens completed responses", _reopen_terminal, _terminal_observe,
        {"reported_usage": 1200, "configured_window": 1000, "provider_window": 2000,
         "terminal_reopenings": 0, "terminal_events": [], "compactions": []},
        "Configured and provider windows differ; reported usage is injected, completion comes from the actual turn.",
        "Keep completed responses out of the compaction retry path.")
    add("C14", "Collapse retries without measurable reduction", _collapse, _collapse_observe,
        {"text": "The synthetic log reports a successful check. " * 6, "max_chars": 64, "reductions": []},
        "A synthetic summary workflow must reduce a text to a character budget. Returned answer length is measured.",
        "Stop with an explicit blocker when a still-oversized summary does not shrink.",
        prompt=lambda world, fixed: f"Summarize this log in at most {world['max_chars']} characters: {world['text']}",
        task="Produce a short accurate summary of the provided synthetic log.",
        limitation="A negative run is expected when the real model successfully shortens the text; length does not prove semantic quality.")
    add("C15", "Fixed overhead prevents useful compaction headroom", _headroom, _headroom_observe,
        {"window": 100, "fixed_overhead": 90, "summary_units": 20,
         "minimum_summary": 10, "required_headroom": 20, "compactions": []},
        "Synthetic fixed overhead plus the minimum retained summary leave less room than the next operation requires.",
        "Check whether the required headroom is attainable before retrying compaction.",
        limitation="Fixed-overhead arithmetic is a candidate mechanism, not a verified Claude Code implementation.")
    add("C16", "Summary serialization loses completion state", _completion_summary, _completion_observe,
        {"phase": "compact", "task_ledger": [{"id": "verify-log", "title": "Verify synthetic log", "completed": True}],
         "restored_tasks": [], "reactivated_tasks": [], "serialized_summary": ""},
        "A synthetic task is already complete. A serializer drops its status and a restore path defaults missing status to pending.",
        "Include completed status in the serialized task summary.",
        limitation="The source Codex PR was a proposal; this deterministic serializer is not a port of that prompt or proof of its efficacy.")
    add("C25", "Restoration queue refills compacted context", _refill, _refill_observe,
        {"context_units": 95, "threshold": 80, "compacted_units": 40,
         "pending_restorations": [{"id": "restore-1", "units": 45}],
         "restoration_applications": [], "compactions": []},
        "An unconsumed synthetic restoration event can be reapplied after each successful compaction.",
        "Consume a restoration event atomically after applying it.",
        limitation="Only a public repair lead exists; queue replay is our explicit hypothetical fixture, not the reported root cause.")
    add("C26", "API failure routes into blocking Stop feedback", _failure_hook, _failure_hook_observe,
        {"injected_api_error": "synthetic_api_failure", "hook_blocks_completion": True,
         "hook_events": [], "feedback_queue": []},
        "A preloaded API failure reaches a synthetic blocking Stop hook; the actual model request need not fail.",
        "Route API failures to StopFailure instead of the blocking Stop feedback path.",
        limitation="Hooks are local state transitions, not actual Claude Code hooks.")
    add("C27", "Blocking Stop feedback has no reentry bound", _blocking_hook, _blocking_observe,
        {"hook_blocks_completion": True, "hook_blocks": 0, "max_hook_blocks": 2, "hook_events": []},
        "A local hook blocks each completion; the outer runner still bounds the whole experiment.",
        "Stop scheduling model turns after the configured consecutive hook-block bound.",
        limitation="The limit of two is a fixture choice, not an asserted historical product value.")
    add("C28", "Fork entrypoint recursively selects the active skill", _skill_fork, _fork_observe,
        {"active_skills": ["review-log"], "requested_skill": "review-log",
         "child_entrypoint": "review-log", "fork_edges": []},
        "A synthetic dispatch request selects the same skill as the active stack; only stack data is expanded.",
        "Reject a skill dispatch already present on the active stack.",
        limitation="No subagents are spawned. This tests dispatch-state recursion, not Deep Agents or Claude Code delegation behavior.")
    add("C29", "Overflow retry preserves the identical failed request", _failed_request, _request_observe,
        {"request": {"overhead": 20, "content": "synthetic log; " * 12},
         "input_limit": 100, "retry_content_cap": 60, "request_attempts": []},
        "A local character-budget gate rejects an application request; identical request hashes make unchanged retries observable.",
        "Reduce the request content after overflow before retrying.",
        limitation="No oversized live provider request is issued; this is a miniature retry controller.")
    add("C35", "Second compaction mixes global and local retention indices", _retention, _retention_observe,
        {"required_ids": ["m1", "m2", "m3", "m4", "m5", "m6"],
         "previous_summary_covers": ["m1", "m2"], "previous_keep_global_index": 2,
         "visible_messages": [{"id": f"m{i}", "content": f"synthetic fact {i}"} for i in range(3, 7)],
         "keep_local_index": 2, "summary_covers": [], "retained_ids": []},
        "The first compaction already covers m1/m2. A second compaction partitions the surviving local list.",
        "Use local slice coordinates when choosing newly summarized messages.",
        limitation="The index mismatch is a candidate fixture mechanism. Message loss is adjacent evidence, not proof of a subsequent agent loop.")
    return cases
