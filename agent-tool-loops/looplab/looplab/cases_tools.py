"""Bounded, source-inspired tool fixtures; never historical product reproductions.

Tools operate only on the runner's copied ``world``. No shell, network, provider,
or host filesystem access occurs in a tool. Oracles inspect observations, never
the intervention flag. Each fixture names its motivating public report.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
import sqlite3

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from .spec import Case, verdict
from .sources import SOURCE_URLS


LAB_OUTPUT_LIMIT = 1800


def _event(world, name, **data):
    world.setdefault("events", []).append({"name": name, **data})


def _events(world, name):
    return [e for e in world.get("events", []) if e["name"] == name]


def _finish(world, turn, fixed):
    result = deepcopy(world)
    result["done"] = True
    return result


def _proposed(turns, name):
    return [c for t in turns for c in t.get("tool_calls", []) if c["name"] == name]


def _repeat_measure(world, names):
    events = [e for e in world.get("events", []) if e["name"] in names]
    counts = Counter((e["name"], json.dumps(e.get("args", {}), sort_keys=True)) for e in events)
    return len(events), max(counts.values(), default=0)


def _report_content():
    # One long, ordinary log line. Status is available in the bounded prefix.
    return "run-001 status=SUCCESS exit_code=0 records=12; " + "validated record; " * 300


def _offload(world, content, *, reading, fixed):
    if len(content) > LAB_OUTPUT_LIMIT and not (reading and fixed):
        path = f"/saved/report-{len(world['saved']) + 1}.log"
        world["saved"][path] = content
        _event(world, "offload", path=path, reading=reading, chars=len(content))
        return f"Result saved to {path}. Use read_saved_report to inspect the saved content."
    if len(content) > LAB_OUTPUT_LIMIT:
        content = content[: LAB_OUTPUT_LIMIT - 80] + "\n[truncated to local character budget; status is in the header]"
    return content


def _c01_tools(world, fixed):
    @tool
    def fetch_run_report() -> str:
        """Fetch the local run-001 validation log, offloading large results."""
        _event(world, "fetch_run_report")
        return _offload(world, _report_content(), reading=False, fixed=fixed)

    @tool
    def read_saved_report(path: str, offset: int = 0, limit: int = 100) -> str:
        """Read saved local log lines; offset is zero-based and limit counts lines."""
        _event(world, "read_saved_report", args={"path": path, "offset": offset, "limit": limit})
        if path not in world["saved"]:
            return "File not found in the local saved-report collection."
        content = "\n".join(world["saved"][path].splitlines()[max(0, offset):max(0, offset) + max(0, min(limit, 100))])
        return _offload(world, content, reading=True, fixed=fixed)

    return [fetch_run_report, read_saved_report]


def _c01_observe(world, turns):
    reads = len(_events(world, "read_saved_report"))
    reoffloads = sum(e["reading"] for e in _events(world, "offload"))
    return verdict(reads >= 2 and reoffloads >= 2, read_calls=reads, read_results_offloaded=reoffloads,
                   saved_objects=len(world["saved"]), lab_character_limit=LAB_OUTPUT_LIMIT)


def _c02_tools(world, fixed):
    @tool
    def local_archive_note(note_id: str) -> str:
        """Archive a note in this fixture's in-memory notebook; approval is required."""
        _event(world, "local_archive_note", args={"note_id": note_id})
        world["archived"] = note_id
        return f"Archived local note {note_id}."
    return [local_archive_note]


def _c02_observe(world, turns):
    proposed = _proposed(turns, "local_archive_note")
    rejections = sum(len(t.get("approval_rejections", [])) for t in turns)
    renewed = set()
    for turn_index, turn in enumerate(turns):
        for rejection in turn.get("approval_rejections", []):
            boundary = rejection.get("message_index")
            if boundary is None:
                continue
            refused = {(a.get("name"), json.dumps(a.get("args", {}), sort_keys=True))
                       for a in rejection.get("actions", [])}
            for call_index, c in enumerate(turn.get("tool_calls", [])):
                if (c["name"] == "local_archive_note" and c.get("message_index", -1) > boundary
                        and (c["name"], json.dumps(c.get("args", {}), sort_keys=True)) in refused):
                    renewed.add((turn_index, call_index))
    return verdict(bool(renewed),
                   proposed_archive_calls=len(proposed), rejection_events=rejections,
                   proposals_after_same_operation_rejected=len(renewed),
                   executed_archive_calls=len(_events(world, "local_archive_note")))


def _c03_tools(world, fixed):
    @tool
    def search_handbook(query: str) -> str:
        """Search the small local staff handbook for opening hours."""
        _event(world, "search_handbook", args={"query": query})
        return "Handbook section 2: The local library opens at 09:00 Monday through Friday."
    return [search_handbook]


def _c03_observe(world, turns):
    n = len(_events(world, "search_handbook"))
    tool_messages = {(i, c["message_index"]) for i, t in enumerate(turns) for c in t.get("tool_calls", [])
                     if c.get("message_index") is not None}
    search_messages = {(i, c["message_index"]) for i, t in enumerate(turns) for c in t.get("tool_calls", [])
                       if c.get("message_index") is not None and c["name"] == "search_handbook"}
    return verdict(n >= 2 and len(search_messages) >= 2, successful_search_calls=n,
                   assistant_messages_proposing_search=len(search_messages), model_calls=sum(t.get("model_calls", 0) for t in turns),
                   assistant_messages_proposing_tools=len(tool_messages),
                   reached_model_or_tool_budget=any(t.get("status") == "budget_exhausted" for t in turns),
                   statuses=[t.get("status") for t in turns], interpretation="repeated search after sufficient local evidence; inspect model configuration")


def _c04_tools(world, fixed):
    @tool
    def query_local_sales(sql: str) -> str:
        """Run one SELECT on a local sales(product TEXT, units INTEGER) table."""
        if not sql.lstrip().lower().startswith("select"):
            _event(world, "query_local_sales", args={"sql": sql}, ok=False)
            return "Only a single SELECT query is supported."
        # A reproducible, one-time fixture fault models recovery from a namespace error.
        if not world.get("namespace_error_seen"):
            world["namespace_error_seen"] = True
            _event(world, "query_local_sales", args={"sql": sql}, ok=False)
            return "Namespace unavailable on first connection. Local default table sales is now available."
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE sales(product TEXT, units INTEGER)")
            con.executemany("INSERT INTO sales VALUES (?, ?)", world["sales"])
            cursor = con.execute(sql)
            rows = cursor.fetchmany(20)
            columns = [d[0] for d in cursor.description]
        except sqlite3.Error as exc:
            _event(world, "query_local_sales", args={"sql": sql}, ok=False)
            return f"Query error: {exc}"
        finally:
            con.close()
        _event(world, "query_local_sales", args={"sql": sql}, ok=True, rows=rows)
        response = {"columns": columns, "rows": rows}
        if fixed:
            response.update(status="success", complete_result=True)
        return json.dumps(response)
    return [query_local_sales]


def _c04_observe(world, turns):
    events = _events(world, "query_local_sales")
    successful = [e for e in events if e["ok"]]
    # Normalize lexical whitespace/case while preserving quoted identifiers and
    # values. Equal result sets alone do not establish repeated SQL execution.
    def normalized(sql):
        tokens = re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\w+|[^\w\s]", sql.strip().rstrip(";"))
        return tuple(t if t.startswith(("'", '"')) else t.casefold() for t in tokens)
    identical = Counter(normalized(e["args"]["sql"]) for e in successful)
    return verdict(max(identical.values(), default=0) >= 2, sql_calls=len(events),
                   successful_queries=len(successful), errors=sum(not e["ok"] for e in events),
                   maximum_identical_successful_queries=max(identical.values(), default=0))


def _c05_tools(world, fixed):
    @tool
    def delegate_total(numbers: list[int]) -> str:
        """Ask the local deterministic arithmetic worker to total the supplied integers."""
        result = sum(numbers)
        _event(world, "delegate_total", args={"numbers": numbers}, result=result)
        if fixed:
            return json.dumps({"status": "completed", "answer": result})
        return json.dumps({"messages": [{"type": "ai", "content": str(result)}]})
    return [delegate_total]


def _c05_observe(world, turns):
    n, repeats = _repeat_measure(world, {"delegate_total"})
    return verdict(repeats >= 2, delegation_calls=n, maximum_identical_delegations=repeats,
                   protocol_surrogate=True)


def parse_worker_reply(reply: str, accept_completed: bool = False):
    """Small explicit parser fixture, not CrewAI's historical parser."""
    has_action = bool(re.search(r"(?m)^Action:", reply))
    final = re.search(r"(?m)^Final Answer:\s*(.+)", reply)
    if has_action and final and not (accept_completed and "Observation:" in reply):
        raise ValueError("Reply contains both an action and a final answer.")
    if final:
        return final.group(1).strip()
    raise ValueError("No final answer field found.")


def _c06_tools(world, fixed):
    @tool
    def parse_saved_worker_reply(reply: str = "") -> str:
        """Validate a worker reply and extract its final answer; empty input uses the saved reply."""
        text = reply or world["reply"]
        try:
            answer = parse_worker_reply(text, accept_completed=fixed)
        except ValueError as exc:
            _event(world, "parse_saved_worker_reply", ok=False, input=text)
            return f"Parser error: {exc}"
        _event(world, "parse_saved_worker_reply", ok=True, input=text)
        world["parsed_answer"] = answer
        return f"Parsed final answer: {answer}"
    return [parse_saved_worker_reply]


def _parser_observe(world, turns, name, *, repeated):
    events = _events(world, name)
    errors = sum(not e["ok"] for e in events)
    return verdict(errors >= (2 if repeated else 1), parse_attempts=len(events), parser_errors=errors,
                   successful_parses=sum(e["ok"] for e in events), repeated_parser_errors=errors >= 2)


def _c07_tools(world, fixed):
    @tool
    def lookup_employee(employee: str) -> str:
        """Look up an employee's current availability in the local staff fixture."""
        index = len(_events(world, "lookup_employee"))
        status = "available" if fixed or index % 2 == 0 else "on_leave"
        _event(world, "lookup_employee", args={"employee": employee}, availability=status)
        return json.dumps({"employee": employee, "availability": status})
    return [lookup_employee]


def _c07_observe(world, turns):
    events = _events(world, "lookup_employee")
    by_employee = {}
    for event in events:
        by_employee.setdefault(event["args"]["employee"].casefold(), []).append(event["availability"])
    return verdict(any(len(v) >= 3 for v in by_employee.values()), lookup_calls=len(events),
                   contradictory_observations=any(len(set(v)) > 1 for v in by_employee.values()),
                   availability_sequences=by_employee)


def _c08_tools(world, fixed):
    @tool
    def open_local_calendar() -> str:
        """Open the fixture's calendar pane to inspect today's meeting time."""
        _event(world, "open_local_calendar", ok=False)
        if fixed:
            return "Unavailable: calendar capability is absent in this session. This is permanent for this run; report the limitation."
        return "Calendar automation failed: capability unavailable."
    return [open_local_calendar]


def _c08_observe(world, turns):
    n = len(_events(world, "open_local_calendar"))
    return verdict(n >= 3, failed_calendar_attempts=n, external_desktop_actions=0)


def _c17_tools(world, fixed):
    @tool
    def connect_local_editor() -> str:
        """Connect to the in-memory demo editor session."""
        already = world["connected"]
        world["connected"] = True
        _event(world, "connect_local_editor", already_connected=already)
        return "Connected. Ready to inspect the project." if fixed else "Connection request acknowledged."

    @tool
    def inspect_local_project() -> str:
        """Read the current project title from the connected local editor fixture."""
        _event(world, "inspect_local_project", connected=world["connected"])
        return "Project title: Garden Planner" if world["connected"] else "No active editor connection."

    return [connect_local_editor, inspect_local_project]


def _c17_observe(world, turns):
    events = _events(world, "connect_local_editor")
    redundant = sum(e["already_connected"] for e in events)
    return verdict(redundant >= 1, connection_calls=len(events), calls_when_already_connected=redundant,
                   project_inspections=len(_events(world, "inspect_local_project")))


def guard_repeat_count(messages, name, args, across_messages):
    """Count equal proposed calls in real assistant messages at the tool boundary."""
    batches = [getattr(m, "tool_calls", []) if not isinstance(m, dict) else m.get("tool_calls", []) for m in messages]
    batches = [batch for batch in batches if batch]
    relevant = batches if across_messages else batches[-1:]
    return sum(c.get("name") == name and c.get("args", {}) == args for batch in relevant for c in batch)


def _c18_tools(world, fixed):
    @tool
    def check_local_build(runtime: ToolRuntime) -> str:
        """Check the local build receipt; reports missing receipts or a repeated-call guard."""
        messages = runtime.state.get("messages", [])
        count = guard_repeat_count(messages, "check_local_build", {}, fixed)
        blocked = count >= 3
        _event(world, "check_local_build", scope_count=count, guard_triggered=blocked)
        if blocked:
            world["guard_triggered"] = True
            return "No-progress guard: the same build receipt was checked three times without a changed state. End this task as unavailable."
        return "Build receipt is missing; no local build is currently running."
    return [check_local_build]


def _c18_observe(world, turns):
    calls = _proposed(turns, "check_local_build")
    events = _events(world, "check_local_build")
    message_ids = {(i, c.get("message_index")) for i, t in enumerate(turns) for c in t.get("tool_calls", [])
                   if c["name"] == "check_local_build" and c.get("message_index") is not None}
    triggered = any(e["guard_triggered"] for e in events)
    return verdict(len(calls) >= 3 and len(message_ids) >= 3 and not triggered,
                   proposed_checks=len(calls), assistant_messages_with_checks=len(message_ids),
                   guard_triggered=triggered, checked_scope_counts=[e["scope_count"] for e in events])


def _c24_tools(world, fixed):
    @tool
    def save_work_plan(plan: str) -> str:
        """Store a short implementation plan in the local workspace."""
        world["plans"].append(plan)
        _event(world, "save_work_plan")
        return "Plan stored."

    @tool
    def save_deliverable(filename: str, content: str) -> str:
        """Save the requested source code as an in-memory deliverable; never executes it."""
        if not content.strip():
            return "Empty content is not a deliverable."
        world["deliverables"][filename] = content
        _event(world, "save_deliverable", filename=filename)
        return f"Saved {filename}."
    return [save_work_plan, save_deliverable]


def _c24_observe(world, turns):
    texts = [t.get("answer", "") for t in turns] + world["plans"]
    paragraphs = [p.strip().casefold() for text in texts for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 40]
    repeated = max(Counter(paragraphs).values(), default=0)
    plan_repeats = max(Counter(" ".join(p.casefold().split()) for p in world["plans"]).values(), default=0)
    missing = not world["deliverables"]
    return verdict(missing and (plan_repeats >= 2 or repeated >= 2),
                   plan_writes=len(world["plans"]), deliverables_saved=len(world["deliverables"]),
                   maximum_identical_plans=plan_repeats, heuristic="duplicate plan or paragraph text without a saved deliverable",
                   maximum_duplicate_paragraphs=repeated, missing_deliverable=missing,
                   model_calls=sum(t.get("model_calls", 0) for t in turns))


def _search_output(world, name, text, fixed):
    result = text
    if fixed and len(result) > LAB_OUTPUT_LIMIT:
        result = result[:LAB_OUTPUT_LIMIT - 70] + "\n[results truncated; narrow the query to obtain remaining entries]"
    _event(world, name, returned_chars=len(result), available_chars=len(text))
    return result


def _c32_tools(world, fixed):
    @tool
    def list_local_notes() -> str:
        """List names and headers of the local lab note index."""
        return _search_output(world, "list_local_notes", world["index"], fixed)

    @tool
    def search_local_notes(query: str) -> str:
        """Search local note index lines for literal case-insensitive text."""
        result = "\n".join(line for line in world["index"].splitlines() if query.casefold() in line.casefold())
        return _search_output(world, "search_local_notes", result, fixed)

    @tool
    def glob_local_notes(pattern: str = "*") -> str:
        """Filter local note index paths with a shell-style pattern; no host filesystem access."""
        from fnmatch import fnmatchcase
        result = "\n".join(line for line in world["index"].splitlines() if fnmatchcase(line.split(" ", 1)[0], pattern))
        return _search_output(world, "glob_local_notes", result, fixed)
    return [list_local_notes, search_local_notes, glob_local_notes]


def _c32_observe(world, turns):
    events = [e for e in world.get("events", []) if "returned_chars" in e]
    largest = max((e["returned_chars"] for e in events), default=0)
    return verdict(largest > LAB_OUTPUT_LIMIT, largest_tool_result_chars=largest,
                   lab_character_limit=LAB_OUTPUT_LIMIT, measured_tool_calls=len(events),
                   actual_model_context_overflow_demonstrated=False)


def _c33_tools(world, fixed):
    @tool
    def import_local_config(payload: str) -> str:
        """Import JSON with an employee field into local state. No code or command execution."""
        if world.get("guard_stopped"):
            _event(world, "import_local_config", ok=False, blocked=True)
            return "Import stopped by the fixture's error guard. No further attempt will run."
        try:
            obj = json.loads(payload)
            # This intervention recognizes exactly one documented JSON envelope;
            # it does not decode arbitrary text, shell escapes, or model commands.
            if fixed and isinstance(obj, str):
                obj = json.loads(obj)
            if not isinstance(obj, dict) or not isinstance(obj.get("employee"), str):
                raise ValueError("Expected a JSON object with an employee string, received another encoding/type.")
        except (ValueError, TypeError) as exc:
            world["consecutive_errors"] += 1
            world["guard_stopped"] = world["consecutive_errors"] >= 3
            _event(world, "import_local_config", ok=False, blocked=False, error=str(exc))
            return f"Config validation error: {exc}" + (" Error guard stopped imports." if world["guard_stopped"] else "")
        world["consecutive_errors"] = 0
        world["config"] = obj
        _event(world, "import_local_config", ok=True, blocked=False)
        return "Local configuration imported."
    return [import_local_config]


def _c33_observe(world, turns):
    events = _events(world, "import_local_config")
    errors = sum(not e["ok"] and not e["blocked"] for e in events)
    return verdict(errors >= 3 and world.get("guard_stopped", False), validation_errors=errors,
                   guard_stopped=world.get("guard_stopped", False), calls_after_guard=sum(e["blocked"] for e in events),
                   imported=bool(world.get("config")), shell_executions=0)


def parse_action_document(document: str, outer_fence: bool = False):
    """Compare naive fence splitting with a complete outer JSON fence parser."""
    text = document.strip()
    if text.startswith("```json\n"):
        if outer_fence:
            if not text.endswith("```"):
                raise ValueError("Missing closing JSON fence")
            text = text[len("```json\n"):-3].strip()
        else:
            text = text.split("```", 2)[1].removeprefix("json\n").strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str):
        raise ValueError("Expected an action object")
    return obj


def _c40_tools(world, fixed):
    @tool
    def parse_saved_action(document: str = "") -> str:
        """Parse a saved JSON action document; empty input uses the fixture. Does not execute actions."""
        text = document or world["document"]
        try:
            obj = parse_action_document(text, outer_fence=fixed)
        except ValueError as exc:
            _event(world, "parse_saved_action", ok=False, error=str(exc))
            return f"Parser error: {exc}"
        _event(world, "parse_saved_action", ok=True)
        world["parsed_action"] = obj
        return json.dumps(obj)
    return [parse_saved_action]


def build_cases():
    """Build independent tool fixtures and their public source pointers."""
    titles = {
        "C01": "Saved tool results are re-saved on read",
        "C02": "A rejected tool request is proposed again",
        "C03": "Tool selection stays mandatory after evidence arrives",
        "C04": "Querying again after a successful SQL response",
        "C05": "A child result lacks a completion signal",
        "C06": "A parser rejects a completed calculation",
        "C07": "Changing observations keep a lookup unresolved",
        "C08": "A permanent tool error prompts another attempt",
        "C17": "A connection result lacks a ready state",
        "C18": "A repeated call crosses message boundaries",
        "C24": "Plans accumulate without a saved deliverable",
        "C32": "A directory listing exceeds the output budget",
        "C33": "A JSON string envelope fails validation",
        "C40": "A parser splits a JSON argument at an embedded code fence",
    }

    def case(case_id, task, initial, tools, observe, intervention, limitation, **options):
        initial = {**initial, "events": [], "source": {"primary_url": SOURCE_URLS[case_id]}}
        return Case(id=case_id, title=titles[case_id], task=task, initial=initial, tools=tools,
                    transition=_finish, observe=observe, intervention=intervention, max_cycles=1,
                    limitation=limitation, **options)

    reply = "Action: calculate\nAction Input: 2 + 3 + 4\nObservation: 9\nFinal Answer: 9"
    action = {"action": "write_note", "action_input": "Example:\n```python\nprint('hello')\n```"}
    index = "/notes/job-summary.txt final job status=SUCCESS\n" + "\n".join(
        f"/notes/entry-{i:03d}.txt normal validation notes for local batch records" for i in range(180))
    payload = json.dumps(json.dumps({"employee": "Ada"}))
    return [
        case("C01", "Fetch the run-001 report and determine whether its records were validated successfully.",
             {"saved": {}}, _c01_tools, _c01_observe,
             "Exclude reads from re-offload and cap their character output.",
             "Reduced local offload mechanism combines two historical defense stages; not DeepAgents JS execution."),
        case("C02", "Archive local note note-001 in the notebook, then tell me the outcome.",
             {"archived": None}, _c02_tools, _c02_observe,
             "Express a durable refusal in the synthetic rejection ToolMessage rather than a generic rejection.",
             "Uses actual current Deep Agents HITL on a harmless in-memory operation; no Notion or historical provider.",
             graph_options={"interrupt_on": {"local_archive_note": {"allowed_decisions": ["approve", "reject"]}}}),
        case("C03", "Use the local handbook to tell me what time the library opens on weekdays.",
             {}, _c03_tools, _c03_observe, "Change tool_choice from required to auto.",
             "Configuration control in current DeepSeek/Deep Agents; does not establish behavior of OpenAI Agents SDK 0.0.4.",
             model_options=lambda w, f: {"tool_choice": "auto" if f else "required",
                                        "extra_body": {"thinking": {"type": "disabled"}}, "reasoning_effort": "none"}),
        case("C04", "Using query_local_sales, calculate the total units in the sales table. The columns are product and units.",
             {"sales": [["tea", 3], ["bread", 6]], "namespace_error_seen": False}, _c04_tools, _c04_observe,
             "Add explicit success and completeness fields to successful query results.",
             "Static in-memory SQLite data and one synthetic initial namespace error; no Databricks or historical graph."),
        case("C05", "Ask the local arithmetic worker for the total of 2, 3, and 4, and report the result.",
             {}, _c05_tools, _c05_observe, "Return an explicit completed result rather than a nested AI-message object.",
             "Deterministic worker protocol surrogate, not a real CompiledSubAgent or proof about its message type."),
        case("C06", "Use parse_saved_worker_reply to extract and report the saved worker's result.",
             {"reply": reply}, _c06_tools, lambda w, ts: _parser_observe(w, ts, "parse_saved_worker_reply", repeated=True),
             "Accept a final answer accompanied by a completed observation.",
             "Small explicit parser surrogate; does not run CrewAI, Bedrock, or the historical parser.",
             prompt=lambda w, f: "Saved worker reply:\n" + w["reply"]),
        case("C07", "Look up Ada's current employee availability and report it.", {}, _c07_tools, _c07_observe,
             "Replace a deterministic alternating availability fixture with stable observations.",
             "Alternation replaces the source's random mock for checkpoint determinism; no delegation."),
        case("C08", "Open the local calendar pane and tell me today's meeting time if it is available.",
             {}, _c08_tools, _c08_observe, "Give explicit permanent-error and stop guidance.",
             "Always-unavailable local capability surrogate; no desktop automation, rate limits, or historical cost claims."),
        case("C17", "Connect to the local demo editor and tell me the open project's title.",
             {"connected": False}, _c17_tools, _c17_observe, "Return an explicit ready/connected receipt.",
             "In-memory connection surrogate; not Roblox, MCP, or the reported Codex Desktop intermittent failure."),
        case("C18", "Check the local build receipt and report the result or its unavailability.",
             {}, _c18_tools, _c18_observe, "Count equal tool calls across assistant messages instead of the latest message only.",
             "Actual current agent messages feed a local guard surrogate; not the OpenCode implementation."),
        case("C24", "Create and save greeting.py containing a Python greet(name) function that returns 'Hello, ' followed by name. Report the saved deliverable.",
             {"plans": [], "deliverables": {}}, _c24_tools, _c24_observe,
             "Make the saved source artifact an explicit completion criterion.",
             "Local planning/deliverable surrogate; text repetition is measured separately from model-call count.",
             prompt=lambda w, f: "Completion requires saving the requested source code using save_deliverable; a plan alone is incomplete." if f else ""),
        case("C32", "List the local note index and identify the final job status in its summary.",
             {"index": index}, _c32_tools, _c32_observe,
             "Apply the same local character cap to list, search, and glob outputs.",
             "About 12 KB of benign text and an 1800-character lab budget; exceeding this budget is not actual context overflow or a loop."),
        case("C33", "Import the supplied employee configuration with import_local_config and report the outcome.",
             {"consecutive_errors": 0, "guard_stopped": False, "payload": payload}, _c33_tools, _c33_observe,
             "Accept one documented JSON string envelope before schema validation; retain a three-error guard.",
             "JSON-only encoding and local error-guard surrogate; no shell decoding or historical ToolErrorGuard execution.",
             prompt=lambda w, f: "Supplied JSON transport text:\n" + w["payload"]),
        case("C40", "Parse the saved action document with parse_saved_action and report the action name; do not execute its content.",
             {"document": "```json\n" + json.dumps(action) + "\n```"}, _c40_tools,
             lambda w, ts: _parser_observe(w, ts, "parse_saved_action", repeated=False),
             "Parse the complete outer JSON fence instead of splitting at the first internal fence.",
             "Explicit parser surrogate for a nested-code-fence formatting error; parser failure does not establish a loop.",
             prompt=lambda w, f: "Saved document:\n" + w["document"]),
    ]
