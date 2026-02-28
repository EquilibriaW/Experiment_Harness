# Experiment Harness v2

## Design Laws

1. Never put an LLM on the critical path of keeping GPUs busy.
2. Data lives in the environment, not the prompt. (RLM principle)
3. REPL output truncation forces symbolic recursion. (Not prompt instructions)
4. Anomaly thresholds are human-set. Anomalies are informational, not auto-kill.

## Two Loops, One List

### Scheduling Loop (event_loop.py)
Fast, deterministic, no LLM (except bounded CLI fixers).
- Pops front of experiment list → launches on free GPUs
- Monitors training subprocesses (run_manager.py)
- Polls live metrics from training logs (metrics_poller.py)
- Monitors GPU state (resource_monitor.py)
- Detects warning conditions (warning_engine.py)
- Decides when to wake the research agent
- Deadlock prevention: if nothing running + nothing queued → force wake

### Research Agent (research_loop.py → dspy.RLM)
Slow, LLM-driven, invoked on triggers.
Uses dspy.RLM (arXiv 2512.24601) with Opus as root, Haiku as sub-LM.

**The agent is consulted, not resident.**

## dspy.RLM Integration

### Why dspy.RLM, not Codex/Claude CLI

The RLM paper identifies three flaws in standard scaffolds (Algorithm 2):

1. **Flaw #1**: Putting data in the prompt → context rot
2. **Flaw #2**: Agent generates output autoregressively → bounded output
3. **Flaw #3**: Appending outputs to history → needs lossy compaction

dspy.RLM fixes all three via architectural constraints:

- `experiment_state` is a **variable** in the REPL, not text in the prompt
- `max_output_chars=10_000` **truncates** REPL stdout, forcing the model
  to store results in variables and delegate to `llm_query()` instead
  of trying to read everything into its own context
- Output via `SUBMIT()` — built up symbolically in variables
- `llm_query(prompt)` available in REPL for **symbolic recursion**

The truncation is the key mechanism. When the agent writes
`print(json.loads(experiment_state))` on a state with 50 runs, it sees
truncated output and *discovers through interaction* that it must write
Python to slice, filter, and delegate. The recursion is forced by
architecture, not by prompt instructions.

Codex/Claude CLI manage their own REPL loops — we can't control output
truncation. So the research agent goes through the API via dspy.RLM.

### Signature

```python
rlm = dspy.RLM(
    "experiment_state, trigger, warnings, budget -> "
    "experiment_list, reflection, hypotheses",
    max_iterations=20,
    max_output_chars=10_000,
    sub_lm=dspy.LM("anthropic/claude-haiku-4-5"),
    tools=[read_log, read_file, list_files, kill_run, apply_code_change],
)
```

### What the agent sees

The agent gets experiment_state as a JSON string variable. To analyze:

```python
# In the RLM's REPL — this is what the AGENT writes, not us
import json
state = json.loads(experiment_state)
print(f"Runs: {len(state['runs'])}, Best: {state.get('best_result', {}).get('run_id')}")

# REPL output is truncated to 10K chars — agent can't paste everything
# So it writes targeted queries:
completed = [r for r in state['runs'] if r['status'] == 'completed']
for r in completed[-3:]:
    print(r['run_id'], r['metrics'].get('loss'), r['hypothesis'])

# For deep analysis, delegate to sub-LLM:
log = read_log(f"run_logs/{completed[-1]['run_id']}.log", tail_lines=200)
analysis = llm_query(f"Why did loss plateau in this run?\n\n{log}")
print(analysis)

# Code changes go through Codex CLI as a tool:
apply_code_change("In train.py, add gradient clipping with max_norm=1.0")

# Submit decisions
SUBMIT({
    "experiment_list": [...],
    "reflection": "...",
    "hypotheses": [...]
})
```

### Tools provided to the RLM

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `read_log(path, tail_lines)` | Read training logs | File I/O |
| `read_file(path)` | Read any file (code, config) | File I/O |
| `list_files(subdir)` | List directory contents | File I/O |
| `kill_run(run_id)` | Request kill of running experiment | Writes _kill_requests.json |
| `apply_code_change(instruction)` | Edit code via Codex/Claude CLI | subprocess call |

These are genuinely external operations (file I/O, process management).
Everything else — parsing state, comparing metrics, finding patterns —
the agent writes its own Python for.

### Cost

~$2.50/invocation with Opus root + Haiku sub.
~15-30 invocations per 24hr session.
~$40-75/day API cost.

## When the Research Agent is Invoked

- Run completed → analyze results, update list
- Run failed → strategic rethinking (after mechanical fix)
- Warnings fired → stagnation, budget thresholds
- List depleted + GPUs idle → provide more experiments
- Deadlock: nothing running + nothing queued → force wake
- Heartbeat (every 15min) → check live metrics on long runs

## File Map

```
remote/
  event_loop.py          Scheduling loop + trigger logic
  research_loop.py       Research agent (dspy.RLM wrapper)
  metrics_poller.py      Live metrics extraction from logs
  warning_engine.py      Condition detection + warning injection
  experiment_state.py    Structured JSON state (source of truth)
  resource_monitor.py    GPU monitoring
  run_manager.py         Training subprocess management
  executor.py            Pre-launch code changes (Codex/Claude CLI)
  budget_guard.py        Time budget
  event_types.py         Event dataclasses
  _agent_utils.py        Agent CLI adapter loading
  bootstrap.sh           Pod setup (installs dspy, deno, CLI agents)
```

## Anomaly Thresholds

Human-set via CLI. Informational only — produce warnings for
the research agent's prompt. The agent decides what to do.

```bash
python event_loop.py --spec spec.md \
  --diverge-ratio 2.0 \
  --plateau-steps 500 \
  --model anthropic/claude-opus-4-5 \
  --sub-model anthropic/claude-haiku-4-5 \
  --research-max-output-chars 10000
```
