# Research Loop Design — The Details That Matter

## The Two Loops

There are two loops, coupled by the experiment list and shared state:

```
┌─────────────────────────────────────────────────────────────┐
│  SCHEDULING LOOP (fast, deterministic, no LLM)              │
│                                                             │
│  while GPUs exist:                                          │
│    wait for event (run finished, GPU freed, tick)           │
│    if list non-empty and GPU free: pop front, launch        │
│    if run failed: invoke fixer (sync, bounded)              │
│    update live metrics from training logs                   │
│    check warning conditions                                 │
│    if research loop should wake: post event                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
              experiment_list + experiment_state.json
              (scheduling writes metrics/status,
               research writes list/hypotheses/code)
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  RESEARCH LOOP (slow, LLM, background thread)               │
│                                                             │
│  wait for trigger event                                     │
│  assemble context: state + live metrics + warnings          │
│  invoke agent (codex/claude CLI)                            │
│  parse output: new list, kills, code changes, reflection    │
│  apply to state                                             │
│  go back to waiting                                         │
└─────────────────────────────────────────────────────────────┘
```

## Question 1: How do errors propagate fast?

Two paths:

### Path A: Scaffold-level fix (fast, no research loop needed)
Training crashes with OOM, missing module, shape mismatch, etc.
The scheduling loop invokes a focused fixer agent (sync, 15 turns max).
The fix is mechanical, not strategic. The research loop doesn't need
to reconsider its hypothesis because of an OOM error.

The fix is logged in experiment_state.json so the research loop sees
it next time it wakes up:
```json
{"run_id": "run_003", "status": "failed", "error": "OOM at step 200",
 "fix_applied": "reduced batch_size 64→32", "fix_relaunch": "run_004"}
```

### Path B: Research-level failure (needs rethinking)
Training runs to completion but loss doesn't decrease. Or a hypothesis
is falsified. Or the approach is fundamentally broken.

This is detected by the WARNING ENGINE (see below) and triggers a
research loop invocation with an injected warning:

```
⚠ WARNING: Hypothesis "larger batch → lower loss" has been falsified.
  - run_003 (batch=32): final_loss=3.41
  - run_005 (batch=64): final_loss=3.52
  - run_007 (batch=128): final_loss=3.78
  Loss is INCREASING with batch size. Reconsider approach.
```

The key distinction: **mechanical failures → scheduling fixer (fast).
Strategic failures → warning injection → research loop (deliberate).**

## Question 2: How does the research loop read live results?

### Metrics Poller (background thread in scheduling loop)

Every 30s, reads the last N lines of each running experiment's log file.
Extracts metrics via regex (loss, step, lr, eval_loss, etc.).
Writes to experiment_state.json:

```json
{
  "run_id": "run_005",
  "status": "running",
  "live_metrics": {
    "step": 1500,
    "loss": 2.34,
    "eval_loss": 2.89,
    "lr": 0.0003,
    "gpu_util_pct": 78,
    "samples_per_sec": 1200,
    "last_updated": 1709012345.6
  },
  "metrics_history": [
    {"step": 0, "loss": 8.12},
    {"step": 500, "loss": 4.56},
    {"step": 1000, "loss": 3.01},
    {"step": 1500, "loss": 2.34}
  ]
}
```

### What the research agent sees when invoked

Not the raw log (3000 truncated chars) — structured data:

```
## Running Experiments

→ run_005 (GPU 0, 23min elapsed)
  Hypothesis: "AdamW with cosine schedule converges faster than constant LR"
  Progress: step 1500/5000, loss 2.34 (↓ from 8.12)
  Trend: loss decreasing steadily, ~0.004/step
  GPU util: 78%, mem: 34.2/80GB
  Kill criteria: loss > 4.0 after step 1000 → NOT TRIGGERED

→ run_006 (GPU 1, 18min elapsed)
  Hypothesis: "Larger model (d=512) outperforms d=256"
  Progress: step 800/5000, loss 3.89 (↓ from 9.45)
  Trend: loss decreasing but slower than run_005
  GPU util: 82%, mem: 61.3/80GB
  Kill criteria: loss > 5.0 after step 500 → NOT TRIGGERED
```

This lets the research agent make informed decisions:
- "run_005 is converging well, keep it running"
- "run_006 is 2x slower convergence — kill and reallocate GPU to a variant of 005's approach"

### Anomaly detection (scaffold, not LLM)

The metrics poller also detects:
- **NaN/Inf**: immediate failure, triggers error path
- **Loss plateau**: loss hasn't decreased >1% in last 500 steps
- **Loss divergence**: loss increased >50% from minimum
- **Underutilization**: GPU util <20% for >2 minutes

These generate warnings injected into the next research agent call,
or immediate events (NaN → kill run immediately).

## Question 3: What is the research loop, concretely?

It is ONE agent invocation (codex CLI) with a rich prompt and
structured output. Not three separate agents.

### Trigger conditions (when to invoke)

```python
RESEARCH_TRIGGERS = {
    # A run finished (completed or failed) — most common
    "run_finished": always,

    # Experiment list is empty or very short
    "list_depleted": lambda state: state.list_size() < 2,

    # Warning engine fired
    "warning_fired": lambda warnings: len(warnings) > 0,

    # Budget milestone
    "budget_milestone": lambda budget: budget.fraction_remaining in [0.75, 0.5, 0.25, 0.1],

    # Periodic heartbeat (every N minutes if nothing else triggered)
    "heartbeat": lambda last_invocation: time.time() - last_invocation > 900,
}
```

### What it does NOT trigger on

- GPU freed (scheduling loop handles this)
- Mechanical training failure (fixer handles this)
- Resource monitoring (scheduling loop handles this)

### The unified prompt

```
You are a research agent conducting ML experiments.

# Goal
{spec}

# Current State
{structured_state_summary}

# Running Experiments (with live metrics)
{running_experiments_with_metrics}

# Completed Experiments
{completed_experiments_table}

# Failed Experiments
{failed_experiments_with_errors}

# Current Experiment List (what will run next)
{current_list}

# ⚠ Warnings
{injected_warnings}

# Your Reflection (from last invocation)
{reflection}

# Budget
{budget_status}

# Instructions

You have three responsibilities:

1. THINK: Update your reflection. What have you learned? What hypotheses
   are confirmed/rejected? What's the most promising direction?

2. DECIDE: Output a new experiment list. This REPLACES the current list.
   Position 1 runs next when a GPU is free. You can also kill running
   experiments that aren't worth continuing.

3. IMPLEMENT: If any experiment needs code changes, make them now.
   The executor is you — there's no separate implementation step.

Write `_research_output.json`:
{
  "reflection": "Updated thinking...",
  "hypotheses": [...],
  "experiment_list": [...],
  "kill_runs": [],
  "reasoning": "Why this plan"
}

You may also directly modify code files in the experiment directory.
```

### What happens if it needs to wait for a run to finish?

**It doesn't wait.** The Codex CLI invocation finishes. The agent says
"I need run_005 to complete before I can decide the next direction"
by writing an experiment list that has:

1. Items that can run NOW (on free GPUs)
2. Items that depend on run_005's results (marked with a note)

The scaffold re-invokes the research agent when run_005 completes.
Between invocations, the scheduling loop keeps GPUs busy from the list.

If the agent truly has nothing to add to the list, it writes an empty
list and the scheduling loop does nothing until the running experiments
finish and trigger a new research invocation.

The critical insight: **the research agent is consulted, not resident.**
It runs, decides, exits. The scaffold decides when to consult it again.

## Question 4: Warning injection

### Warning engine (scaffold-level pattern detection)

```python
class WarningEngine:
    def check(self, state) -> list[Warning]:
        warnings = []

        # Hypothesis falsified repeatedly
        for h in state.hypotheses:
            if h.consecutive_failures >= 3:
                warnings.append(Warning(
                    severity="high",
                    message=f"Hypothesis '{h.description}' falsified "
                            f"{h.consecutive_failures} times. Consider "
                            f"abandoning this direction.",
                    evidence=[...run results...],
                ))

        # No improvement in N rounds
        if state.rounds_since_improvement >= 3:
            warnings.append(Warning(
                severity="high",
                message=f"No improvement in {state.rounds_since_improvement} "
                        f"rounds. Best result is still {state.best_result}. "
                        f"Consider a fundamentally different approach.",
            ))

        # Budget warnings
        remaining = state.budget.fraction_remaining
        if remaining < 0.25:
            warnings.append(Warning(
                severity="medium",
                message=f"Only {remaining*100:.0f}% budget remaining. "
                        f"Prioritize consolidating results.",
            ))

        # Live metric anomalies
        for run in state.running_runs:
            if run.loss_plateaued:
                warnings.append(Warning(
                    severity="low",
                    message=f"Run {run.run_id} loss plateaued at "
                            f"{run.current_loss:.4f} for {run.plateau_steps} "
                            f"steps. Consider early stopping.",
                ))

        return warnings
```

Warnings are formatted and injected into the research agent prompt.
They're also logged to experiment_state.json for traceability.

## Question 5: Is the research loop just an ongoing loop?

No. It's event-driven, same as the scheduling loop.

```
Research agent invoked when:
  1. Run finishes (completed/failed) → analyze results, update list
  2. Warning fires → address the warning
  3. List depleted → need more experiments
  4. Periodic heartbeat → check on long-running experiments
  5. Budget milestone → strategic reassessment

Research agent NOT invoked when:
  - GPUs are busy and list is full (nothing to decide)
  - Only mechanical failures happened (fixer handles it)
  - Between events (it's not a persistent process)
```

The gap between invocations is FINE. The scheduling loop keeps GPUs
busy. The research agent is expensive (minutes per call, $$) — invoke
it only when there's a real decision to make.

## Implementation Plan

### New files
- `research_loop.py` — unified research agent (replaces planner.py, analyst.py)
- `metrics_poller.py` — live metrics extraction from training logs
- `warning_engine.py` — condition-based warning detection

### Modified files
- `event_loop.py` — integrate research loop, metrics poller, warnings
- `experiment_state.py` — add live_metrics, metrics_history to RunRecord

### Removed roles
- `planner.py` → absorbed into research_loop.py
- `analyst.py` → absorbed into research_loop.py
- `executor.py` → the research agent implements code changes directly

### Kept as-is
- `run_manager.py` — process management
- `resource_monitor.py` — GPU monitoring
- `budget_guard.py` — time tracking
- `event_types.py` — events
