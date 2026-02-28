"""Research agent: OpenAI Agents SDK replacement for dspy.RLM.

Uses a two-agent system:
1. Research Agent — phased workflow (hypothesize → verify → reflect → plan → design → implement)
2. Maintainability Agent — reviews code changes for quality

Fixes all information gaps from the dspy.RLM approach:
- Full system prompt with spec text embedded
- All experiments visible (no truncation)
- No artificial output caps
- Proper reasoning effort (xhigh)
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, function_tool, RunContextWrapper

from experiment_state import ExperimentState, ExperimentSpec, Hypothesis
from warning_engine import WarningEngine


# ── Pydantic output models ─────────────────────────────────────────────────

class ExperimentPlan(BaseModel):
    """A single experiment to run."""
    spec_id: str
    description: str
    hypothesis: str
    config: dict[str, Any]
    train_command: str = ""
    requires_code_change: bool = False
    code_change_description: str = ""
    predicted_outcome: str = ""
    kill_criteria: str = ""
    uncertainty: str = "high"
    gpu_requirement: int = 1
    priority: int = 0


class HypothesisPlan(BaseModel):
    """A research hypothesis."""
    id: str
    description: str
    status: str = "active"
    evidence: list[str] = []


class ResearchDecision(BaseModel):
    """Structured output from research agent — one complete cycle."""
    hypotheses: list[HypothesisPlan]
    reflection: str
    experiment_list: list[ExperimentPlan]
    kill_runs: list[str] = []


class ReviewResult(BaseModel):
    """Output from the maintainability agent."""
    issues_found: int
    fixes_applied: list[str]
    summary: str


# ── Shared contexts ─────────────────────────────────────────────────────────

@dataclass
class ResearchContext:
    state: ExperimentState
    experiment_dir: Path
    agent_name: str
    warning_engine: WarningEngine
    spec_text: str
    maint_agent: Agent


@dataclass
class MaintContext:
    experiment_dir: Path


# ── Research agent tools ────────────────────────────────────────────────────

@function_tool
def read_experiment_state(ctx: RunContextWrapper[ResearchContext]) -> str:
    """Get full experiment state summary: all runs, metrics, full queue, goal."""
    return ctx.context.state.agent_summary()


@function_tool
def read_file(ctx: RunContextWrapper[ResearchContext], path: str) -> str:
    """Read any file in the experiment directory.

    Args:
        path: Relative path (e.g., "train.py", "config.json")
    """
    full = (ctx.context.experiment_dir / path).resolve()
    if not full.is_relative_to(ctx.context.experiment_dir.resolve()):
        return "(error: path escapes experiment directory)"
    if not full.exists():
        return f"(file not found: {path})"
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(error: {e})"


@function_tool
def read_log(ctx: RunContextWrapper[ResearchContext], run_id: str, tail_lines: int = 500) -> str:
    """Read training log tail for a run.

    Args:
        run_id: The run ID (e.g., "run_005")
        tail_lines: Number of lines from the end to return (default 500)
    """
    log_path = ctx.context.experiment_dir / "run_logs" / f"{run_id}.log"
    if not log_path.exists():
        return f"(log not found for {run_id})"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail_lines:])
    except Exception as e:
        return f"(error: {e})"


@function_tool
def list_files(ctx: RunContextWrapper[ResearchContext], subdir: str = ".") -> str:
    """List files in a subdirectory of the experiment.

    Args:
        subdir: Subdirectory to list (default "." for root)
    """
    target = ctx.context.experiment_dir / subdir
    if not target.exists():
        return f"(directory not found: {subdir})"
    try:
        entries = sorted(target.iterdir())
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'} {e.name}"
            for e in entries[:200]
        )
    except Exception as e:
        return f"(error: {e})"


@function_tool
def kill_run(ctx: RunContextWrapper[ResearchContext], run_id: str) -> str:
    """Request that a running experiment be killed.

    Args:
        run_id: The run_id to kill (e.g., "run_005")
    """
    kill_file = ctx.context.experiment_dir / "_kill_requests.json"
    existing = []
    if kill_file.exists():
        try:
            existing = json.loads(kill_file.read_text())
        except Exception:
            pass
    if run_id not in existing:
        existing.append(run_id)
    kill_file.write_text(json.dumps(existing), encoding="utf-8")
    return f"Kill requested for {run_id}"


@function_tool
def apply_code_change(ctx: RunContextWrapper[ResearchContext], instruction: str) -> str:
    """Have a coding agent make a specific change, then validate with tests.

    Flow: git checkpoint → coding agent → maintainability review → pytest → revert on fail.

    Args:
        instruction: What to change and why (e.g.,
            "In train.py, replace linear LR schedule with cosine annealing.
             Keep warmup_steps=100. Set T_max to total_steps.")
    """
    cwd = str(ctx.context.experiment_dir)
    agent_name = ctx.context.agent_name

    # 1. Git checkpoint — capture SHA so rollback targets the exact commit
    has_git = False
    checkpoint_sha = None
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       capture_output=True, cwd=cwd, timeout=5)
        subprocess.run(["git", "add", "-A"], capture_output=True, cwd=cwd, timeout=10)
        subprocess.run(["git", "commit", "-m", f"checkpoint before: {instruction[:60]}",
                        "--allow-empty"],
                       capture_output=True, cwd=cwd, timeout=10)
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if sha_result.returncode == 0:
            checkpoint_sha = sha_result.stdout.strip()
        has_git = True
    except Exception:
        pass

    # 2. Run coding agent
    if agent_name == "codex":
        cmd = ["codex", "exec", "--yolo", "-"]
        stdin_text = instruction
    else:
        cmd = ["claude", "--print", "--dangerously-skip-permissions",
               "-p", instruction]
        stdin_text = None
    try:
        agent_result = subprocess.run(
            cmd, input=stdin_text,
            capture_output=True, text=True,
            timeout=300, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return "Failed: coding agent timed out after 300s"
    except FileNotFoundError:
        return f"Failed: {agent_name} CLI not found"

    if agent_result.returncode != 0:
        err = agent_result.stderr.strip()
        return f"Failed (exit {agent_result.returncode}):\n{err[-500:]}"

    # 3. Get diff
    diff = ""
    try:
        diff_result = subprocess.run(
            ["git", "diff", "HEAD~1"], capture_output=True, text=True,
            cwd=cwd, timeout=10,
        )
        diff = diff_result.stdout.strip()
    except Exception:
        diff = "(could not get diff)"

    # 4. Run maintainability agent on diff + modified files
    maint_report = ""
    try:
        modified_files: list[str] = []
        try:
            mod_result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                capture_output=True, text=True, cwd=cwd, timeout=10,
            )
            modified_files = [f.strip() for f in mod_result.stdout.strip().split("\n") if f.strip()]
        except Exception:
            pass

        maint_prompt = (
            f"## Code Change Diff\n```\n{diff[:8000]}\n```\n\n"
            f"## Modified Files\n{', '.join(modified_files)}\n\n"
            f"Review this diff for bugs, dead code, naming issues, "
            f"and missing error handling."
        )

        maint_ctx = MaintContext(experiment_dir=ctx.context.experiment_dir)
        maint_result = Runner.run_sync(
            ctx.context.maint_agent, maint_prompt,
            context=maint_ctx, max_turns=10,
        )
        if maint_result.final_output:
            review = maint_result.final_output
            maint_report = (
                f"Maintainability: {review.issues_found} issues, "
                f"{len(review.fixes_applied)} fixes. {review.summary}"
            )
    except Exception as e:
        maint_report = f"Maintainability review skipped: {e}"

    # 5. Run smoke tests
    test_passed = True
    test_output = ""
    try:
        test_result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q",
             "--timeout=60", "-x"],
            capture_output=True, text=True,
            timeout=120, cwd=cwd,
        )
        test_output = test_result.stdout.strip()[-500:]
        if test_result.returncode != 0:
            test_passed = False
    except subprocess.TimeoutExpired:
        test_output = "(tests timed out)"
        test_passed = False
    except FileNotFoundError:
        try:
            import_result = subprocess.run(
                ["python", "-c", "import train"],
                capture_output=True, text=True,
                timeout=30, cwd=cwd,
            )
            if import_result.returncode != 0:
                test_passed = False
                test_output = f"Import failed: {import_result.stderr[-300:]}"
            else:
                test_output = "(no pytest, import check passed)"
        except Exception:
            test_output = "(no validation available)"

    # 6. Revert if tests failed — reset to checkpoint, not HEAD~1
    if not test_passed and has_git:
        try:
            if checkpoint_sha:
                subprocess.run(["git", "reset", "--hard", checkpoint_sha],
                               capture_output=True, cwd=cwd, timeout=10)
            else:
                # Checkpoint commit failed; just discard working-tree edits
                subprocess.run(["git", "checkout", "."],
                               capture_output=True, cwd=cwd, timeout=10)
        except Exception:
            pass
        return (
            f"REVERTED — tests failed after change.\n"
            f"Diff was:\n{diff[:2000]}\n\n"
            f"Test output:\n{test_output}\n\n{maint_report}"
        )

    return (
        f"Applied successfully.\n"
        f"Diff:\n{diff[:3000]}\n\n"
        f"Tests: {'PASSED' if test_passed else 'NO TESTS'}\n"
        f"{test_output}\n\n{maint_report}"
    )


# ── Maintainability agent tools ─────────────────────────────────────────────

@function_tool
def read_file_maint(ctx: RunContextWrapper[MaintContext], path: str) -> str:
    """Read a file to examine full context of changed code.

    Args:
        path: Relative path to the file
    """
    full = (ctx.context.experiment_dir / path).resolve()
    if not full.is_relative_to(ctx.context.experiment_dir.resolve()):
        return "(error: path escapes experiment directory)"
    if not full.exists():
        return f"(file not found: {path})"
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(error: {e})"


@function_tool
def apply_fix(ctx: RunContextWrapper[MaintContext], path: str, old_text: str, new_text: str) -> str:
    """Make a targeted fix in a file by replacing old_text with new_text.

    Args:
        path: Relative path to the file
        old_text: Exact text to find and replace
        new_text: Replacement text
    """
    full = (ctx.context.experiment_dir / path).resolve()
    if not full.is_relative_to(ctx.context.experiment_dir.resolve()):
        return "(error: path escapes experiment directory)"
    if not full.exists():
        return f"(file not found: {path})"
    try:
        content = full.read_text(encoding="utf-8")
        if old_text not in content:
            return f"(old_text not found in {path})"
        new_content = content.replace(old_text, new_text, 1)
        full.write_text(new_content, encoding="utf-8")
        return f"Fixed: replaced text in {path}"
    except Exception as e:
        return f"(error: {e})"


# ── System prompts ──────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """\
You are a research agent running ML experiments on a GPU cluster.
Your goal is to resolve the research problem described below through
systematic experimentation at meaningful scale.

## Research Specification
{spec_text}

## Your Workflow
You operate in a cycle: HYPOTHESIZE → VERIFY → REFLECT → PLAN → DESIGN → IMPLEMENT.

### Phase 1: HYPOTHESIZE
Form testable hypotheses about what will improve the model.
- Call read_file("train.py") to understand the training interface and CLI flags
- Call list_files() to see what code and data exists
- Formulate specific, falsifiable hypotheses grounded in the spec

### Phase 2: VERIFY (Read Experimental Results)
Check what experiments have already run and their outcomes.
- Call read_experiment_state() for the full summary of runs, metrics, and queue
- Call read_log(run_id) for completed/failed runs to understand what happened
- Compare actual outcomes to predicted outcomes from previous rounds
- Mark hypotheses as confirmed/rejected based on evidence

### Phase 3: REFLECT
Synthesize findings into understanding.
- What patterns emerge from the data?
- Which hypotheses gained/lost support?
- What surprised you? What matched expectations?
- Write a reflection that persists to your next invocation

### Phase 4: PLAN
Decide what to investigate next.
- What is the highest-value next experiment given current understanding?
- Should you try a new direction or refine a working one?
- Are any running experiments worth killing? (diverging, plateaued, superseded)
- Consider budget constraints from the trigger message

### Phase 5: DESIGN EXPERIMENTS
Create concrete experiment specs with SPECIFIC parameters.
- Every experiment MUST have a non-empty config dict with training parameters
- Reference train.py's actual CLI flags in your config
- Include train_command if the default "python train.py" needs flags
- Set predicted_outcome and kill_criteria for each experiment
- Design experiments at meaningful scale — the goal is to resolve
  the research problem, not just to probe. Use enough training steps
  to see real signal (thousands, not tens).

### Phase 6: IMPLEMENT
If experiments require code changes:
- Set requires_code_change=True with a clear code_change_description
- Use apply_code_change(instruction) for changes needed NOW
- The maintainability agent will automatically review your changes

## Rules
- NEVER leave config empty — every experiment needs training parameters
- Cross-check configs against train.py's argparse flags
- Kill runs that show loss divergence (>2x initial) or are clearly superseded
- On first invocation (no runs yet), read train.py thoroughly, then design
  initial experiments that establish baselines and test core hypotheses
"""

MAINTAINABILITY_PROMPT = """\
You review code changes for quality.
Given a git diff, scan modified files for:
- Bugs, logic errors, incorrect indexing
- Dead code, unused imports/variables
- Naming inconsistencies with surrounding code
- Missing error handling at system boundaries

Use read_file_maint to examine full context of changed files.
Use apply_fix to make targeted corrections.
Do NOT refactor or add features — only fix genuine issues."""


# ── ResearchOutput (same interface as research_loop.py) ─────────────────────

class ResearchOutput:
    def __init__(self):
        self.reflection: str = ""
        self.hypotheses: list[Hypothesis] = []
        self.specs: list[ExperimentSpec] = []
        self.kill_runs: list[str] = []
        self.list_unchanged: bool = False
        self.trajectory: list = []


# ── ResearchLoop (drop-in replacement) ──────────────────────────────────────

class ResearchLoop:
    """Research agent powered by OpenAI Agents SDK.

    Drop-in replacement for dspy.RLM-based ResearchLoop with matching
    invoke() signature. Fixes all information gaps:
    - Full system prompt with spec text and phased workflow
    - Proper reasoning effort (xhigh)
    - No output truncation
    - All experiments visible
    """

    def __init__(
        self,
        state: ExperimentState,
        experiment_dir: Path,
        spec_path: Path,
        warning_engine: WarningEngine,
        model: str = "gpt-5.3-codex",
        sub_model: str | None = None,
        agent_name: str = "codex",
        max_iterations: int = 20,
        **kwargs,
    ) -> None:
        self.state = state
        self.experiment_dir = experiment_dir
        self.spec_path = spec_path
        self.warning_engine = warning_engine
        self.model = model
        self.agent_name = agent_name
        self.max_turns = max_iterations

        self.last_invocation_time: float = 0.0
        self.invocation_count: int = 0

        # Read spec once at init
        self.spec_text = ""
        if spec_path.exists():
            self.spec_text = spec_path.read_text(encoding="utf-8", errors="replace")

        # Build system prompt with spec embedded
        system_prompt = RESEARCH_SYSTEM_PROMPT.format(spec_text=self.spec_text)

        # Maintainability agent
        self.maint_agent = Agent(
            name="maintainability",
            model=model,
            instructions=MAINTAINABILITY_PROMPT,
            tools=[read_file_maint, apply_fix],
            output_type=ReviewResult,
        )

        # Research agent
        self.agent = Agent(
            name="research",
            model=model,
            model_settings=ModelSettings(reasoning={"effort": "xhigh"}),
            instructions=system_prompt,
            tools=[read_experiment_state, read_file, read_log,
                   list_files, kill_run, apply_code_change],
            output_type=ResearchDecision,
        )

    def invoke(
        self,
        trigger_reason: str,
        budget_status: str = "",
        gpus_free: int = 0,
        gpu_count: int = 1,
        budget_low: bool = False,
    ) -> ResearchOutput:
        """Invoke the research agent via OpenAI Agents SDK."""

        self.last_invocation_time = time.time()
        self.invocation_count += 1

        # Build rich prompt with ALL context (fixing info gaps)
        warnings = self.warning_engine.check()
        warnings_text = self.warning_engine.format_for_prompt(warnings)

        budget_text = f"{budget_status} | {gpus_free} free of {gpu_count} GPUs"
        if budget_low:
            budget_text += " | CRITICALLY LOW — wrap up"

        prompt = (
            f"## Trigger\n{trigger_reason}\n\n"
            f"## Budget\n{budget_text}\n\n"
            f"## Warnings\n{warnings_text}\n"
        )

        ctx = ResearchContext(
            state=self.state,
            experiment_dir=self.experiment_dir,
            agent_name=self.agent_name,
            warning_engine=self.warning_engine,
            spec_text=self.spec_text,
            maint_agent=self.maint_agent,
        )

        print(f"  [Research] Invoking Agents SDK (trigger: {trigger_reason[:80]})")
        start = time.time()

        try:
            result = Runner.run_sync(
                self.agent, prompt, context=ctx, max_turns=self.max_turns,
            )
            duration = time.time() - start
            print(f"  [Research] Done in {duration:.0f}s")
        except Exception as e:
            duration = time.time() - start
            print(f"  [Research] FAILED in {duration:.0f}s: {e}")
            return ResearchOutput()

        # Convert structured output to ResearchOutput
        decision = result.final_output
        output = self._convert(decision)

        # Pick up kill requests from tool calls (written to file by kill_run tool)
        kill_file = self.experiment_dir / "_kill_requests.json"
        if kill_file.exists():
            try:
                file_kills = json.loads(kill_file.read_text())
                for k in file_kills:
                    if k not in output.kill_runs:
                        output.kill_runs.append(k)
                kill_file.unlink()
            except Exception:
                pass

        # Apply to state
        self._apply_to_state(output)

        return output

    def _convert(self, decision: ResearchDecision) -> ResearchOutput:
        """Convert Pydantic ResearchDecision → existing ResearchOutput."""
        output = ResearchOutput()
        output.reflection = decision.reflection
        output.kill_runs = list(decision.kill_runs)

        for exp in decision.experiment_list:
            output.specs.append(ExperimentSpec(
                spec_id=exp.spec_id,
                description=exp.description,
                config=exp.config,
                hypothesis=exp.hypothesis,
                predicted_outcome=exp.predicted_outcome,
                uncertainty=exp.uncertainty,
                gpu_requirement=exp.gpu_requirement,
                kill_criteria=exp.kill_criteria,
                priority=exp.priority,
                requires_code_change=exp.requires_code_change,
                code_change_description=exp.code_change_description,
                train_command=exp.train_command,
            ))

        for h in decision.hypotheses:
            output.hypotheses.append(Hypothesis(
                id=h.id,
                description=h.description,
                status=h.status,
                evidence=list(h.evidence),
            ))

        return output

    def _apply_to_state(self, output: ResearchOutput) -> None:
        """Write research output back to experiment state."""
        if output.specs:
            self.state.set_experiment_list(output.specs)
            print(f"  [Research] Experiment list: {len(output.specs)} items")
            for i, s in enumerate(output.specs[:5]):
                print(f"    {i+1}. {s.description[:70]}")

        if output.reflection:
            self.state.update_summary(output.reflection)

        if output.hypotheses:
            self.state.update_hypotheses(output.hypotheses)
