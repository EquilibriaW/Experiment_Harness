"""The core autonomous experiment loop (Ralph Wiggum Mode).

Runs entirely on the remote pod inside tmux (survives SSH disconnect).

Each round:
  1. IMPLEMENT  — fresh agent reads spec + reflection.md, modifies code
  2. VALIDATE   — smoke test (pytest)
  3. TRAIN      — run training (skips if results already exist)
  4. REFLECT    — fresh agent rewrites reflection.md with data/analysis/next steps
  5. REVIEW     — reviewer↔implementer cycle (max N turns), each turn rewrites reflection.md
  6. BUDGET     — check limits, next round or stop

Every agent call is a fresh subprocess = fresh context.
reflection.md is the ONLY persistent memory between calls.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from budget_guard import BudgetGuard
from log_manager import ReflectionManager


# ── Prompt templates ────────────────────────────────────────────────────────

IMPLEMENT_PROMPT = """\
You are an ML experiment implementer agent working in {experiment_dir}.

# Goal

The point of this experiment is to train a model that BEATS nanochat baseline.
Every round should make progress toward that goal.

CRITICAL CONSTRAINTS — read carefully:
- You are running on an NVIDIA H100 80GB GPU. USE IT. Do not run synthetic or
  tiny models. Use the REAL nanochat dataset and enough iterations to get
  meaningful loss curves (1000+ iterations minimum).
- GPU utilization should be HIGH (>50%). If your config uses <10% of an H100,
  your batch size or model is way too small. Scale up.
- Do NOT run toy/synthetic/smoke configurations for actual experiments. Those
  are only for pytest smoke tests. Real training = real data, real scale.
- You are paying for this GPU by the hour. Every minute at low utilization is
  wasted money. Be aggressive with scale.

# Experiment Spec

{spec}

# Instructions

Read reflection.md in your working directory for full context — it contains all
prior results, analysis, and a prioritized list of what to try next.

Your job: implement the next improvements to the experiment. This includes BOTH
code changes AND training configuration:

- Update training scripts, configs, and launch arguments so the next training
  run uses appropriate scale (iterations, batch size, model size, etc.).
  The train phase will call `python train.py` with no args by default — if the
  default args produce a toy/smoke run, FIX the defaults or create a proper
  launch config.
- Implement code changes from reflection.md's Next Steps.
- Fix errors mentioned in reflection.md.

Rules:
- For ambiguous issues, let data from training runs inform your decisions — don't guess.
- Your code must pass `pytest` (smoke tests verify imports and train.py exists).
- Budget remaining: {budget_info}
- Focus on what reflection.md says to try next.
"""

TRAIN_PROMPT_EXPLICIT = """\
You are an ML training runner agent working in {experiment_dir}.

# Instructions

Your ONLY job is to run training and capture the results.

1. Run this training command:
   ```
   {train_command}
   ```
2. If the command fails, diagnose the error and fix it. Common fixes:
   - Wrong device (cpu vs cuda) → fix device config
   - Missing directory → create it
   - Shape mismatch → fix tensor dimensions
   - OOM → reduce batch size
   - Missing dependency → install it
   - Bad file path → correct it
3. After fixing, re-run the training command. Repeat up to 3 times.
4. If after 3 attempts training still fails, STOP and write a summary of
   what went wrong to stdout so the reflect phase can address it.

Rules:
- You CAN modify code, but ONLY to fix errors that block training from running.
- Do NOT restructure the experiment, change hyperparameters for "improvement",
  or refactor code. Only fix what's broken.
- Capture and print the full training output including any metrics.
- If training succeeds, print "TRAINING COMPLETE" at the end.
"""

TRAIN_PROMPT_AUTO = """\
You are an ML training runner agent working in {experiment_dir}.

# Instructions

Your job is to run a REAL training run and capture the results.

1. Read the training scripts and reflection.md to understand how to launch training.
2. Determine the correct command with appropriate flags:
   - Use CUDA/GPU if available (`nvidia-smi` to check). NEVER default to CPU when a GPU exists.
   - Use the real dataset, not synthetic/toy/smoke defaults.
   - Use reasonable hyperparameters for a real run (read reflection.md for guidance).
3. Run the training command.
4. If the command fails, diagnose the error and fix it. Common fixes:
   - Wrong device (cpu vs cuda) → fix device config
   - Missing directory → create it
   - Shape mismatch → fix tensor dimensions
   - OOM → reduce batch size
   - Missing dependency → install it
   - Bad file path → correct it
5. After fixing, re-run the training command. Repeat up to 3 times.
6. If after 3 attempts training still fails, STOP and write a summary of
   what went wrong to stdout so the reflect phase can address it.

Rules:
- You CAN modify code, but ONLY to fix errors that block training from running.
- Do NOT restructure the experiment, change hyperparameters for "improvement",
  or refactor code. Only fix what's broken.
- Capture and print the full training output including any metrics.
- If training succeeds, print "TRAINING COMPLETE" at the end.
"""

REFLECT_PROMPT = """\
You are a scientific experiment documenter working in {experiment_dir}.

# Experiment Spec

{spec}

# Results from This Round

{results_section}

# Instructions

Read the current code files and the results above. Then REWRITE reflection.md
completely. This file is the ONLY persistent memory — every future agent call
starts fresh and reads only this file for context.

reflection.md MUST contain:
1. **Experiment Goal** — brief summary of what we're building and why
2. **Current Architecture** — what the code does right now, key design choices
3. **Results** — data from this and prior rounds (preserve important numbers
   from the current reflection.md, drop noise)
4. **Analysis** — what worked, what didn't, why
5. **Errors / Problems** — any failures, OOM, divergence, etc. that need fixing
6. **Next Steps** — concrete, prioritized list of what to try next

HARD LIMIT: {word_limit} words maximum. Summarize ruthlessly. Keep data, drop prose.
Output ONLY the markdown content for reflection.md, nothing else.
"""

REVIEWER_PROMPT = """\
You are a senior ML researcher reviewing an experiment implementation.

# Experiment Spec

{spec}

# Instructions

Read reflection.md and ALL Python files in this directory. Another agent wrote
this code. Provide a thorough, constructive code review.

Cover:
1. **Correctness** — bugs, logic errors, wrong implementations
2. **ML Best Practices** — training procedure, loss functions, evaluation, data handling
3. **Experiment Design** — are we testing the right hypothesis? Missing controls/ablations?
4. **Code Quality** — readability, organization, unnecessary complexity
5. **Top 3 Priority Changes** — ranked by expected impact on experiment quality

Be specific. Reference file names and describe exact issues. Be constructive.

DO NOT modify any files. Output ONLY your review as text.
"""

REVIEW_IMPLEMENT_PROMPT = """\
You are an ML experiment implementer agent working in {experiment_dir}.

# Experiment Spec

{spec}

# Code Review

{review_text}

# Instructions

Read reflection.md and the code review above. Decide which reviewer suggestions
to implement — focus on the highest-impact changes.

Make the code modifications. Your code must still pass `pytest`.

After making changes, REWRITE reflection.md to note:
- What you changed and why (based on the review)
- Updated analysis and next steps

HARD LIMIT on reflection.md: {word_limit} words. Summarize ruthlessly.
"""


# ── Agent loading ───────────────────────────────────────────────────────────

def get_agent_adapter(agent_name: str, max_turns: int = 30):
    """Dynamically import and return the right agent adapter."""
    if agent_name == "codex":
        from codex_adapter import CodexAdapter
        return CodexAdapter()
    elif agent_name == "claude":
        from claude_adapter import ClaudeAdapter
        return ClaudeAdapter(max_turns=max_turns)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ── Loop runner ─────────────────────────────────────────────────────────────

class LoopRunner:
    """The autonomous experiment loop."""

    def __init__(
        self,
        spec_path: str,
        experiment_dir: str,
        agent_name: str = "codex",
        reviewer_agent_name: Optional[str] = None,
        max_hours: float = 24.0,
        reflection_word_limit: int = 8000,
        review_turns: int = 2,
        agent_max_turns: int = 30,
        train_command: Optional[str] = None,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.experiment_dir = Path(experiment_dir)
        self.backup_dir = self.experiment_dir / "backups"
        self.logs_dir = self.experiment_dir / "logs"
        self.reflection_path = self.experiment_dir / "reflection.md"

        self.agent_name = agent_name
        self.reviewer_agent_name = reviewer_agent_name or agent_name
        self.agent_max_turns = agent_max_turns
        self.review_turns = review_turns
        self.train_command = train_command

        self.budget = BudgetGuard(max_hours=max_hours)
        self.reflection = ReflectionManager(
            self.reflection_path, word_limit=reflection_word_limit,
        )
        self.reflection_word_limit = reflection_word_limit

    # ── Main loop ───────────────────────────────────────────────

    def run(self) -> None:
        print(f"=== Experiment Loop Starting ===")
        print(f"Implementer: {self.agent_name}")
        print(f"Reviewer:    {self.reviewer_agent_name}")
        print(f"Spec:        {self.spec_path}")
        print(f"Dir:         {self.experiment_dir}")
        print(f"Review turns per round: {self.review_turns}")
        print(f"Budget:      {self.budget.status()}")
        print()

        self._setup_dirs()

        while True:
            stop_reason = self.budget.check()
            if stop_reason:
                print(f"\n=== STOPPING: {stop_reason} ===")
                break

            round_num = self.budget.rounds_completed + 1
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}")
            print(f"  {self.budget.status()}")
            print(f"{'='*60}\n")

            self._run_round(round_num)

        print(f"\n=== Experiment Complete ===")
        print(f"Final status: {self.budget.status()}")
        print(f"Reflection: {self.reflection_path}")

    def _run_round(self, round_num: int) -> None:
        self._safety_checks(round_num)

        # ── Phase 1: IMPLEMENT (fresh context) ──────────────────
        print(f"[Round {round_num}] Phase 1: IMPLEMENT")
        self._phase_implement(round_num)

        # ── Phase 2: VALIDATE (smoke test only) ─────────────────
        print(f"[Round {round_num}] Phase 2: VALIDATE")
        smoke_ok, smoke_output = self._phase_validate(round_num)

        # ── Phase 3: TRAIN ────────────────────────────────────────
        print(f"[Round {round_num}] Phase 3: TRAIN")
        train_ran, train_output = self._phase_train(round_num)

        # ── Phase 4: REFLECT (receives smoke + train results) ───
        print(f"[Round {round_num}] Phase 4: REFLECT")
        self._phase_reflect(round_num, smoke_ok, smoke_output,
                            train_ran=train_ran, train_output=train_output)

        # ── Phase 5: REVIEW CYCLE ──────────────────────────────
        print(f"[Round {round_num}] Phase 5: REVIEW ({self.review_turns} turns max)")
        self._phase_review_cycle(round_num)

        # ── Budget ──────────────────────────────────────────────
        self.budget.record_round()
        print(f"[Round {round_num}] Complete. {self.budget.status()}")

    # ── Phase implementations ───────────────────────────────────

    def _phase_implement(self, round_num: int) -> None:
        """Spawn a fresh implementer: reads spec + reflection.md, modifies code."""
        agent = get_agent_adapter(self.agent_name, self.agent_max_turns)
        spec = self._read_spec()

        prompt = IMPLEMENT_PROMPT.format(
            experiment_dir=self.experiment_dir,
            spec=spec,
            budget_info=self.budget.status(),
        )

        result = agent.run(
            prompt=prompt,
            working_dir=str(self.experiment_dir),
        )
        print(f"  Implementer: exit={result.exit_code}, "
              f"time={result.duration_seconds:.0f}s")

        self._log_phase(
            round_num, "implement",
            agent=self.agent_name, prompt=prompt,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )

    def _phase_validate(self, round_num: int) -> tuple[bool, str]:
        """Smoke test only. Training is the agent's responsibility."""
        print(f"  Running smoke tests...")
        smoke_ok, smoke_output = self._run_smoke_test()
        if not smoke_ok:
            print(f"  Smoke test FAILED — attempting auto-fix")
            self._auto_fix_smoke(round_num, smoke_output)
            smoke_ok, smoke_output = self._run_smoke_test()
            if not smoke_ok:
                print(f"  Smoke test still failing after fix attempt")

        status = "PASSED" if smoke_ok else "FAILED"
        print(f"  Smoke test: {status}")

        self._log_phase(
            round_num, "smoke_test",
            stdout=smoke_output,
            extra={"passed": smoke_ok},
        )

        return smoke_ok, smoke_output

    def _phase_train(self, round_num: int) -> tuple[bool, str]:
        """Run training."""
        # Determine the training command
        train_cmd = self.train_command
        if not train_cmd:
            # No explicit command — check there's something trainable, then let agent decide
            has_train_script = (
                (self.experiment_dir / "train.py").exists()
                or (self.experiment_dir / "main.py").exists()
                or list(self.experiment_dir.glob("scripts/train*.py"))
                or list(self.experiment_dir.glob("scripts/*train*.py"))
            )
            if not has_train_script:
                print(f"  No train_command set and no training scripts found, skipping")
                return False, "No training script found"

        agent = get_agent_adapter(self.agent_name, self.agent_max_turns)

        if train_cmd:
            print(f"  Running training: {train_cmd}")
            prompt = TRAIN_PROMPT_EXPLICIT.format(
                experiment_dir=self.experiment_dir,
                train_command=train_cmd,
            )
        else:
            print(f"  Training: agent will determine command")
            prompt = TRAIN_PROMPT_AUTO.format(
                experiment_dir=self.experiment_dir,
            )

        result = agent.run(
            prompt=prompt,
            working_dir=str(self.experiment_dir),
        )

        print(f"  Training agent: exit={result.exit_code}, "
              f"time={result.duration_seconds:.0f}s")

        self._log_phase(
            round_num, "train",
            agent=self.agent_name, prompt=prompt,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )

        # Extract human-readable text for reflect (NDJSON → readable)
        train_text = self._extract_agent_text(result.stdout) if result.stdout else ""
        if result.stderr and result.stderr.strip():
            train_text += "\n\nSTDERR:\n" + result.stderr.strip()
        if not train_text.strip():
            train_text = "(no output captured)"

        return True, train_text

    def _phase_reflect(
        self,
        round_num: int,
        smoke_ok: bool,
        smoke_output: str,
        train_ran: bool = False,
        train_output: str = "",
    ) -> None:
        """Spawn a fresh agent to rewrite reflection.md with results."""
        agent = get_agent_adapter(self.agent_name, self.agent_max_turns)
        spec = self._read_spec()

        # Build results section
        results_parts = [f"Round: {round_num}"]
        if not smoke_ok:
            results_parts.append(f"Smoke test FAILED:\n```\n{smoke_output[-1500:]}\n```")
        else:
            results_parts.append("Smoke test: PASSED")

        if train_ran and train_output.strip():
            # Include training output (truncated to last 3000 chars for context)
            truncated = train_output.strip()[-3000:]
            results_parts.append(
                f"Training output:\n```\n{truncated}\n```"
            )
        elif train_ran:
            results_parts.append("Training ran but produced no output.")

        # Snapshot reflection.md before agent runs so we can detect changes
        pre_reflection = self.reflection.read()

        prompt = REFLECT_PROMPT.format(
            experiment_dir=self.experiment_dir,
            spec=spec,
            results_section="\n\n".join(results_parts),
            word_limit=self.reflection_word_limit,
        )

        result = agent.run(
            prompt=prompt,
            working_dir=str(self.experiment_dir),
        )

        self._log_phase(
            round_num, "reflect",
            agent=self.agent_name, prompt=prompt,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )

        # The agent writes reflection.md directly on disk. Read it back
        # and enforce the word limit. Don't parse stdout — it's NDJSON for
        # Codex and would corrupt the file.
        post_reflection = self.reflection.read()
        if post_reflection != pre_reflection and post_reflection.strip():
            # Agent updated it — just enforce word limit
            self.reflection.write(post_reflection)
            print(f"  Reflection updated ({self.reflection.word_count()} words)")
        else:
            # Agent didn't touch the file — append smoke test results as fallback
            fallback_section = (
                f"\n\n## Round {round_num} Results\n\n"
                + "\n\n".join(results_parts) + "\n"
            )
            self._append_to_reflection(fallback_section)
            print(f"  Reflection: fallback append (agent didn't write it)")

    def _phase_review_cycle(self, round_num: int) -> None:
        """Reviewer↔implementer cycle, max N turns. Each turn rewrites reflection.md."""
        for turn in range(1, self.review_turns + 1):
            print(f"  Review turn {turn}/{self.review_turns}")

            # ── Reviewer (fresh context, reads code in temp copy) ───
            review_text = self._run_reviewer(round_num, turn)
            if not review_text or len(review_text.strip()) < 20:
                print(f"    Reviewer produced no meaningful output, skipping cycle")
                break

            print(f"    Review: {len(review_text)} chars")

            # ── Implementer responds to review (fresh context) ──────
            self._run_review_implementer(round_num, turn, review_text)

            # ── Quick smoke test to verify review changes ───────────
            smoke_ok, smoke_output = self._run_smoke_test()
            if not smoke_ok:
                print(f"    Post-review smoke test FAILED, attempting fix")
                self._auto_fix_smoke(round_num, smoke_output)

        print(f"  Review cycle complete")

    # ── Reviewer / review-implementer ───────────────────────────

    def _run_reviewer(self, round_num: int, turn: int) -> str:
        """Run reviewer in a temp copy so it can't modify the real code."""
        review_dir = Path(tempfile.mkdtemp(prefix=f"review_r{round_num}_t{turn}_"))
        try:
            # Copy source files only — skip heavy data/checkpoint dirs
            shutil.copytree(
                self.experiment_dir, review_dir, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "runs", "checkpoints*", "backups", "logs", "metrics",
                    "__pycache__", ".git", "*.pt", "*.pth", "*.ckpt",
                    "*.bin", "*.safetensors", "*.npy", "*.npz", "wandb",
                ),
            )

            agent = get_agent_adapter(self.reviewer_agent_name, self.agent_max_turns)
            spec = self._read_spec()
            prompt = REVIEWER_PROMPT.format(spec=spec)

            result = agent.run(
                prompt=prompt,
                working_dir=str(review_dir),
            )

            self._log_phase(
                round_num, f"review_t{turn}_reviewer",
                agent=self.reviewer_agent_name, prompt=prompt,
                stdout=result.stdout, stderr=result.stderr,
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
            )

            return self._extract_agent_text(result.stdout) if result.stdout else ""
        finally:
            shutil.rmtree(review_dir, ignore_errors=True)

    def _run_review_implementer(
        self, round_num: int, turn: int, review_text: str,
    ) -> None:
        """Fresh implementer: reads review + reflection.md, makes changes."""
        agent = get_agent_adapter(self.agent_name, self.agent_max_turns)
        spec = self._read_spec()

        prompt = REVIEW_IMPLEMENT_PROMPT.format(
            experiment_dir=self.experiment_dir,
            spec=spec,
            review_text=review_text,
            word_limit=self.reflection_word_limit,
        )

        result = agent.run(
            prompt=prompt,
            working_dir=str(self.experiment_dir),
        )
        print(f"    Review-implementer: exit={result.exit_code}, "
              f"time={result.duration_seconds:.0f}s")

        self._log_phase(
            round_num, f"review_t{turn}_implementer",
            agent=self.agent_name, prompt=prompt,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )

    # ── Auto-fix on smoke failure ───────────────────────────────

    def _auto_fix_smoke(self, round_num: int, smoke_output: str) -> None:
        """Spawn a one-shot agent to fix smoke test failures."""
        agent = get_agent_adapter(self.agent_name, self.agent_max_turns)
        spec = self._read_spec()

        prompt = (
            f"You are working in {self.experiment_dir}.\n\n"
            f"# Experiment Spec\n\n{spec}\n\n"
            f"# Problem\n\n"
            f"The smoke test (`pytest`) failed with this output:\n\n"
            f"```\n{smoke_output[-2000:]}\n```\n\n"
            f"Fix the errors so that `pytest` passes.\n\n"
            f"Rules:\n"
            f"- Only fix syntax errors, import errors, and missing files.\n"
            f"- Do not change experiment logic or hyperparameters.\n"
            f"- If a test fails due to a timeout or runtime error during actual "
            f"training, that is NOT a smoke test issue — leave it alone."
        )

        result = agent.run(
            prompt=prompt,
            working_dir=str(self.experiment_dir),
        )

        self._log_phase(
            round_num, "auto_fix_smoke",
            agent=self.agent_name, prompt=prompt,
            stdout=result.stdout, stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _archive_previous_run(self) -> None:
        """Archive logs from a previous run into traces/run_{timestamp}/."""
        if not self.logs_dir.exists():
            return
        round_dirs = [d for d in self.logs_dir.iterdir()
                      if d.is_dir() and d.name.startswith("round_")]
        if not round_dirs:
            return

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        traces_dir = self.experiment_dir / "traces"
        archive = traces_dir / f"run_{ts}"
        archive.mkdir(parents=True, exist_ok=True)

        # Move logs/ contents into archive/logs/
        archive_logs = archive / "logs"
        shutil.move(str(self.logs_dir), str(archive_logs))
        print(f"  Archived {len(round_dirs)} rounds → {archive.relative_to(self.experiment_dir)}")

        # Copy reflection.md into archive (copy, not move — keep as starting context)
        if self.reflection_path.exists():
            shutil.copy2(str(self.reflection_path), str(archive / "reflection.md"))

        # Copy loop.log if it exists
        loop_log = self.experiment_dir / "loop.log"
        if loop_log.exists():
            shutil.copy2(str(loop_log), str(archive / "loop.log"))

    def _setup_dirs(self) -> None:
        self._archive_previous_run()

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if not self.reflection_path.exists():
            spec_text = self._read_spec()[:500]
            self.reflection.write(
                f"# Reflection\n\n"
                f"## Experiment Goal\n\n{spec_text}\n\n"
                f"## Current Architecture\n\nNo implementation yet.\n\n"
                f"## Results\n\nNo training runs completed.\n\n"
                f"## Next Steps\n\n"
                f"1. Read the experiment spec\n"
                f"2. Create initial implementation\n"
                f"3. Run first training\n"
            )

    def _log_phase(
        self,
        round_num: int,
        phase: str,
        *,
        agent: str = "",
        prompt: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration_seconds: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a structured JSON trace for one phase/step."""
        round_dir = self.logs_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Auto-number files so they sort chronologically
        existing = sorted(round_dir.glob("*.json"))
        seq = len(existing) + 1

        entry: dict[str, Any] = {
            "phase": phase,
            "round": round_num,
            "seq": seq,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "exit_code": exit_code,
            "prompt": prompt,
            "stdout": stdout,
            "stderr": stderr,
        }
        if extra:
            entry.update(extra)

        path = round_dir / f"{seq:02d}_{phase}.json"
        path.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")

    def _read_spec(self) -> str:
        if self.spec_path.exists():
            return self.spec_path.read_text(encoding="utf-8")
        return "(no spec found)"

    def _append_to_reflection(self, text: str) -> None:
        current = self.reflection.read()
        self.reflection.write(current + text)

    def _safety_checks(self, round_num: int) -> None:
        round_backup = self.backup_dir / f"round_{round_num}"
        round_backup.mkdir(parents=True, exist_ok=True)

        if self.spec_path.exists():
            shutil.copy2(self.spec_path, round_backup / "spec.md")
        if self.reflection_path.exists():
            shutil.copy2(self.reflection_path, round_backup / "reflection.md")

        try:
            stat = shutil.disk_usage(self.experiment_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 5.0:
                print(f"  [WARNING] Low disk space: {free_gb:.1f} GB free")
                self._cleanup_old_checkpoints()
        except OSError:
            pass

    def _cleanup_old_checkpoints(self) -> None:
        ckpt_dirs = sorted(self.experiment_dir.glob("checkpoint*"))
        if len(ckpt_dirs) > 2:
            for d in ckpt_dirs[:-2]:
                print(f"  [Cleanup] Removing old checkpoint: {d}")
                shutil.rmtree(d, ignore_errors=True)

    def _run_smoke_test(self) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-x", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                cwd=str(self.experiment_dir),
            )
            output = proc.stdout + "\n" + proc.stderr
            return proc.returncode == 0, output
        except FileNotFoundError:
            return False, "pytest not found"

    def _extract_agent_text(self, agent_output: str) -> str:
        """Extract human-readable text from agent output.

        Handles Codex NDJSON (extracts agent_message events) and plain text.
        Used for reviewer output where the agent writes to stdout, not files.
        """
        lines = agent_output.strip().splitlines()
        if not lines:
            return ""

        # Check if it looks like NDJSON (first line parses as JSON)
        try:
            json.loads(lines[0])
            is_ndjson = True
        except (json.JSONDecodeError, TypeError):
            is_ndjson = False

        if is_ndjson:
            messages = []
            for line in lines:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        etype = event.get("type", "")
                        if etype == "agent_message":
                            content = event.get("content", "")
                            if content:
                                messages.append(content)
                        elif etype == "message" and "content" in event:
                            messages.append(event["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
            if messages:
                return "\n\n".join(messages)
            # No agent_message events — fall through to raw text

        # Plain text — strip code fences
        text = agent_output.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        return text


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Experiment loop runner")
    p.add_argument("--spec", required=True)
    p.add_argument("--experiment-dir", default="/workspace/experiment")
    p.add_argument("--agent", default="codex", choices=["codex", "claude"])
    p.add_argument("--reviewer-agent", default=None, choices=["codex", "claude"])
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--reflection-word-limit", type=int, default=8000)
    p.add_argument("--review-turns", type=int, default=2)
    p.add_argument("--agent-max-turns", type=int, default=30)
    p.add_argument("--train-command", default=None,
                   help="Command to run training (e.g. 'python train.py')")

    args = p.parse_args()

    runner = LoopRunner(
        spec_path=args.spec,
        experiment_dir=args.experiment_dir,
        agent_name=args.agent,
        reviewer_agent_name=args.reviewer_agent,
        max_hours=args.max_hours,
        reflection_word_limit=args.reflection_word_limit,
        review_turns=args.review_turns,
        agent_max_turns=args.agent_max_turns,
        train_command=args.train_command,
    )
    runner.run()


if __name__ == "__main__":
    main()
