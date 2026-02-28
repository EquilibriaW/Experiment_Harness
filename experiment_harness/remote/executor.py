"""Executor agent: implements code/config changes for an experiment spec.

The executor is only invoked when an experiment spec requires code changes.
For config-only experiments, the run manager launches training directly.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from experiment_state import ExperimentState


EXECUTOR_PROMPT = """\
You are an ML experiment implementer. You implement ONE specific change.

# Experiment Spec

{spec_text}

# What To Implement

{code_change_description}

# Current State

{state_summary}

# Instructions

Make the code changes described above. Be surgical — change only what's needed.

Rules:
1. Your code must pass `pytest -x --tb=short -q` after changes.
2. Do NOT change hyperparameters or configs beyond what the spec requires.
3. Do NOT refactor, reorganize, or "improve" unrelated code.
4. If the change requires new dependencies, add them to requirements.txt.
5. The training command will be: {train_command}
   Make sure it works with your changes.

Work in {experiment_dir}.
"""


class Executor:
    """Invokes a coding agent to implement code changes."""

    def __init__(
        self,
        state: ExperimentState,
        experiment_dir: Path,
        spec_path: Path,
        agent_name: str = "codex",
        agent_max_turns: int = 20,
    ) -> None:
        self.state = state
        self.experiment_dir = experiment_dir
        self.spec_path = spec_path
        self.agent_name = agent_name
        self.agent_max_turns = agent_max_turns

    def execute(self, spec: dict) -> bool:
        """Implement code changes for a spec. Returns True on success."""
        from _agent_utils import get_agent, read_file

        spec_text = read_file(self.spec_path)
        state_summary = self.state.agent_summary(include_list=False)
        train_command = spec.get("train_command", "python train.py --config $RUN_CONFIG")

        prompt = EXECUTOR_PROMPT.format(
            spec_text=spec_text,
            code_change_description=spec.get("code_change_description", spec.get("description", "")),
            state_summary=state_summary,
            train_command=train_command,
            experiment_dir=self.experiment_dir,
        )

        agent = get_agent(self.agent_name, self.agent_max_turns)
        print(f"  [Executor] Invoking {self.agent_name}...")
        start = time.time()
        result = agent.run(prompt=prompt, working_dir=str(self.experiment_dir))
        duration = time.time() - start
        print(f"  [Executor] Done in {duration:.0f}s (exit={result.exit_code})")

        self.state.record_agent_call(cost_usd=result.estimated_cost_usd or 0.0)

        # Validate with smoke test
        smoke_ok = self._smoke_test()
        if not smoke_ok:
            print(f"  [Executor] Smoke test failed after changes, attempting fix...")
            self._auto_fix_smoke()
            smoke_ok = self._smoke_test()
            if not smoke_ok:
                print(f"  [Executor] Smoke test still failing")

        return smoke_ok

    def _smoke_test(self) -> bool:
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-x", "--tb=short", "-q"],
                capture_output=True, text=True,
                cwd=str(self.experiment_dir),
                timeout=120,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _auto_fix_smoke(self) -> None:
        """Quick agent call to fix smoke test failures."""
        from _agent_utils import get_agent

        proc = subprocess.run(
            ["python", "-m", "pytest", "-x", "--tb=long", "-q"],
            capture_output=True, text=True,
            cwd=str(self.experiment_dir),
            timeout=120,
        )

        prompt = (
            f"The smoke test failed. Fix the errors.\n\n"
            f"```\n{proc.stdout[-2000:]}\n{proc.stderr[-500:]}\n```\n\n"
            f"Only fix syntax/import/missing-file errors. "
            f"Do not change experiment logic."
        )

        agent = get_agent(self.agent_name, 10)
        agent.run(prompt=prompt, working_dir=str(self.experiment_dir))
        self.state.record_agent_call()
