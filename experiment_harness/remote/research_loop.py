"""Research loop: unified research agent via dspy.RLM.

Uses DSPy's RLM module (arXiv 2512.24601) to let the research agent
programmatically explore experiment state through a sandboxed REPL.

The RLM enforces the right architecture:
  - max_output_chars truncates REPL output → forces symbolic access
  - llm_query() available in REPL → enables recursive sub-calls
  - SUBMIT() writes output via environment → not autoregressive
  - State is a variable in the REPL, not text in the prompt

The harness just calls rlm(experiment_state=..., trigger=...) and
gets back structured decisions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from experiment_state import ExperimentState, ExperimentSpec, Hypothesis
from warning_engine import WarningEngine


def _make_read_log(experiment_dir: Path):
    """Factory: returns a read_log tool bound to the experiment dir."""
    def read_log(path: str, tail_lines: int = 100) -> str:
        """Read a log file from the experiment directory.
        
        Args:
            path: Relative path to the log file (e.g., "run_logs/run_005.log")
            tail_lines: Number of lines from the end to return (default 100)
        """
        full = experiment_dir / path
        if not full.exists():
            return f"(file not found: {path})"
        try:
            lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-tail_lines:])
        except Exception as e:
            return f"(error: {e})"
    return read_log


def _make_read_file(experiment_dir: Path):
    """Factory: returns a read_file tool bound to the experiment dir."""
    def read_file(path: str) -> str:
        """Read any file in the experiment directory.
        
        Args:
            path: Relative path (e.g., "train.py", "config.json")
        """
        full = experiment_dir / path
        if not full.exists():
            return f"(file not found: {path})"
        try:
            return full.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"(error: {e})"
    return read_file


def _make_list_files(experiment_dir: Path):
    """Factory: returns a list_files tool."""
    def list_files(subdir: str = ".") -> str:
        """List files in a subdirectory of the experiment.
        
        Args:
            subdir: Subdirectory to list (default "." for root)
        """
        target = experiment_dir / subdir
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
    return list_files


def _make_kill_run(experiment_dir: Path):
    """Factory: returns a kill_run tool that writes kill requests."""
    def kill_run(run_id: str) -> str:
        """Request that a running experiment be killed.
        
        Args:
            run_id: The run_id to kill (e.g., "run_005")
        """
        kill_file = experiment_dir / "_kill_requests.json"
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
    return kill_run


def _make_apply_code_change(experiment_dir: Path, agent_name: str):
    """Factory: returns a tool that applies code changes with validation.

    Unlike a raw CLI call, this:
    1. Creates a git checkpoint before changes
    2. Runs the coding agent to make the change
    3. Runs smoke tests (pytest) to validate
    4. Reverts on test failure
    Returns the diff + test result, not just "done."
    """
    import subprocess as _sp

    def apply_code_change(instruction: str) -> str:
        """Have a coding agent make a specific change, then validate with tests.

        The change is validated before being accepted:
        1. Git checkpoint created (if git available)
        2. Coding agent makes the change
        3. Smoke tests run to verify nothing broke
        4. If tests fail, changes are reverted

        Args:
            instruction: What to change and why (e.g.,
                "In train.py, replace linear LR schedule with cosine annealing.
                 Keep warmup_steps=100. Set T_max to total_steps.")
        """
        cwd = str(experiment_dir)

        # 1. Git checkpoint
        has_git = False
        try:
            _sp.run(["git", "rev-parse", "--is-inside-work-tree"],
                     capture_output=True, cwd=cwd, timeout=5)
            _sp.run(["git", "add", "-A"], capture_output=True, cwd=cwd, timeout=10)
            _sp.run(["git", "commit", "-m", f"pre-change: {instruction[:60]}",
                      "--allow-empty"],
                     capture_output=True, cwd=cwd, timeout=10)
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
            agent_result = _sp.run(
                cmd, input=stdin_text,
                capture_output=True, text=True,
                timeout=300, cwd=cwd,
            )
        except _sp.TimeoutExpired:
            return "Failed: coding agent timed out after 300s"
        except FileNotFoundError:
            return f"Failed: {agent_name} CLI not found"

        if agent_result.returncode != 0:
            err = agent_result.stderr.strip()
            return f"Failed (exit {agent_result.returncode}):\n{err[-500:]}"

        # 3. Get diff
        diff = ""
        try:
            diff_result = _sp.run(
                ["git", "diff", "--stat"], capture_output=True, text=True,
                cwd=cwd, timeout=10,
            )
            diff = diff_result.stdout.strip()
        except Exception:
            diff = "(could not get diff)"

        # 4. Run smoke tests
        test_passed = True
        test_output = ""
        try:
            test_result = _sp.run(
                ["python", "-m", "pytest", "--tb=short", "-q",
                 "--timeout=60", "-x"],
                capture_output=True, text=True,
                timeout=120, cwd=cwd,
            )
            test_output = test_result.stdout.strip()[-500:]
            if test_result.returncode != 0:
                test_passed = False
        except _sp.TimeoutExpired:
            test_output = "(tests timed out)"
            test_passed = False
        except FileNotFoundError:
            # No pytest — try basic import check
            try:
                import_result = _sp.run(
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

        # 5. Revert if tests failed
        if not test_passed and has_git:
            try:
                _sp.run(["git", "checkout", "."], capture_output=True,
                         cwd=cwd, timeout=10)
            except Exception:
                pass
            return (f"REVERTED — tests failed after change.\n"
                    f"Diff was:\n{diff}\n\nTest output:\n{test_output}")

        return (f"Applied successfully.\n"
                f"Diff:\n{diff}\n\nTests: {'PASSED' if test_passed else 'NO TESTS'}\n"
                f"{test_output}")
    return apply_code_change


class ResearchLoop:
    """Research agent powered by dspy.RLM.
    
    The RLM treats experiment_state.json as external data that the LLM
    explores programmatically via a sandboxed REPL. Output truncation
    (max_output_chars) forces the LLM to use symbolic access patterns
    and sub-LLM calls instead of trying to read everything at once.
    """

    def __init__(
        self,
        state: ExperimentState,
        experiment_dir: Path,
        spec_path: Path,
        warning_engine: WarningEngine,
        model: str = "anthropic/claude-opus-4-5",
        sub_model: str = "anthropic/claude-haiku-4-5",
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        agent_name: str = "codex",
    ) -> None:
        self.state = state
        self.experiment_dir = experiment_dir
        self.spec_path = spec_path
        self.warning_engine = warning_engine
        self.model = model
        self.sub_model = sub_model
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.agent_name = agent_name

        self.last_invocation_time: float = 0.0
        self.invocation_count: int = 0

        self._rlm = None  # lazy init (dspy may not be installed)

    def _get_rlm(self):
        """Lazy-init the dspy.RLM module."""
        if self._rlm is not None:
            return self._rlm

        import dspy

        dspy.configure(lm=dspy.LM(self.model, reasoning_effort="extra_high"))

        if self.sub_model.startswith("cli:"):
            from cli_lm import make_cli_lm
            sub_lm = make_cli_lm(self.sub_model)
        else:
            sub_lm = dspy.LM(self.sub_model)

        tools = [
            _make_read_log(self.experiment_dir),
            _make_read_file(self.experiment_dir),
            _make_list_files(self.experiment_dir),
            _make_kill_run(self.experiment_dir),
            _make_apply_code_change(self.experiment_dir, self.agent_name),
        ]

        # The signature: state + trigger in, decisions out
        # experiment_list is JSON array of experiment specs
        # reflection is the agent's persistent memory
        # hypotheses is JSON array of hypothesis objects
        self._rlm = dspy.RLM(
            "experiment_state, trigger, warnings, budget -> "
            "experiment_list, reflection, hypotheses",
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            max_output_chars=self.max_output_chars,
            sub_lm=sub_lm,
            tools=tools,
        )

        return self._rlm

    def invoke(
        self,
        trigger_reason: str,
        budget_status: str = "",
        gpus_free: int = 0,
        gpu_count: int = 1,
        budget_low: bool = False,
    ) -> ResearchOutput:
        """Invoke the research agent via dspy.RLM."""

        self.last_invocation_time = time.time()
        self.invocation_count += 1

        # ── Serialize state as the "P" variable ─────────────────
        current_state = self.state.read()
        state_json = json.dumps(current_state, default=str)

        # Warnings are small and scaffold-generated — include directly
        warnings = self.warning_engine.check()
        warnings_text = self.warning_engine.format_for_prompt(warnings)

        budget_text = (
            f"{budget_status} | {gpus_free} free of {gpu_count} GPUs"
        )
        if budget_low:
            budget_text += " | CRITICALLY LOW — wrap up"

        # ── Call the RLM ─────────────────────────────────────────
        rlm = self._get_rlm()
        print(f"  [Research] Invoking RLM (trigger: {trigger_reason[:80]})")
        start = time.time()

        try:
            result = rlm(
                experiment_state=state_json,
                trigger=trigger_reason,
                warnings=warnings_text,
                budget=budget_text,
            )
            duration = time.time() - start
            print(f"  [Research] Done in {duration:.0f}s")

            # Debug: check if state leaked into the prompt
            self._debug_check_context(result, len(state_json))

        except Exception as e:
            duration = time.time() - start
            print(f"  [Research] FAILED in {duration:.0f}s: {e}")
            return ResearchOutput()

        # ── Parse output ─────────────────────────────────────────
        # Debug: show raw RLM output fields
        for field in ("experiment_list", "reflection", "hypotheses"):
            val = getattr(result, field, None)
            if val is not None:
                preview = str(val)[:200]
                print(f"  [Research] Raw {field}: {preview}")
            else:
                print(f"  [Research] Raw {field}: None")

        output = self._parse_result(result)

        # ── Pick up kill requests from tool calls ────────────────
        kill_file = self.experiment_dir / "_kill_requests.json"
        if kill_file.exists():
            try:
                output.kill_runs = json.loads(kill_file.read_text())
                kill_file.unlink()
            except Exception:
                pass

        # ── Apply to state ───────────────────────────────────────
        self._apply_to_state(output)

        return output

    def _debug_check_context(self, result, state_bytes: int) -> None:
        """Check whether dspy.RLM smuggled the state into the prompt.

        If the state is large and appears to be entering the LM context
        directly (rather than as a REPL variable), warn and log.
        """
        trajectory = getattr(result, "trajectory", [])
        if not trajectory:
            return

        # Estimate: if the first iteration's prompt is very large relative
        # to the state, the state is probably being included verbatim
        try:
            import dspy
            lm = dspy.settings.lm
            history = getattr(lm, "history", [])
            if history:
                last_call = history[-1]
                prompt_tokens = last_call.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = last_call.get("usage", {}).get("completion_tokens", 0)
                print(f"  [Research] Tokens: {prompt_tokens} in, "
                      f"{completion_tokens} out (state: {state_bytes} bytes)")

                # Heuristic: if prompt tokens >> expected metadata size,
                # the state is likely in the prompt
                # ~4 chars per token, state + overhead should be < 5K tokens
                # if prompt > state_bytes/4, state is probably verbatim
                state_token_estimate = state_bytes // 4
                if prompt_tokens > state_token_estimate * 0.8 and state_bytes > 20_000:
                    print(f"  [Research] WARNING: prompt tokens ({prompt_tokens}) "
                          f"suggest state ({state_bytes} bytes ≈ "
                          f"{state_token_estimate} tokens) may be in the "
                          f"prompt rather than the REPL. Consider using "
                          f"file-based access instead.")
        except Exception:
            pass

    def invoke_file_based(
        self,
        trigger_reason: str,
        **kwargs,
    ) -> ResearchOutput:
        """Fallback: pass state as a file path, not a string.

        Use this if _debug_check_context warns that state is entering
        the prompt. The agent reads it via read_file() in the REPL.
        """
        import dspy

        state_path = self.experiment_dir / "experiment_state.json"
        state_size = state_path.stat().st_size if state_path.exists() else 0

        # Minimal metadata — agent must read the file
        metadata = (
            f"Experiment state is in experiment_state.json ({state_size} bytes). "
            f"Use read_file('experiment_state.json') to access it."
        )

        # Rebuild RLM with path-based signature if needed
        if not hasattr(self, '_rlm_file_based'):
            dspy.configure(lm=dspy.LM(self.model, reasoning_effort="extra_high"))
            if self.sub_model.startswith("cli:"):
                from cli_lm import make_cli_lm
                sub_lm = make_cli_lm(self.sub_model)
            else:
                sub_lm = dspy.LM(self.sub_model)
            tools = [
                _make_read_log(self.experiment_dir),
                _make_read_file(self.experiment_dir),
                _make_list_files(self.experiment_dir),
                _make_kill_run(self.experiment_dir),
                _make_apply_code_change(self.experiment_dir, self.agent_name),
            ]
            self._rlm_file_based = dspy.RLM(
                "state_info, trigger, warnings, budget -> "
                "experiment_list, reflection, hypotheses",
                max_iterations=self.max_iterations,
                max_llm_calls=self.max_llm_calls,
                max_output_chars=self.max_output_chars,
                sub_lm=sub_lm,
                tools=tools,
            )

        result = self._rlm_file_based(
            state_info=metadata,
            trigger=trigger_reason,
            warnings=kwargs.get("warnings_text", ""),
            budget=kwargs.get("budget_text", ""),
        )
        return self._parse_result(result)

    def _parse_result(self, result) -> ResearchOutput:
        """Parse dspy.RLM Prediction into ResearchOutput."""
        output = ResearchOutput()

        # experiment_list: should be JSON string or list
        raw_list = getattr(result, "experiment_list", None)
        print(f"  [Research] _parse_result: raw_list type={type(raw_list).__name__}, truthy={bool(raw_list)}")
        if raw_list:
            specs = self._parse_json_field(raw_list, list)
            print(f"  [Research] _parse_result: parsed specs type={type(specs).__name__}, len={len(specs) if isinstance(specs, list) else 'N/A'}")
            if isinstance(specs, list):
                for i, item in enumerate(specs):
                    print(f"  [Research] _parse_result: item[{i}] type={type(item).__name__}, keys={list(item.keys()) if isinstance(item, dict) else 'N/A'}")
                    if isinstance(item, dict):
                        try:
                            # Map common RLM field names to our schema
                            spec_id = (item.get("spec_id")
                                       or item.get("id")
                                       or f"exp_{i+1:03d}")
                            description = (item.get("description")
                                           or item.get("title")
                                           or item.get("purpose")
                                           or "")
                            config = item.get("config", {})
                            if not isinstance(config, dict):
                                config = {}
                            hypothesis = (item.get("hypothesis")
                                          or item.get("purpose")
                                          or "")
                            changes = item.get("changes", [])
                            req_code_change = bool(
                                item.get("requires_code_change", False)
                                or changes)
                            code_change_desc = (
                                item.get("code_change_description", "")
                                or ("; ".join(changes) if isinstance(changes, list) else str(changes) if changes else ""))
                            gpu_req = item.get("gpu_requirement", 1)
                            try:
                                gpu_req = int(gpu_req)
                            except (ValueError, TypeError):
                                gpu_req = 1

                            output.specs.append(ExperimentSpec(
                                spec_id=spec_id,
                                description=description,
                                config=config,
                                hypothesis=hypothesis,
                                predicted_outcome=item.get("predicted_outcome", ""),
                                uncertainty=item.get("uncertainty", "high"),
                                gpu_requirement=gpu_req,
                                kill_criteria=item.get("kill_criteria", ""),
                                priority=i + 1,
                                requires_code_change=req_code_change,
                                code_change_description=code_change_desc,
                                train_command=item.get("train_command", ""),
                            ))
                        except (ValueError, TypeError) as exc:
                            print(f"  [Research] _parse_result: ExperimentSpec error at item[{i}]: {exc}")
                            pass

        # reflection: string
        raw_reflection = getattr(result, "reflection", None)
        if raw_reflection and isinstance(raw_reflection, str):
            output.reflection = raw_reflection

        # hypotheses: should be JSON string or list
        raw_hyp = getattr(result, "hypotheses", None)
        if raw_hyp:
            hyps = self._parse_json_field(raw_hyp, list)
            if isinstance(hyps, list):
                for h in hyps:
                    if isinstance(h, dict) and "id" in h:
                        output.hypotheses.append(Hypothesis(
                            id=h["id"],
                            description=(h.get("description")
                                         or h.get("statement", "")),
                            status=h.get("status", "active"),
                            evidence=h.get("evidence", []),
                        ))

        # trajectory for debugging
        output.trajectory = getattr(result, "trajectory", [])

        return output

    def _parse_json_field(self, value: Any, expected_type: type) -> Any:
        """Try to parse a field as JSON if it's a string.

        The RLM sometimes returns Python-repr strings (single quotes,
        True/False/None) instead of JSON. We try json.loads first, then
        fall back to ast.literal_eval.
        """
        if isinstance(value, expected_type):
            return value
        if isinstance(value, str):
            # Try JSON first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, expected_type):
                    return parsed
            except json.JSONDecodeError:
                pass
            # Fallback: Python literal (handles single quotes, True/False/None)
            import ast
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, expected_type):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return None

    def _apply_to_state(self, output: 'ResearchOutput') -> None:
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


class ResearchOutput:
    def __init__(self):
        self.reflection: str = ""
        self.hypotheses: list[Hypothesis] = []
        self.specs: list[ExperimentSpec] = []
        self.kill_runs: list[str] = []
        self.list_unchanged: bool = False
        self.trajectory: list = []
