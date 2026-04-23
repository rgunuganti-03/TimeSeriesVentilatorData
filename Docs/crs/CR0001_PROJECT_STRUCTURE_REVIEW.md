# CR0001 - Project Structure Review & Immediate Next Steps

**Author:** Codex
**Date:** 2026-04-09
**Status:** Ready for Review
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Purpose

This document reviews the current project structure, identifies the main issues
in the repository as received, and defines the next set of cleanup and
discipline tasks needed before the project should be treated as a serious
working prototype.

This is not a criticism of the prototype itself. The prototype is real and the
core idea is valid. The main gap is project hygiene, reproducibility, and
evidence of technical exploration.

---

## What Is Working

The intern has produced a real prototype with a coherent direction:

- A rule-based waveform generator exists in `generator/waveforms.py`
- An ODE-based generator exists in `generator/ode_solver.py`
- Both are wired into a common Streamlit UI in `ui/dashboard.py`
- The project has a useful top-level framing in `PROBLEM_STATEMENT.md`
- The architecture has been thought about in modular terms

This means the work is not just presentation. There is an actual implementation
to build on.

---

## Main Findings

### 1. The folder is large because the virtual environment was included

The project folder is about 500 MB primarily because `venv/` is present inside
the project and contains installed packages, binaries, assets, and test files.

Observed issue:

- `venv/` is approximately 482 MB
- The actual source code is small by comparison

Why this matters:

- The project becomes hard to share cleanly
- The environment is machine-specific and should not be treated as source
- It hides what the real project size actually is

---

### 2. The included environment is not portable

The `venv` appears to have been created on macOS and then copied into this
project folder.

Observed issue:

- `venv/pyvenv.cfg` points to `/opt/anaconda3/bin/python3.12`
- The layout uses `venv/bin/` rather than a Windows-native environment layout
- The copied environment could not be used directly from this Windows machine

Why this matters:

- The project cannot be assumed to run cleanly on another machine
- It prevents reproducible setup and testing

---

### 3. Project hygiene is currently weak

Observed issues:

- No `.gitignore` is present
- `.DS_Store` files are present
- `__pycache__` artifacts are present
- A machine-built virtual environment is present

Why this matters:

- It suggests the project is being handled more like a local folder than a
  maintainable engineering repository
- It makes review harder and creates noise around the real work

---

### 4. Documentation is stronger than the evidence trail

The current Markdown files are organized and readable, but they mainly describe
the intended architecture and current story of the prototype. They do not yet
show enough of the actual exploration path.

Missing evidence that would strengthen the work:

- What hypotheses were tried
- What parameter choices were tested
- What failed or looked unrealistic
- What was changed after those observations
- What source material or references informed the condition presets

Why this matters:

- A technical prototype should show not just what was built, but how reasoning
  improved through iteration
- This is especially important for physiology-oriented synthetic data work

---

### 5. Testing is incomplete relative to the apparent maturity of the docs

Observed issues:

- `tests/test_ode_solver.py` contains substantive tests
- `tests/test_waveforms.py` exists but is empty
- The project therefore does not yet test both major generator paths with the
  same level of rigor

Why this matters:

- The rule-based model is part of the claimed product surface
- The simpler model should have tests before the project expands further

---

### 6. Scope statements need to stay aligned with implementation

There is some mismatch between the stated phase framing and the actual contents
of the project. For example, some docs still describe ODE work as future scope,
while the repository already includes an ODE implementation and tests.

Why this matters:

- Phase language should help manage work, not confuse status
- Once a feature exists in the codebase, the docs should clearly mark whether it
  is experimental, partial, or ready

---

## Immediate Required Actions

The next round of work should not be "add more features first." It should be:

1. Add a `.gitignore`
2. Remove `venv/`, `.DS_Store`, and `__pycache__/` artifacts from the project
3. Confirm the project can be rebuilt from `requirements.txt` on a clean machine
4. Fill in `tests/test_waveforms.py`
5. Reconcile all phase/status language across `README.md`,
   `PROBLEM_STATEMENT.md`, and `ARCHITECTURE.md`
6. Add an experiment log documenting what has been tried and learned so far

---

## Recommended Working Method Going Forward

The project would benefit from a more explicit change-request workflow.

Recommended pattern:

1. Keep the high-level problem statement stable
2. Break concrete work into small Markdown change requests
3. Each CR should define:
   - problem
   - current state
   - proposed change
   - acceptance criteria
   - files likely to be touched
   - status
4. CRs can originate from multiple sources:
   - Prashanth (mentor)
   - Riya (intern)
   - Codex / ChatGPT
   - Claude
5. If the intern identifies a useful direction independently, that idea should be
   converted into a CR before implementation so the scope is explicit
6. The intern should pick up one CR at a time and implement against it
7. Each completed CR should leave behind both code changes and updated docs

This is a good fit for an agent-assisted workflow using Claude Code or Codex.
It gives the model a bounded task and gives the reviewer a clean unit of
progress to assess.

---

## Additional Documents To Create Next

The following documents should be added in the next phase:

- `EXPERIMENT_LOG.md`
  - chronological notes on parameter sweeps, unrealistic outputs, fixes, and
    open questions
- `VALIDATION.md`
  - define what "physiologically plausible" means for this prototype
- `CR_TRACKER.md`
  - a simple table of CR ID, title, owner, status, and notes

---

## Review Summary

The prototype is promising, but the current state is still a sandbox. The most
important next step is not deeper modeling yet. It is converting the work from
"a good local prototype folder" into "a clean, reproducible, reviewable
project."

That means:

- remove environment noise
- tighten repo discipline
- make tests consistent
- record the technical learning path
- use change requests to structure the next iteration

Once those are in place, the next week of work will be much more valuable.
