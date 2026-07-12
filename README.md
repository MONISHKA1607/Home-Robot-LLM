# 🤖 Home Robot Language Agent

A natural-language control layer for a simulated home robot — turns plain-English requests like *"bring me the towel"* into safe, grounded, recoverable robot behavior, backed by a real LLM reasoning loop and a code-level safety firewall.

> The robot doesn't just execute plans blindly. It only acts on what it's actually sensed, refuses unsafe requests on its own terms, asks when a request is genuinely ambiguous, and recovers gracefully when its own gripper fails.

---

## Why this exists

Most "LLM controls a robot" demos wire a model straight up to actuators and hope the prompt holds. That's fragile — a model will happily hallucinate an object that isn't there, "complete" a delivery it never made, or hand over a kitchen knife because nothing stopped it. This project treats the LLM as a *proposer*, never an *executor*: every action it suggests passes through a deterministic validator before it ever reaches the robot. The LLM reasons; the code decides.

## Architecture

```
 Person's request
        │
        ▼
 ┌─────────────────┐
 │   LLM (Groq /    │  proposes ONE action at a time,
 │ Anthropic/Gemini/│  sees a live "available actions" menu
 │     Ollama)      │  built fresh from real robot state
 └────────┬─────────┘
          ▼
 ┌─────────────────┐
 │    Validator     │  hard, rule-based firewall —
 │ (no LLM calls)   │  unsafe categories, unknown skills,
 │                  │  ungrounded objects rejected here,
 │                  │  regardless of what the model decided
 └────────┬─────────┘
          ▼
 ┌─────────────────┐
 │  Simulator       │  real Result (ok/fail) fed back
 │  (robot body)    │  before the next action is proposed
 └────────┬─────────┘
          ▼
 ┌─────────────────┐
 │  Failure         │  tags every failure: perception /
 │  Classification  │  execution / capability / safety /
 │                  │  planning — each recovers differently
 └────────┬─────────┘
          ▼
 ┌─────────────────┐
 │  Mission Trace   │  structured log + console report:
 │                  │  rooms visited, retries, outcome
 └─────────────────┘
```

Four layers, each doing one job:

| Layer | Responsibility |
|---|---|
| **Tool awareness** | A live menu of OK / BLOCKED actions, computed from real robot state, shown to the model every step — makes "verify before acting" something the model reasons about explicitly, not something buried in a paragraph of rules |
| **Validator** | The only thing allowed to say "no" on the LLM's behalf. Kept deliberately dumb: no LLM calls, no ambiguity, just hard checks that can't be argued with |
| **Failure classification** | A failed grasp gets retried; an unsensed object triggers more exploration, not a retry; an out-of-scope request just gets declined — different failures, different recoveries |
| **Mission trace** | Every run produces a structured, inspectable log — never spoken to the person, purely for debugging and explainability |

## What it actually handles

- ✅ **Grounding** — will only `pick()` an object it has genuinely sensed via a real `navigate_to()` call. Never assumes where something is.
- ✅ **Safety** — sharp objects and medication are hard-blocked in code (`UNSAFE_CATEGORIES`), not just discouraged in the prompt, and surfaced pre-emptively in the action menu so the model doesn't even have to be told "no" after the fact.
- ✅ **Ambiguity** — "get me something to drink" with both a water bottle and juice box sensed → the robot stops and asks, rather than guessing.
- ✅ **Recovery** — the gripper fails ~30% of the time by design. Transient grasp failures are retried a bounded number of times, then honestly reported — never silently pretended away.
- ✅ **Loop protection** — a 3-in-a-row or alternating-pattern detector catches a model that's stuck, and ends the turn gracefully instead of burning the whole step budget on repetition.
- ✅ **Scope discipline** — asked to open a window or turn on a TV, it says plainly it can't, and doesn't keep exploring afterward looking for a workaround.

## Running it

```bash
pip install -r requirements.txt

export GROQ_API_KEY=...            # or ANTHROPIC_API_KEY / GEMINI_API_KEY / OLLAMA_HOST

python run.py --test               # full test suite
python run.py                      # interactive mode
python run.py --gui                # interactive, with a live visual window
```

## What's next

The architecture is intentionally provider-agnostic and stateless-by-design, which makes it a clean base to extend:

- **Richer world memory** — a confidence-scored belief state instead of a binary "sensed / not sensed," so the robot can reason about objects it hasn't seen in a while.
- **Multi-robot coordination** — the validator/executor split generalizes cleanly to a fleet, since each robot's actuator boundary is already isolated from the reasoning layer.
- **Learned search priors** — replace the shuffled, bias-free location menu with a lightweight learned model of "where things usually live," once there's enough real usage data to justify it without overfitting to a handful of scenarios.
