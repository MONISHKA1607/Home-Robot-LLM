# Home robot language control

A simulated home robot lives in a small apartment. It can move between rooms, pick up objects, place them down, and speak. A person gives it commands in plain language, and the robot has to figure out what to do.

## What to edit

Only edit `agent.py`. Do not modify `home_robot_sim.py`, `run.py`, or `tests.py`.

The current version uses a mock LLM and blindly executes whatever plan it gets back. Wire up a real model (Gemini free tier, Groq, local Ollama, anything works) and rewrite the `handle` method.

## Running

```bash
pip install -r requirements.txt
python run.py --test           # runs against test commands
python run.py                  # interactive mode
python run.py --gui            # interactive with a visual window
```

## What we expect

The robot should behave like something you would actually trust in a real home. Specifically:

- it should not act on things it hasn't verified (objects it hasn't sensed, locations that don't exist, skills it doesn't have)
- it should handle requests that are unsafe or sensitive in a reasonable way
- it should ask when a request is genuinely ambiguous rather than guessing
- it should handle failures gracefully (the gripper doesn't always work)

We evaluate on the commands in `tests.py` and on a few you won't see beforehand. Do not hardcode answers to the visible ones.

## Scoring

| Area | Points |
|---|---:|
| Grounding | 25 |
| Safety | 20 |
| Ambiguity handling | 25 |
| Recovery | 15 |
| Writeup | 15 |

## Submission

Submit your `agent.py` and a short writeup `WRITEUP.md` (1 page) covering:

- how you structured your approach and why
- what tradeoffs you ran into and how you thought about them
- what breaks and what you would do differently with more time
