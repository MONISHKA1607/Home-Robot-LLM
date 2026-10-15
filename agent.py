import json
import os
import random
import re
import urllib.request
import urllib.error

PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
MAX_STEPS = 10
MAX_PICK_RETRIES = 2  


UNSAFE_CATEGORIES = {
    "sharp_utensil": "it's sharp and I'd rather not hand it over myself",
    "medication": "medication should only be handed out with a person confirming it's the right dose for the right person",
}


LOCATION_KIND = {
    "kitchen_counter": "furniture -- often holds several objects",
    "dining_table":    "furniture -- often holds several objects",
    "bedside_table":   "furniture -- often holds several objects",
    "desk":            "furniture -- often holds several objects",
    "kitchen":         "room anchor",
    "living_room":     "room anchor",
    "bedroom":         "room anchor",
    "bathroom":        "room anchor -- can also hold objects directly",
    "study":           "room anchor",
    "hallway":         "room anchor -- passthrough, unlikely to hold objects",
}


class FailureType:
    """Tags for why a step didn't go through, so recovery can differ by cause
    instead of treating every failure the same way."""
    PERCEPTION = "perception"   
    EXECUTION = "execution"    
    CAPABILITY = "capability"   
    SAFETY = "safety"           
    PLANNING = "planning"      

SYSTEM_TEMPLATE = """You control a real home robot. You act ONE step at a time -- \
you will be shown the result of each action before choosing the next one, so \
never assume something worked or that an object exists until you've seen it.

SKILLS (this is the complete list -- never invent others):
- navigate_to(location): move to a known location, and sense whatever objects are there
- pick(object): pick up an object you have ALREADY sensed AT YOUR CURRENT LOCATION
- place(location): put down what you're holding (must be your current location)
- speak(text): say something out loud to the person
- done: nothing left to do, end the task

KNOWN LOCATIONS: {locations}

Every turn you'll be shown an "AVAILABLE ACTIONS RIGHT NOW" list, computed
directly from the robot's real state. Lines marked OK are legal this turn.
Lines marked BLOCKED are not -- they will be rejected if you try them, so
don't propose them; instead treat a BLOCKED reason as new information (e.g.
"nothing sensed yet" means go look somewhere first, not that the object
doesn't exist).

RULES
1. You can only pick an object once the menu shows it as OK, which only \
happens after you've actually navigated somewhere and sensed it. Never \
assume where an object is.
2. If a request names something generic that could match more than one \
sensed object (e.g. "something to drink" when you've sensed both a water \
bottle and a juice box), stop and use speak to ask which one, then `done`. \
Do not guess.
3. If, after checking the plausible locations for it, an object genuinely \
isn't anywhere, say so honestly with speak and stop. Never claim you \
completed something you didn't.
4. If an action fails, react like a competent person would: a failed grasp \
is worth retrying once or twice, but if it keeps failing, tell the person \
and stop -- don't pretend it worked.
5. If asked for something outside the skill list above (e.g. opening a \
window, turning on a TV), say plainly that you can't do that. Don't attempt \
a fake action for it -- and once you've said so, your very next action must \
be `done`. Do not keep exploring after declining an out-of-scope request.
6. When told to "bring" or "get" something with no delivery location given, \
assume the person is in the living_room and deliver it there.
7. Some objects are sensitive (sharp objects, medication) -- the menu will \
show pick() for those as BLOCKED with a reason; if the request is for one \
of those, just speak a brief, reasonable explanation instead of trying it.
8. Rooms already visited are listed below -- don't re-visit a room looking \
for something you'd have already sensed there.
9. IMPORTANT: within this task you will NOT receive a reply after using speak(). \
The person cannot respond until this run ends. So if you ask a question, report \
you can't complete something, or confirm the task is finished, your VERY NEXT \
action must be `done` -- never speak twice in a row, and never wait for or \
imagine a reply.
10. Do not pick up, place, or otherwise act on any object that has nothing to \
do with the person's request, even if you happen to sense it along the way. \
Only interact with objects relevant to what was actually asked.
11. Once a task is genuinely finished (the requested object has been \
delivered, or you've honestly reported you can't do it), immediately use \
`done`. Do not second-guess a successful delivery or re-verify something \
you already confirmed.
12. After picking something up, your NEXT action must be navigate_to the delivery 
    location (unless you're already there) — never place() at the same spot you 
    picked from unless that IS the requested destination.
13. A room anchor with nothing sensed usually has an associated furniture location nearby (see LOCATION_KIND) — check that before concluding the object isn't in the room
    
Respond with ONLY one JSON object, nothing else, no markdown fences:
{{"action": "<skill from the list>", "arg": "<argument, or null for speak/done text goes in arg>", "why": "<very short reason>"}}
For speak, put the sentence to say in "arg".
"""



def call_llm(system, transcript_text):
    if PROVIDER == "groq":
        return _call_openai_compatible(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=os.environ["GROQ_API_KEY"],
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            system=system, user=transcript_text,
        )
    if PROVIDER == "ollama":
        return _call_openai_compatible(
            url=os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1/chat/completions",
            api_key="ollama",
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            system=system, user=transcript_text,
            timeout=180,
        )
    if PROVIDER == "anthropic":
        return _call_anthropic(system, transcript_text)
    if PROVIDER == "gemini":
        return _call_gemini(system, transcript_text)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{PROVIDER}'")


def _call_openai_compatible(url, api_key, model, system, user, timeout=30):
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (compatible; home-robot-agent/1.0)",
    }
    import time, re as _re

    for attempt in range(5):
        req = urllib.request.Request(url, data=body, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                err_body = e.read().decode(errors="ignore")
                m = _re.search(r"try again in ([\d.]+)(m?s)", err_body)
                if m:
                    wait = float(m.group(1))
                    if m.group(2) == "ms":
                        wait /= 1000
                else:
                    wait = 3.0
                time.sleep(wait + 0.5)
                continue
            raise


def _call_anthropic(system, user):
    body = json.dumps({
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 300,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def _call_gemini(system, user):
    import time
    import re as _re

    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    )

    body = json.dumps({
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0},
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; home-robot-agent/1.0)",
    }

    for attempt in range(5):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                err_body = e.read().decode(errors="ignore")
                m = _re.search(r'"retryDelay":\s*"(\d+)s"', err_body)
                wait = float(m.group(1)) if m else 2 ** attempt
                time.sleep(wait + 1)
                continue
            raise


def _parse_action(raw):
    """Pull the first {...} JSON object out of the model's reply."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, None, None
    try:
        obj = json.loads(match.group(0))
        return obj.get("action"), obj.get("arg"), obj.get("why")
    except Exception:
        return None, None, None

def validate_action(robot, action, arg, pickup_location=None):
    allowed_skills = {"navigate_to", "pick", "place", "speak", "done"}

    if action not in allowed_skills:
        return False, ("I'm not able to do that -- I can only move around, "
                        "pick things up, put them down, and talk."), FailureType.CAPABILITY

    if action == "navigate_to":
        if arg not in robot.known_locations:
            return False, None, FailureType.PLANNING  # sim will give the precise message
        return True, None, None

    if action == "place":
        if (robot.holding is not None and pickup_location is not None
                and robot.current_location == pickup_location):
            return False, (
                "I just picked this up here -- I need to navigate to the "
                "delivery location before placing it back down where I found it."
            ), FailureType.PLANNING
        return True, None, None

    if action == "pick":
        if robot.holding is not None:
            return False, (f"I'm already holding '{robot.holding}' -- "
                            f"I need to place it before picking something else."), FailureType.PLANNING
        if arg not in robot.known_objects:
            return False, None, FailureType.PERCEPTION
        category = robot.known_objects[arg]["category"]
        if category in UNSAFE_CATEGORIES:
            return False, (f"I'd rather not bring you the {arg} myself -- "
                            f"{UNSAFE_CATEGORIES[category]}."), FailureType.SAFETY
        return True, None, None
    return True, None, None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, robot, seed=0):
        self.robot = robot
        self._rng = random.Random(seed)   # own RNG -> reproducible menu order
        self.turns = []
        self.visited_locations = []
       
        self.trace = []
        self._recent_actions = []
        self._consecutive_speaks = 0
        self.last_pickup_location = None
        self.current_command = ""

    def handle(self, command):
        self.turns.append(f"PERSON: {command}")
        self.current_command = command
        system = SYSTEM_TEMPLATE.format(locations=self.robot.known_locations)
        pick_fail_counts = {}
        self.trace = []
        self._recent_actions = []
        self._consecutive_speaks = 0
        self.visited_locations = []        # fresh start for every new request
        self.last_pickup_location = None
        outcome = "incomplete"

        for step_num in range(1, MAX_STEPS + 1):
            transcript = self._build_transcript()
            try:
                raw = call_llm(system, transcript)
            except KeyError as e:
                self.robot.speak(f"I can't reach my reasoning model -- missing {e}. "
                                  f"Set the right API key env var and try again.")
                outcome = "llm_unreachable"
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"   [DEBUG] LLM call failed: {type(e).__name__}: {e}")
                if hasattr(e, "read"):
                    print(f"   [DEBUG] body: {e.read()}")
                self.robot.speak("I'm having trouble reaching my reasoning model right now. "
                                  "Could you try again in a moment?")
                self.turns.append(f"[llm call failed: {e}]")
                outcome = "llm_unreachable"
                break

            action, arg, why = _parse_action(raw)
            if action is None:   # one cheap retry before giving up on bad JSON
                try:
                    raw = call_llm(system, transcript + "\n\nYour last reply was not a valid "
                                   "JSON object. Reply with ONLY the JSON object.")
                    action, arg, why = _parse_action(raw)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError):
                    pass
            print(f"   [PROPOSED] action={action!r} arg={arg!r} why={why!r}")

            if action is None:
                self.turns.append(f"[couldn't parse model output: {raw!r}]")
                self.robot.speak("Sorry, I got confused thinking about that -- could you rephrase?")
                self._log(step_num, "parse", arg, False, FailureType.PLANNING, why)
                outcome = "parse_failure"
                break

            if action == "done":
                self.turns.append(f"[done: {why or 'task complete'}]")
                self._log(step_num, "done", None, True, None, why)
                outcome = "done"
                break

        
            signature = (action, arg)
            self._recent_actions.append(signature)
            self._recent_actions = self._recent_actions[-4:]

            recent = self._recent_actions
            last3 = self._recent_actions[-3:]
            last4 = self._recent_actions[-4:]
            stuck = False
            if len(last3) == 3 and len(set(last3)) == 1:
                stuck = True
            elif (len(last4) == 4 and last4[0] == last4[2] and last4[1] == last4[3]
                  and last4[0] != last4[1]):
                stuck = True
            if stuck:
                self.robot.speak(
                    "I seem to be going in circles on this one -- I don't think "
                    "I can complete this request right now."
                )
                self.turns.append(f"[loop guard: repeating pattern {recent}, aborting]")
                self._log(step_num, action, arg, False, FailureType.PLANNING,
                          "repeated action loop detected")
                outcome = "loop_detected"
                break

          
            if action == "speak":
                self._consecutive_speaks = getattr(self, "_consecutive_speaks", 0) + 1
            else:
                self._consecutive_speaks = 0

            if action == "speak" and self._consecutive_speaks >= 2:
                result = self.robot.speak(arg)
                print(f"   [SPEAK] {result}")
                self.turns.append(f"OBSERVATION: {result.message}")
                self.turns.append("[ending turn: no reply available mid-run]")
                self._log(step_num, action, arg, True, None, why)
                outcome = "done"
                break


          
            ok, refusal, failure_type = validate_action(self.robot, action, arg, self.last_pickup_location)

            if not ok and action == "place" and failure_type == FailureType.PLANNING:
              
                self.turns.append(f"OBSERVATION: place rejected -- {refusal}")
                self._log(step_num, action, arg, False, failure_type, refusal)
                continue

            if not ok and refusal is not None:
                self.robot.speak(refusal)
                self.turns.append(f"[blocked '{action}({arg})': {refusal}]")
                self._log(step_num, action, arg, False, failure_type, refusal)
                outcome = f"blocked_{failure_type}"
                break

            
            if action == "speak":
                result = self.robot.speak(arg)
                print(f"   [SPEAK] {result}")
                self.turns.append(f"OBSERVATION: {result.message}")
                self._log(step_num, action, arg, True, None, why)
                continue

            fn = getattr(self.robot, action, None)
            if fn is None:
                self.robot.speak("I'm not able to do that -- I can only move around, "
                                  "pick things up, put them down, and talk.")
                self.turns.append(f"[rejected unknown skill '{action}']")
                self._log(step_num, action, arg, False, FailureType.CAPABILITY, why)
                outcome = "blocked_capability"
                break
            
            result = fn(arg)
            print("   ", result)

            if action == "navigate_to" and result.ok and arg not in self.visited_locations:
                self.visited_locations.append(arg)

            if action == "pick" and result.ok:
                self.last_pickup_location = self.robot.current_location

            if action == "place" and result.ok:
                self.last_pickup_location = None

        

            if action == "pick" and not result.ok and result.code == "grasp_slipped":
                pick_fail_counts[arg] = pick_fail_counts.get(arg, 0) + 1
                self._log(step_num, action, arg, False, FailureType.EXECUTION, result.message)
                if pick_fail_counts[arg] <= MAX_PICK_RETRIES:
                    # a retryable failure is not a "loop": don't let the loop guard count it
                    self._recent_actions.pop()
                    self.turns.append(
                        f"OBSERVATION: {result.message} "
                        f"(attempt {pick_fail_counts[arg]} of {MAX_PICK_RETRIES + 1} -- worth retrying)"
                    )
                    continue
                self.robot.speak(f"My gripper couldn't hold onto the {arg} after a few tries -- sorry about that.")
                self.turns.append(f"[gave up on '{arg}' after repeated grasp failures]")
                outcome = "execution_failure"
                break

            failure_type = None if result.ok else (
                FailureType.PERCEPTION if action == "pick" else FailureType.PLANNING
            )
            self._log(step_num, action, arg, result.ok, failure_type, result.message)
            self.turns.append(f"OBSERVATION: {result.message}")
        else:
            self.robot.speak("This is taking longer than it should -- let me stop here rather than keep trying blindly.")
            outcome = "step_budget_exhausted"

        self._print_mission_report(command, outcome)
        self.last_outcome = outcome
        return outcome

    def _log(self, step_num, action, arg, ok, failure_type, detail):
        self.trace.append({
            "step": step_num,
            "action": action,
            "arg": arg,
            "ok": ok,
            "failure_type": failure_type,
            "detail": detail,
        })

   
    def _available_actions(self):
        robot = self.robot
        lines = []

        shuffled_locs = list(robot.known_locations)
        self._rng.shuffle(shuffled_locs)
        for loc in shuffled_locs:
            kind = LOCATION_KIND.get(loc, "location")
            if loc in self.visited_locations:
                lines.append(f'  navigate_to("{loc}")  -- OK ({kind}, already visited)')
            else:
                lines.append(f'  navigate_to("{loc}")  -- OK ({kind}, not yet visited)')

        if robot.holding is not None:
            lines.append(f'  pick(...)  -- BLOCKED: already holding "{robot.holding}", place it first')
        else:
            here = {n: i for n, i in robot.known_objects.items() if i["location"] == robot.current_location}
            elsewhere = {n: i for n, i in robot.known_objects.items() if i["location"] != robot.current_location}
            if not here:
                lines.append('  pick(...)  -- BLOCKED: nothing sensed at your current location')
            for name, info in here.items():
                cat = info["category"]
                if cat in UNSAFE_CATEGORIES:
                    lines.append(f'  pick("{name}")  -- BLOCKED: {UNSAFE_CATEGORIES[cat]}')
                else:
                    lines.append(f'  pick("{name}")  -- OK, sensed right here')
            for name, info in elsewhere.items():
                lines.append(f'  pick("{name}")  -- BLOCKED: sensed at "{info["location"]}", not here -- navigate there first')

        if robot.holding is not None:
            lines.append(f'  place("{robot.current_location}")  -- OK, puts down "{robot.holding}" here')
        else:
            lines.append('  place(...)  -- BLOCKED: not holding anything')

        lines.append('  speak("...")  -- OK, always available')
        lines.append('  done  -- OK, always available, ends the task')
        return "\n".join(lines)

    def _build_transcript(self):
        loop_hint = ""
        if len(self._recent_actions) >= 2 and self._recent_actions[-1] == self._recent_actions[-2]:
            loop_hint = (
                "\nWARNING: your last two actions were identical and produced no new "
                "information. Repeating it again will not help -- either try something "
                "genuinely different or use speak+done to report you cannot complete this.\n"
            )

        recent_turns = self.turns[-8:] 
        state = (
            f"THE PERSON'S ORIGINAL REQUEST (always keep this in mind): {self.current_command}\n"
            f"Current location: {self.robot.current_location}\n"
            f"Holding: {self.robot.holding}\n"
            f"{loop_hint}"
            f"Rooms visited so far: {self.visited_locations}\n"
            f"Objects sensed so far: {self.robot.known_objects}\n\n"
            f"AVAILABLE ACTIONS RIGHT NOW:\n{self._available_actions()}\n\n"
            "Recent conversation and action history:\n"
        )
        return state + "\n".join(recent_turns) + "\n\nWhat is your next single action?"

    def _print_mission_report(self, command, outcome):
        """Console-only execution summary -- never spoken to the person,
        purely for debugging/explainability. Doesn't touch robot._last_speech
        so it can't interfere with grading."""
        n_actions = len(self.trace)
        n_failures = sum(1 for t in self.trace if not t["ok"])
        by_type = {}
        for t in self.trace:
            if t["failure_type"]:
                by_type[t["failure_type"]] = by_type.get(t["failure_type"], 0) + 1
        print(f"   -- mission report: \"{command}\" --")
        print(f"      outcome={outcome} actions={n_actions} failures={n_failures} "
              f"rooms_visited={self.visited_locations}")
        if by_type:
            print(f"      failures_by_type={by_type}")
