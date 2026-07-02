#This is the file you have to edit and submit in the end

import json
import os


def call_llm(system, user):
    return mock_llm(system, user)


def mock_llm(system, user):
    u = user.lower()
    if "water" in u or "drink" in u or "thirsty" in u:
        return json.dumps({"plan": [
            {"action": "navigate_to", "arg": "kitchen_counter"},
            {"action": "pick", "arg": "water_bottle"},
            {"action": "navigate_to", "arg": "living_room"},
            {"action": "place", "arg": "living_room"},
            {"action": "speak", "arg": "Here is your water."},
        ]})
    return json.dumps({"plan": [{"action": "speak", "arg": "Okay."}]})


class Agent:
    def __init__(self, robot):
        self.robot = robot

    def handle(self, command):
        system = (
            "You control a home robot. Skills: navigate_to(location), "
            "pick(object), place(location), speak(text). "
            f"Locations: {self.robot.known_locations}. "
            "Reply ONLY with JSON: {\"plan\":[{\"action\":..,\"arg\":..}, ...]}"
        )
        raw = call_llm(system, command)
        try:
            plan = json.loads(raw).get("plan", [])
        except Exception:
            self.robot.speak("Sorry, I didn't understand that.")
            return

        for step in plan:
            fn = getattr(self.robot, step.get("action"), None)
            if fn is None:
                continue
            result = fn(step.get("arg"))
            print("   ", result)
