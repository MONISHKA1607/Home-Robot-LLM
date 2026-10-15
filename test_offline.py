"""Offline tests: a scripted FAKE LLM replaces the real model, so these run
instantly, need no API key, and are fully deterministic.

Run:  python -m unittest test_offline -v
"""
import json
import unittest

import agent as agent_mod
import home_robot_sim as sim
from agent import Agent


def fake_llm(*steps):
    """Build a fake call_llm that replays the given replies in order.
    A step can be a dict (turned into JSON) or a raw string."""
    it = iter(steps)

    def _call(system, transcript):
        step = next(it)
        return step if isinstance(step, str) else json.dumps(step)
    return _call


def act(action, arg=None, why="test"):
    return {"action": action, "arg": arg, "why": why}


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self._orig = agent_mod.call_llm

    def tearDown(self):
        agent_mod.call_llm = self._orig

    def run_agent(self, steps, fail=0.0, seed=0, command="test"):
        agent_mod.call_llm = fake_llm(*steps)
        robot = sim.Robot(grasp_fail_prob=fail, seed=seed)
        agent = Agent(robot, seed=seed)
        outcome = agent.handle(command)
        return robot, agent, outcome

    # 1. the bug we fixed: 3 grasp attempts really happen, then honest report
    def test_retry_gets_all_three_attempts(self):
        robot, _, outcome = self.run_agent(
            [act("navigate_to", "kitchen_counter")] + [act("pick", "water_bottle")] * 3,
            fail=1.0)
        picks = [l for l in robot._log if l.startswith("pick")]
        self.assertEqual(len(picks), 3)
        self.assertEqual(outcome, "execution_failure")
        self.assertIn("gripper", robot._last_speech)

    # 2. the loop guard must still catch genuine loops
    def test_loop_guard_still_works(self):
        _, _, outcome = self.run_agent([act("navigate_to", "kitchen")] * 3)
        self.assertEqual(outcome, "loop_detected")

    # 3. safety: even if the LLM asks for the knife, code blocks it
    def test_knife_blocked_in_code(self):
        robot, _, outcome = self.run_agent(
            [act("navigate_to", "kitchen_counter"), act("pick", "kitchen_knife")])
        self.assertEqual(outcome, "blocked_safety")
        self.assertIsNone(robot.holding)
        self.assertEqual(robot._objects["kitchen_knife"].location, "kitchen_counter")

    def test_medication_blocked_in_code(self):
        robot, _, outcome = self.run_agent(
            [act("navigate_to", "bedside_table"), act("pick", "pill_bottle")])
        self.assertEqual(outcome, "blocked_safety")
        self.assertIsNone(robot.holding)

    # 4. grounding: can't pick something never sensed
    def test_ungrounded_pick_rejected(self):
        robot, _, _ = self.run_agent([act("pick", "towel"), act("done")])
        self.assertIsNone(robot.holding)

    # 5. a delivery really changes the world (the place() bug we fixed)
    def test_delivery_moves_object(self):
        robot, _, outcome = self.run_agent([
            act("navigate_to", "desk"), act("pick", "book"),
            act("navigate_to", "living_room"), act("place", "living_room"),
            act("done")])
        self.assertEqual(outcome, "done")
        self.assertEqual(robot._objects["book"].location, "living_room")
        self.assertIsNone(robot.holding)

    # 6. one retry on malformed JSON
    def test_bad_json_gets_one_retry(self):
        _, _, outcome = self.run_agent(["sorry, I think we should...", act("done")])
        self.assertEqual(outcome, "done")

    def test_bad_json_twice_aborts(self):
        _, _, outcome = self.run_agent(["nope", "still nope"])
        self.assertEqual(outcome, "parse_failure")

    # 7. place() right where we picked up is rejected softly, run continues
    def test_place_at_pickup_spot_rejected_softly(self):
        robot, _, outcome = self.run_agent([
            act("navigate_to", "desk"), act("pick", "book"),
            act("place", "desk"),                       # rejected, not fatal
            act("navigate_to", "living_room"), act("place", "living_room"),
            act("done")])
        self.assertEqual(outcome, "done")
        self.assertEqual(robot._objects["book"].location, "living_room")

    # 8. unknown skill is refused
    def test_unknown_skill_refused(self):
        _, _, outcome = self.run_agent([act("open_window", "bedroom")])
        self.assertEqual(outcome, "blocked_capability")

    # 9. reproducibility: same seed -> same menu order
    def test_same_seed_same_menu(self):
        a1 = Agent(sim.Robot(), seed=7)._available_actions()
        a2 = Agent(sim.Robot(), seed=7)._available_actions()
        self.assertEqual(a1, a2)

    # 10. visited rooms reset on each new command
    def test_visited_resets_between_commands(self):
        agent_mod.call_llm = fake_llm(act("navigate_to", "kitchen"), act("done"),
                                      act("done"))
        agent = Agent(sim.Robot(), seed=0)
        agent.handle("first")
        self.assertEqual(agent.visited_locations, ["kitchen"])
        agent.handle("second")
        self.assertEqual(agent.visited_locations, [])

    # 12. REGRESSION: two robots must not share world state
    def test_robots_do_not_share_world_state(self):
        r1 = sim.Robot()
        r1._objects["book"].location = "living_room"
        r2 = sim.Robot()
        self.assertEqual(r2._objects["book"].location, "desk")

    # 13. REGRESSION: the original request must survive a long run
    def test_original_request_never_scrolls_out(self):
        a = Agent(sim.Robot(), seed=0)
        a.current_command = "Bring me my phone charger"
        a.turns = ["PERSON: Bring me my phone charger"] + [f"OBSERVATION: step {i}" for i in range(30)]
        self.assertIn("phone charger", a._build_transcript())

    # 11. step budget: a model that never finishes gets stopped
    def test_step_budget(self):
        locs = ["kitchen", "bedroom", "study", "bathroom", "living_room"]
        steps = [act("navigate_to", locs[i % 5]) for i in range(agent_mod.MAX_STEPS)]
        _, _, outcome = self.run_agent(steps)
        self.assertEqual(outcome, "step_budget_exhausted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
