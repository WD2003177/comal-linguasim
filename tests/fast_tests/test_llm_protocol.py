import random
import unittest

from flow.controllers.llm_controller import LLMController
from flow.core.params import SumoCarFollowingParams
from flow.envs.base import Env
from flow.utils.agents_network import message_pool
from flow.utils.highway_scene import assign_roles_from_geometry
from flow.utils.highway_scene import generate_custom_highway_opening
from flow.utils.highway_scene import sample_and_freeze_scene
from flow.utils.highway_scene import scene_gate_metrics
from flow.utils.exceptions import FatalFlowError


def make_payload(veh_id, lane, rel_x, speed=24.0, ego_lane=1, ego_pos=900.0, ego_speed=23.0):
    return {
        "veh_id": veh_id,
        "type_id": "ego" if veh_id == "ego_0" else ("llm" if "llm" in veh_id else "human"),
        "edge": "highway_0",
        "lane": lane,
        "pos": ego_pos + rel_x,
        "speed": speed,
        "ego_edge": "highway_0",
        "ego_lane": ego_lane,
        "ego_pos": ego_pos,
        "ego_speed": ego_speed,
        "rel_lane_to_ego": lane - ego_lane,
        "rel_x_to_ego": rel_x,
        "rel_speed_to_ego": speed - ego_speed,
    }


class DummyEnv(object):
    def __init__(self, step=10, rollout_id=1):
        self.time_counter = step
        self.message_pool = message_pool()
        self.message_pool.reset_rollout("scenario", rollout_id)


class FakeNetwork(object):
    def num_lanes(self, edge):
        return 4

    def edge_length(self, edge):
        return 1800.0


class FakeKernel(object):
    def __init__(self):
        self.network = FakeNetwork()


class FakeSceneEnv(object):
    def __init__(self):
        self.initial_ids = ["ego_0", "llm_0", "llm_1"] + ["human_{}".format(i) for i in range(8)]
        self.initial_state = {}
        for veh_id in self.initial_ids:
            if veh_id == "ego_0":
                type_id = "ego"
            elif "llm" in veh_id:
                type_id = "llm"
            else:
                type_id = "human"
            self.initial_state[veh_id] = (type_id, "highway_0", 1, 900.0, 0.0)
        self.k = FakeKernel()


class FakeInitialConfig(object):
    def __init__(self):
        self.shuffle = False
        self.spacing = "custom"
        self.additional_params = {"start_positions": [], "start_lanes": []}


class FakeFreezableSceneEnv(FakeSceneEnv):
    def __init__(self, fail_resets=1):
        super(FakeFreezableSceneEnv, self).__init__()
        self.initial_config = FakeInitialConfig()
        self.network = type("Network", (), {"initial_config": FakeInitialConfig()})()
        self._remaining_failures = int(fail_resets)

    def reset(self):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise FatalFlowError(msg="spawn failed")
        return None


class FakePlanningVehicleKernel(object):
    def __init__(self, ordered_ids, controllers):
        self._ordered_ids = ordered_ids
        self._controllers = controllers

    def get_controlled_ids(self):
        return list(self._ordered_ids)

    def get_acc_controller(self, veh_id):
        return self._controllers[veh_id]


class FakePlanningEnv(object):
    _get_coordinated_llm_controllers = Env._get_coordinated_llm_controllers
    _run_controlled_planning = Env._run_controlled_planning

    def __init__(self, ordered_ids, controllers, step=4):
        self.k = type("Kernel", (), {"vehicle": FakePlanningVehicleKernel(ordered_ids, controllers)})()
        self.message_pool = message_pool()
        self.message_pool.reset_rollout("scenario", 1)
        self.time_counter = step


class PlanningControllerStub(object):
    def __init__(self, veh_id, role, call_log):
        self.veh_id = veh_id
        self.attack_role = role
        self._call_log = call_log

    def uses_coordinated_structured_protocol(self):
        return True

    def prepare_for_coordinated_step(self, env, snapshot=None):
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        self._call_log.append((self.veh_id, active_owner_plan.get("plan_id", "")))
        if self.attack_role == "Blocker":
            env.message_pool.publish({
                "sender": self.veh_id,
                "owner": self.veh_id,
                "kind": "commit",
                "plan_id": "p1",
                "phase": "compress",
                "intent": "block",
                "step": env.time_counter,
                "expires_at_step": env.time_counter + 4,
                "control_cycle_step": env.time_counter,
            })


class TestHighwaySceneGate(unittest.TestCase):
    def test_scene_gate_rejects_bad_opening(self):
        geometry = {
            "ego_0": make_payload("ego_0", 1, 0.0),
            "llm_0": make_payload("llm_0", 1, 220.0),
            "llm_1": make_payload("llm_1", 2, -10.0),
            "human_0": make_payload("human_0", 0, -30.0),
            "human_1": make_payload("human_1", 3, 35.0),
            "human_2": make_payload("human_2", 0, -60.0),
            "human_3": make_payload("human_3", 3, 60.0),
        }

        gate = scene_gate_metrics(geometry)

        self.assertFalse(gate["accepted"])
        self.assertEqual(gate["reason"], "blocker_rel_x_out_of_band")

    def test_scene_gate_accepts_and_assigns_roles(self):
        geometry = {
            "ego_0": make_payload("ego_0", 1, 0.0),
            "llm_0": make_payload("llm_0", 0, 12.0),
            "llm_1": make_payload("llm_1", 2, -10.0),
            "human_0": make_payload("human_0", 2, -70.0),
            "human_1": make_payload("human_1", 3, -35.0),
            "human_2": make_payload("human_2", 1, 25.0),
            "human_3": make_payload("human_3", 3, 65.0),
        }

        gate = scene_gate_metrics(geometry)
        role_map = assign_roles_from_geometry(geometry)

        self.assertTrue(gate["accepted"])
        self.assertEqual(role_map["llm_0"], "Blocker")
        self.assertEqual(role_map["llm_1"], "Striker")

    def test_custom_scene_sampler_generates_valid_openings(self):
        env = FakeSceneEnv()
        rng = random.Random(7)
        for _ in range(100):
            layout = generate_custom_highway_opening(env, rng=rng)
            self.assertIsNotNone(layout)
            gate = scene_gate_metrics(layout["geometry"])
            self.assertTrue(gate["accepted"])
            self.assertLess(layout["geometry"]["ego_0"]["pos"], 320.0)

            key_positions = [
                layout["geometry"]["ego_0"]["pos"],
                layout["geometry"][gate["blocker_id"]]["pos"],
                layout["geometry"][gate["striker_id"]]["pos"],
            ]
            self.assertGreater(len(set(round(pos, 1) for pos in key_positions)), 2)

    def test_scene_freeze_retries_after_spawn_failure(self):
        env = FakeFreezableSceneEnv(fail_resets=1)
        scenario_context = sample_and_freeze_scene(env, "20260403-000000")
        self.assertTrue(scenario_context["scenario_id"])
        self.assertEqual(scenario_context["sampling_strategy"], "custom_highway_opening")
        self.assertTrue(scenario_context["scene_gate_status"]["accepted"])


class TestMessagePool(unittest.TestCase):
    def test_disengaged_owner_plan_is_not_active(self):
        pool = message_pool()
        pool.set_scenario_context({"role_map": {"llm_1": "Blocker", "llm_0": "Striker"}})
        pool.reset_rollout("scenario", 1)
        pool.set_scenario_context({"role_map": {"llm_1": "Blocker", "llm_0": "Striker"}})

        pool.publish({
            "sender": "llm_1",
            "owner": "llm_1",
            "kind": "commit",
            "plan_id": "p1",
            "phase": "compress",
            "intent": "box_in",
            "expires_at_step": 50,
        })
        active_snapshot = pool.snapshot(10)
        self.assertEqual(active_snapshot["active_owner_plan"]["plan_id"], "p1")
        self.assertEqual(active_snapshot["scenario_context"]["role_map"]["llm_1"], "Blocker")

        pool.publish({
            "sender": "llm_1",
            "owner": "llm_1",
            "kind": "replan",
            "plan_id": "p1",
            "phase": "disengage",
            "intent": "abort",
            "expires_at_step": 60,
        })
        terminal_snapshot = pool.snapshot(11)
        self.assertIsNone(terminal_snapshot["active_owner_plan"])

    def test_control_cycle_entries_are_exposed(self):
        pool = message_pool()
        pool.reset_rollout("scenario", 1)
        pool.begin_control_cycle(4)
        pool.publish({
            "sender": "llm_0",
            "owner": "llm_0",
            "kind": "commit",
            "plan_id": "p1",
            "phase": "compress",
            "intent": "block",
            "step": 4,
            "expires_at_step": 8,
            "control_cycle_step": 4,
        })

        snapshot = pool.snapshot(4)
        self.assertEqual(snapshot["control_cycle_step"], 4)
        self.assertEqual(len(snapshot["cycle_entries"]), 1)


class TestLLMControllerProtocol(unittest.TestCase):
    def make_controller(self, veh_id):
        controller = LLMController(
            veh_id=veh_id,
            car_following_params=SumoCarFollowingParams(),
        )
        return controller

    def test_owner_plan_reuse_and_phase_normalization(self):
        controller = self.make_controller("llm_1")
        controller.role_map = {"llm_1": "Blocker", "llm_0": "Striker"}
        controller.refresh_attack_role()

        env = DummyEnv(step=20, rollout_id=3)
        first = controller._normalize_structured_decision({
            "kind": "commit",
            "phase": "setup",
            "intent": "compress_now",
            "trigger_code": "none",
            "done_code": "phase_complete",
            "fallback": "hold_lane",
            "confidence": 0.8,
            "mode": "track_pose",
            "desired_rel_lane": 0,
            "desired_rel_x": 8.0,
            "desired_rel_s": 3.0,
            "horizon_steps": 12,
        }, env, {"active_owner_plan": {}, "scenario_context": {"role_map": controller.role_map}})

        self.assertEqual(first["message"]["owner"], "llm_1")
        self.assertEqual(first["message"]["kind"], "commit")
        self.assertEqual(first["message"]["phase"], "compress")
        self.assertEqual(first["message"]["plan_id"], "p1")

        env.time_counter = 30
        second = controller._normalize_structured_decision({
            "kind": "commit",
            "phase": "setup",
            "intent": "tighten",
            "trigger_code": "none",
            "done_code": "phase_complete",
            "fallback": "hold_lane",
            "confidence": 0.8,
            "mode": "track_pose",
            "desired_rel_lane": 0,
            "desired_rel_x": 4.0,
            "desired_rel_s": 2.5,
            "horizon_steps": 10,
        }, env, {
            "active_owner_plan": {
                "owner": "llm_1",
                "sender": "llm_1",
                "plan_id": "p1",
                "phase": "strike",
                "expires_at_step": 60,
            },
            "scenario_context": {"role_map": controller.role_map},
        })

        self.assertEqual(second["message"]["kind"], "replan")
        self.assertEqual(second["message"]["plan_id"], "p1")
        self.assertEqual(second["message"]["phase"], "strike")

    def test_structured_normalization_repairs_common_errors(self):
        controller = self.make_controller("llm_0")
        controller.role_map = {"llm_0": "Blocker", "llm_1": "Striker"}
        controller.refresh_attack_role()
        env = DummyEnv(step=16, rollout_id=2)

        decision = controller._normalize_structured_decision({
            "kind": "invalid",
            "phase": "status",
            "intent": "tighten",
            "trigger_code": "none",
            "done_code": "self_at_rel_pose",
            "fallback": "none",
            "confidence": 0.7,
            "mode": "stay",
            "desired_rel_lane": 0,
            "desired_rel_x": 8.0,
            "desired_rel_s": 4.0,
            "horizon_steps": 4,
        }, env, {"active_owner_plan": {}, "scenario_context": {"role_map": controller.role_map}})

        self.assertEqual(decision["message"]["kind"], "commit")
        self.assertEqual(decision["message"]["phase"], "compress")
        self.assertEqual(decision["message"]["done_code"], "reached_rel_pose")
        self.assertEqual(decision["message"]["fallback"], "hold_lane")
        self.assertEqual(decision["control"]["mode"], "hold_lane")
        self.assertGreater(controller.normalization_repairs, 0)

    def test_prompt_summaries_are_deambiguated(self):
        controller = self.make_controller("llm_0")
        controller.role_map = {"llm_0": "Blocker", "llm_1": "Striker"}
        controller.refresh_attack_role()

        snapshot = {
            "control_cycle_step": 4,
            "active_owner_plan": {
                "owner": "llm_0",
                "sender": "llm_0",
                "plan_id": "p4",
                "phase": "compress",
                "intent": "squeeze",
                "expires_at_step": 40,
            },
            "latest_commit": {
                "owner": "llm_0",
                "sender": "llm_0",
                "plan_id": "p4",
            },
            "latest_status_by_agent": {
                "llm_0": {"plan_id": "p4", "intent": "squeeze"},
                "llm_1": {"plan_id": "p4", "intent": "ready"},
            },
            "trigger_eval": {
                "llm_0": {"satisfied": True},
                "llm_1": {"satisfied": False},
            },
            "recent_events": [
                {"sender": "llm_0", "kind": "commit", "phase": "compress", "intent": "squeeze"},
                {"sender": "llm_1", "kind": "ack", "phase": "compress", "intent": "ready"},
            ],
        }
        blackboard_summary = controller._build_blackboard_summary(snapshot)

        self.assertIn("owner_plan_present=yes", blackboard_summary)
        self.assertIn("teammate_reply_seen=yes", blackboard_summary)
        self.assertNotIn("aligned", blackboard_summary)
        self.assertNotIn("stale_or_missing", blackboard_summary)
        self.assertNotIn("status/", blackboard_summary)

    def test_same_tick_planning_runs_blocker_before_striker(self):
        call_log = []
        controllers = {
            "llm_0": PlanningControllerStub("llm_0", "Blocker", call_log),
            "llm_1": PlanningControllerStub("llm_1", "Striker", call_log),
        }
        env = FakePlanningEnv(["llm_1", "llm_0"], controllers, step=4)

        env._run_controlled_planning()

        self.assertEqual(call_log[0][0], "llm_0")
        self.assertEqual(call_log[1], ("llm_1", "p1"))


if __name__ == "__main__":
    unittest.main()
