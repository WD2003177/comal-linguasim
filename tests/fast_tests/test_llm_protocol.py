import copy
import csv
import os
import random
import tempfile
import unittest
from unittest.mock import patch

from examples.simulate import resolve_roles_for_rollout
from flow.controllers.llm_controller import DriverAgent
from flow.controllers.llm_controller import LLMController
from flow.core.params import SumoCarFollowingParams
from flow.utils.agents_network import message_pool
from flow.utils.exceptions import FatalFlowError
from flow.utils.highway_scene import assign_roles_from_geometry
from flow.utils.highway_scene import generate_custom_highway_opening
from flow.utils.highway_scene import generate_three_car_fixed_opening
from flow.utils.highway_scene import sample_and_freeze_scene
from flow.utils.highway_scene import scene_gate_metrics

try:
    from flow.core.kernel.simulation.traci import TraCISimulation
except Exception:  # pragma: no cover - optional runtime dependency in tests
    TraCISimulation = None


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
    def __init__(self, num_humans=8):
        self.initial_ids = ["ego_0", "llm_0", "llm_1"] + ["human_{}".format(i) for i in range(num_humans)]
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
    def __init__(self, fail_resets=1, num_humans=8):
        super(FakeFreezableSceneEnv, self).__init__(num_humans=num_humans)
        self.initial_config = FakeInitialConfig()
        self.network = type("Network", (), {"initial_config": FakeInitialConfig()})()
        self._remaining_failures = int(fail_resets)
        self.sim_params = type("SimParams", (), {"emission_path": None})()
        self.k.simulation = type("Simulation", (), {"emission_path": None})()
        self.initial_vehicles = type("Vehicles", (), {"get_type": lambda self, veh_id: "human"})()

    def reset(self):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise FatalFlowError(msg="spawn failed")
        return None

    def restart_simulation(self, sim_params):
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
    def __init__(self, ordered_ids, controllers, step=4):
        self.k = type("Kernel", (), {"vehicle": FakePlanningVehicleKernel(ordered_ids, controllers)})()
        self.message_pool = message_pool()
        self.message_pool.reset_rollout("scenario", 1)
        self.time_counter = step

    def _get_coordinated_llm_controllers(self):
        controllers = []
        for veh_id in self.k.vehicle.get_controlled_ids():
            controller = self.k.vehicle.get_acc_controller(veh_id)
            if controller is None:
                continue
            if not hasattr(controller, "uses_coordinated_planning"):
                continue
            if not controller.uses_coordinated_planning():
                continue
            controllers.append(controller)
        controllers.sort(key=lambda controller: getattr(controller, "veh_id", ""))
        return controllers

    def _run_controlled_planning(self):
        controllers = self._get_coordinated_llm_controllers()
        if not controllers:
            return

        step = int(self.time_counter)
        self.message_pool.begin_control_cycle(step)
        snapshot = self.message_pool.snapshot(step, viewer_id="coordinator")
        for controller in controllers:
            controller.run_coordinated_step(self, snapshot=snapshot)
            snapshot = self.message_pool.snapshot(step, viewer_id=controller.veh_id)


class PlanningControllerStub(object):
    def __init__(self, veh_id, call_log):
        self.veh_id = veh_id
        self._call_log = call_log

    def uses_coordinated_planning(self):
        return True

    def run_coordinated_step(self, env, snapshot=None):
        self._call_log.append((self.veh_id, snapshot.get("viewer_id", "")))
        return True


class NegotiationControllerStub(object):
    def __init__(self, veh_id, proposals):
        self.veh_id = veh_id
        self._proposals = list(proposals)
        self.calls = 0
        self.role_map = {}
        self.role_source = ""
        self.geometry_role_hint = {}
        self.attack_role = "Undecided"
        self.active_phase = "negotiation"
        self.role_resolution_fallback_used = 0

    def negotiate_role(self, env):
        index = min(self.calls, len(self._proposals) - 1)
        self.calls += 1
        return dict(self._proposals[index])

    def set_role_assignment(self, role_map, role_source=""):
        self.role_map = dict(role_map)
        self.role_source = str(role_source or "")
        self.attack_role = self.role_map.get(self.veh_id, "Undecided")


class FakeHighwayVehicleKernel(object):
    def __init__(self, vehicles):
        self._vehicles = dict(vehicles)

    def get_ids(self):
        return list(self._vehicles.keys())

    def get_speed(self, veh_id):
        return float(self._vehicles[veh_id]["speed"])

    def get_previous_speed(self, veh_id):
        return float(self._vehicles[veh_id].get("prev_speed", self.get_speed(veh_id)))

    def get_lane(self, veh_id):
        return int(self._vehicles[veh_id]["lane"])

    def get_x_by_id(self, veh_id):
        return float(self._vehicles[veh_id]["pos"])

    def get_position(self, veh_id):
        return float(self._vehicles[veh_id]["pos"])

    def get_edge(self, veh_id):
        return str(self._vehicles[veh_id].get("edge", "highway_0"))

    def get_length(self, veh_id):
        return float(self._vehicles[veh_id].get("length", 5.0))

    def get_headway(self, veh_id):
        leader_id = self.get_leader(veh_id)
        if not leader_id:
            return 1e6
        return max(
            0.0,
            self.get_position(leader_id)
            - self.get_position(veh_id)
            - self.get_length(leader_id),
        )

    def get_leader(self, veh_id):
        lane = self.get_lane(veh_id)
        edge = self.get_edge(veh_id)
        pos = self.get_position(veh_id)
        leaders = [
            other_id for other_id in self.get_ids()
            if other_id != veh_id
            and self.get_edge(other_id) == edge
            and self.get_lane(other_id) == lane
            and self.get_position(other_id) > pos
        ]
        if not leaders:
            return ""
        return min(leaders, key=lambda other_id: self.get_position(other_id))

    def get_follower(self, veh_id):
        lane = self.get_lane(veh_id)
        edge = self.get_edge(veh_id)
        pos = self.get_position(veh_id)
        followers = [
            other_id for other_id in self.get_ids()
            if other_id != veh_id
            and self.get_edge(other_id) == edge
            and self.get_lane(other_id) == lane
            and self.get_position(other_id) < pos
        ]
        if not followers:
            return ""
        return max(followers, key=lambda other_id: self.get_position(other_id))


class FakeHighwayEnv(object):
    def __init__(self, vehicles, step=4):
        self.time_counter = step
        self.time_step = step
        self.sim_step = 0.1
        self.message_pool = message_pool()
        self.message_pool.reset_rollout("scenario", 1)
        self.k = type("Kernel", (), {})()
        self.k.vehicle = FakeHighwayVehicleKernel(vehicles)
        self.k.network = FakeNetwork()
        api_vehicle = type("ApiVehicle", (), {})()
        api_vehicle.change_calls = []
        api_vehicle.relative_change_calls = []
        api_vehicle.speed_mode_calls = []
        api_vehicle.lane_change_mode_calls = []
        api_vehicle.parameter_calls = []
        api_vehicle.sublane_calls = []

        def set_max_speed(veh_id, speed):
            self.k.vehicle._vehicles[veh_id]["max_speed_cmd"] = float(speed)

        def set_min_gap(veh_id, gap):
            self.k.vehicle._vehicles[veh_id]["min_gap_cmd"] = float(gap)

        def set_speed_mode(veh_id, mode):
            api_vehicle.speed_mode_calls.append((veh_id, int(mode)))
            self.k.vehicle._vehicles[veh_id]["speed_mode_cmd"] = int(mode)

        def set_lane_change_mode(veh_id, mode):
            api_vehicle.lane_change_mode_calls.append((veh_id, int(mode)))
            self.k.vehicle._vehicles[veh_id]["lane_change_mode_cmd"] = int(mode)

        def set_parameter(veh_id, key, value):
            api_vehicle.parameter_calls.append((veh_id, str(key), str(value)))

        def change_lane(veh_id, target_lane, duration):
            self.k.vehicle._vehicles[veh_id]["lane"] = int(target_lane)
            api_vehicle.change_calls.append((veh_id, int(target_lane), int(duration)))

        def change_lane_relative(veh_id, delta, duration):
            api_vehicle.relative_change_calls.append((veh_id, int(delta), int(duration)))

        def change_sublane(veh_id, delta):
            api_vehicle.sublane_calls.append((veh_id, float(delta)))

        api_vehicle.setMaxSpeed = set_max_speed
        api_vehicle.setMinGap = set_min_gap
        api_vehicle.setSpeedMode = set_speed_mode
        api_vehicle.setLaneChangeMode = set_lane_change_mode
        api_vehicle.setParameter = set_parameter
        api_vehicle.changeLane = change_lane
        api_vehicle.changeLaneRelative = change_lane_relative
        api_vehicle.changeSublane = change_sublane
        self.k.kernel_api = type("KernelApi", (), {"vehicle": api_vehicle})()


class FakeKernelApi(object):
    def close(self):
        return None


class FakeEmissionMasterKernel(object):
    def __init__(self, name="highway"):
        net = type("Net", (), {"orig_name": name, "name": name})()
        self.network = type("NetworkWrapper", (), {"network": net})()


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

    def test_scene_gate_accepts_and_assigns_geometry_hint(self):
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
        for _ in range(50):
            layout = generate_custom_highway_opening(env, rng=rng)
            self.assertIsNotNone(layout)
            gate = scene_gate_metrics(layout["geometry"])
            self.assertTrue(gate["accepted"])
            self.assertLess(layout["geometry"]["ego_0"]["pos"], 320.0)

    def test_three_car_fixed_sampler_generates_valid_opening_without_humans(self):
        env = FakeSceneEnv(num_humans=0)

        layout = generate_three_car_fixed_opening(env)

        self.assertIsNotNone(layout)
        gate = scene_gate_metrics(layout["geometry"], scene_mode="three_car_fixed")
        self.assertTrue(gate["accepted"])
        self.assertEqual(len(layout["geometry"]), 3)
        self.assertEqual(gate["background_nearby_count"], 0)
        self.assertTrue(-5.5 <= layout["geometry"]["llm_1"]["rel_x_to_ego"] <= -4.5)
        self.assertGreaterEqual(
            layout["geometry"]["llm_1"]["speed"] - layout["geometry"]["ego_0"]["speed"],
            2.5,
        )

    def test_scene_freeze_returns_empty_role_map_and_geometry_hint(self):
        env = FakeFreezableSceneEnv(fail_resets=1)
        with patch.dict(os.environ, {"FLOW_HIGHWAY_SCENE_MODE": "traffic"}, clear=False):
            scenario_context = sample_and_freeze_scene(env, "20260403-000000")
        self.assertTrue(scenario_context["scenario_id"])
        self.assertEqual(scenario_context["sampling_strategy"], "custom_highway_opening")
        self.assertTrue(scenario_context["scene_gate_status"]["accepted"])
        self.assertEqual(scenario_context["role_map"], {})
        self.assertEqual(scenario_context["role_source"], "")
        self.assertTrue(scenario_context["geometry_role_hint"])
        self.assertNotIn("role_map", scenario_context["scene_gate_status"])

    def test_scene_freeze_returns_three_car_fixed_context(self):
        env = FakeFreezableSceneEnv(fail_resets=0, num_humans=0)

        with patch.dict(os.environ, {"FLOW_HIGHWAY_SCENE_MODE": "three_car_fixed"}, clear=False):
            scenario_context = sample_and_freeze_scene(env, "20260403-000000")

        self.assertTrue(scenario_context["scenario_id"])
        self.assertEqual(scenario_context["sampling_strategy"], "three_car_fixed_opening")
        self.assertEqual(scenario_context["scene_gate_status"]["scene_mode"], "three_car_fixed")
        self.assertTrue(scenario_context["scene_gate_status"]["accepted"])
        self.assertEqual(len(scenario_context["frozen_geometry"]), 3)


class TestCoordinatedHook(unittest.TestCase):
    def test_env_planning_uses_generic_hook_and_vehicle_order(self):
        call_log = []
        controllers = {
            "llm_0": PlanningControllerStub("llm_0", call_log),
            "llm_1": PlanningControllerStub("llm_1", call_log),
        }
        env = FakePlanningEnv(["llm_1", "llm_0"], controllers, step=4)

        env._run_controlled_planning()

        self.assertEqual(call_log[0][0], "llm_0")
        self.assertEqual(call_log[1][0], "llm_1")


class TestRoleResolution(unittest.TestCase):
    def make_env(self, controllers):
        return FakePlanningEnv(sorted(controllers.keys()), controllers, step=4)

    def test_role_resolution_uses_geometry_lock_by_default(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "decision": "confirm",
                "message": "Geometry role is fine.",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "decision": "confirm",
                "message": "Keep the locked split.",
            }]),
        }
        env = self.make_env(controllers)
        scenario_context = {
            "scenario_id": "scenario",
            "role_map": {},
            "role_source": "",
            "geometry_role_hint": {"llm_0": "Blocker", "llm_1": "Striker"},
        }

        role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration=1)

        self.assertEqual(role_map, {"llm_0": "Blocker", "llm_1": "Striker"})
        self.assertEqual(scenario_context["role_source"], "llm_confirmed_geometry")
        self.assertEqual(controllers["llm_0"].attack_role, "Blocker")
        self.assertEqual(controllers["llm_1"].attack_role, "Striker")

    def test_role_resolution_keeps_geometry_when_only_one_requests_swap(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "decision": "swap",
                "message": "Swap would fit me better.",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "decision": "confirm",
                "message": "Do not swap.",
            }]),
        }
        env = self.make_env(controllers)
        scenario_context = {
            "scenario_id": "scenario",
            "role_map": {},
            "role_source": "",
            "geometry_role_hint": {"llm_0": "Blocker", "llm_1": "Striker"},
        }

        role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration=1)

        self.assertEqual(role_map, {"llm_0": "Blocker", "llm_1": "Striker"})
        self.assertEqual(scenario_context["role_source"], "geometry_locked")
        self.assertEqual(controllers["llm_0"].role_resolution_fallback_used, 0)
        self.assertEqual(controllers["llm_1"].role_resolution_fallback_used, 0)

    def test_role_resolution_swaps_only_when_both_swap(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "decision": "swap",
                "message": "I should be striker.",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "decision": "swap",
                "message": "I should be blocker.",
            }]),
        }
        env = self.make_env(controllers)
        scenario_context = {
            "scenario_id": "scenario",
            "role_map": {},
            "role_source": "",
            "geometry_role_hint": {"llm_0": "Blocker", "llm_1": "Striker"},
        }

        role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration=1)

        self.assertEqual(role_map, {"llm_0": "Striker", "llm_1": "Blocker"})
        self.assertEqual(scenario_context["role_source"], "llm_swapped")

    def test_existing_role_map_is_reused_without_new_negotiation(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "decision": "swap",
                "message": "unused",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "decision": "confirm",
                "message": "unused",
            }]),
        }
        env = self.make_env(controllers)
        scenario_context = {
            "scenario_id": "scenario",
            "role_map": {"llm_0": "Blocker", "llm_1": "Striker"},
            "role_source": "llm_negotiated",
            "geometry_role_hint": {"llm_0": "Striker", "llm_1": "Blocker"},
        }

        role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration=2)

        self.assertEqual(role_map, {"llm_0": "Blocker", "llm_1": "Striker"})
        self.assertEqual(controllers["llm_0"].calls, 0)
        self.assertEqual(controllers["llm_1"].calls, 0)

    def test_negotiated_role_resolution_freezes_contract(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "proposed_role": "Blocker",
                "pass_side": "left",
                "message": "I will seal the right escape side.",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "proposed_role": "Striker",
                "pass_side": "left",
                "message": "I will cut in from ego_0's left side.",
            }]),
        }
        env = self.make_env(controllers)
        scenario_context = {
            "scenario_id": "scenario",
            "role_map": {},
            "role_source": "",
            "geometry_role_hint": {"llm_0": "Blocker", "llm_1": "Striker"},
            "frozen_geometry": {
                "ego_0": make_payload("ego_0", 1, 0.0),
                "llm_0": make_payload("llm_0", 2, 14.0),
                "llm_1": make_payload("llm_1", 0, -5.0),
            },
            "scene_gate_status": {"scene_mode": "three_car_fixed_negotiated"},
        }

        role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration=1)

        self.assertEqual(role_map, {"llm_0": "Blocker", "llm_1": "Striker"})
        self.assertEqual(scenario_context["role_source"], "llm_negotiated")
        self.assertEqual(scenario_context["contract_source"], "negotiated")
        self.assertEqual(scenario_context["pass_side"], "left")
        self.assertEqual(scenario_context["block_side"], "right")
        self.assertEqual(env.message_pool.negotiated_contract["pass_side"], "left")
        self.assertEqual(env.message_pool.negotiated_contract["block_side"], "right")


class TestNegotiatedMessagePool(unittest.TestCase):
    def test_negotiated_namespace_is_isolated_from_legacy_blackboard(self):
        pool = message_pool()
        pool.reset_rollout("scenario", 1)
        pool.begin_control_cycle(4)
        pool.join("llm_0", "legacy message")
        pool.publish_negotiated_negotiation({
            "sender": "llm_0",
            "proposed_role": "Blocker",
            "pass_side": "left",
            "message": "seal side",
            "step": 0,
            "control_cycle_step": 4,
        })
        pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Striker",
            "primitive": "merge_commit",
            "eta": 1,
            "step": 4,
            "control_cycle_step": 4,
        })

        self.assertEqual(pool.get_all_msg()["llm_0"], "legacy message")
        negotiated_snapshot = pool.negotiated_snapshot(4, viewer_id="llm_0")
        self.assertIn("llm_0", negotiated_snapshot["latest_negotiation_by_agent"])
        self.assertEqual(
            negotiated_snapshot["latest_primitive_by_agent"]["llm_1"]["primitive"],
            "merge_commit",
        )
        self.assertNotIn("latest_primitive_by_agent", pool.snapshot(4, viewer_id="llm_0"))


class TestLLMControllerSimpleProtocol(unittest.TestCase):
    def make_controller(self, veh_id):
        return LLMController(
            veh_id=veh_id,
            map="highway",
            car_following_params=SumoCarFollowingParams(),
        )

    def test_highway_controller_uses_generic_hook_not_structured(self):
        controller = self.make_controller("llm_0")
        self.assertTrue(controller.uses_coordinated_planning())
        self.assertFalse(controller.uses_coordinated_structured_protocol())

    def test_feedback_prompt_is_one_line_summary(self):
        controller = self.make_controller("llm_0")
        controller.previous_feedback = {
            "result": "too_safe",
            "feedback_summary": "ego_escaped_right",
            "ego_max_decel": 1.75,
        }
        summary = controller._build_feedback_prompt()
        self.assertIn("too_safe", summary)
        self.assertIn("ego_escaped_right", summary)
        self.assertNotIn("memory=", summary)

    def test_rollout_diagnostics_keep_simple_compat_fields(self):
        controller = self.make_controller("llm_0")
        controller.role_resolution_fallback_used = 1
        controller.role_source = "geometry_locked"
        controller.geometry_role_hint = {"llm_0": "Blocker", "llm_1": "Striker"}
        controller.intent = "gain_lead"
        controller.intent_urgency = "high"
        controller.executor_state = "chase"
        controller.target_abs_lane = 1
        controller.target_v = 28.0
        controller.target_s = 0.8
        controller.lead_acquired = False
        controller.brake_armed = False
        controller.striker_completed_cut_in = False
        controller.striker_became_ego_leader = True
        diagnostics = controller.get_rollout_diagnostics()
        self.assertEqual(diagnostics["role_resolution_fallback_used"], 1)
        self.assertEqual(diagnostics["role_source"], "geometry_locked")
        self.assertIn("geometry_role_hint", diagnostics)
        self.assertEqual(diagnostics["intent"], "gain_lead")
        self.assertEqual(diagnostics["urgency"], "high")
        self.assertEqual(diagnostics["executor_state"], "chase")
        self.assertEqual(diagnostics["target_abs_lane"], 1)
        self.assertTrue(diagnostics["striker_became_ego_leader"])


class TestHighwayControllerExecutor(unittest.TestCase):
    def make_controller(self, veh_id, role_map):
        controller = LLMController(
            veh_id=veh_id,
            map="highway",
            car_following_params=SumoCarFollowingParams(),
        )
        controller.set_role_assignment(role_map, role_source="geometry_locked")
        controller.scene_gate_status = {"scene_mode": "traffic"}
        return controller

    def set_step(self, env, step):
        env.time_counter = int(step)
        env.time_step = int(step)

    def test_highway_llm_collaborate_uses_new_json_schema(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -6.0, speed=21.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.DA.collaborate = lambda *args, **kwargs: (
            "{\"intent\":\"cut_in\",\"urgency\":\"high\",\"message\":\"move now\"}"
        )

        decision = controller.llm_collaborate(env)

        self.assertEqual(
            decision,
            {"intent": "cut_in", "urgency": "high", "message": "move now"},
        )
        self.assertIn("intent=cut_in", env.message_pool.get_all_msg()["llm_0"])

    def test_negotiated_highway_llm_collaborate_uses_structured_namespace(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Blocker",
            "primitive": "hold_side_front",
            "step": 4,
            "control_cycle_step": 4,
        })
        controller._begin_rollout_if_needed(env)
        controller.DA.collaborate_highway_primitive = lambda *args, **kwargs: (
            "{\"primitive\":\"merge_commit\",\"eta\":1,\"message\":\"go\"}"
        )

        decision = controller.llm_collaborate(env)

        self.assertEqual(decision["primitive"], "merge_commit")
        self.assertEqual(env.message_pool.get_all_msg(), {})
        negotiated_snapshot = env.message_pool.negotiated_snapshot(4, viewer_id="llm_0")
        self.assertEqual(
            negotiated_snapshot["latest_requested_primitive_by_agent"]["llm_0"]["primitive"],
            "merge_commit",
        )
        self.assertNotIn("llm_0", negotiated_snapshot["latest_primitive_by_agent"])

    def test_negotiated_channel_publishes_executor_commit_not_raw_request(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -8.0, speed=24.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 14.0, speed=23.2, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "primitive": "merge_commit",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "merge now",
        }
        controller._publish_negotiated_request(env, controller.current_decision)
        controller.last_control_step = env.time_step

        controller.get_accel(env)

        negotiated_snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        self.assertEqual(
            negotiated_snapshot["latest_requested_primitive_by_agent"]["llm_0"]["primitive"],
            "merge_commit",
        )
        self.assertEqual(
            negotiated_snapshot["latest_primitive_by_agent"]["llm_0"]["primitive"],
            "gain_lead",
        )

    def test_highway_perception_includes_geometry_fields(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -6.0, speed=21.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )

        perception = controller.get_perception(env)

        self.assertIn("ego_lane=2", perception)
        self.assertIn("self_lane=1", perception)
        self.assertIn("delta_to_ego_lane=-1", perception)
        self.assertIn("is_adjacent_to_ego_lane=yes", perception)
        self.assertIn("lead_gap_if_same_lane=n/a", perception)

    def test_striker_behind_rewrites_cut_in_to_gain_lead_and_targets_passing_side(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -8.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "cut now",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "gain_lead")
        self.assertIn(controller.executor_state, ("chase", "align"))
        self.assertGreater(controller.target_v, vehicles["ego_0"]["speed"])
        self.assertEqual(controller.target_abs_lane, 1)

    def test_three_car_striker_can_cut_in_without_full_lead(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.5, speed=25.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 12.0, speed=23.8, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "force merge now",
        }

        controller._apply_highway_executor(env)

        self.assertFalse(controller.lead_acquired)
        self.assertEqual(controller.intent, "cut_in")
        self.assertEqual(controller.executor_state, "cut_in")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertTrue(controller._allow_aggressive_cut_in)
        self.assertAlmostEqual(controller.target_s, 0.6)

    def test_three_car_merge_commit_persists_after_llm_switches_back_to_gain_lead(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.5, speed=25.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 12.0, speed=23.8, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "force merge now",
        }

        controller._apply_highway_executor(env)
        self.assertGreater(controller._merge_commit_until_step, env.time_step)

        self.set_step(env, 8)
        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] - 0.3
        controller.current_decision = {
            "intent": "gain_lead",
            "urgency": "mid",
            "message": "keep pressure",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "cut_in")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertIn(controller.executor_state, ("cut_in", "merge_commit"))
        self.assertTrue(controller._allow_aggressive_cut_in)

    def test_three_car_striker_auto_brakes_after_merge(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 2.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 9.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=12)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "gain_lead",
            "urgency": "mid",
            "message": "front trap",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "brake_pulse")
        self.assertEqual(controller.executor_state, "front_brake")
        self.assertTrue(controller.brake_armed)
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])
        self.assertGreater(controller._pulse_end_step, env.time_step)

    def test_three_car_blocker_reclaims_side_front_aggressively(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 5.0, speed=21.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -5.0, speed=25.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "hold_side_front",
            "urgency": "high",
            "message": "seal side",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "claim_side")
        self.assertEqual(controller.executor_state, "claim")
        self.assertEqual(controller.target_abs_lane, 3)
        self.assertGreater(controller.target_v, vehicles["ego_0"]["speed"] + 3.0)
        self.assertAlmostEqual(controller.target_s, 0.8)

    def test_negotiated_front_brake_triggers_from_merge_contract_completion(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.4, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=12)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Blocker",
            "primitive": "seal_escape",
            "step": 12,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller._prev_self_lane = 1
        controller.current_decision = {
            "primitive": "merge_commit",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "commit",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")
        self.assertTrue(controller._front_brake_triggered)
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_negotiated_front_brake_accepts_blocker_geometry_support(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 2.4, speed=23.8, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.2, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=12)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Blocker",
            "primitive": "hold_side_front",
            "step": 12,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller._prev_self_lane = 1
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 18
        controller.current_decision = {
            "primitive": "gain_lead",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "finish attack",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")
        self.assertTrue(controller._front_brake_triggered)

    def test_negotiated_merge_commit_persists_without_repeated_llm_confirmation(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -1.2, speed=24.6, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.5, speed=23.4, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Blocker",
            "primitive": "hold_side_front",
            "step": 8,
            "control_cycle_step": 8,
        })
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "primitive": "merge_commit",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "go",
        }

        controller._apply_highway_executor(env)
        self.assertEqual(controller.intent, "merge_commit")
        self.assertGreater(controller._merge_commit_until_step, env.time_step)

        env.time_step = 12
        env.time_counter = 12
        controller.current_decision = {
            "primitive": "gain_lead",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "keep pressure",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "merge_commit")
        self.assertEqual(controller.executor_state, "merge_commit")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertTrue(controller._allow_aggressive_cut_in)

    def test_negotiated_blocker_auto_seals_from_striker_merge_request(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.4, speed=23.1, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.4, speed=24.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=12)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Blocker", "llm_1": "Striker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_request({
            "sender": "llm_1",
            "role": "Striker",
            "primitive": "merge_commit",
            "step": 12,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "primitive": "hold_side_front",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "hold",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "seal_escape")
        self.assertEqual(controller.executor_state, "seal")
        self.assertGreater(controller._seal_escape_until_step, env.time_step)

    def test_negotiated_blocker_does_not_upgrade_to_seal_on_front_brake_alone(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.0, speed=23.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 2, 3.0, speed=22.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=12)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": {"llm_0": "Blocker", "llm_1": "Striker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        env.message_pool.publish_negotiated_primitive({
            "sender": "llm_1",
            "role": "Striker",
            "primitive": "front_brake",
            "step": 12,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "primitive": "seal_escape",
            "eta": 0,
            "target_v": None,
            "target_s": None,
            "message": "seal now",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "hold_side_front")
        self.assertEqual(controller.executor_state, "hold")

    def test_highway_runtime_trace_records_aggressive_cut_in_and_lane_change_attempt(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 12.0, speed=23.8, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -2.5, speed=25.2, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=16)
        controller = self.make_controller(
            "llm_1",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "force merge now",
        }
        controller.last_control_step = env.time_step

        controller.get_accel(env)

        runtime_entry = controller.llm_trace_entries[-1]
        self.assertEqual(runtime_entry["protocol"], "highway_runtime")
        self.assertAlmostEqual(runtime_entry["rel_x"], -2.5, places=2)
        self.assertGreaterEqual(runtime_entry["speed_adv"], 2.0)
        self.assertTrue(runtime_entry["aggressive_cut_in_ready"])
        self.assertEqual(runtime_entry["target_lc"], 1)
        self.assertTrue(runtime_entry["lane_change_attempted"])
        self.assertEqual(env.k.vehicle.get_lane("llm_1"), 2)
        self.assertIn(("llm_1", 0), env.k.kernel_api.vehicle.speed_mode_calls)
        self.assertIn(("llm_1", 0), env.k.kernel_api.vehicle.lane_change_mode_calls)
        self.assertTrue(any(call[0] == "llm_1" and call[2] >= 3 for call in env.k.kernel_api.vehicle.change_calls))

    def test_striker_chase_speed_scales_with_rel_x_and_uses_chase_idm_profile(self):
        near_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -5.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        })
        far_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -15.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        })
        near_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        far_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        near_controller._begin_rollout_if_needed(near_env)
        far_controller._begin_rollout_if_needed(far_env)

        near_controller.current_decision = {"intent": "gain_lead", "urgency": "high", "message": "close in"}
        far_controller.current_decision = {"intent": "gain_lead", "urgency": "high", "message": "close in"}

        near_controller._apply_highway_executor(near_env)
        far_controller._apply_highway_executor(far_env)

        self.assertGreater(far_controller.target_v, near_controller.target_v)
        self.assertAlmostEqual(far_controller.idm_a, 1.8)
        self.assertAlmostEqual(far_controller.T, 0.6)

    def test_highway_executor_resets_phase_profile_before_early_return(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -10.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {"intent": "gain_lead", "urgency": "high", "message": "chase"}

        controller._apply_highway_executor(env)

        self.assertAlmostEqual(controller.idm_a, 1.8)
        self.assertAlmostEqual(controller.T, 0.6)

        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] + 5.0
        env.k.vehicle._vehicles["llm_0"]["lane"] = 2
        controller.current_decision = {"intent": "abort", "urgency": "low", "message": "back off"}

        controller._apply_highway_executor(env)

        self.assertAlmostEqual(controller.idm_a, controller.base_idm_a)
        self.assertAlmostEqual(controller.T, controller.base_T)

    def test_striker_cannot_enter_brake_before_lead(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, -1.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, 8.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "brake_pulse",
            "urgency": "high",
            "message": "brake now",
        }

        controller._apply_highway_executor(env)

        self.assertNotEqual(controller.executor_state, "brake")
        self.assertEqual(controller.intent, "gain_lead")
        self.assertFalse(controller.brake_armed)

    def test_lane_direction_comes_from_target_lane_not_message_text(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 7.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 9.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
            "human_1": make_payload("human_1", 2, -30.0, speed=21.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "change right immediately",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "cut_in")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 1)

    def test_cut_in_window_arms_once_and_expires_after_four_sim_steps(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 7.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 9.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
            "human_1": make_payload("human_1", 2, -30.0, speed=21.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "merge now",
        }

        controller._apply_highway_executor(env)
        self.assertTrue(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, 7)
        self.assertAlmostEqual(controller.target_s, 1.3)

        self.set_step(env, 5)
        controller._apply_highway_executor(env)
        self.assertEqual(controller._cut_in_aggressive_until_step, 7)
        self.assertAlmostEqual(controller.target_s, 1.3)

        self.set_step(env, 7)
        controller._apply_highway_executor(env)
        self.assertAlmostEqual(controller.target_s, 1.3)

        self.set_step(env, 8)
        controller._apply_highway_executor(env)
        self.assertTrue(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, 7)
        self.assertAlmostEqual(controller.target_s, 2.0)

    def test_cut_in_episode_clears_when_transitioning_to_brake_pulse(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 4.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, 8.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
            "human_1": make_payload("human_1", 2, -30.0, speed=21.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller._cut_in_episode_active = True
        controller._cut_in_aggressive_until_step = 12
        controller.current_decision = {
            "intent": "brake_pulse",
            "urgency": "high",
            "message": "brake now",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "brake_pulse")
        self.assertFalse(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, -1)

    def test_active_brake_pulse_path_clears_cut_in_episode_state(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 4.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, 8.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller._cut_in_episode_active = True
        controller._cut_in_aggressive_until_step = 12
        controller._pulse_end_step = 10

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "brake_pulse")
        self.assertFalse(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, -1)

    def test_finished_brake_pulse_path_clears_cut_in_episode_state(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 4.0, speed=20.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, 8.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=11)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller._cut_in_episode_active = True
        controller._cut_in_aggressive_until_step = 12
        controller._pulse_end_step = 10

        controller._apply_highway_executor(env)

        self.assertEqual(controller.intent, "abort")
        self.assertFalse(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, -1)

    def test_gain_lead_transition_clears_cut_in_episode_and_allows_rearm(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 7.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 9.0, speed=24.0, ego_lane=2),
            "human_0": make_payload("human_0", 2, 20.0, speed=22.0, ego_lane=2),
            "human_1": make_payload("human_1", 2, -30.0, speed=21.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "cut_in",
            "urgency": "high",
            "message": "merge now",
        }

        controller._apply_highway_executor(env)
        self.assertTrue(controller._cut_in_episode_active)

        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] - 3.0
        self.set_step(env, 5)
        controller._apply_highway_executor(env)
        self.assertEqual(controller.intent, "gain_lead")
        self.assertFalse(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, -1)

        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] + 7.0
        self.set_step(env, 6)
        controller._apply_highway_executor(env)
        self.assertTrue(controller._cut_in_episode_active)
        self.assertEqual(controller._cut_in_aggressive_until_step, 9)

    def test_blocker_does_not_enter_ego_lane_or_striker_side(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 6.0, speed=23.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, -5.0, speed=21.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "seal_escape",
            "urgency": "mid",
            "message": "hold left side",
        }

        controller._apply_highway_executor(env)

        self.assertEqual(controller.target_abs_lane, 1)
        self.assertNotEqual(controller.target_abs_lane, 2)
        self.assertNotEqual(controller.target_abs_lane, 3)

    def test_blocker_hold_side_front_uses_window_feedback(self):
        near_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 6.2, speed=23.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, -5.0, speed=21.0, ego_lane=2),
        })
        far_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 10.0, speed=23.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, -5.0, speed=21.0, ego_lane=2),
        })
        near_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        far_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        near_controller._begin_rollout_if_needed(near_env)
        far_controller._begin_rollout_if_needed(far_env)
        near_controller.current_decision = {
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold window",
        }
        far_controller.current_decision = {
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold window",
        }

        near_controller._apply_highway_executor(near_env)
        far_controller._apply_highway_executor(far_env)

        self.assertEqual(near_controller.intent, "hold_side_front")
        self.assertGreater(near_controller.target_v, near_env.k.vehicle.get_speed("ego_0"))
        self.assertLess(far_controller.target_v, far_env.k.vehicle.get_speed("ego_0"))
        self.assertAlmostEqual(near_controller.target_s, 1.3)

    def test_blocker_seal_escape_gap_is_tighter_than_hold(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 6.0, speed=23.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, -5.0, speed=21.0, ego_lane=2),
        }
        hold_env = FakeHighwayEnv(copy.deepcopy(vehicles))
        seal_env = FakeHighwayEnv(copy.deepcopy(vehicles))
        hold_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        seal_controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        hold_controller._begin_rollout_if_needed(hold_env)
        seal_controller._begin_rollout_if_needed(seal_env)
        hold_controller.current_decision = {
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold window",
        }
        seal_controller.current_decision = {
            "intent": "seal_escape",
            "urgency": "mid",
            "message": "seal now",
        }

        hold_controller._apply_highway_executor(hold_env)
        seal_controller._apply_highway_executor(seal_env)

        self.assertAlmostEqual(hold_controller.target_s, 1.3)
        self.assertAlmostEqual(seal_controller.target_s, 1.2)
        self.assertLess(seal_controller.target_s, hold_controller.target_s)


class TestDriverAgentPrompts(unittest.TestCase):
    def test_highway_action_prompt_drops_structured_context_blocks(self):
        agent = DriverAgent("llm_0")
        captured = {}

        def fake_call(system_message, user_message):
            captured["system"] = system_message
            captured["user"] = user_message
            return "{\"intent\":\"claim_side\",\"urgency\":\"mid\",\"message\":\"hold side\"}"

        agent.call = fake_call
        agent.collaborate(
            "highway",
            "perception",
            {"llm_1": "ready"},
            "Blocker",
            "ego_0",
            "Last iteration: too safe.",
            phase_instruction="Seal the outside lane.",
        )

        self.assertNotIn("blackboard=", captured["user"])
        self.assertNotIn("constraints=", captured["user"])
        self.assertNotIn("memory=", captured["user"])
        self.assertIn("Locked role: Blocker", captured["user"])
        self.assertIn("ego_lane", captured["system"])
        self.assertIn("lead_gap_if_same_lane", captured["system"])


class TestEmissionPersistence(unittest.TestCase):
    def _make_stored_data(self, speed):
        return {
            "ego_0": {
                0.1: {
                    "x": 1.0,
                    "y": 0.0,
                    "speed": speed,
                    "headway": 10.0,
                    "leader_id": "",
                    "target_accel_with_noise_with_failsafe": 0.0,
                    "target_accel_no_noise_no_failsafe": 0.0,
                    "target_accel_with_noise_no_failsafe": 0.0,
                    "target_accel_no_noise_with_failsafe": 0.0,
                    "realized_accel": 0.0,
                    "road_grade": 0.0,
                    "edge_id": "highway_0",
                    "lane_number": 1,
                    "distance": 1.0,
                    "relative_position": 1.0,
                    "follower_id": "",
                    "leader_rel_speed": 0.0,
                }
            }
        }

    def test_complete_and_partial_emissions_are_saved_separately(self):
        if TraCISimulation is None:
            self.skipTest("TraCI runtime dependencies are unavailable in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            simulation = TraCISimulation(FakeEmissionMasterKernel())
            simulation.kernel_api = FakeKernelApi()
            simulation.emission_path = temp_dir

            simulation.stored_data = self._make_stored_data(20.0)
            simulation.save_emission("iter00")

            complete_path = "{}/highway_iter00_emission.csv".format(temp_dir)
            with open(complete_path) as csv_file:
                complete_rows = list(csv.reader(csv_file))

            simulation.stored_data = self._make_stored_data(12.0)
            simulation.save_emission("iter00_partial")

            partial_path = "{}/highway_iter00_partial_emission.csv".format(temp_dir)
            simulation.close()

            self.assertTrue(complete_rows)
            self.assertTrue(complete_path.endswith("iter00_emission.csv"))
            self.assertTrue(partial_path.endswith("iter00_partial_emission.csv"))
            with open(complete_path) as csv_file:
                self.assertEqual(list(csv.reader(csv_file)), complete_rows)
            with open(partial_path) as csv_file:
                partial_rows = list(csv.reader(csv_file))
            self.assertNotEqual(partial_rows[1][4], complete_rows[1][4])


if __name__ == "__main__":
    unittest.main()
