import copy
import csv
import os
import random
import tempfile
import unittest
from unittest.mock import patch

from examples.simulate import init_ego_motion_metrics
from examples.simulate import resolve_roles_for_rollout
from examples.simulate import summarize_ego_motion_metrics
from examples.simulate import update_ego_motion_metrics
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
        controllers.sort(
            key=lambda controller: (
                controller.get_coordinated_planning_order_key()
                if hasattr(controller, "get_coordinated_planning_order_key")
                else (1, getattr(controller, "veh_id", ""))
            )
        )
        return controllers

    def _run_controlled_planning(self):
        controllers = self._get_coordinated_llm_controllers()
        if not controllers:
            return

        step = int(self.time_counter)
        self.message_pool.begin_control_cycle(step)
        if self.message_pool.is_control_cycle_planned(step):
            return
        structured_controllers = []
        generic_controllers = []
        for controller in controllers:
            if (
                    hasattr(controller, "uses_coordinated_structured_protocol")
                    and controller.uses_coordinated_structured_protocol()):
                structured_controllers.append(controller)
            else:
                generic_controllers.append(controller)

        negotiated_snapshot = {}
        if structured_controllers and hasattr(self.message_pool, "negotiated_snapshot"):
            negotiated_snapshot = self.message_pool.negotiated_snapshot(step, viewer_id="coordinator")

        for controller in structured_controllers:
            controller.prepare_intent_for_coordinated_step(self, snapshot=negotiated_snapshot)
            negotiated_snapshot = self.message_pool.negotiated_snapshot(step, viewer_id=controller.veh_id)

        if structured_controllers:
            negotiated_snapshot = self.message_pool.negotiated_snapshot(step, viewer_id="coordinator")

        for controller in structured_controllers:
            controller.prepare_for_coordinated_step(self, snapshot=negotiated_snapshot)
            negotiated_snapshot = self.message_pool.negotiated_snapshot(step, viewer_id=controller.veh_id)

        for controller in generic_controllers:
            controller.run_coordinated_step(self)

        self.message_pool.mark_control_cycle_planned(step)


class PlanningControllerStub(object):
    def __init__(self, veh_id, call_log):
        self.veh_id = veh_id
        self._call_log = call_log

    def uses_coordinated_planning(self):
        return True

    def run_coordinated_step(self, env, snapshot=None):
        snapshot = snapshot or {}
        self._call_log.append((self.veh_id, snapshot.get("viewer_id", "")))
        return True


class StructuredPlanningControllerStub(object):
    def __init__(self, veh_id, priority, call_log):
        self.veh_id = veh_id
        self._priority = int(priority)
        self._call_log = call_log

    def uses_coordinated_planning(self):
        return True

    def uses_coordinated_structured_protocol(self):
        return True

    def get_coordinated_planning_order_key(self):
        return self._priority, self.veh_id

    def prepare_intent_for_coordinated_step(self, env, snapshot=None):
        self._call_log.append(("intent", self.veh_id, snapshot.get("viewer_id", "")))
        return True

    def prepare_for_coordinated_step(self, env, snapshot=None):
        self._call_log.append(("action", self.veh_id, snapshot.get("viewer_id", "")))
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
        api_vehicle.move_to_calls = []

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

        def move_to(veh_id, lane_id, pos):
            target_lane = int(str(lane_id).rsplit("_", 1)[-1])
            self.k.vehicle._vehicles[veh_id]["lane"] = target_lane
            self.k.vehicle._vehicles[veh_id]["pos"] = float(pos)
            api_vehicle.move_to_calls.append((veh_id, str(lane_id), float(pos)))

        api_vehicle.setMaxSpeed = set_max_speed
        api_vehicle.setMinGap = set_min_gap
        api_vehicle.setSpeedMode = set_speed_mode
        api_vehicle.setLaneChangeMode = set_lane_change_mode
        api_vehicle.setParameter = set_parameter
        api_vehicle.changeLane = change_lane
        api_vehicle.changeLaneRelative = change_lane_relative
        api_vehicle.changeSublane = change_sublane
        api_vehicle.moveTo = move_to
        self.k.kernel_api = type("KernelApi", (), {"vehicle": api_vehicle})()


class SlowLaneChangeHighwayEnv(FakeHighwayEnv):
    def __init__(self, vehicles, step=4):
        super(SlowLaneChangeHighwayEnv, self).__init__(vehicles, step=step)

        def change_lane(veh_id, target_lane, duration):
            self.k.vehicle._vehicles[veh_id]["pending_lane_target"] = int(target_lane)
            self.k.vehicle._vehicles[veh_id]["pending_lane_duration"] = int(duration)
            self.k.kernel_api.vehicle.change_calls.append((veh_id, int(target_lane), int(duration)))

        self.k.kernel_api.vehicle.changeLane = change_lane


class FakeKernelApi(object):
    def close(self):
        return None


class FakeEmissionMasterKernel(object):
    def __init__(self, name="highway"):
        net = type("Net", (), {"orig_name": name, "name": name})()
        self.network = type("NetworkWrapper", (), {"network": net})()


class TestEgoMotionMetrics(unittest.TestCase):
    def test_tracks_jerk_and_linguasim_style_comfort(self):
        metrics = init_ego_motion_metrics()
        for speed in (10.0, 11.0, 11.0, 9.0):
            update_ego_motion_metrics(metrics, speed, 1.0)

        summary = summarize_ego_motion_metrics(metrics)

        self.assertEqual(summary["ego_accel_samples"], 3)
        self.assertEqual(summary["ego_jerk_samples"], 2)
        self.assertEqual(summary["ego_comfort_samples"], 2)
        self.assertAlmostEqual(summary["ego_max_decel"], 2.0)
        self.assertAlmostEqual(summary["ego_mean_abs_accel"], 1.0)
        self.assertAlmostEqual(summary["ego_max_abs_jerk"], 2.0)
        self.assertAlmostEqual(summary["ego_mean_abs_jerk"], 1.5)
        self.assertAlmostEqual(summary["ego_rms_jerk"], 1.5811388300841898)
        self.assertAlmostEqual(summary["ego_comfort"], (0.5 + (1.0 / 3.0)) / 2.0)

    def test_constant_speed_has_perfect_comfort_and_zero_jerk(self):
        metrics = init_ego_motion_metrics()
        for speed in (12.0, 12.0, 12.0):
            update_ego_motion_metrics(metrics, speed, 1.0)

        summary = summarize_ego_motion_metrics(metrics)

        self.assertEqual(summary["ego_accel_samples"], 2)
        self.assertEqual(summary["ego_comfort_samples"], 0)
        self.assertAlmostEqual(summary["ego_comfort"], 1.0)
        self.assertAlmostEqual(summary["ego_mean_abs_jerk"], 0.0)


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

    def test_structured_planning_runs_two_passes_with_owner_first_action_order(self):
        call_log = []
        controllers = {
            "llm_0": StructuredPlanningControllerStub("llm_0", 1, call_log),
            "llm_1": StructuredPlanningControllerStub("llm_1", 0, call_log),
        }
        env = FakePlanningEnv(["llm_0", "llm_1"], controllers, step=4)

        env._run_controlled_planning()

        self.assertEqual(
            call_log,
            [
                ("intent", "llm_1", "coordinator"),
                ("intent", "llm_0", "llm_1"),
                ("action", "llm_1", "coordinator"),
                ("action", "llm_0", "llm_1"),
            ],
        )


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

    def test_negotiated_role_resolution_repairs_geometry_conflicting_swap_without_fallback(self):
        controllers = {
            "llm_0": NegotiationControllerStub("llm_0", [{
                "proposed_role": "Striker",
                "pass_side": "right",
                "message": "I should dive in from the front-right slot.",
            }]),
            "llm_1": NegotiationControllerStub("llm_1", [{
                "proposed_role": "Blocker",
                "pass_side": "right",
                "message": "I will cover the left escape lane from the rear.",
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
        self.assertEqual(scenario_context["role_source"], "llm_negotiated_geometry_constrained")
        self.assertEqual(scenario_context["contract_source"], "negotiated")
        self.assertEqual(scenario_context["pass_side"], "left")
        self.assertEqual(scenario_context["block_side"], "right")
        self.assertEqual(env.message_pool.negotiated_contract["pass_side"], "left")
        self.assertEqual(env.message_pool.negotiated_contract["role_map"], role_map)


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
        pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "merge",
            "step": 4,
            "expires_at_step": 16,
            "control_cycle_step": 4,
        })
        pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "mode": "track_pose",
            "lane_policy": "ego_lane",
            "gap_band": "tight",
            "speed_band": "surge",
            "message": "merge",
            "step": 4,
            "expires_at_step": 16,
            "control_cycle_step": 4,
        })

        self.assertEqual(pool.get_all_msg()["llm_0"], "legacy message")
        negotiated_snapshot = pool.negotiated_snapshot(4, viewer_id="llm_0")
        self.assertIn("llm_0", negotiated_snapshot["latest_negotiation_by_agent"])
        self.assertEqual(
            negotiated_snapshot["latest_tactic_by_agent"]["llm_1"]["intent"],
            "merge_commit",
        )
        self.assertNotIn("latest_tactic_by_agent", pool.get_all_msg())


class TestLLMControllerSimpleProtocol(unittest.TestCase):
    def make_controller(self, veh_id):
        controller = LLMController(
            veh_id=veh_id,
            map="highway",
            car_following_params=SumoCarFollowingParams(),
        )
        controller.scene_gate_status = {"scene_mode": "traffic"}
        return controller

    def test_highway_controller_uses_generic_hook_not_structured(self):
        controller = self.make_controller("llm_0")
        self.assertTrue(controller.uses_coordinated_planning())
        self.assertFalse(controller.uses_coordinated_structured_protocol())

    def test_negotiated_highway_uses_structured_protocol(self):
        controller = self.make_controller("llm_0")
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        self.assertTrue(controller.uses_coordinated_structured_protocol())

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

    def attach_planning_hooks(self, env, controllers, ordered_ids=None):
        ordered_ids = list(ordered_ids or sorted(controllers.keys()))
        env.k.vehicle.get_controlled_ids = lambda: list(ordered_ids)
        env.k.vehicle.get_acc_controller = lambda veh_id: controllers[veh_id]
        env._get_coordinated_llm_controllers = FakePlanningEnv._get_coordinated_llm_controllers.__get__(env, FakePlanningEnv)
        env._run_controlled_planning = FakePlanningEnv._run_controlled_planning.__get__(env, FakePlanningEnv)

    def make_negotiated_controller(self, veh_id, role_map):
        controller = self.make_controller(veh_id, role_map)
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller.set_highway_contract({
            "role_map": copy.deepcopy(role_map),
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        return controller

    def prime_negotiated_striker_refresh_state(
            self,
            step=4,
            striker_lane=1,
            striker_rel_x=0.8,
            striker_speed=25.0,
            blocker_lane=3,
            blocker_rel_x=8.0):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", striker_lane, striker_rel_x, speed=striker_speed, ego_lane=2),
            "llm_1": make_payload("llm_1", blocker_lane, blocker_rel_x, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=step)
        controller = self.make_negotiated_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        controller._begin_rollout_if_needed(env)
        env.message_pool.begin_control_cycle(step)
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "seal_escape",
            "urgency": "mid",
            "message": "seal lane",
            "step": step,
            "expires_at_step": step + 24,
            "control_cycle_step": step,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "seal_escape",
            "mode": "track_pose",
            "lane_policy": "block_side",
            "gap_band": "tight",
            "speed_band": "press",
            "message": "seal lane",
            "step": step,
            "expires_at_step": step + 24,
            "control_cycle_step": step,
        })
        intent_plan = {
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "goal": "commit the cut in",
            "message": "merge now",
        }
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            intent_plan,
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="Merge under blocker cover.",
        )
        controller.current_decision["control"]["horizon_steps"] = 20
        controller.current_decision["message"]["expires_at_step"] = step + 20
        controller._publish_negotiated_runtime_decision(
            env,
            controller.current_decision,
            {"satisfied": True, "trigger_code": "none", "details": {}},
        )
        snapshot = env.message_pool.negotiated_snapshot(step, viewer_id="llm_0")
        controller._finalize_highway_action_refresh_state(
            env,
            snapshot,
            controller.current_decision,
            reused=False,
            reason="initial",
        )
        controller.last_control_step = -1
        return env, controller

    def publish_blocker_intent(self, env, step, intent="seal_escape", phase="strike"):
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": phase,
            "intent": intent,
            "urgency": "mid",
            "message": intent,
            "step": step,
            "expires_at_step": step + 24,
            "control_cycle_step": step,
        })
        lane_policy = "block_side" if intent in ("seal_escape", "hold_side_front") else "hold_current"
        mode = "track_pose" if intent != "abort" else "disengage"
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": phase,
            "intent": intent,
            "mode": mode,
            "lane_policy": lane_policy,
            "gap_band": "tight" if intent == "seal_escape" else "medium",
            "speed_band": "press" if intent != "abort" else "yield",
            "message": intent,
            "step": step,
            "expires_at_step": step + 24,
            "control_cycle_step": step,
        })

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

    def test_negotiated_highway_prepare_publishes_structured_decision_only(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 8.0, speed=23.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -2.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
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
        controller._begin_rollout_if_needed(env)
        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"compress\",\"intent\":\"hold_side_front\",\"urgency\":\"mid\","
            "\"goal\":\"seal the side-front window\",\"message\":\"hold side\"}"
        )
        controller.DA.collaborate_highway_tactic = lambda *args, **kwargs: (
            "{\"message\":\"Hold the block-side slot.\","
            "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"block_side\","
            "\"gap_band\":\"medium\",\"speed_band\":\"press\"}}"
        )

        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(controller.current_decision["message"]["intent"], "hold_side_front")
        self.assertEqual(controller.current_decision["tactic"]["lane_policy"], "block_side")
        self.assertEqual(controller.current_decision["message_text"], "Hold the block-side slot.")
        self.assertEqual(controller.latest_intent_plan["intent"], "hold_side_front")
        self.assertIn("llm_0", env.message_pool.get_all_msg())
        negotiated_snapshot = env.message_pool.negotiated_snapshot(4, viewer_id="llm_0")
        self.assertEqual(
            negotiated_snapshot["latest_phase_by_agent"]["llm_0"]["intent"],
            "hold_side_front",
        )
        self.assertEqual(
            negotiated_snapshot["latest_tactic_by_agent"]["llm_0"]["lane_policy"],
            "block_side",
        )
        self.assertNotIn("owner", controller.current_decision["message"])
        self.assertNotIn("plan_id", controller.current_decision["message"])
        protocols = [entry["protocol"] for entry in controller.llm_trace_entries]
        self.assertIn("highway_intent", protocols)
        self.assertIn("highway_tactic", protocols)

    def test_negotiated_cycle_planning_shares_phase_and_tactic_before_striker_step(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -3.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        blocker = self.make_controller(
            "llm_1",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        striker = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        for controller in (blocker, striker):
            controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
            controller.set_highway_contract({
                "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
                "pass_side": "left",
                "block_side": "right",
                "contract_source": "negotiated",
            })
            controller._begin_rollout_if_needed(env)
        env.message_pool.set_negotiated_contract(blocker.highway_contract)
        self.attach_planning_hooks(env, {"llm_0": striker, "llm_1": blocker}, ordered_ids=["llm_0", "llm_1"])

        blocker.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"compress\",\"intent\":\"hold_side_front\",\"urgency\":\"mid\","
            "\"goal\":\"hold the block side\",\"message\":\"block now\"}"
        )
        striker.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"strike\",\"intent\":\"merge_commit\",\"urgency\":\"high\","
            "\"goal\":\"cut in from the pass side\",\"message\":\"merge now\"}"
        )
        blocker.DA.collaborate_highway_tactic = lambda *args, **kwargs: (
            "{\"message\":\"Block the right side-front slot.\","
            "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"block_side\","
            "\"gap_band\":\"medium\",\"speed_band\":\"press\"}}"
        )
        captured = {}

        def striker_tactic(*args):
            captured["teammate_state"] = copy.deepcopy(args[5])
            captured["local_ready"] = copy.deepcopy(args[6])
            return (
                "{\"message\":\"Use blocker cover, merge, then prepare to brake.\","
                "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"ego_lane\","
                "\"gap_band\":\"tight\",\"speed_band\":\"surge\"}}"
            )

        striker.DA.collaborate_highway_tactic = striker_tactic

        env._run_controlled_planning()

        teammate_state = captured["teammate_state"]
        local_ready = captured["local_ready"]
        self.assertEqual(teammate_state["teammate_phase"]["intent"], "hold_side_front")
        self.assertEqual(teammate_state["teammate_tactic"]["lane_policy"], "block_side")
        self.assertNotIn("teammate_id", teammate_state)
        self.assertNotIn("recent_event", teammate_state)
        self.assertEqual(
            set(local_ready.keys()),
            {
                "approach_window_ready",
                "cut_in_gap_ready",
                "clean_cut_in_gap_ready",
                "body_gap_after_merge",
                "projected_body_gap_after_merge",
                "required_body_gap_after_merge",
                "effective_ego_gap_after_merge",
                "merged_into_ego_lane",
                "front_brake_window_ready",
                "overshoot",
            },
        )

    def test_negotiated_get_accel_keeps_blackboard_only_without_negotiated_commit(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 10.0, speed=24.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -8.0, speed=23.2, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
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
        controller._begin_rollout_if_needed(env)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "medium",
                "speed_band": "press",
            },
            message_text="Hold the block side.",
        )
        controller._publish_negotiated_runtime_decision(
            env,
            controller.current_decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )
        controller.last_control_step = env.time_step

        controller.get_accel(env)

        negotiated_snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        self.assertEqual(
            negotiated_snapshot["latest_tactic_by_agent"]["llm_0"]["lane_policy"],
            "block_side",
        )
        self.assertEqual(controller.intent, "hold_side_front")
        self.assertEqual(controller.executor_state, "hold")
        self.assertGreater(controller.target_v, controller.bounds["v_min"])

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

    def test_negotiated_perception_includes_role_pair_ttc(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=20.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -5.0, speed=25.0, ego_lane=2, ego_speed=20.0),
            "llm_1": make_payload("llm_1", 3, 10.0, speed=18.0, ego_lane=2, ego_speed=20.0),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_negotiated_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )

        perception = controller.get_perception(env)

        self.assertNotIn("self_ego_ttc", perception)
        self.assertNotIn("teammate_ego_ttc", perception)
        self.assertIn(
            "role_ttc_sec striker_ego_ttc_projection=1.00 striker_ego_ttc_same_lane=inf blocker_ego_ttc_projection=5.00 blocker_ego_ttc_same_lane=inf",
            perception,
        )

    def test_negotiated_highway_perception_defaults_to_two_neighbors_and_honors_override(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
            "human_0": make_payload("human_0", 2, 6.0, speed=22.0, ego_lane=2),
            "human_1": make_payload("human_1", 1, -12.0, speed=21.5, ego_lane=2),
            "human_2": make_payload("human_2", 0, 15.0, speed=25.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles)
        controller = self.make_negotiated_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )

        with patch.dict(os.environ, {}, clear=False):
            perception = controller.get_perception(env)

        near_lines = [line for line in perception.splitlines() if line.startswith("near=")]
        self.assertEqual(len(near_lines), 2)

        with patch.dict(os.environ, {"FLOW_LLM_NEIGHBOR_K": "3"}, clear=False):
            override_perception = controller.get_perception(env)

        override_near_lines = [line for line in override_perception.splitlines() if line.startswith("near=")]
        self.assertEqual(len(override_near_lines), 3)

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

    def test_negotiated_front_brake_triggers_from_local_window_and_blocker_support(self):
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
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "seal_escape",
            "urgency": "high",
            "message": "seal",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "seal_escape",
            "mode": "track_pose",
            "lane_policy": "block_side",
            "gap_band": "tight",
            "speed_band": "press",
            "message": "seal",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller._prev_self_lane = 1
        controller._merged_into_ego_lane = True
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "brake_pulse",
                "intent": "front_brake",
                "urgency": "high",
                "goal": "front brake",
                "message": "brake now",
            },
            tactic_profile={
                "mode": "pulse_brake",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "brake",
            },
        )
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")
        trigger_eval = controller._evaluate_negotiated_tactic_trigger(env, snapshot, decision)
        controller._set_execution_targets(env, decision, trigger_eval)

        self.assertTrue(trigger_eval["satisfied"])
        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")
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
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "hold_side_front",
            "mode": "track_pose",
            "lane_policy": "block_side",
            "gap_band": "medium",
            "speed_band": "press",
            "message": "hold",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        controller._prev_self_lane = 1
        controller._merged_into_ego_lane = True
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "brake_pulse",
                "intent": "front_brake",
                "urgency": "high",
                "goal": "front brake",
                "message": "finish attack",
            },
            tactic_profile={
                "mode": "pulse_brake",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "brake",
            },
        )
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")
        trigger_eval = controller._evaluate_negotiated_tactic_trigger(env, snapshot, decision)
        controller._set_execution_targets(env, decision, trigger_eval)

        self.assertTrue(trigger_eval["satisfied"])
        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")

    def test_negotiated_merge_commit_executes_without_owner_plan(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.6, ego_lane=2),
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
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold",
            "step": 8,
            "expires_at_step": 20,
            "control_cycle_step": 8,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Blocker",
            "phase": "strike",
            "intent": "hold_side_front",
            "mode": "track_pose",
            "lane_policy": "block_side",
            "gap_band": "medium",
            "speed_band": "press",
            "message": "hold",
            "step": 8,
            "expires_at_step": 20,
            "control_cycle_step": 8,
        })
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "go",
                "message": "go",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
        )
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        trigger_eval = controller._evaluate_negotiated_tactic_trigger(env, snapshot, decision)
        controller._set_execution_targets(env, decision, trigger_eval)

        self.assertTrue(trigger_eval["satisfied"])
        self.assertEqual(controller.intent, "merge_commit")
        self.assertEqual(controller.executor_state, "cut_in_commit")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertGreater(controller.target_v, vehicles["ego_0"]["speed"])
        self.assertTrue(controller._allow_aggressive_cut_in)

    def test_negotiated_blocker_auto_seals_from_striker_phase_support(self):
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
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "merge",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "mode": "track_pose",
            "lane_policy": "ego_lane",
            "gap_band": "tight",
            "speed_band": "surge",
            "message": "merge",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "seal_escape",
                "urgency": "high",
                "goal": "seal",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "tight",
                "speed_band": "press",
            },
        )
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")
        trigger_eval = controller._evaluate_negotiated_tactic_trigger(env, snapshot, decision)
        controller._set_execution_targets(env, decision, trigger_eval)

        self.assertTrue(trigger_eval["satisfied"])
        self.assertEqual(controller.intent, "seal_escape")
        self.assertEqual(controller.executor_state, "seal")
        self.assertEqual(controller.target_abs_lane, 3)

    def test_negotiated_blocker_does_not_drop_behind_when_far_ahead(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 14.0, speed=23.4, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.0, speed=24.4, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "seal_escape",
                "urgency": "high",
                "goal": "seal",
                "message": "seal",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "tight",
                "speed_band": "press",
            },
        )

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertGreaterEqual(controller.target_v, vehicles["ego_0"]["speed"] - 0.6)
        self.assertEqual(controller.target_abs_lane, 3)

    def test_negotiated_blocker_hold_stays_hold_without_merge_support(self):
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
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "brake_pulse",
            "intent": "front_brake",
            "urgency": "high",
            "message": "brake",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "brake_pulse",
            "intent": "front_brake",
            "mode": "hold_lane",
            "lane_policy": "hold_current",
            "gap_band": "tight",
            "speed_band": "brake",
            "message": "brake",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "medium",
                "speed_band": "press",
            },
        )
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")
        trigger_eval = controller._evaluate_negotiated_tactic_trigger(env, snapshot, decision)
        controller._set_execution_targets(env, decision, trigger_eval)

        self.assertEqual(controller.intent, "hold_side_front")
        self.assertEqual(controller.executor_state, "hold")

    def test_negotiated_intent_failure_falls_back_to_default_intent_then_runs_action(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.0, speed=23.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -4.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        captured = {}

        def fake_tactic(*args):
            captured["intent_plan"] = copy.deepcopy(args[4])
            return (
                "{\"message\":\"Fallback intent still produced a blocker hold.\","
                "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"block_side\","
                "\"gap_band\":\"medium\",\"speed_band\":\"press\"}}"
            )

        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: "not-json"
        controller.DA.collaborate_highway_tactic = fake_tactic

        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(captured["intent_plan"]["intent"], "hold_side_front")
        self.assertEqual(controller.latest_intent_plan["intent"], "hold_side_front")
        self.assertIn("highway_intent_fallback", [entry["protocol"] for entry in controller.llm_trace_entries])

    def test_negotiated_action_failure_falls_back_to_intent_structured_decision(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 0.8, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.2, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        env.message_pool.begin_control_cycle(4)
        self.publish_blocker_intent(env, 4)
        controller._begin_rollout_if_needed(env)
        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"strike\",\"intent\":\"merge_commit\",\"urgency\":\"high\","
            "\"goal\":\"commit the cut-in\",\"message\":\"go now\"}"
        )
        controller.DA.collaborate_highway_tactic = lambda *args, **kwargs: "bad-action"

        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(controller.current_decision["message"]["intent"], "merge_commit")
        self.assertTrue(controller.current_decision["action_sequence_text"])
        self.assertEqual(controller.current_decision["intent_plan"]["intent"], "merge_commit")
        self.assertIn("highway_tactic_fallback", [entry["protocol"] for entry in controller.llm_trace_entries])

    def test_tactic_fallback_resyncs_structured_diagnostics_to_final_decision(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 8.0, speed=23.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -2.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"compress\",\"intent\":\"hold_side_front\"}"
        )
        controller.DA.collaborate_highway_tactic = lambda *args, **kwargs: "bad-tactic"

        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(controller.latest_intent_plan["intent"], "hold_side_front")
        self.assertEqual(controller.latest_tactic_profile["lane_policy"], "block_side")
        self.assertTrue(controller.latest_action_sequence_text)

    def test_negotiated_role_history_context_is_compact(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.0, speed=23.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -4.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        controller = self.make_negotiated_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.previous_feedback = {"result": "too_safe", "feedback_summary": "ego escaped"}
        controller.case_memory = [{
            "scenario_id": controller.scenario_id,
            "role": "Blocker",
            "result": "clean_success",
            "iteration": 1,
            "feedback_summary": "seal earlier",
            "min_ttc": 1.2,
            "ego_max_decel": 2.9,
            "state_signature_bucketed": {"phase": "compress"},
        }]
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        controller._begin_rollout_if_needed(env)
        captured = {}

        def fake_negotiate(perception, snapshot, target_vehicle, history_context=None):
            captured["history_context"] = history_context
            return "{\"proposed_role\":\"Blocker\",\"pass_side\":\"left\",\"message\":\"seal\"}"

        controller.DA.negotiate_highway_contract = fake_negotiate

        proposal = controller.negotiate_role(env)

        self.assertEqual(proposal["proposed_role"], "Blocker")
        self.assertIn("feedback=", captured["history_context"])
        self.assertIn("memory=", captured["history_context"])

    def test_negotiated_intent_history_is_only_injected_on_new_phase(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.0, speed=23.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -4.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
        controller = self.make_negotiated_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.previous_feedback = {"result": "too_safe", "feedback_summary": "ego escaped"}
        controller.case_memory = [{
            "scenario_id": controller.scenario_id,
            "role": "Blocker",
            "result": "clean_success",
            "iteration": 1,
            "feedback_summary": "seal earlier",
            "min_ttc": 1.2,
            "ego_max_decel": 2.9,
            "state_signature_bucketed": {"phase": "compress"},
        }]
        env.message_pool.set_negotiated_contract(controller.highway_contract)
        controller._begin_rollout_if_needed(env)
        history_calls = []
        teammate_states = []
        local_ready_payloads = []

        def fake_intent(*args):
            teammate_states.append(copy.deepcopy(args[4]))
            local_ready_payloads.append(copy.deepcopy(args[5]))
            history_calls.append(args[6])
            phase = "strike" if controller.active_phase == "strike" else "compress"
            intent = "seal_escape" if phase == "strike" else "hold_side_front"
            return (
                "{{\"phase\":\"{}\",\"intent\":\"{}\",\"urgency\":\"mid\","
                "\"goal\":\"goal\",\"message\":\"msg\"}}".format(phase, intent)
            )

        controller.DA.collaborate_highway_intent = fake_intent

        env.message_pool.begin_control_cycle(4)
        controller.prepare_intent_for_coordinated_step(env)

        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        controller.prepare_intent_for_coordinated_step(env)

        controller.active_phase = "strike"
        self.set_step(env, 12)
        env.message_pool.begin_control_cycle(12)
        controller.prepare_intent_for_coordinated_step(env)

        self.assertIn("feedback=", history_calls[0])
        self.assertIsNone(history_calls[1])
        self.assertIn("feedback=", history_calls[2])
        self.assertEqual(set(teammate_states[0].keys()), {"teammate_phase"})
        self.assertEqual(
            set(local_ready_payloads[0].keys()),
            {
                "approach_window_ready",
                "cut_in_gap_ready",
                "clean_cut_in_gap_ready",
                "body_gap_after_merge",
                "projected_body_gap_after_merge",
                "required_body_gap_after_merge",
                "effective_ego_gap_after_merge",
                "merged_into_ego_lane",
                "front_brake_window_ready",
                "overshoot",
            },
        )

    def test_highway_action_refresh_reason_detects_material_changes(self):
        cases = (
            ("phase_changed", lambda env, controller, intent_plan: ({"phase": "compress", "intent": "gain_lead"}, None)),
            ("intent_changed", lambda env, controller, intent_plan: ({"phase": "strike", "intent": "front_brake"}, None)),
            ("self_rel_lane_changed", lambda env, controller, intent_plan: (intent_plan, env.k.vehicle._vehicles["llm_0"].update({"lane": 2}))),
            ("merged_into_ego_lane_changed", lambda env, controller, intent_plan: (intent_plan, setattr(controller, "_merged_into_ego_lane", True))),
            ("approach_window_ready_changed", lambda env, controller, intent_plan: (intent_plan, env.k.vehicle._vehicles["llm_0"].update({"speed": 22.0}))),
            ("trigger_satisfied_changed", lambda env, controller, intent_plan: (
                intent_plan,
                setattr(controller, "_last_published_trigger_satisfied", False),
            )),
            ("teammate_phase_changed", lambda env, controller, intent_plan: (
                intent_plan,
                env.message_pool.publish_negotiated_phase({
                    "sender": "llm_1",
                    "role": "Blocker",
                    "phase": "compress",
                    "intent": "hold_side_front",
                    "urgency": "mid",
                    "message": "hold",
                    "step": 8,
                    "expires_at_step": 32,
                    "control_cycle_step": 8,
                }),
            )),
            ("teammate_intent_changed", lambda env, controller, intent_plan: (
                intent_plan,
                env.message_pool.publish_negotiated_phase({
                    "sender": "llm_1",
                    "role": "Blocker",
                    "phase": "strike",
                    "intent": "hold_side_front",
                    "urgency": "mid",
                    "message": "hold",
                    "step": 8,
                    "expires_at_step": 32,
                    "control_cycle_step": 8,
                }),
            )),
            ("teammate_mode_changed", lambda env, controller, intent_plan: (
                intent_plan,
                env.message_pool.publish_negotiated_tactic({
                    "sender": "llm_1",
                    "role": "Blocker",
                    "phase": "strike",
                    "intent": "seal_escape",
                    "mode": "hold_lane",
                    "lane_policy": "block_side",
                    "gap_band": "tight",
                    "speed_band": "press",
                    "message": "hold",
                    "step": 8,
                    "expires_at_step": 32,
                    "control_cycle_step": 8,
                }),
            )),
            ("decision_expired", lambda env, controller, intent_plan: (
                intent_plan,
                controller.current_decision["message"].update({"expires_at_step": 7}),
            )),
            ("control_mode_changed", lambda env, controller, intent_plan: (
                intent_plan,
                controller.current_decision["tactic"].update({"mode": "hold_lane"}),
            )),
            ("bad_merge_reason_changed", lambda env, controller, intent_plan: (
                intent_plan,
                setattr(controller, "_bad_merge_reason", "late_merge_ahead"),
            )),
        )

        for expected_reason, mutate in cases:
            with self.subTest(expected_reason=expected_reason):
                env, controller = self.prime_negotiated_striker_refresh_state()
                self.set_step(env, 8)
                env.message_pool.begin_control_cycle(8)
                self.publish_blocker_intent(env, 8)
                intent_plan = copy.deepcopy(controller.current_decision["intent_plan"])
                next_intent_plan, _ = mutate(env, controller, intent_plan)
                snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
                reason = controller._get_highway_action_refresh_reason(env, snapshot, next_intent_plan)
                self.assertEqual(reason, expected_reason)

    def test_highway_action_refresh_reason_uses_rel_x_hysteresis(self):
        env, controller = self.prime_negotiated_striker_refresh_state(
            striker_rel_x=-2.1,
            striker_speed=25.0,
        )
        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        self.publish_blocker_intent(env, 8)

        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] - 1.6
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        no_refresh_reason = controller._get_highway_action_refresh_reason(
            env,
            snapshot,
            controller.current_decision["intent_plan"],
        )

        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] - 1.1
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        refresh_reason = controller._get_highway_action_refresh_reason(
            env,
            snapshot,
            controller.current_decision["intent_plan"],
        )

        self.assertEqual(no_refresh_reason, "")
        self.assertEqual(refresh_reason, "rel_x_bucket_changed")

    def test_highway_action_refresh_reason_covers_front_brake_window_and_overshoot(self):
        env, controller = self.prime_negotiated_striker_refresh_state(
            striker_lane=2,
            striker_rel_x=0.0,
            striker_speed=25.0,
        )
        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        self.publish_blocker_intent(env, 8)
        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] + 0.4
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        front_brake_reason = controller._get_highway_action_refresh_reason(
            env,
            snapshot,
            controller.current_decision["intent_plan"],
        )

        env, controller = self.prime_negotiated_striker_refresh_state(
            striker_lane=2,
            striker_rel_x=7.8,
            striker_speed=25.0,
        )
        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        self.publish_blocker_intent(env, 8)
        env.k.vehicle._vehicles["llm_0"]["pos"] = env.k.vehicle._vehicles["ego_0"]["pos"] + 8.8
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")
        overshoot_reason = controller._get_highway_action_refresh_reason(
            env,
            snapshot,
            controller.current_decision["intent_plan"],
        )

        self.assertEqual(front_brake_reason, "front_brake_window_ready_changed")
        self.assertEqual(overshoot_reason, "overshoot_changed")

    def test_negotiated_action_reuses_previous_decision_when_signature_is_unchanged(self):
        env, controller = self.prime_negotiated_striker_refresh_state()
        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        self.publish_blocker_intent(env, 8)
        action_calls = {"count": 0}
        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"strike\",\"intent\":\"merge_commit\",\"urgency\":\"high\","
            "\"goal\":\"commit the cut in\",\"message\":\"merge now\"}"
        )

        def fake_tactic(*args, **kwargs):
            action_calls["count"] += 1
            return "should-not-run"

        controller.DA.collaborate_highway_tactic = fake_tactic
        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(action_calls["count"], 0)
        self.assertEqual(controller.current_decision["message"]["step"], 8)
        self.assertGreater(controller.current_decision["message"]["expires_at_step"], 8)
        self.assertTrue(controller.get_rollout_diagnostics()["last_action_reused"])
        self.assertIn("merge_commit", controller.last_raw_response)
        self.assertNotIn("signature_unchanged", controller.last_raw_response)

    def test_negotiated_action_hard_refreshes_after_two_reuse_cycles(self):
        env, controller = self.prime_negotiated_striker_refresh_state()
        controller.DA.collaborate_highway_intent = lambda *args, **kwargs: (
            "{\"phase\":\"strike\",\"intent\":\"merge_commit\",\"urgency\":\"high\","
            "\"goal\":\"commit the cut in\",\"message\":\"merge now\"}"
        )
        action_calls = {"count": 0}

        def fake_tactic(*args, **kwargs):
            action_calls["count"] += 1
            return (
                "{\"message\":\"Refresh the merge target.\","
                "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"ego_lane\","
                "\"gap_band\":\"tight\",\"speed_band\":\"surge\"}}"
            )

        controller.DA.collaborate_highway_tactic = fake_tactic

        for step in (8, 12):
            self.set_step(env, step)
            env.message_pool.begin_control_cycle(step)
            self.publish_blocker_intent(env, step)
            planned = controller.prepare_for_coordinated_step(env)
            self.assertTrue(planned)

        self.assertEqual(action_calls["count"], 0)

        self.set_step(env, 16)
        env.message_pool.begin_control_cycle(16)
        self.publish_blocker_intent(env, 16)
        planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(planned)
        self.assertEqual(action_calls["count"], 1)
        self.assertEqual(controller.get_rollout_diagnostics()["last_action_refresh_reason"], "hard_refresh")

    def test_terminal_latch_reuses_local_abort_without_llm_calls(self):
        env, controller = self.prime_negotiated_striker_refresh_state()
        self.set_step(env, 8)
        env.message_pool.begin_control_cycle(8)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "disengage",
                "intent": "abort",
                "urgency": "low",
                "goal": "abort",
                "message": "abort",
            },
            tactic_profile={
                "mode": "disengage",
                "lane_policy": "hold_current",
                "gap_band": "loose",
                "speed_band": "yield",
            },
            message_text="abort",
        )
        controller.active_phase = "disengage"
        controller.active_control_mode = "disengage"
        llm_calls = {"intent": 0, "tactic": 0}

        def fail_intent(*args, **kwargs):
            llm_calls["intent"] += 1
            raise AssertionError("intent LLM should not be called")

        def fail_tactic(*args, **kwargs):
            llm_calls["tactic"] += 1
            raise AssertionError("tactic LLM should not be called")

        controller.DA.collaborate_highway_intent = fail_intent
        controller.DA.collaborate_highway_tactic = fail_tactic

        intent_planned = controller.prepare_intent_for_coordinated_step(env)
        action_planned = controller.prepare_for_coordinated_step(env)

        self.assertTrue(intent_planned)
        self.assertTrue(action_planned)
        self.assertEqual(llm_calls, {"intent": 0, "tactic": 0})
        self.assertTrue(controller._terminal_plan_locked)
        self.assertEqual(controller.current_decision["message"]["phase"], "disengage")
        self.assertEqual(controller.get_rollout_diagnostics()["last_action_refresh_reason"], "terminal_latch")

    def test_striker_phase_intent_contract_repairs_nonmatching_intents(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -4.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(4, viewer_id="llm_0")

        strike_plan = controller._normalize_highway_intent_plan({
            "phase": "strike",
            "intent": "gain_lead",
        }, env, snapshot)
        brake_plan = controller._normalize_highway_intent_plan({
            "phase": "brake_pulse",
            "intent": "merge_commit",
        }, env, snapshot)
        compress_plan = controller._normalize_highway_intent_plan({
            "phase": "compress",
            "intent": "front_brake",
        }, env, snapshot)

        self.assertEqual(strike_plan["intent"], "gain_lead")
        self.assertEqual(brake_plan["phase"], "compress")
        self.assertEqual(brake_plan["intent"], "gain_lead")
        self.assertEqual(compress_plan["intent"], "gain_lead")

    def test_blocker_phase_floor_follows_striker_strike_support(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.2, speed=23.1, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.3, speed=24.4, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "merge",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")

        repaired = controller._normalize_highway_intent_plan({
            "phase": "compress",
            "intent": "hold_side_front",
        }, env, snapshot)

        self.assertEqual(repaired["phase"], "strike")
        self.assertEqual(repaired["intent"], "seal_escape")

    def test_striker_geometry_guard_aborts_stale_ahead_merge_commit(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 17.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(4, viewer_id="llm_0")

        repaired = controller._normalize_highway_intent_plan({
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "cut in now",
        }, env, snapshot)

        self.assertEqual(repaired["phase"], "compress")
        self.assertEqual(repaired["intent"], "gain_lead")
        self.assertTrue(controller._stale_merge_candidate)
        self.assertTrue(controller._clean_merge_failed)

    def test_striker_execution_recovers_when_merge_window_is_missed(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 8.2, speed=27.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        vehicles["llm_0"]["length"] = 7.5
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
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge now",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.executor_state, "merge_wait_gap")
        self.assertEqual(controller.intent, "merge_commit")
        self.assertEqual(controller.target_abs_lane, 1)
        self.assertEqual(controller.target_lc, 0)
        self.assertGreater(controller.target_v, vehicles["ego_0"]["speed"])
        self.assertFalse(controller._terminal_plan_locked)

    def test_striker_execution_marks_stale_candidate_without_terminal_lock(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 17.0, speed=28.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge now",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertFalse(controller._terminal_plan_locked)
        self.assertEqual(controller._terminal_lock_reason, "")
        self.assertEqual(controller._bad_merge_reason, "stale_merge_ahead_far")
        self.assertTrue(controller._stale_merge_candidate)
        self.assertTrue(controller._clean_merge_failed)
        self.assertEqual(controller.executor_state, "disengage")
        self.assertEqual(controller.target_lc, 0)

    def test_blocker_geometry_guard_blocks_seal_escape_from_rear_position(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, -4.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")

        repaired = controller._normalize_highway_intent_plan({
            "phase": "strike",
            "intent": "seal_escape",
            "urgency": "high",
            "message": "seal now",
        }, env, snapshot)

        self.assertEqual(repaired["phase"], "compress")
        self.assertEqual(repaired["intent"], "hold_side_front")

    def test_striker_geometry_guard_keeps_merge_commit_during_active_cut_in_episode(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 3.1, speed=23.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40
        snapshot = env.message_pool.negotiated_snapshot(24, viewer_id="llm_0")

        repaired = controller._normalize_highway_intent_plan({
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "keep merging",
        }, env, snapshot)

        self.assertEqual(repaired["phase"], "strike")
        self.assertEqual(repaired["intent"], "merge_commit")

    def test_blocker_geometry_guard_keeps_seal_escape_from_far_side_front_support(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 14.0, speed=23.4, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -2.0, speed=24.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        env.message_pool.publish_negotiated_phase({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "message": "merge",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        env.message_pool.publish_negotiated_tactic({
            "sender": "llm_1",
            "role": "Striker",
            "phase": "strike",
            "intent": "merge_commit",
            "mode": "track_pose",
            "lane_policy": "ego_lane",
            "gap_band": "tight",
            "speed_band": "surge",
            "message": "merge",
            "step": 12,
            "expires_at_step": 24,
            "control_cycle_step": 12,
        })
        snapshot = env.message_pool.negotiated_snapshot(12, viewer_id="llm_0")

        repaired = controller._normalize_highway_intent_plan({
            "phase": "compress",
            "intent": "hold_side_front",
            "urgency": "mid",
            "message": "hold",
        }, env, snapshot)

        self.assertEqual(repaired["phase"], "strike")
        self.assertEqual(repaired["intent"], "seal_escape")

    def test_negotiated_tactic_normalizer_forces_merge_into_ego_lane(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -0.8, speed=24.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.8, speed=23.4, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")

        decision = controller._normalize_highway_tactic_output({
            "message": "merge into ego lane",
            "tactic": {
                "mode": "track_pose",
                "lane_policy": "pass_side",
                "gap_band": "tight",
                "speed_band": "press",
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "goal": "merge",
            "message": "merge",
        })

        self.assertEqual(decision["tactic"]["lane_policy"], "ego_lane")
        self.assertEqual(decision["control"]["mode"], "track_pose")

    def test_negotiated_tactic_normalizer_forces_aggressive_speed_bands_for_attack_actions(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -0.8, speed=24.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.8, speed=23.4, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        striker = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        striker.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        striker.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        striker._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_0")

        striker_decision = striker._normalize_highway_tactic_output({
            "message": "merge now",
            "tactic": {
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "match",
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "goal": "merge",
            "message": "merge",
        })

        blocker = self.make_controller(
            "llm_1",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        blocker.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        blocker.set_highway_contract({
            "role_map": {"llm_0": "Striker", "llm_1": "Blocker"},
            "pass_side": "left",
            "block_side": "right",
            "contract_source": "negotiated",
        })
        blocker._begin_rollout_if_needed(env)
        blocker_snapshot = env.message_pool.negotiated_snapshot(8, viewer_id="llm_1")
        blocker_decision = blocker._normalize_highway_tactic_output({
            "message": "seal now",
            "tactic": {
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "tight",
                "speed_band": "match",
            },
        }, env, {
            "phase": "strike",
            "intent": "seal_escape",
            "urgency": "high",
            "goal": "seal",
            "message": "seal",
        })

        self.assertEqual(striker_decision["tactic"]["speed_band"], "surge")
        self.assertEqual(blocker_decision["tactic"]["speed_band"], "press")

    def test_negotiated_tactic_normalizer_keeps_valid_sequence_and_relative_hints(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)

        decision = controller._normalize_highway_tactic_output({
            "message": "commit with small speed advantage",
            "tactic": {
                "style": "aggressive",
                "sequence": ["cap_speed_advantage", "commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 1.2,
                "lead_gap_hint_m": 2.8,
                "hold_cycles": 3,
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "message": "merge",
        })

        self.assertEqual(decision["tactic"]["style"], "aggressive")
        self.assertEqual(
            decision["tactic"]["sequence"],
            ["cap_speed_advantage", "commit_lane_change", "front_brake"],
        )
        self.assertEqual(decision["tactic"]["speed_delta_hint_mps"], 1.2)
        self.assertEqual(decision["tactic"]["lead_gap_hint_m"], 2.8)
        self.assertEqual(decision["tactic"]["hold_cycles"], 3)
        self.assertEqual(decision["tactic"]["tactic_hints"]["speed_delta_hint_mps"], 1.2)

    def test_negotiated_tactic_normalizer_clamps_invalid_hints_and_ignores_absolute_targets(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)

        repairs_before = controller.normalization_repairs
        decision = controller._normalize_highway_tactic_output({
            "message": "bad absolute target",
            "tactic": {
                "style": "fast",
                "sequence": ["commit_lane_change", "teleport"],
                "speed_delta_hint_mps": 8.5,
                "lead_gap_hint_m": 8.0,
                "hold_cycles": 9,
                "target_speed": 30.0,
                "lane_policy": "pass_side",
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "message": "merge",
        })

        self.assertEqual(decision["tactic"]["style"], "normal")
        self.assertEqual(decision["tactic"]["sequence"], ["commit_lane_change", "front_brake"])
        self.assertEqual(decision["tactic"]["speed_delta_hint_mps"], 1.5)
        self.assertEqual(decision["tactic"]["lead_gap_hint_m"], 4.0)
        self.assertEqual(decision["tactic"]["hold_cycles"], 3)
        self.assertEqual(decision["tactic"]["lane_policy"], "ego_lane")
        self.assertGreater(controller.normalization_repairs, repairs_before)

    def test_merge_commit_sequence_auto_appends_front_brake(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 1.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        repairs_before = controller.normalization_repairs

        decision = controller._normalize_highway_tactic_output({
            "message": "commit then stabilize",
            "tactic": {
                "style": "normal",
                "sequence": ["gain_lead", "commit_lane_change", "stabilize_same_lane"],
                "speed_delta_hint_mps": 1.2,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "message": "merge",
        })

        self.assertEqual(
            decision["tactic"]["sequence"],
            ["commit_lane_change", "stabilize_same_lane", "front_brake"],
        )
        self.assertGreater(controller.normalization_repairs, repairs_before)

    def test_gain_lead_sequence_filters_lane_change_tokens(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -1.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        repairs_before = controller.normalization_repairs

        decision = controller._normalize_highway_tactic_output({
            "message": "gain lead before merge",
            "tactic": {
                "style": "aggressive",
                "sequence": ["gain_lead", "cap_speed_advantage", "commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 4.0,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {
            "phase": "compress",
            "intent": "gain_lead",
            "message": "gain",
        })

        self.assertEqual(decision["tactic"]["sequence"], ["gain_lead", "cap_speed_advantage"])
        self.assertGreater(controller.normalization_repairs, repairs_before)

    def test_striker_tactic_mapping_keeps_merge_close_and_pressure_gap(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        snapshot = env.message_pool.negotiated_snapshot(4, viewer_id="llm_0")

        strike_decision = controller._normalize_highway_tactic_output({
            "message": "merge now",
            "tactic": {
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "urgency": "high",
            "goal": "cut in front of ego_0",
            "message": "cut in front of ego_0",
        })
        brake_decision = controller._normalize_highway_tactic_output({
            "message": "brake now",
            "tactic": {
                "mode": "pulse_brake",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "brake",
            },
        }, env, {
            "phase": "brake_pulse",
            "intent": "front_brake",
            "urgency": "high",
            "goal": "front brake after the merge",
            "message": "front brake after the merge",
        })

        controller._set_execution_targets(
            env,
            strike_decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )
        strike_target_v = controller.target_v
        strike_target_s = controller.target_s
        strike_target_lane = controller.target_abs_lane

        controller._set_execution_targets(
            env,
            brake_decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertEqual(strike_decision["tactic"]["lane_policy"], "ego_lane")
        self.assertEqual(strike_target_lane, 2)
        self.assertLessEqual(strike_target_s, 1.05)
        self.assertGreater(strike_target_v, vehicles["ego_0"]["speed"])
        self.assertLessEqual(strike_target_v, vehicles["ego_0"]["speed"] + 1.5)
        self.assertEqual(brake_decision["tactic"]["mode"], "pulse_brake")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertLessEqual(controller.target_s, 0.85)
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_merge_commit_execution_uses_clamped_relative_hints(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        decision = controller._normalize_highway_tactic_output({
            "message": "commit with clamped relative hints",
            "tactic": {
                "style": "aggressive",
                "sequence": ["cap_speed_advantage", "commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 9.0,
                "lead_gap_hint_m": 4.0,
                "hold_cycles": 9,
            },
        }, env, {
            "phase": "strike",
            "intent": "merge_commit",
            "message": "merge",
        })

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertEqual(controller.executor_state, "cut_in_commit")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertAlmostEqual(controller.target_v, vehicles["ego_0"]["speed"] + 0.6)
        self.assertAlmostEqual(controller.target_s, 0.6)
        self.assertEqual(
            controller._merge_window_hold_until_step,
            env.time_step + controller._control_cycles_to_steps(3),
        )

    def test_merge_commit_waits_on_pass_side_when_too_close_for_cut_in(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 0.6, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        decision = controller._normalize_highway_tactic_output({
            "message": "wait for safe cut-in gap",
            "tactic": {
                "style": "aggressive",
                "sequence": ["commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 1.5,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {"phase": "strike", "intent": "merge_commit", "message": "merge"})

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertEqual(controller.executor_state, "merge_wait_gap")
        self.assertEqual(controller.target_abs_lane, 1)
        self.assertEqual(controller.target_lc, 0)
        self.assertGreaterEqual(controller.target_v, vehicles["ego_0"]["speed"] + 0.3)
        self.assertLessEqual(controller.target_v, vehicles["ego_0"]["speed"] + 0.8)
        self.assertEqual(controller._last_force_cut_in_blocked_reason, "too_close")

    def test_merge_commit_uses_body_gap_for_change_lane_but_effective_gap_for_clean_label(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 11.5, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        decision = controller._normalize_highway_tactic_output({
            "message": "wait for safe cut-in gap",
            "tactic": {
                "style": "aggressive",
                "sequence": ["commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 1.5,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {"phase": "strike", "intent": "merge_commit", "message": "merge"})

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertAlmostEqual(controller._ego_lead_gap_after_merge(env), 6.5, places=2)
        self.assertAlmostEqual(controller._effective_ego_gap_after_merge(env), 3.5, places=2)
        self.assertLessEqual(
            controller._dynamic_body_cut_in_gap(env),
            controller._ego_lead_gap_after_merge(env),
        )
        self.assertTrue(controller._body_safe_cut_in_ready(env))
        self.assertTrue(controller._clean_cut_in_gap_ready(env))
        self.assertEqual(controller._classify_merge_event(11.5, env=env, ctx=controller._get_relative_context(env)), "valid")
        self.assertEqual(controller.executor_state, "cut_in_commit")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 1)

    def test_merge_wait_gap_stays_sticky_and_accelerates_when_next_llm_says_gain_lead(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 4.6, speed=22.7, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=20)
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
        controller._begin_rollout_if_needed(env)
        controller.executor_state = "merge_wait_gap"
        decision = controller._normalize_highway_tactic_output({
            "message": "gain lead but continue preparing",
            "tactic": {
                "style": "aggressive",
                "sequence": ["gain_lead", "cap_speed_advantage", "commit_lane_change"],
                "speed_delta_hint_mps": 4.0,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {"phase": "compress", "intent": "gain_lead", "message": "gain"})

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertEqual(controller.intent, "merge_commit")
        self.assertEqual(controller.executor_state, "merge_wait_gap")
        self.assertEqual(controller.target_abs_lane, 1)
        self.assertEqual(controller.target_lc, 0)
        self.assertGreaterEqual(controller.target_v, vehicles["ego_0"]["speed"] + 2.4)

    def test_merge_wait_gap_times_out_to_recover(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 11.9, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        vehicles["llm_0"]["length"] = 12.0
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)
        controller._merge_wait_gap_cycles = 999
        decision = controller._normalize_highway_tactic_output({
            "message": "wait for safe cut-in gap",
            "tactic": {
                "style": "aggressive",
                "sequence": ["commit_lane_change", "front_brake"],
                "speed_delta_hint_mps": 1.5,
                "lead_gap_hint_m": 2.5,
                "hold_cycles": 2,
            },
        }, env, {"phase": "strike", "intent": "merge_commit", "message": "merge"})

        controller._set_execution_targets(
            env,
            decision,
            {"satisfied": True, "trigger_code": "negotiated_local", "details": {}},
        )

        self.assertEqual(controller.intent, "gain_lead")
        self.assertEqual(controller.executor_state, "merge_recover")
        self.assertEqual(controller.target_abs_lane, 1)
        self.assertEqual(controller.target_lc, 0)
        self.assertEqual(controller._last_force_cut_in_blocked_reason, "wait_timeout")

    def test_late_body_safe_gap_still_allows_clean_merge_commit(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 11.5, speed=24.6, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        local_ready = controller._build_negotiated_local_ready(env, controller._get_relative_context(env))
        allowed = controller._allowed_highway_intent_space(env, local_ready=local_ready)
        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge now",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertTrue(local_ready["cut_in_gap_ready"])
        self.assertTrue(local_ready["clean_cut_in_gap_ready"])
        self.assertEqual(allowed["intents"], ["merge_commit"])
        self.assertEqual(controller.executor_state, "cut_in_commit")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 1)

    def test_late_body_safe_merge_event_is_clean_valid_before_stale_threshold(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 11.5, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._prev_self_lane = 1

        controller._refresh_negotiated_runtime_state(env, controller._get_relative_context(env))

        self.assertTrue(controller.striker_completed_cut_in)
        self.assertFalse(controller._clean_merge_failed)
        self.assertFalse(controller._bad_merge_event)
        self.assertEqual(controller._bad_merge_reason, "")

    def test_gain_lead_uses_lead_gap_hint_as_desired_rel_x(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=4)
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
        controller._begin_rollout_if_needed(env)

        small_gap_decision = controller._normalize_highway_tactic_output({
            "message": "gain a small lead",
            "tactic": {
                "style": "normal",
                "sequence": ["gain_lead", "cap_speed_advantage"],
                "speed_delta_hint_mps": 2.0,
                "lead_gap_hint_m": 0.8,
                "hold_cycles": 1,
            },
        }, env, {"phase": "compress", "intent": "gain_lead", "message": "gain"})
        large_gap_decision = controller._normalize_highway_tactic_output({
            "message": "gain a larger lead",
            "tactic": {
                "style": "normal",
                "sequence": ["gain_lead", "cap_speed_advantage"],
                "speed_delta_hint_mps": 2.0,
                "lead_gap_hint_m": 4.0,
                "hold_cycles": 1,
            },
        }, env, {"phase": "compress", "intent": "gain_lead", "message": "gain"})

        controller._set_execution_targets(env, small_gap_decision, {"satisfied": True, "trigger_code": "none", "details": {}})
        small_gap_target_v = controller.target_v
        controller._set_execution_targets(env, large_gap_decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertGreater(controller.target_v, small_gap_target_v)
        self.assertEqual(controller.target_abs_lane, 1)

    def test_front_brake_uses_lead_gap_hint_as_trigger_window(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 4.6, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._merged_into_ego_lane = True
        controller.striker_completed_cut_in = True
        controller._merge_commit_until_step = 20

        late_brake_decision = controller._normalize_highway_tactic_output({
            "message": "brake from a larger lead gap",
            "tactic": {
                "style": "aggressive",
                "sequence": ["stabilize_same_lane", "front_brake"],
                "speed_delta_hint_mps": -7.0,
                "lead_gap_hint_m": 4.0,
                "hold_cycles": 1,
            },
        }, env, {"phase": "brake_pulse", "intent": "front_brake", "message": "brake"})

        controller._set_execution_targets(env, late_brake_decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_event_gated_phase_progression_blocks_early_brake_and_advances_after_merge_window(self):
        early_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -6.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }, step=24)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller._begin_rollout_if_needed(early_env)
        early_ctx = controller._get_relative_context(early_env)
        early_ready = controller._build_negotiated_local_ready(early_env, early_ctx)

        early_phase = controller._apply_phase_progression(
            "brake_pulse",
            24,
            ctx=early_ctx,
            local_ready=early_ready,
        )
        self.assertEqual(early_phase, "compress")

        merged_env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }, step=24)
        controller.active_phase = "compress"
        merged_ctx = controller._get_relative_context(merged_env)
        merged_ready = controller._build_negotiated_local_ready(merged_env, merged_ctx)

        merged_phase = controller._apply_phase_progression(
            "compress",
            24,
            ctx=merged_ctx,
            local_ready=merged_ready,
        )
        self.assertEqual(merged_phase, "brake_pulse")

    def test_blocker_event_gated_phase_progression_advances_on_late_merge_ahead(self):
        env = FakeHighwayEnv({
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 7.0, speed=23.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 2, 12.0, speed=24.5, ego_lane=2),
        }, step=24)
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
        controller._begin_rollout_if_needed(env)
        ctx = controller._get_relative_context(env)
        ready = controller._build_negotiated_local_ready(env, ctx)

        phase = controller._apply_phase_progression(
            "compress",
            24,
            ctx=ctx,
            local_ready=ready,
        )

        self.assertEqual(phase, "brake_pulse")

    def test_structured_lane_change_in_progress_is_not_overridden_by_hold_current_lane(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = SlowLaneChangeHighwayEnv(vehicles, step=5)
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
        controller._begin_rollout_if_needed(env)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "hold_lane",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "match",
            },
            message_text="hold before completing the merge",
        )
        controller.current_trigger_eval = {"satisfied": True, "trigger_code": "none", "details": {}}
        controller.last_control_step = 4
        controller._lane_change_in_progress_until_step = 8
        controller._lane_change_target_lane = 2

        controller.get_accel(env)

        self.assertEqual(env.k.kernel_api.vehicle.change_calls, [])
        self.assertTrue(controller._lane_change_in_progress(env))

    def test_structured_lane_change_in_progress_does_not_reissue_same_merge_command_for_negotiated_striker(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.6, ego_lane=2),
        }
        env = SlowLaneChangeHighwayEnv(vehicles, step=5)
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
        controller._begin_rollout_if_needed(env)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="commit the merge now",
        )
        controller.current_trigger_eval = {"satisfied": True, "trigger_code": "none", "details": {}}
        controller.last_control_step = env.time_step

        controller.get_accel(env)
        self.assertEqual(len(env.k.kernel_api.vehicle.change_calls), 1)
        self.assertTrue(controller._lane_change_in_progress(env))
        self.assertEqual(controller._lane_change_in_progress_until_step, 15)

        env.time_counter = 9
        env.time_step = 9
        controller.get_accel(env)

        self.assertEqual(len(env.k.kernel_api.vehicle.change_calls), 1)
        self.assertTrue(controller._lane_change_in_progress(env))
        self.assertEqual(env.k.vehicle.get_lane("llm_0"), 1)
        self.assertEqual(env.k.kernel_api.vehicle.move_to_calls, [])

    def test_structured_merge_commit_retries_lane_change_between_control_cycles(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.6, ego_lane=2),
        }
        env = SlowLaneChangeHighwayEnv(vehicles, step=9)
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
        controller._begin_rollout_if_needed(env)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge now",
        )
        controller.current_trigger_eval = {"satisfied": True, "trigger_code": "none", "details": {}}
        controller.last_control_step = 8
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 20
        controller._lane_change_in_progress_until_step = -1
        controller._lane_change_target_lane = None
        controller._set_execution_targets(
            env,
            controller.current_decision,
            controller.current_trigger_eval,
        )
        controller.pending_lane_change = 0

        controller.get_accel(env)

        self.assertEqual(len(env.k.kernel_api.vehicle.change_calls), 1)
        self.assertEqual(env.k.kernel_api.vehicle.change_calls[0][1], 2)
        self.assertGreaterEqual(controller._merge_attempt_steps, 1)

    def test_lane_change_stall_requires_multiple_attempts_after_in_progress_clears(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 9.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.6, ego_lane=2),
        }
        env = SlowLaneChangeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.target_abs_lane = 2
        controller._merge_attempt_steps = 1

        controller._update_merge_stall_state(env, controller._get_relative_context(env))

        self.assertFalse(controller._lane_change_stalled)
        self.assertEqual(controller._merge_stall_cycles, 0)

        env.time_counter = 28
        env.time_step = 28
        controller._merge_attempt_steps = 2
        controller._merge_stall_cycles = 7
        controller._update_merge_stall_state(env, controller._get_relative_context(env))

        self.assertTrue(controller._lane_change_stalled)
        self.assertEqual(controller._bad_merge_reason, "lane_change_stalled")

    def test_negotiated_merge_commit_allows_body_safe_ego_lane_cut_in(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "merge_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40

        self.assertTrue(controller._lane_change_is_safe(env, "highway_0", 2, aggressive=True))

    def test_negotiated_merge_commit_does_not_bypass_lane_change_safe_gate_with_too_small_front_buffer(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 0.2, speed=23.4, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "cut_in_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40

        self.assertFalse(controller._negotiated_cut_in_override_active(env, target_lane=2))
        self.assertFalse(controller._negotiated_unsafe_cut_in_active(env, target_lane=2))

    def test_negotiated_merge_commit_does_not_bypass_lane_change_safe_gate_from_rear_position(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -1.2, speed=25.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "merge_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40

        self.assertFalse(controller._lane_change_is_safe(env, "highway_0", 2, aggressive=True))

    def test_striker_execution_uses_lead_gain_before_front_cut_in(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, -2.2, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge later once ahead",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.executor_state, "lead_gain")
        self.assertEqual(controller.target_abs_lane, 1)
        self.assertEqual(controller.target_lc, 0)
        self.assertGreaterEqual(controller.target_v, vehicles["ego_0"]["speed"] + 5.5)

    def test_blocker_execution_clamps_side_front_band_during_teammate_strike(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 14.0, speed=24.2, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.0, speed=25.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._last_teammate_phase = "strike"
        controller._last_teammate_intent = "merge_commit"

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "tight",
                "speed_band": "press",
            },
            message_text="hold",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.intent, "seal_escape")
        self.assertEqual(controller.executor_state, "seal")
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_blocker_execution_accelerates_back_into_side_front_band(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 3.0, speed=22.4, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -1.0, speed=25.5, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._last_teammate_phase = "strike"
        controller._last_teammate_intent = "merge_commit"

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "tight",
                "speed_band": "press",
            },
            message_text="hold",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.intent, "seal_escape")
        self.assertEqual(controller.target_abs_lane, 3)
        self.assertGreater(controller.target_v, vehicles["ego_0"]["speed"])

    def test_striker_execution_deterministically_front_brakes_after_valid_merge_window(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 2.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.4, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._merged_into_ego_lane = True
        controller.striker_completed_cut_in = True
        controller._merge_commit_until_step = 20

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "press",
            },
            message_text="merge now",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.intent, "front_brake")
        self.assertEqual(controller.executor_state, "front_brake")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 0)
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_striker_execution_recovers_late_merge_before_front_brake(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 5.2, speed=24.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 7.0, speed=23.4, ego_lane=2),
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
        controller._begin_rollout_if_needed(env)
        controller._merged_into_ego_lane = True
        controller.striker_completed_cut_in = True
        controller._merge_commit_until_step = 20

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "press",
            },
            message_text="recover then brake",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertEqual(controller.intent, "merge_commit")
        self.assertEqual(controller.executor_state, "front_brake_setup")
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 0)
        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])

    def test_tactic_relative_speed_hint_influences_execution_targets(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 8.0, speed=23.5, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, -2.0, speed=24.0, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller._begin_rollout_if_needed(env)

        match_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "medium",
                "speed_band": "match",
                "tactic_hints": {
                    "sequence": ["hold_side_front", "match_ego"],
                    "speed_delta_hint_mps": -1.0,
                    "lead_gap_hint_m": 3.5,
                    "hold_cycles": 1,
                },
            },
            message_text="match speed while holding side-front",
        )
        surge_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "compress",
                "intent": "hold_side_front",
                "urgency": "mid",
                "goal": "hold",
                "message": "hold",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "medium",
                "speed_band": "surge",
                "tactic_hints": {
                    "sequence": ["hold_side_front", "match_ego"],
                    "speed_delta_hint_mps": 2.5,
                    "lead_gap_hint_m": 3.5,
                    "hold_cycles": 1,
                },
            },
            message_text="press forward into the side-front slot",
        )

        controller._set_execution_targets(env, match_decision, {"satisfied": True, "trigger_code": "none", "details": {}})
        match_target_v = controller.target_v
        controller._set_execution_targets(env, surge_decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertGreater(controller.target_v, match_target_v)
        self.assertEqual(controller.target_abs_lane, 3)
        self.assertEqual(controller.intent, "hold_side_front")
        self.assertEqual(controller.executor_state, "hold")

    def test_front_brake_execution_caps_speed_after_late_merge(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 2, 7.0, speed=30.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller._begin_rollout_if_needed(env)
        controller._bad_merge_reason = "late_merge_ahead"

        decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "brake_pulse",
                "intent": "front_brake",
                "urgency": "high",
                "goal": "brake",
                "message": "brake",
            },
            tactic_profile={
                "mode": "pulse_brake",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "brake",
            },
            message_text="brake hard after the merge",
        )

        controller._set_execution_targets(env, decision, {"satisfied": True, "trigger_code": "none", "details": {}})

        self.assertLess(controller.target_v, vehicles["ego_0"]["speed"])
        self.assertGreaterEqual(controller.target_v, vehicles["ego_0"]["speed"] - 4.5)
        self.assertEqual(controller.target_abs_lane, 2)
        self.assertEqual(controller.target_lc, 0)
        self.assertTrue(controller.brake_armed)
        self.assertLessEqual(controller.target_s, 0.65)

    def test_structured_road_end_guard_disables_lane_change_and_aborts_low_speed_terminal_state(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2, ego_pos=1790.0),
            "llm_0": make_payload("llm_0", 1, 0.0, speed=0.5, ego_lane=2, ego_pos=1790.0),
            "llm_1": make_payload("llm_1", 3, 6.0, speed=23.0, ego_lane=2, ego_pos=1790.0),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Striker", "llm_1": "Blocker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed_negotiated"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = controller._build_negotiated_runtime_decision(
            env,
            {
                "phase": "strike",
                "intent": "merge_commit",
                "urgency": "high",
                "goal": "merge",
                "message": "merge",
            },
            tactic_profile={
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            },
            message_text="merge before road end",
        )
        controller._merge_commit_until_step = 20
        controller._seal_escape_until_step = 20
        controller._pulse_end_step = 20
        controller.brake_armed = True
        controller.last_control_step = env.time_step

        controller.get_accel(env)

        self.assertEqual(controller.intent, "abort")
        self.assertEqual(controller.executor_state, "disengage")
        self.assertEqual(controller.target_lc, 0)
        self.assertEqual(controller._merge_commit_until_step, -1)
        self.assertEqual(controller._seal_escape_until_step, -1)
        self.assertEqual(controller._pulse_end_step, -1)
        self.assertFalse(controller.brake_armed)
        self.assertTrue(all(call[1] == 1 for call in env.k.kernel_api.vehicle.change_calls))

    def test_road_end_guard_does_not_change_simple_highway_runtime(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2, ego_pos=1790.0),
            "llm_0": make_payload("llm_0", 1, 6.0, speed=0.5, ego_lane=2, ego_pos=1790.0),
            "llm_1": make_payload("llm_1", 3, -4.0, speed=24.0, ego_lane=2, ego_pos=1790.0),
        }
        env = FakeHighwayEnv(vehicles, step=8)
        controller = self.make_controller(
            "llm_0",
            {"llm_0": "Blocker", "llm_1": "Striker"},
        )
        controller.scene_gate_status = {"scene_mode": "three_car_fixed"}
        controller._begin_rollout_if_needed(env)
        controller.current_decision = {
            "intent": "seal_escape",
            "urgency": "mid",
            "message": "seal now",
        }
        controller.last_control_step = env.time_step

        controller.get_accel(env)

        self.assertNotEqual(controller.intent, "abort")
        self.assertNotEqual(controller.executor_state, "disengage")

    def test_highway_runtime_trace_records_aggressive_cut_in_and_lane_change_attempt(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 3, 12.0, speed=23.8, ego_lane=2),
            "llm_1": make_payload("llm_1", 1, 6.8, speed=25.2, ego_lane=2),
        }
        vehicles["ego_0"]["min_gap"] = 0.0
        vehicles["llm_1"]["length"] = 4.0
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
        self.assertAlmostEqual(runtime_entry["rel_x"], 6.8, places=2)
        self.assertGreaterEqual(runtime_entry["speed_adv"], 2.0)
        self.assertTrue(runtime_entry["aggressive_cut_in_ready"])
        self.assertEqual(runtime_entry["target_lc"], 1)
        self.assertTrue(runtime_entry["lane_change_attempted"])
        self.assertEqual(env.k.vehicle.get_lane("llm_1"), 2)
        self.assertIn(("llm_1", 0), env.k.kernel_api.vehicle.speed_mode_calls)
        self.assertIn(("llm_1", 0), env.k.kernel_api.vehicle.lane_change_mode_calls)
        self.assertTrue(any(call[0] == "llm_1" and call[2] == 1 for call in env.k.kernel_api.vehicle.change_calls))

    def test_negotiated_force_cut_in_uses_supported_lane_change_parameters_only(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.0, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "cut_in_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40
        controller._lane_change_in_progress_until_step = 30
        controller._lane_change_target_lane = 2
        controller._merge_attempt_steps = 1

        controller._force_negotiated_cut_in(env, "highway_0", 2)

        parameter_keys = [call[1] for call in env.k.kernel_api.vehicle.parameter_calls]
        self.assertIn("laneChangeModel.lcAssertive", parameter_keys)
        self.assertIn("laneChangeModel.lcPushy", parameter_keys)
        self.assertIn("laneChangeModel.lcImpatience", parameter_keys)
        self.assertNotIn("laneChangeModel.lcPushyGap", parameter_keys)
        self.assertEqual(env.k.vehicle.get_lane("llm_0"), 2)
        self.assertEqual(env.k.kernel_api.vehicle.move_to_calls[-1][1], "highway_0_2")

    def test_negotiated_force_cut_in_blocks_move_to_when_too_close(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 1.2, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "cut_in_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40
        controller._lane_change_in_progress_until_step = 30
        controller._lane_change_target_lane = 2
        controller._merge_attempt_steps = 1

        self.assertFalse(controller._force_negotiated_cut_in(env, "highway_0", 2))
        self.assertEqual(env.k.vehicle.get_lane("llm_0"), 1)
        self.assertEqual(env.k.kernel_api.vehicle.move_to_calls, [])
        self.assertEqual(controller._last_force_cut_in_blocked_reason, "too_close")

    def test_negotiated_force_cut_in_blocks_move_to_outside_force_window(self):
        vehicles = {
            "ego_0": make_payload("ego_0", 2, 0.0, speed=23.0, ego_lane=2),
            "llm_0": make_payload("llm_0", 1, 12.5, speed=24.0, ego_lane=2),
            "llm_1": make_payload("llm_1", 3, 8.0, speed=23.5, ego_lane=2),
        }
        env = FakeHighwayEnv(vehicles, step=24)
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
        controller._begin_rollout_if_needed(env)
        controller.intent = "merge_commit"
        controller.executor_state = "cut_in_commit"
        controller._cut_in_episode_active = True
        controller._merge_commit_until_step = 40
        controller._lane_change_in_progress_until_step = 30
        controller._lane_change_target_lane = 2
        controller._merge_attempt_steps = 1

        self.assertTrue(controller._body_safe_cut_in_ready(env))
        self.assertFalse(controller._force_negotiated_cut_in(env, "highway_0", 2))
        self.assertEqual(env.k.kernel_api.vehicle.move_to_calls, [])
        self.assertEqual(controller._last_force_cut_in_blocked_reason, "outside_force_window")

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

    def test_structured_highway_tactic_prompt_omits_feedback_and_memory(self):
        agent = DriverAgent("llm_0")
        captured = {}

        def fake_call(system_message, user_message):
            captured["system"] = system_message
            captured["user"] = user_message
            return (
                "{\"message\":\"Merge now.\","
                "\"tactic\":{\"mode\":\"track_pose\",\"lane_policy\":\"ego_lane\","
                "\"gap_band\":\"tight\",\"speed_band\":\"surge\"}}"
            )

        agent.call = fake_call
        agent.collaborate_highway_tactic(
            "perception",
            "Striker",
            "ego_0",
            {"pass_side": "left"},
            {"phase": "strike", "intent": "merge_commit"},
            {"teammate_id": "llm_1", "teammate_phase": {"intent": "seal_escape"}},
            {"approach_window_ready": True},
        )

        self.assertIn("intent_plan=", captured["user"])
        self.assertNotIn("feedback=", captured["user"])
        self.assertNotIn("memory=", captured["user"])
        self.assertNotIn("desired_rel_x", captured["user"])
        self.assertNotIn("speed_target_hint", captured["user"])
        self.assertIn("sequence", captured["system"])
        self.assertIn("speed_delta_hint_mps", captured["system"])
        self.assertIn("lead_gap_hint_m", captured["system"])
        self.assertIn("hold_cycles", captured["system"])
        self.assertIn("Do not output mode, lane_policy", captured["system"])
        self.assertIn("allowed_sequence_tokens", captured["user"])
        self.assertIn("relative_hint_ranges", captured["user"])

    def test_structured_highway_intent_prompt_includes_allowed_next_space(self):
        agent = DriverAgent("llm_0")
        captured = {}

        def fake_call(system_message, user_message):
            captured["system"] = system_message
            captured["user"] = user_message
            return "{\"phase\":\"strike\",\"intent\":\"merge_commit\"}"

        agent.call = fake_call
        agent.collaborate_highway_intent(
            "perception",
            "Striker",
            "ego_0",
            {"pass_side": "left"},
            {"teammate_phase": {"intent": "hold_side_front"}},
            {"approach_window_ready": True},
            allowed_next={
                "phases": ["strike"],
                "intents": ["merge_commit"],
                "phase_intents": [{"phase": "strike", "intent": "merge_commit"}],
                "reason": "approach_window_ready",
            },
        )

        self.assertNotIn("allowed_next_phases", captured["user"])
        self.assertIn("allowed_phase_intents", captured["user"])
        self.assertIn("merge_commit", captured["user"])
        self.assertIn("allowed_next_intents", captured["system"])


class TestDriverAgentClientConfig(unittest.TestCase):
    def test_driver_agent_prefers_deepseek_defaults_when_key_present(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deepseek"}, clear=True):
            with patch("flow.controllers.llm_controller.OpenAI") as openai_cls:
                DriverAgent("llm_0")

        kwargs = openai_cls.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "sk-deepseek")
        self.assertEqual(kwargs["base_url"], "https://api.deepseek.com")

    def test_driver_agent_base_url_override_beats_deepseek_default(self):
        env = {
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "FLOW_LLM_BASE_URL": "https://example.com/v1",
            "FLOW_LLM_MODEL": "custom-model",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("flow.controllers.llm_controller.OpenAI") as openai_cls:
                agent = DriverAgent("llm_0")

        kwargs = openai_cls.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://example.com/v1")
        self.assertEqual(agent.llm_model, "custom-model")

    def test_driver_agent_default_model_is_deepseek_chat(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("flow.controllers.llm_controller.OpenAI"):
                agent = DriverAgent("llm_0")

        self.assertEqual(agent.llm_model, "deepseek-chat")


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
