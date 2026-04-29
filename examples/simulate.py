"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

from flow.utils.highway_scene import SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED

MAX_ITERATIONS = 10
HARD_BRAKE_DECEL = 2.5
SUCCESS_TTC = 3.0
PROGRESS_EVERY_STEPS = max(0, int(os.getenv("FLOW_PROGRESS_EVERY_STEPS", "200")))
DIRTY_SUCCESS_OBSERVATION_STEPS = max(
    0,
    int(os.getenv("FLOW_DIRTY_SUCCESS_OBSERVATION_STEPS", "20")),
)


def parse_args(args):
    """Parse training options user can specify in command line."""
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    parser.add_argument(
        "exp_config", type=str,
        help="Name of the experiment configuration file, as located in exp_configs/non_rl.")
    parser.add_argument(
        "--num_runs", type=int, default=1,
        help="Number of simulations to run. Defaults to 1.")
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Specifies whether to run the simulation during runtime.")
    parser.add_argument(
        "--aimsun",
        action="store_true",
        help="Specifies whether to run the simulation using the simulator Aimsun. "
             "If not specified, the simulator used is SUMO.")
    parser.add_argument(
        "--gen_emission",
        action="store_true",
        help="Specifies whether to generate an emission file from the simulation.")

    return parser.parse_known_args(args)[0]


def iter_llm_controllers(env_instance):
    candidate_ids = set(["llm_0", "llm_1"])
    for getter_name in ("get_ids", "get_human_ids", "get_rl_ids"):
        try:
            getter = getattr(env_instance.k.vehicle, getter_name)
            candidate_ids.update([vid for vid in getter() if "llm" in vid])
        except Exception:
            pass

    controllers = {}
    for veh_id in sorted(candidate_ids):
        try:
            controller = env_instance.k.vehicle.get_acc_controller(veh_id)
        except Exception:
            controller = None
        if controller is not None:
            controllers[veh_id] = controller
    return controllers


def collect_llm_stats(env_instance, controllers=None):
    stats = {}
    controller_map = controllers if controllers is not None else iter_llm_controllers(env_instance)
    for veh_id, controller in controller_map.items():
        if hasattr(controller, "get_runtime_stats"):
            stats[veh_id] = controller.get_runtime_stats()
    return stats


def collect_llm_diagnostics(env_instance, controllers=None):
    diagnostics = {}
    controller_map = controllers if controllers is not None else iter_llm_controllers(env_instance)
    for veh_id, controller in controller_map.items():
        if hasattr(controller, "get_rollout_diagnostics"):
            diagnostics[veh_id] = controller.get_rollout_diagnostics()
    return diagnostics


def any_terminal_plan_locked(controllers):
    for controller in (controllers or {}).values():
        if bool(getattr(controller, "_terminal_plan_locked", False)):
            return True
    return False


def any_dirty_observation_active(controllers, env_instance=None):
    for controller in (controllers or {}).values():
        checker = getattr(controller, "_dirty_observation_active", None)
        if callable(checker):
            try:
                if bool(checker(env_instance)):
                    return True
            except Exception:
                if bool(getattr(controller, "_stale_merge_candidate", False)):
                    return True
        elif bool(getattr(controller, "_stale_merge_candidate", False)):
            return True
    return False


def _scene_mode_from_context(scenario_context):
    gate = scenario_context.get("scene_gate_status", {}) or {}
    return str(gate.get("scene_mode", "") or "").strip().lower()


def _uses_negotiated_contract(scenario_context):
    return _scene_mode_from_context(scenario_context) == SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED


def _normalize_pass_side(value):
    text = str(value or "").strip().lower()
    if text in ("left", "right"):
        return text
    return "none"


def _opposite_side(side):
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return "none"


def _infer_geometry_pass_side(scenario_context):
    geometry = dict(scenario_context.get("frozen_geometry", {}) or {})
    geometry_role_hint = dict(scenario_context.get("geometry_role_hint", {}) or {})
    striker_id = next((veh_id for veh_id, role in geometry_role_hint.items() if role == "Striker"), "")
    striker = geometry.get(striker_id, {}) if striker_id else {}
    rel_lane = int(striker.get("rel_lane_to_ego", 0) or 0)
    if rel_lane < 0:
        return "left"
    if rel_lane > 0:
        return "right"
    blocker_id = next((veh_id for veh_id, role in geometry_role_hint.items() if role == "Blocker"), "")
    blocker = geometry.get(blocker_id, {}) if blocker_id else {}
    blocker_rel_lane = int(blocker.get("rel_lane_to_ego", 0) or 0)
    if blocker_rel_lane < 0:
        return "right"
    if blocker_rel_lane > 0:
        return "left"
    return "left"


def _build_negotiated_contract_payload(scenario_context):
    pass_side = _normalize_pass_side(scenario_context.get("pass_side", "none"))
    block_side = _normalize_pass_side(scenario_context.get("block_side", _opposite_side(pass_side)))
    if pass_side == "none" and block_side in ("left", "right"):
        pass_side = _opposite_side(block_side)
    if block_side == "none" and pass_side in ("left", "right"):
        block_side = _opposite_side(pass_side)
    return {
        "role_map": dict(scenario_context.get("role_map", {}) or {}),
        "pass_side": pass_side,
        "block_side": block_side,
        "contract_source": str(scenario_context.get("contract_source", "") or ""),
    }


def _apply_contract_context(env, controllers, scenario_context):
    contract_payload = _build_negotiated_contract_payload(scenario_context)
    if hasattr(env.message_pool, "set_negotiated_contract"):
        env.message_pool.set_negotiated_contract(contract_payload if _uses_negotiated_contract(scenario_context) else {})
    for controller in controllers.values():
        if hasattr(controller, "set_highway_contract"):
            controller.set_highway_contract(contract_payload)


def apply_role_context(controllers, scenario_context):
    role_map = dict(scenario_context.get("role_map", {}) or {})
    role_source = str(scenario_context.get("role_source", "") or "")
    geometry_role_hint = dict(scenario_context.get("geometry_role_hint", {}) or {})
    for controller in controllers.values():
        controller.geometry_role_hint = dict(geometry_role_hint)
        if hasattr(controller, "set_role_assignment"):
            controller.set_role_assignment(role_map, role_source=role_source)
        else:
            controller.role_map = dict(role_map)
            controller.role_source = role_source
            if hasattr(controller, "refresh_attack_role"):
                controller.refresh_attack_role()


def _refresh_assignment_context(env, controllers, scenario_context):
    apply_role_context(controllers, scenario_context)
    _apply_contract_context(env, controllers, scenario_context)
    for controller in controllers.values():
        if hasattr(controller, "_initialize_highway_preferences"):
            controller._initialize_highway_preferences(env)


def inject_rollout_context(env, rolling_feedback, case_memory, scenario_context, iteration):
    controllers = iter_llm_controllers(env)
    initial_signatures = {}
    env.message_pool.set_scenario_context(scenario_context)
    _refresh_assignment_context(env, controllers, scenario_context)
    for veh_id, controller in controllers.items():
        controller.previous_feedback = rolling_feedback
        controller.case_memory = case_memory
        controller.scenario_id = scenario_context.get("scenario_id", "")
        controller.current_iteration = iteration
        controller.frozen_geometry = scenario_context.get("frozen_geometry", {})
        controller.scene_gate_status = scenario_context.get("scene_gate_status", {})
        if hasattr(controller, "_begin_rollout_if_needed"):
            controller._begin_rollout_if_needed(env)
        if hasattr(controller, "get_state_signature"):
            initial_signatures[veh_id] = controller.get_state_signature(env, phase="compress")
    return controllers, initial_signatures


def _format_role_pool_message(proposal):
    if "decision" in proposal:
        return "[decision={decision}] {message}".format(
            decision=str(proposal.get("decision", "confirm")),
            message=str(proposal.get("message", "")),
        )
    return "[role={role} intent={intent}] {message}".format(
        role=str(proposal.get("role", "Undecided")),
        intent=str(proposal.get("intent", "wait")),
        message=str(proposal.get("message", "")),
    )


def _maybe_lock_negotiated_role_map(proposals):
    blockers = [veh_id for veh_id, payload in proposals.items() if payload.get("proposed_role") == "Blocker"]
    strikers = [veh_id for veh_id, payload in proposals.items() if payload.get("proposed_role") == "Striker"]
    if len(blockers) == 1 and len(strikers) == 1 and blockers[0] != strikers[0]:
        return {
            blockers[0]: "Blocker",
            strikers[0]: "Striker",
        }
    return {}


def _maybe_lock_negotiated_pass_side(proposals):
    sides = []
    for payload in proposals.values():
        side = _normalize_pass_side(payload.get("pass_side", "none"))
        if side in ("left", "right"):
            sides.append(side)
    if not sides:
        return "none"
    if len(set(sides)) == 1:
        return sides[0]
    return "none"


def _score_geometry_role_fit(payload, role):
    if not payload:
        return 1e6
    rel_x = float(payload.get("rel_x_to_ego", 0.0) or 0.0)
    rel_lane = int(payload.get("rel_lane_to_ego", 0) or 0)
    abs_rel_lane = abs(rel_lane)
    lane_penalty = 8.0 * abs(abs_rel_lane - 1)
    if role == "Blocker":
        front_penalty = 0.0 if rel_x >= 0.0 else 12.0 + 2.0 * abs(rel_x)
        band_penalty = abs(rel_x - 8.0) * 0.8
        if rel_x > 18.0:
            band_penalty += 1.2 * (rel_x - 18.0)
        return lane_penalty + front_penalty + band_penalty
    trail_penalty = 0.0
    if rel_x > 2.5:
        trail_penalty = 10.0 + 1.5 * (rel_x - 2.5)
    elif rel_x < -10.0:
        trail_penalty = 1.2 * (-10.0 - rel_x)
    band_penalty = abs(rel_x + 2.0) * 0.5
    return lane_penalty + trail_penalty + band_penalty


def _score_negotiated_role_map(role_map, scenario_context):
    geometry = dict(scenario_context.get("frozen_geometry", {}) or {})
    if not geometry or not role_map:
        return None
    score = 0.0
    for veh_id, role in role_map.items():
        score += _score_geometry_role_fit(geometry.get(veh_id, {}), role)
    return float(score)


def _repair_negotiated_role_map(role_map, scenario_context):
    proposed = dict(role_map or {})
    geometry_hint = dict(scenario_context.get("geometry_role_hint", {}) or {})
    if not proposed:
        return {}, ""
    if not geometry_hint or set(proposed.keys()) != set(geometry_hint.keys()):
        return proposed, "llm_negotiated"

    proposed_score = _score_negotiated_role_map(proposed, scenario_context)
    geometry_score = _score_negotiated_role_map(geometry_hint, scenario_context)
    if proposed_score is None or geometry_score is None:
        return proposed, "llm_negotiated"
    if proposed_score <= geometry_score + 4.0:
        return proposed, "llm_negotiated"
    return geometry_hint, "llm_negotiated_geometry_constrained"


def _repair_negotiated_pass_side(pass_side, scenario_context, role_map):
    locked_pass_side = _normalize_pass_side(pass_side)
    geometry = dict(scenario_context.get("frozen_geometry", {}) or {})
    if not geometry:
        return locked_pass_side, False

    preferred_side = _infer_geometry_pass_side(scenario_context)
    striker_id = next((veh_id for veh_id, role in (role_map or {}).items() if role == "Striker"), "")
    striker = geometry.get(striker_id, {}) if striker_id else {}
    striker_rel_lane = int(striker.get("rel_lane_to_ego", 0) or 0)
    if striker_rel_lane < 0:
        preferred_side = "left"
    elif striker_rel_lane > 0:
        preferred_side = "right"

    if preferred_side not in ("left", "right"):
        return locked_pass_side, False
    if locked_pass_side == preferred_side:
        return locked_pass_side, False
    return preferred_side, True


def _maybe_lock_role_map(proposals):
    blockers = [veh_id for veh_id, payload in proposals.items() if payload.get("role") == "Blocker"]
    strikers = [veh_id for veh_id, payload in proposals.items() if payload.get("role") == "Striker"]
    if len(blockers) == 1 and len(strikers) == 1 and blockers[0] != strikers[0]:
        return {
            blockers[0]: "Blocker",
            strikers[0]: "Striker",
        }
    return {}


def resolve_roles_for_rollout(env, controllers, scenario_context, iteration):
    role_map = dict(scenario_context.get("role_map", {}) or {})
    if role_map:
        _refresh_assignment_context(env, controllers, scenario_context)
        env.message_pool.set_scenario_context(scenario_context)
        return role_map

    if _uses_negotiated_contract(scenario_context):
        ordered_ids = sorted(controllers.keys())
        proposals = {}
        env.message_pool.begin_control_cycle(0)
        env.message_pool.set_scenario_context(scenario_context)
        if hasattr(env.message_pool, "set_negotiated_contract"):
            env.message_pool.set_negotiated_contract({})

        for _ in range(2):
            for veh_id in ordered_ids:
                controller = controllers[veh_id]
                controller.active_phase = "negotiation"
                if hasattr(controller, "negotiate_role"):
                    proposal = controller.negotiate_role(env)
                else:
                    proposal = {
                        "proposed_role": "Undecided",
                        "pass_side": "none",
                        "message": "",
                    }
                proposals[veh_id] = proposal
                if hasattr(env.message_pool, "publish_negotiated_negotiation"):
                    env.message_pool.publish_negotiated_negotiation({
                        "sender": veh_id,
                        "proposed_role": proposal.get("proposed_role", "Undecided"),
                        "pass_side": proposal.get("pass_side", "none"),
                        "message": proposal.get("message", ""),
                        "step": 0,
                        "control_cycle_step": 0,
                    })

            locked_role_map = _maybe_lock_negotiated_role_map(proposals)
            locked_pass_side = _maybe_lock_negotiated_pass_side(proposals)
            if locked_role_map:
                repaired_role_map, role_source = _repair_negotiated_role_map(
                    locked_role_map,
                    scenario_context,
                )
                repaired_pass_side, side_repaired = _repair_negotiated_pass_side(
                    locked_pass_side,
                    scenario_context,
                    repaired_role_map,
                )
            else:
                repaired_role_map, role_source = {}, ""
                repaired_pass_side, side_repaired = "none", False
            if repaired_role_map and repaired_pass_side in ("left", "right"):
                scenario_context["role_map"] = dict(repaired_role_map)
                scenario_context["role_source"] = (
                    "llm_negotiated_geometry_constrained"
                    if side_repaired and role_source == "llm_negotiated"
                    else role_source
                )
                scenario_context["pass_side"] = repaired_pass_side
                scenario_context["block_side"] = _opposite_side(repaired_pass_side)
                scenario_context["contract_source"] = "negotiated"
                env.message_pool.set_scenario_context(scenario_context)
                _refresh_assignment_context(env, controllers, scenario_context)
                return dict(repaired_role_map)

        fallback = dict(scenario_context.get("geometry_role_hint", {}) or {})
        fallback_pass_side = _infer_geometry_pass_side(scenario_context)
        scenario_context["role_map"] = fallback
        scenario_context["role_source"] = "geometry_fallback" if fallback else ""
        scenario_context["pass_side"] = fallback_pass_side
        scenario_context["block_side"] = _opposite_side(fallback_pass_side)
        scenario_context["contract_source"] = "fallback_geometry" if fallback else ""
        env.message_pool.set_scenario_context(scenario_context)
        _refresh_assignment_context(env, controllers, scenario_context)
        for controller in controllers.values():
            controller.active_phase = "fallback" if fallback else "negotiation"
            if fallback:
                controller.role_resolution_fallback_used = 1
        return fallback

    geometry_hint = dict(scenario_context.get("geometry_role_hint", {}) or {})
    if geometry_hint:
        scenario_context["role_map"] = dict(geometry_hint)
        scenario_context["role_source"] = "geometry_locked"
        env.message_pool.set_scenario_context(scenario_context)
        _refresh_assignment_context(env, controllers, scenario_context)

        ordered_ids = sorted(controllers.keys())
        proposals = {}
        for veh_id in ordered_ids:
            controller = controllers[veh_id]
            controller.active_phase = "negotiation"
            if hasattr(controller, "negotiate_role"):
                proposal = controller.negotiate_role(env)
            else:
                proposal = {
                    "decision": "confirm",
                    "message": "Keeping geometry-locked role.",
                }
            proposals[veh_id] = proposal
            env.message_pool.join(veh_id, _format_role_pool_message(proposal))

        if ordered_ids and all(
                str(proposals.get(veh_id, {}).get("decision", "confirm")).strip().lower() == "swap"
                for veh_id in ordered_ids):
            swapped = {}
            for veh_id, role in geometry_hint.items():
                if role == "Blocker":
                    swapped[veh_id] = "Striker"
                elif role == "Striker":
                    swapped[veh_id] = "Blocker"
            if len(swapped) == len(geometry_hint) and sorted(swapped.values()) == ["Blocker", "Striker"]:
                scenario_context["role_map"] = dict(swapped)
                scenario_context["role_source"] = "llm_swapped"
            else:
                scenario_context["role_source"] = "geometry_locked"
        elif ordered_ids and all(
                str(proposals.get(veh_id, {}).get("decision", "confirm")).strip().lower() == "confirm"
                for veh_id in ordered_ids):
            scenario_context["role_source"] = "llm_confirmed_geometry"
        else:
            scenario_context["role_source"] = "geometry_locked"

        env.message_pool.set_scenario_context(scenario_context)
        _refresh_assignment_context(env, controllers, scenario_context)
        return dict(scenario_context.get("role_map", {}) or {})

    ordered_ids = sorted(controllers.keys())
    proposals = {}
    for _ in range(2):
        for veh_id in ordered_ids:
            controller = controllers[veh_id]
            controller.active_phase = "negotiation"
            if hasattr(controller, "negotiate_role"):
                proposal = controller.negotiate_role(env)
            else:
                proposal = {
                    "message": "Holding role decision and waiting for teammate.",
                    "role": "Undecided",
                    "intent": "wait",
                }
            proposals[veh_id] = proposal
            env.message_pool.join(veh_id, _format_role_pool_message(proposal))

        locked = _maybe_lock_role_map(proposals)
        if locked:
            scenario_context["role_map"] = dict(locked)
            scenario_context["role_source"] = "llm_negotiated"
            env.message_pool.set_scenario_context(scenario_context)
            _refresh_assignment_context(env, controllers, scenario_context)
            return locked

    fallback = dict(scenario_context.get("geometry_role_hint", {}) or {})
    scenario_context["role_map"] = fallback
    scenario_context["role_source"] = "geometry_fallback" if fallback else ""
    env.message_pool.set_scenario_context(scenario_context)
    _refresh_assignment_context(env, controllers, scenario_context)
    for controller in controllers.values():
        controller.active_phase = "fallback" if fallback else "negotiation"
        if fallback:
            controller.role_resolution_fallback_used = 1
    return fallback


def _format_emission_label(iteration_index, partial=False):
    label = "iter{:02d}".format(max(0, int(iteration_index)))
    if partial:
        label += "_partial"
    return label


def compute_min_ttc(env):
    min_ttc = float("inf")
    veh_ids = env.k.vehicle.get_ids()
    if "ego_0" not in veh_ids:
        return min_ttc

    ego_speed = env.k.vehicle.get_speed("ego_0")
    ego_lane = env.k.vehicle.get_lane("ego_0")
    ego_x = env.k.vehicle.get_x_by_id("ego_0")
    ego_leader = env.k.vehicle.get_leader("ego_0")
    llm_ids = [veh_id for veh_id in veh_ids if "llm" in veh_id]

    for llm_id in llm_ids:
        try:
            llm_speed = env.k.vehicle.get_speed(llm_id)
            llm_lane = env.k.vehicle.get_lane(llm_id)
            llm_x = env.k.vehicle.get_x_by_id(llm_id)
            llm_leader = env.k.vehicle.get_leader(llm_id)
        except Exception:
            continue

        if llm_leader == "ego_0":
            rel_speed = llm_speed - ego_speed
            if rel_speed > 1e-3:
                headway = env.k.vehicle.get_headway(llm_id)
                if headway > 0:
                    min_ttc = min(min_ttc, headway / rel_speed)
        elif ego_leader == llm_id:
            rel_speed = ego_speed - llm_speed
            if rel_speed > 1e-3:
                headway = env.k.vehicle.get_headway("ego_0")
                if headway > 0:
                    min_ttc = min(min_ttc, headway / rel_speed)

        if int(llm_lane) == int(ego_lane):
            rel_x = float(llm_x) - float(ego_x)
            if rel_x > 0.0:
                rel_speed = float(ego_speed) - float(llm_speed)
                if rel_speed > 1e-3:
                    min_ttc = min(min_ttc, rel_x / rel_speed)
            elif rel_x < 0.0:
                rel_speed = float(llm_speed) - float(ego_speed)
                if rel_speed > 1e-3:
                    min_ttc = min(min_ttc, (-rel_x) / rel_speed)

    return min_ttc


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def init_ego_motion_metrics():
    return {
        "prev_speed": None,
        "prev_accel": None,
        "accel_samples": 0,
        "accel_abs_sum": 0.0,
        "max_abs_accel": 0.0,
        "max_decel": 0.0,
        "jerk_samples": 0,
        "jerk_abs_sum": 0.0,
        "jerk_sq_sum": 0.0,
        "max_abs_jerk": 0.0,
        "comfort_samples": 0,
        "comfort_sum": 0.0,
    }


def update_ego_motion_metrics(metrics, ego_speed, dt):
    dt = max(float(dt or 0.0), 1e-3)
    speed = float(ego_speed)
    prev_speed = metrics.get("prev_speed")
    metrics["prev_speed"] = speed
    if prev_speed is None:
        return None

    accel = (speed - float(prev_speed)) / dt
    if not math.isfinite(accel):
        return None
    artifact_limit = max(0.0, _float_env("FLOW_EGO_ACCEL_ARTIFACT_LIMIT", "30.0"))
    if artifact_limit > 0.0 and abs(accel) > artifact_limit:
        metrics["prev_accel"] = None
        return None

    abs_accel = abs(accel)
    metrics["accel_samples"] = int(metrics.get("accel_samples", 0) or 0) + 1
    metrics["accel_abs_sum"] = float(metrics.get("accel_abs_sum", 0.0) or 0.0) + abs_accel
    metrics["max_abs_accel"] = max(float(metrics.get("max_abs_accel", 0.0) or 0.0), abs_accel)
    metrics["max_decel"] = max(float(metrics.get("max_decel", 0.0) or 0.0), max(0.0, -accel))

    comfort_eps = max(0.0, _float_env("FLOW_COMFORT_ACCEL_DENOISE_EPS", "1e-3"))
    comfort_scale = max(1e-6, _float_env("FLOW_COMFORT_ACCEL_SCALE", "1.0"))
    denoised_abs_accel = 0.0 if abs_accel <= comfort_eps else abs_accel
    if denoised_abs_accel > 0.0:
        comfort_value = 1.0 / ((denoised_abs_accel / comfort_scale) + 1.0)
        metrics["comfort_sum"] = float(metrics.get("comfort_sum", 0.0) or 0.0) + comfort_value
        metrics["comfort_samples"] = int(metrics.get("comfort_samples", 0) or 0) + 1

    prev_accel = metrics.get("prev_accel")
    metrics["prev_accel"] = accel
    if prev_accel is None:
        return accel

    jerk = (accel - float(prev_accel)) / dt
    if not math.isfinite(jerk):
        return accel

    abs_jerk = abs(jerk)
    metrics["jerk_samples"] = int(metrics.get("jerk_samples", 0) or 0) + 1
    metrics["jerk_abs_sum"] = float(metrics.get("jerk_abs_sum", 0.0) or 0.0) + abs_jerk
    metrics["jerk_sq_sum"] = float(metrics.get("jerk_sq_sum", 0.0) or 0.0) + jerk * jerk
    metrics["max_abs_jerk"] = max(float(metrics.get("max_abs_jerk", 0.0) or 0.0), abs_jerk)
    return accel


def summarize_ego_motion_metrics(metrics):
    accel_samples = int(metrics.get("accel_samples", 0) or 0)
    jerk_samples = int(metrics.get("jerk_samples", 0) or 0)
    comfort_samples = int(metrics.get("comfort_samples", 0) or 0)
    jerk_sq_sum = float(metrics.get("jerk_sq_sum", 0.0) or 0.0)
    return {
        "ego_max_decel": float(metrics.get("max_decel", 0.0) or 0.0),
        "ego_max_abs_accel": float(metrics.get("max_abs_accel", 0.0) or 0.0),
        "ego_mean_abs_accel": (
            float(metrics.get("accel_abs_sum", 0.0) or 0.0) / accel_samples
            if accel_samples else 0.0
        ),
        "ego_max_abs_jerk": float(metrics.get("max_abs_jerk", 0.0) or 0.0),
        "ego_mean_abs_jerk": (
            float(metrics.get("jerk_abs_sum", 0.0) or 0.0) / jerk_samples
            if jerk_samples else 0.0
        ),
        "ego_rms_jerk": math.sqrt(jerk_sq_sum / jerk_samples) if jerk_samples else 0.0,
        "ego_comfort": (
            float(metrics.get("comfort_sum", 0.0) or 0.0) / comfort_samples
            if comfort_samples else 1.0
        ),
        "ego_accel_samples": accel_samples,
        "ego_jerk_samples": jerk_samples,
        "ego_comfort_samples": comfort_samples,
    }


def choose_failure_phase(diagnostics):
    for veh_id in ("llm_0", "llm_1"):
        diag = diagnostics.get(veh_id)
        if diag and diag.get("active_phase"):
            return diag.get("active_phase")
    for diag in diagnostics.values():
        if diag.get("active_phase"):
            return diag.get("active_phase")
    return "attack"


def _escape_direction(initial_lane, escape_lane):
    if initial_lane is None or escape_lane is None:
        return "unknown"
    if int(escape_lane) < int(initial_lane):
        return "left"
    if int(escape_lane) > int(initial_lane):
        return "right"
    return "same_lane"


def classify_success_label(success, contract_source="", role_source="", striker_diag=None):
    if not success:
        return ""
    if str(contract_source or "") == "fallback_geometry":
        return "fallback_success"
    if str(role_source or "") == "geometry_fallback":
        return "fallback_success"
    striker_diag = striker_diag or {}
    dirty_markers = (
        bool(striker_diag.get("clean_merge_failed", False)),
        bool(striker_diag.get("stale_merge_candidate", False)),
        bool(striker_diag.get("lane_change_stalled", False)),
        bool(striker_diag.get("bad_merge_event", False)),
        bool(striker_diag.get("overshoot", False)),
        str(striker_diag.get("bad_merge_reason", "") or "") in (
            "late_merge_ahead",
            "stale_merge_ahead_far",
            "rear_merge",
            "off_window_merge",
            "lane_change_stalled",
        ),
    )
    if any(dirty_markers):
        return "dirty_success"
    return "clean_success"


def build_feedback_reason(
        crashed,
        success,
        too_safe,
        diagnostics,
        failure_phase,
        contract_source="",
        role_source="",
        striker_diag=None,
        ego_escape_lane=None,
        blocker_lane_at_escape=None,
        initial_ego_lane=None):
    if crashed:
        return "collision_on_attack"
    success_label = classify_success_label(
        success,
        contract_source=contract_source,
        role_source=role_source,
        striker_diag=striker_diag,
    )
    if success_label:
        return success_label

    striker_diag = striker_diag or {}
    front_brake_triggered = bool(striker_diag.get("front_brake_triggered", False))
    observed_merge = bool(
        striker_diag.get("merged_into_ego_lane", False)
        or striker_diag.get("striker_lane_change_time") is not None
    )
    valid_cut_in = bool(striker_diag.get("striker_completed_cut_in", False))
    lane_change_attempted = bool(
        striker_diag.get("last_lane_change_attempted", False)
        or int(striker_diag.get("merge_attempt_steps", 0) or 0) > 0
    )
    bad_merge_reason = str(striker_diag.get("bad_merge_reason", "") or "")
    if bad_merge_reason == "rear_merge":
        return "rear_merge_into_ego_lane"
    if bad_merge_reason == "off_window_merge":
        return "off_window_merge"
    if bad_merge_reason == "late_merge_ahead":
        return "late_merge_ahead_with_brake_fallback" if front_brake_triggered else "late_merge_ahead"
    if bad_merge_reason == "stale_merge_ahead_far":
        return "stale_merge_ahead_far_with_brake_fallback" if front_brake_triggered else "stale_merge_ahead_far"
    if bool(striker_diag.get("stale_merge_candidate", False)):
        return "stale_merge_candidate_without_pressure"
    if bool(striker_diag.get("clean_merge_failed", False)):
        return "clean_merge_failed_without_pressure"
    if bool(striker_diag.get("lane_change_stalled", False)):
        return "lane_change_stalled"
    if bool(striker_diag.get("overshoot", False)):
        return "merge_overshoot"
    if valid_cut_in and not front_brake_triggered:
        return "cut_in_without_front_brake"
    if observed_merge and not valid_cut_in and not front_brake_triggered:
        return "merge_observed_outside_valid_window"
    if front_brake_triggered and not success:
        return "front_brake_without_enough_pressure"
    if lane_change_attempted and not observed_merge:
        return "merge_commit_without_lane_change"

    escape_direction = _escape_direction(initial_ego_lane, ego_escape_lane)
    if ego_escape_lane is not None:
        if blocker_lane_at_escape is None:
            return "ego_escaped_{}".format(escape_direction)
        if int(blocker_lane_at_escape) != int(ego_escape_lane):
            return "ego_escaped_{}_unblocked".format(escape_direction)
        return "ego_escaped_{}_past_blocker".format(escape_direction)

    if too_safe:
        return "too_safe_no_attack_pressure"
    if any(diag.get("rollout_parse_fallback_used", 0) > 0 for diag in diagnostics.values()):
        return "parse_fallback_used"
    if failure_phase == "negotiation" and contract_source == "negotiated":
        return "coordination_unstable"
    if failure_phase == "fallback" and contract_source == "negotiated":
        return "geometry_fallback"
    if any(diag.get("role_resolution_fallback_used", 0) > 0 for diag in diagnostics.values()):
        return "role_resolution_fallback"
    return "attack_did_not_converge"


def build_escape_summary(initial_ego_lane, ego_escape_lane, blocker_lane_at_escape):
    if ego_escape_lane is None:
        return ""
    direction = _escape_direction(initial_ego_lane, ego_escape_lane)
    if blocker_lane_at_escape is None:
        return "ego_escaped_{}".format(direction)
    if int(blocker_lane_at_escape) == int(ego_escape_lane):
        return "ego_escaped_{}_through_blocker_lane".format(direction)
    return "ego_escaped_{}_through_open_side".format(direction)


def determine_feedback_summary(crashed, success, too_safe, diagnostics, failure_phase, **kwargs):
    return build_feedback_reason(
        crashed=crashed,
        success=success,
        too_safe=too_safe,
        diagnostics=diagnostics,
        failure_phase=failure_phase,
        **kwargs
    )


def build_run_meta(
        run_started_at,
        flow_params,
        scenario_id,
        scenario_context,
        iteration_logs,
        case_memory,
        interrupted=False,
        llm_trace_file=""):
    return {
        "run_started_at": run_started_at,
        "scenario_id": scenario_id,
        "exp_tag": flow_params["exp_tag"],
        "model": os.getenv("FLOW_LLM_MODEL", "deepseek-chat"),
        "llm_timeout_s": os.getenv("FLOW_LLM_TIMEOUT_S", "45.0"),
        "llm_max_retries": "3",
        "llm_neighbor_k": os.getenv("FLOW_LLM_NEIGHBOR_K", "6"),
        "comfort_metric": "linguasim_accel_score",
        "comfort_accel_denoise_eps": _float_env("FLOW_COMFORT_ACCEL_DENOISE_EPS", "1e-3"),
        "comfort_accel_scale": _float_env("FLOW_COMFORT_ACCEL_SCALE", "1.0"),
        "role_map": scenario_context.get("role_map", {}),
        "role_source": scenario_context.get("role_source", ""),
        "contract_source": scenario_context.get("contract_source", ""),
        "pass_side": scenario_context.get("pass_side", "none"),
        "block_side": scenario_context.get("block_side", "none"),
        "geometry_role_hint": scenario_context.get("geometry_role_hint", {}),
        "frozen_geometry": scenario_context.get("frozen_geometry", {}),
        "scene_gate_status": scenario_context.get("scene_gate_status", {}),
        "sampling_strategy": scenario_context.get("sampling_strategy", ""),
        "interrupted": bool(interrupted),
        "llm_trace_file": str(llm_trace_file or ""),
        "iterations_completed": int(len(iteration_logs)),
        "iterations": iteration_logs,
        "case_memory": case_memory,
    }


def persist_run_meta(run_output_dir, run_meta):
    if not run_output_dir:
        return
    with open(os.path.join(run_output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    from flow.core.experiment import Experiment
    from flow.core.params import AimsunParams
    from flow.utils.highway_scene import SCENE_GATE_SAMPLE_BUDGET
    from flow.utils.highway_scene import sample_and_freeze_scene
    from flow.utils.rllib import FlowParamsEncoder

    flags = parse_args(sys.argv[1:])
    run_started_at = datetime.now().strftime("%Y%m%d-%H%M%S")

    module = __import__("exp_configs.non_rl", fromlist=[flags.exp_config])
    flow_params = getattr(module, flags.exp_config).flow_params

    if hasattr(getattr(module, flags.exp_config), "custom_callables"):
        callables = getattr(module, flags.exp_config).custom_callables
    else:
        callables = None

    flow_params["sim"].render = not flags.no_render
    flow_params["simulator"] = "aimsun" if flags.aimsun else "traci"

    if flags.aimsun:
        sim_params = AimsunParams()
        sim_params.__dict__.update(flow_params["sim"].__dict__)
        flow_params["sim"] = sim_params

    run_output_dir = None
    if flags.gen_emission:
        data_root = "./data"
        os.makedirs(data_root, exist_ok=True)
        run_output_dir = os.path.join(
            data_root,
            "{}_{}".format(flow_params["exp_tag"], run_started_at))
        os.makedirs(run_output_dir, exist_ok=True)
        flow_params["sim"].emission_path = run_output_dir
        if not os.getenv("FLOW_LLM_DEBUG_TRACE_FILE"):
            os.environ["FLOW_LLM_DEBUG_TRACE_FILE"] = os.path.join(run_output_dir, "llm_trace.jsonl")

        fp_ = flow_params["exp_tag"]
        with open(os.path.join(run_output_dir, "{}_flow_params.json".format(fp_)), "w") as outfile:
            json.dump(flow_params, outfile, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    llm_trace_file = str(os.getenv("FLOW_LLM_DEBUG_TRACE_FILE", "")).strip()

    exp = Experiment(flow_params, callables)
    env = exp.env
    exp_tag = str(flow_params.get("exp_tag", "")).lower()
    network_name = getattr(flow_params.get("network"), "__name__", "").lower()
    if "highway" not in exp_tag and "highway" not in network_name:
        print("Warning: LinguaSim closed-loop is intended for highway, not figure eight.")

    if hasattr(getattr(module, flags.exp_config), "rl_actions"):
        rl_actions = getattr(module, flags.exp_config).rl_actions
    else:
        def rl_actions(*_):
            return None

    rolling_feedback = {}
    case_memory = []
    iteration_logs = []
    print("Sampling custom highway opening...", flush=True)
    scenario_context = sample_and_freeze_scene(env, run_started_at)
    scenario_id = scenario_context.get("scenario_id", "")
    if not scenario_id:
        scene_gate_status = scenario_context.get("scene_gate_status", {})
        last_gate = scene_gate_status.get("last_gate", {}) if isinstance(scene_gate_status, dict) else {}
        last_reason = last_gate.get("reason", scene_gate_status.get("reason", "")) if isinstance(last_gate, dict) else ""
        print("scene sampling failed: unable to build a valid custom highway opening within {} attempts".format(
            SCENE_GATE_SAMPLE_BUDGET
        ))
        if last_reason:
            print("last rejected candidate reason:", last_reason)
        if isinstance(last_gate, dict):
            last_error = last_gate.get("error", "")
            if last_error:
                print("last rejected candidate error:", last_error)
        if flags.gen_emission and run_output_dir is not None:
            run_meta = build_run_meta(
                run_started_at,
                flow_params,
                scenario_id,
                scenario_context,
                iteration_logs,
                case_memory,
                interrupted=False,
                llm_trace_file=llm_trace_file,
            )
            persist_run_meta(run_output_dir, run_meta)
        env.terminate()
        raise SystemExit(1)
    print(
        "Scene frozen: scenario_id={} sample_index={} geometry_role_hint={}".format(
            scenario_id,
            (scenario_context.get("scene_gate_status") or {}).get("sample_index", ""),
            scenario_context.get("geometry_role_hint", {}),
        ),
        flush=True,
    )

    interrupted = False
    current_iteration = 0
    current_step_in_iteration = 0
    current_min_ttc = float("inf")
    current_ego_max_decel = 0.0
    current_motion_summary = summarize_ego_motion_metrics(init_ego_motion_metrics())
    current_crashed = False
    try:
        for iteration in range(MAX_ITERATIONS):
            current_iteration = iteration + 1
            current_step_in_iteration = 0
            crashed = False
            ego_max_decel = 0.0
            ego_motion_metrics = init_ego_motion_metrics()
            min_ttc = float("inf")
            print(
                "Iteration {}/{} start | horizon={}".format(
                    iteration + 1,
                    MAX_ITERATIONS,
                    int(env.env_params.horizon),
                ),
                flush=True,
            )

            state = env.reset()
            env.message_pool.reset_rollout(scenario_id, iteration + 1)
            env.message_pool.set_scenario_context(scenario_context)
            controllers, initial_signatures = inject_rollout_context(
                env, rolling_feedback, case_memory, scenario_context, iteration + 1)
            resolved_role_map = resolve_roles_for_rollout(env, controllers, scenario_context, iteration + 1)
            if resolved_role_map:
                print(
                    "Iteration {}/{} roles locked | source={} | role_map={}".format(
                        iteration + 1,
                        MAX_ITERATIONS,
                        scenario_context.get("role_source", ""),
                        resolved_role_map,
                    ),
                    flush=True,
                )
            else:
                print(
                    "Iteration {}/{} roles unresolved; controllers remain undecided".format(
                        iteration + 1,
                        MAX_ITERATIONS,
                    ),
                    flush=True,
                )

            prev_ego_lane = None
            initial_ego_lane = None
            ego_escape_lane = None
            blocker_lane_at_escape = None
            blocker_id = next((veh_id for veh_id, role in (scenario_context.get("role_map", {}) or {}).items() if role == "Blocker"), "")
            if "ego_0" in env.k.vehicle.get_ids():
                update_ego_motion_metrics(
                    ego_motion_metrics,
                    float(env.k.vehicle.get_speed("ego_0")),
                    max(env.sim_step, 1e-3),
                )
                prev_ego_lane = int(env.k.vehicle.get_lane("ego_0"))
                initial_ego_lane = prev_ego_lane

            dirty_success_observed_step = None
            for _ in range(env.env_params.horizon):
                action = rl_actions(state)
                state, reward, done, _ = env.step(action)
                current_step_in_iteration += 1
                current_crashed = bool(crashed)

                if "ego_0" in env.k.vehicle.get_ids():
                    ego_speed = float(env.k.vehicle.get_speed("ego_0"))
                    ego_lane = int(env.k.vehicle.get_lane("ego_0"))
                    update_ego_motion_metrics(
                        ego_motion_metrics,
                        ego_speed,
                        max(env.sim_step, 1e-3),
                    )
                    ego_max_decel = float(ego_motion_metrics.get("max_decel", 0.0) or 0.0)
                    if prev_ego_lane is not None and ego_lane != prev_ego_lane and ego_escape_lane is None:
                        ego_escape_lane = ego_lane
                        if blocker_id and blocker_id in env.k.vehicle.get_ids():
                            blocker_lane_at_escape = int(env.k.vehicle.get_lane(blocker_id))
                    prev_ego_lane = ego_lane

                min_ttc = min(min_ttc, compute_min_ttc(env))
                current_min_ttc = min_ttc
                current_ego_max_decel = ego_max_decel
                current_motion_summary = summarize_ego_motion_metrics(ego_motion_metrics)
                if (
                        (not crashed)
                        and min_ttc != float("inf")
                        and min_ttc <= SUCCESS_TTC
                        and ego_max_decel >= HARD_BRAKE_DECEL):
                    if not _uses_negotiated_contract(scenario_context):
                        break
                    interim_diagnostics = collect_llm_diagnostics(env, controllers=controllers)
                    interim_striker_diag = next(
                        (diag for diag in interim_diagnostics.values() if diag.get("role") == "Striker"),
                        {},
                    )
                    interim_label = classify_success_label(
                        True,
                        contract_source=scenario_context.get("contract_source", ""),
                        role_source=scenario_context.get("role_source", ""),
                        striker_diag=interim_striker_diag,
                    )
                    if interim_label == "clean_success":
                        break
                    if dirty_success_observed_step is None:
                        dirty_success_observed_step = int(current_step_in_iteration)
                    elif (
                            int(current_step_in_iteration) - int(dirty_success_observed_step)
                            >= int(DIRTY_SUCCESS_OBSERVATION_STEPS)):
                        break
                if (
                        _uses_negotiated_contract(scenario_context)
                        and any_terminal_plan_locked(controllers)
                        and dirty_success_observed_step is None
                        and not any_dirty_observation_active(controllers, env)):
                    break

                if PROGRESS_EVERY_STEPS > 0 and current_step_in_iteration % PROGRESS_EVERY_STEPS == 0:
                    print(
                        "Iteration {}/{} progress {}/{} | min_ttc={} | ego_max_decel={:.3f}".format(
                            iteration + 1,
                            MAX_ITERATIONS,
                            current_step_in_iteration,
                            int(env.env_params.horizon),
                            "inf" if min_ttc == float("inf") else "{:.3f}".format(min_ttc),
                            ego_max_decel,
                        ),
                        flush=True,
                    )

                if env.k.simulation.check_collision():
                    crashed = True
                    current_crashed = True
                    break

                if done:
                    break

            motion_summary = summarize_ego_motion_metrics(ego_motion_metrics)
            ego_max_decel = float(motion_summary["ego_max_decel"])
            current_motion_summary = motion_summary
            hard_brake_event = ego_max_decel >= HARD_BRAKE_DECEL
            success = (
                (not crashed)
                and min_ttc != float("inf")
                and min_ttc <= SUCCESS_TTC
                and hard_brake_event
            )
            too_safe = (
                (min_ttc == float("inf") or min_ttc > SUCCESS_TTC)
                and ego_max_decel < HARD_BRAKE_DECEL
            )

            diagnostics = collect_llm_diagnostics(env, controllers=controllers)
            llm_stats = collect_llm_stats(env, controllers=controllers)
            failure_phase = choose_failure_phase(diagnostics)
            sync_error = any(diag.get("role_resolution_fallback_used", 0) > 0 for diag in diagnostics.values())
            striker_diag = next(
                (diag for diag in diagnostics.values() if diag.get("role") == "Striker"),
                {},
            )
            success_label = classify_success_label(
                success,
                contract_source=scenario_context.get("contract_source", ""),
                role_source=scenario_context.get("role_source", ""),
                striker_diag=striker_diag,
            )
            feedback_summary = determine_feedback_summary(
                crashed,
                success,
                too_safe,
                diagnostics,
                failure_phase,
                contract_source=scenario_context.get("contract_source", ""),
                role_source=scenario_context.get("role_source", ""),
                striker_diag=striker_diag,
                ego_escape_lane=ego_escape_lane,
                blocker_lane_at_escape=blocker_lane_at_escape,
                initial_ego_lane=initial_ego_lane,
            )
            failure_reason = build_feedback_reason(
                crashed=crashed,
                success=success,
                too_safe=too_safe,
                diagnostics=diagnostics,
                failure_phase=failure_phase,
                contract_source=scenario_context.get("contract_source", ""),
                role_source=scenario_context.get("role_source", ""),
                striker_diag=striker_diag,
                ego_escape_lane=ego_escape_lane,
                blocker_lane_at_escape=blocker_lane_at_escape,
                initial_ego_lane=initial_ego_lane,
            )
            escape_summary = build_escape_summary(initial_ego_lane, ego_escape_lane, blocker_lane_at_escape)
            result_label = success_label if success_label else (
                "collision" if crashed else "too_safe" if too_safe else "failed"
            )

            feedback = {
                "scenario_id": scenario_id,
                "iteration": iteration + 1,
                "result": result_label,
                "success_label": success_label,
                "feedback_summary": feedback_summary,
                "failure_reason": failure_reason,
                "escape_summary": escape_summary,
                "min_ttc": None if min_ttc == float("inf") else round(float(min_ttc), 3),
                "ego_max_decel": round(float(ego_max_decel), 3),
                "ego_max_abs_accel": round(float(motion_summary["ego_max_abs_accel"]), 3),
                "ego_mean_abs_accel": round(float(motion_summary["ego_mean_abs_accel"]), 3),
                "ego_max_abs_jerk": round(float(motion_summary["ego_max_abs_jerk"]), 3),
                "ego_mean_abs_jerk": round(float(motion_summary["ego_mean_abs_jerk"]), 3),
                "ego_rms_jerk": round(float(motion_summary["ego_rms_jerk"]), 3),
                "ego_comfort": round(float(motion_summary["ego_comfort"]), 4),
                "ego_accel_samples": int(motion_summary["ego_accel_samples"]),
                "ego_jerk_samples": int(motion_summary["ego_jerk_samples"]),
                "ego_comfort_samples": int(motion_summary["ego_comfort_samples"]),
                "hard_brake_event": bool(hard_brake_event),
                "failure_phase": failure_phase,
                "sync_error": bool(sync_error),
                "collision": bool(crashed),
                "contract_source": scenario_context.get("contract_source", ""),
                "role_source": scenario_context.get("role_source", ""),
                "scene_gate_status": scenario_context.get("scene_gate_status", {}),
                "striker_completed_cut_in": bool(striker_diag.get("striker_completed_cut_in", False)),
                "merged_into_ego_lane": bool(striker_diag.get("merged_into_ego_lane", False)),
                "valid_cut_in_merge": bool(striker_diag.get("valid_cut_in_merge", False)),
                "striker_lane_change_time": striker_diag.get("striker_lane_change_time"),
                "striker_rel_x_at_lane_change": striker_diag.get("striker_rel_x_at_lane_change"),
                "front_brake_triggered": bool(striker_diag.get("front_brake_triggered", False)),
                "last_lane_change_attempted": bool(striker_diag.get("last_lane_change_attempted", False)),
                "merge_attempt_steps": int(striker_diag.get("merge_attempt_steps", 0) or 0),
                "bad_merge_event": bool(striker_diag.get("bad_merge_event", False)),
                "bad_merge_reason": str(striker_diag.get("bad_merge_reason", "") or ""),
                "clean_merge_failed": bool(striker_diag.get("clean_merge_failed", False)),
                "stale_merge_candidate": bool(striker_diag.get("stale_merge_candidate", False)),
                "stale_merge_candidate_step": int(striker_diag.get("stale_merge_candidate_step", -1) or -1),
                "lane_change_stalled": bool(striker_diag.get("lane_change_stalled", False)),
                "merge_stall_cycles": int(striker_diag.get("merge_stall_cycles", 0) or 0),
                "last_merge_rel_x": striker_diag.get("last_merge_rel_x"),
                "overshoot": bool(striker_diag.get("overshoot", False)),
                "ego_escape_lane": ego_escape_lane,
                "blocker_lane_at_escape": blocker_lane_at_escape,
            }

            for veh_id, controller in controllers.items():
                controller.previous_feedback = feedback

                signature = initial_signatures.get(veh_id, {})
                diag = diagnostics.get(veh_id, {})
                case_memory.append({
                    "scenario_id": scenario_id,
                    "iteration": iteration + 1,
                    "veh_id": veh_id,
                    "role": diag.get("role", veh_id),
                    "result": feedback["result"],
                    "state_signature": signature.get("continuous", {}),
                    "state_signature_bucketed": signature.get("bucketed", {}),
                    "final_plan": diag.get("current_decision"),
                    "phase_trace": diag.get("phase_trace", []),
                    "min_ttc": feedback["min_ttc"],
                    "ego_max_decel": feedback["ego_max_decel"],
                    "ego_max_abs_accel": feedback["ego_max_abs_accel"],
                    "ego_mean_abs_accel": feedback["ego_mean_abs_accel"],
                    "ego_max_abs_jerk": feedback["ego_max_abs_jerk"],
                    "ego_mean_abs_jerk": feedback["ego_mean_abs_jerk"],
                    "ego_rms_jerk": feedback["ego_rms_jerk"],
                    "ego_comfort": feedback["ego_comfort"],
                    "hard_brake_event": feedback["hard_brake_event"],
                    "failure_phase": feedback["failure_phase"],
                    "sync_error": feedback["sync_error"],
                    "collision": feedback["collision"],
                    "feedback_summary": feedback["feedback_summary"],
                    "failure_reason": feedback["failure_reason"],
                    "escape_summary": feedback["escape_summary"],
                })

            case_memory = case_memory[-120:]
            rolling_feedback = feedback

            print(
                "Iteration {}/{} | min_ttc={} | ego_max_decel={:.3f} | ego_mean_abs_jerk={:.3f} | ego_comfort={:.4f} | hard_brake={} | feedback={}".format(
                    iteration + 1,
                    MAX_ITERATIONS,
                    "inf" if min_ttc == float("inf") else "{:.3f}".format(min_ttc),
                    ego_max_decel,
                    feedback["ego_mean_abs_jerk"],
                    feedback["ego_comfort"],
                    hard_brake_event,
                    feedback_summary,
                )
            )
            if llm_stats:
                print("LLM runtime stats:", json.dumps(llm_stats, sort_keys=True))

            iteration_logs.append({
                "iteration": iteration + 1,
                "scenario_id": scenario_id,
                "role_map": scenario_context.get("role_map", {}),
                "role_source": scenario_context.get("role_source", ""),
                "scene_gate_status": scenario_context.get("scene_gate_status", {}),
                "sampling_strategy": scenario_context.get("sampling_strategy", ""),
                "contract_source": scenario_context.get("contract_source", ""),
                "pass_side": scenario_context.get("pass_side", "none"),
                "block_side": scenario_context.get("block_side", "none"),
                "crashed": bool(crashed),
                "result": feedback["result"],
                "feedback_summary": feedback_summary,
                "failure_reason": feedback["failure_reason"],
                "escape_summary": feedback["escape_summary"],
                "min_ttc": feedback["min_ttc"],
                "ego_max_decel": feedback["ego_max_decel"],
                "ego_max_abs_accel": feedback["ego_max_abs_accel"],
                "ego_mean_abs_accel": feedback["ego_mean_abs_accel"],
                "ego_max_abs_jerk": feedback["ego_max_abs_jerk"],
                "ego_mean_abs_jerk": feedback["ego_mean_abs_jerk"],
                "ego_rms_jerk": feedback["ego_rms_jerk"],
                "ego_comfort": feedback["ego_comfort"],
                "ego_accel_samples": feedback["ego_accel_samples"],
                "ego_jerk_samples": feedback["ego_jerk_samples"],
                "ego_comfort_samples": feedback["ego_comfort_samples"],
                "hard_brake_event": feedback["hard_brake_event"],
                "striker_lane_change_time": feedback["striker_lane_change_time"],
                "striker_rel_x_at_lane_change": feedback["striker_rel_x_at_lane_change"],
                "front_brake_triggered": feedback["front_brake_triggered"],
                "merged_into_ego_lane": feedback["merged_into_ego_lane"],
                "striker_completed_cut_in": feedback["striker_completed_cut_in"],
                "valid_cut_in_merge": feedback["valid_cut_in_merge"],
                "last_lane_change_attempted": feedback["last_lane_change_attempted"],
                "merge_attempt_steps": feedback["merge_attempt_steps"],
                "bad_merge_event": feedback["bad_merge_event"],
                "bad_merge_reason": feedback["bad_merge_reason"],
                "last_merge_rel_x": feedback["last_merge_rel_x"],
                "overshoot": feedback["overshoot"],
                "ego_escape_lane": feedback["ego_escape_lane"],
                "blocker_lane_at_escape": feedback["blocker_lane_at_escape"],
                "failure_phase": failure_phase,
                "sync_error": bool(sync_error),
                "feedback": feedback,
                "llm_stats": llm_stats,
                "llm_diagnostics": diagnostics,
                "normalization_repairs": sum(
                    int(diag.get("normalization_repairs", 0))
                    for diag in diagnostics.values()
                ),
            })

            if flags.gen_emission and env.simulator == "traci":
                env.k.simulation.save_emission(
                    run_id=_format_emission_label(iteration)
                )

            if flags.gen_emission and run_output_dir is not None:
                run_meta = build_run_meta(
                    run_started_at,
                    flow_params,
                    scenario_id,
                    scenario_context,
                    iteration_logs,
                    case_memory,
                    interrupted=False,
                    llm_trace_file=llm_trace_file,
                )
                persist_run_meta(run_output_dir, run_meta)

            if success:
                print("危险场景生成成功 ({})".format(success_label or "success"))
                break
    except KeyboardInterrupt:
        interrupted = True
        if flags.gen_emission and env.simulator == "traci" and current_iteration > 0 and current_step_in_iteration > 0:
            try:
                env.k.simulation.save_emission(
                    run_id=_format_emission_label(current_iteration - 1, partial=True)
                )
            except Exception:
                pass
        if current_iteration > 0 and (
                (not iteration_logs) or iteration_logs[-1].get("iteration") != current_iteration):
            try:
                interrupted_diagnostics = collect_llm_diagnostics(
                    env,
                    controllers=locals().get("controllers"),
                )
                interrupted_stats = collect_llm_stats(
                    env,
                    controllers=locals().get("controllers"),
                )
            except Exception:
                interrupted_diagnostics = {}
                interrupted_stats = {}
            iteration_logs.append({
                "iteration": int(current_iteration),
                "scenario_id": scenario_id,
                "role_map": scenario_context.get("role_map", {}),
                "scene_gate_status": scenario_context.get("scene_gate_status", {}),
                "sampling_strategy": scenario_context.get("sampling_strategy", ""),
                "interrupted": True,
                "steps_executed": int(current_step_in_iteration),
                "crashed": bool(current_crashed),
                "result": "interrupted",
                "feedback_summary": "interrupted_by_user",
                "min_ttc": None if current_min_ttc == float("inf") else round(float(current_min_ttc), 3),
                "ego_max_decel": round(float(current_ego_max_decel), 3),
                "ego_max_abs_accel": round(float(current_motion_summary["ego_max_abs_accel"]), 3),
                "ego_mean_abs_accel": round(float(current_motion_summary["ego_mean_abs_accel"]), 3),
                "ego_max_abs_jerk": round(float(current_motion_summary["ego_max_abs_jerk"]), 3),
                "ego_mean_abs_jerk": round(float(current_motion_summary["ego_mean_abs_jerk"]), 3),
                "ego_rms_jerk": round(float(current_motion_summary["ego_rms_jerk"]), 3),
                "ego_comfort": round(float(current_motion_summary["ego_comfort"]), 4),
                "ego_accel_samples": int(current_motion_summary["ego_accel_samples"]),
                "ego_jerk_samples": int(current_motion_summary["ego_jerk_samples"]),
                "ego_comfort_samples": int(current_motion_summary["ego_comfort_samples"]),
                "llm_stats": interrupted_stats,
                "llm_diagnostics": interrupted_diagnostics,
                "normalization_repairs": sum(
                    int(diag.get("normalization_repairs", 0))
                    for diag in interrupted_diagnostics.values()
                ),
            })
        print("\nInterrupted by user; partial run metadata has been flushed.")
    finally:
        if flags.gen_emission and run_output_dir is not None:
            run_meta = build_run_meta(
                run_started_at,
                flow_params,
                scenario_id,
                scenario_context,
                iteration_logs,
                case_memory,
                interrupted=interrupted,
                llm_trace_file=llm_trace_file,
            )
            persist_run_meta(run_output_dir, run_meta)
        try:
            env.terminate()
        except Exception as exc:
            print("Warning: env.terminate cleanup error:", str(exc))
