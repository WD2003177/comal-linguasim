import os
import random
from copy import deepcopy

from flow.utils.exceptions import FatalFlowError


SCENE_GATE_SAMPLE_BUDGET = 48
SCENE_MODE_ENV_VAR = "FLOW_HIGHWAY_SCENE_MODE"
SCENE_MODE_TRAFFIC = "traffic"
SCENE_MODE_THREE_CAR_FIXED = "three_car_fixed"
SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED = "three_car_fixed_negotiated"
SCENE_MODE_THREE_CAR_RANDOM = "three_car_random"
SCENE_DEFAULT_MODE = SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED
SCENE_SAMPLING_STRATEGY = "custom_highway_opening"
SCENE_SAMPLING_STRATEGIES = {
    SCENE_MODE_TRAFFIC: SCENE_SAMPLING_STRATEGY,
    SCENE_MODE_THREE_CAR_FIXED: "three_car_fixed_opening",
    SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED: "three_car_fixed_negotiated_opening",
    SCENE_MODE_THREE_CAR_RANDOM: "three_car_random_opening",
}
SCENE_EDGE_ID = "highway_0"
SCENE_MIN_SAME_LANE_GAP = 22.0
SCENE_MAX_LANE_LOAD = 3
SCENE_EGO_LANES = (1, 2)
SCENE_BASE_X_RANGE = (150.0, 220.0)
SCENE_BLOCKER_REL_X_RANGE = (10.0, 22.0)
SCENE_STRIKER_REL_X_RANGE = (-20.0, 10.0)
SCENE_BLOCKER_SAME_LANE_PROB = 0.55
SCENE_THREE_CAR_FIXED_BLOCKER_REL_X_RANGE = (12.0, 16.0)
SCENE_THREE_CAR_FIXED_STRIKER_REL_X_RANGE = (-5.5, -4.5)
SCENE_THREE_CAR_FIXED_STRIKER_SPEED_ADV_RANGE = (2.5, 3.5)
SCENE_THREE_CAR_RANDOM_BLOCKER_REL_X_RANGE = (10.0, 18.0)
SCENE_THREE_CAR_RANDOM_STRIKER_REL_X_RANGE = (-10.0, -3.0)
SCENE_THREE_CAR_STRIKER_SPEED_ADV_RANGE = (1.5, 3.0)
SCENE_THREE_CAR_BLOCKER_SPEED_ADV_RANGE = (0.2, 1.2)
SCENE_THREE_CAR_MIN_KEY_SPACING = 2.5
SCENE_THREE_CAR_AGGRESSIVE_CUT_IN_REL_X = (-3.0, 1.5)
SCENE_THREE_CAR_STRIKER_GATE_REL_X_MAX = -3.0
SCENE_THREE_CAR_GATE_PROGRESS_FACTOR = 2.0
SCENE_THREE_CAR_BACKGROUND_OFFSETS = (
    -176.0, -148.0, -120.0, -92.0, 92.0, 120.0, 148.0, 176.0,
)


def is_three_car_fixed_like_mode(scene_mode):
    return str(scene_mode or "").strip().lower() in (
        SCENE_MODE_THREE_CAR_FIXED,
        SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED,
    )


def get_highway_scene_mode():
    mode = str(os.getenv(SCENE_MODE_ENV_VAR, SCENE_DEFAULT_MODE) or SCENE_DEFAULT_MODE).strip().lower()
    if mode in SCENE_SAMPLING_STRATEGIES:
        return mode
    return SCENE_DEFAULT_MODE


def get_highway_sampling_strategy(scene_mode=None):
    mode = str(scene_mode or get_highway_scene_mode() or SCENE_DEFAULT_MODE).strip().lower()
    return SCENE_SAMPLING_STRATEGIES.get(mode, SCENE_SAMPLING_STRATEGY)


def _resolve_scene_mode(scene_mode=None, geometry=None):
    if scene_mode:
        return str(scene_mode).strip().lower()
    if isinstance(geometry, dict) and any("human" in veh_id for veh_id in geometry.keys()):
        return SCENE_MODE_TRAFFIC
    return get_highway_scene_mode()


def default_custom_start_params(scene_mode=None):
    """Provide a safe placeholder layout for the initial env bootstrap."""
    resolved_mode = _resolve_scene_mode(scene_mode=scene_mode)
    if resolved_mode in (
            SCENE_MODE_THREE_CAR_FIXED,
            SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED,
            SCENE_MODE_THREE_CAR_RANDOM):
        base_pos = 220.0
        return {
            "start_positions": [
                (SCENE_EDGE_ID, base_pos),
                (SCENE_EDGE_ID, base_pos + 14.0),
                (SCENE_EDGE_ID, base_pos - 5.0),
            ],
            "start_lanes": [1, 2, 0],
        }

    start_positions = []
    start_lanes = []
    base_pos = 220.0
    lane_cycle = [1, 2, 0, 3]
    rel_offsets = [0.0, 24.0, -26.0, 52.0, -54.0, 86.0, -88.0, 124.0, -126.0, 166.0, -168.0]
    for idx, rel_x in enumerate(rel_offsets):
        start_positions.append((SCENE_EDGE_ID, base_pos + rel_x))
        start_lanes.append(lane_cycle[idx % len(lane_cycle)])
    return {
        "start_positions": start_positions,
        "start_lanes": start_lanes,
    }


def build_initial_geometry(env):
    geometry = {}
    if "ego_0" not in env.initial_state:
        return geometry

    _, ego_edge, ego_lane, ego_pos, ego_speed = env.initial_state["ego_0"]
    ego_lane = int(ego_lane)
    ego_pos = float(ego_pos)
    ego_speed = float(ego_speed)

    for veh_id in env.initial_ids:
        type_id, edge, lane, pos, speed = env.initial_state[veh_id]
        lane = int(lane)
        pos = float(pos)
        speed = float(speed)
        geometry[veh_id] = {
            "veh_id": veh_id,
            "type_id": type_id,
            "edge": edge,
            "lane": lane,
            "pos": pos,
            "speed": speed,
            "ego_edge": ego_edge,
            "ego_lane": ego_lane,
            "ego_pos": ego_pos,
            "ego_speed": ego_speed,
            "rel_lane_to_ego": lane - ego_lane,
            "rel_x_to_ego": pos - ego_pos,
            "rel_speed_to_ego": speed - ego_speed,
        }
    return geometry


def score_blocker_candidate(payload):
    rel_x = float(payload["rel_x_to_ego"])
    rel_lane = abs(int(payload["rel_lane_to_ego"]))
    front_bias = 0 if rel_x >= 0.0 else 1000
    return (front_bias, -rel_x if rel_x >= 0.0 else abs(rel_x), rel_lane, payload["veh_id"])


def assign_roles_from_geometry(geometry):
    llm_entries = sorted(
        [payload for veh_id, payload in geometry.items() if "llm" in veh_id],
        key=score_blocker_candidate,
    )
    role_map = {}
    if not llm_entries:
        return role_map

    blocker_id = llm_entries[0]["veh_id"]
    role_map[blocker_id] = "Blocker"
    for payload in llm_entries[1:]:
        role_map[payload["veh_id"]] = "Striker"
    return role_map


def _key_vehicle_spacing_ok(geometry, key_ids, min_gap):
    for i, veh_a in enumerate(key_ids):
        for veh_b in key_ids[i + 1:]:
            if abs(float(geometry[veh_a]["pos"]) - float(geometry[veh_b]["pos"])) < float(min_gap):
                return False
    return True


def _same_lane_spacing_ok(geometry, min_gap):
    for veh_a, payload_a in geometry.items():
        for veh_b, payload_b in geometry.items():
            if veh_a >= veh_b:
                continue
            if int(payload_a["lane"]) != int(payload_b["lane"]):
                continue
            if abs(float(payload_a["pos"]) - float(payload_b["pos"])) < float(min_gap):
                return False
    return True


def _traffic_scene_gate_metrics(geometry):
    llm_entries = [payload for veh_id, payload in geometry.items() if "llm" in veh_id]
    ego_payload = geometry.get("ego_0")
    if ego_payload is None:
        return {
            "accepted": False,
            "reason": "missing_ego_vehicle",
            "scene_mode": SCENE_MODE_TRAFFIC,
            "sampling_strategy": get_highway_sampling_strategy(SCENE_MODE_TRAFFIC),
        }
    if len(llm_entries) < 2:
        return {
            "accepted": False,
            "reason": "missing_llm_agents",
            "llm_entries": llm_entries,
            "scene_mode": SCENE_MODE_TRAFFIC,
            "sampling_strategy": get_highway_sampling_strategy(SCENE_MODE_TRAFFIC),
        }

    role_map = assign_roles_from_geometry(geometry)
    blocker_id = next((veh_id for veh_id, role in role_map.items() if role == "Blocker"), "")
    striker_id = next((veh_id for veh_id, role in role_map.items() if role == "Striker"), "")
    blocker = geometry.get(blocker_id) or {}
    striker = geometry.get(striker_id) or {}

    if not blocker or not striker:
        return {
            "accepted": False,
            "reason": "role_assignment_failed",
            "role_map": role_map,
            "scene_mode": SCENE_MODE_TRAFFIC,
            "sampling_strategy": get_highway_sampling_strategy(SCENE_MODE_TRAFFIC),
        }

    blocker_rel_x = float(blocker["rel_x_to_ego"])
    striker_rel_x = float(striker["rel_x_to_ego"])
    blocker_lane_ok = abs(int(blocker["rel_lane_to_ego"])) <= 1
    striker_lane_ok = abs(int(striker["rel_lane_to_ego"])) <= 1
    blocker_x_ok = SCENE_BLOCKER_REL_X_RANGE[0] <= blocker_rel_x <= SCENE_BLOCKER_REL_X_RANGE[1]
    striker_x_ok = SCENE_STRIKER_REL_X_RANGE[0] <= striker_rel_x <= SCENE_STRIKER_REL_X_RANGE[1]

    key_ids = ["ego_0", blocker_id, striker_id]
    key_spacing_ok = _key_vehicle_spacing_ok(geometry, key_ids, min_gap=2.0)
    same_lane_spacing_ok = _same_lane_spacing_ok(geometry, min_gap=SCENE_MIN_SAME_LANE_GAP)

    human_nearby = 0
    for veh_id, payload in geometry.items():
        if "human" not in veh_id:
            continue
        if abs(float(payload["rel_x_to_ego"])) <= 90.0:
            human_nearby += 1

    accepted = all((
        blocker_lane_ok,
        striker_lane_ok,
        blocker_x_ok,
        striker_x_ok,
        key_spacing_ok,
        same_lane_spacing_ok,
        human_nearby >= 4,
    ))

    reason = "accepted"
    if not blocker_lane_ok:
        reason = "blocker_lane_out_of_band"
    elif not blocker_x_ok:
        reason = "blocker_rel_x_out_of_band"
    elif not striker_lane_ok:
        reason = "striker_lane_out_of_band"
    elif not striker_x_ok:
        reason = "striker_rel_x_out_of_band"
    elif not key_spacing_ok:
        reason = "critical_vehicles_too_aligned"
    elif not same_lane_spacing_ok:
        reason = "same_lane_spacing_failed"
    elif human_nearby < 4:
        reason = "insufficient_background_traffic"

    return {
        "accepted": accepted,
        "reason": reason,
        "scene_mode": SCENE_MODE_TRAFFIC,
        "sampling_strategy": get_highway_sampling_strategy(SCENE_MODE_TRAFFIC),
        "role_map": dict(role_map),
        "blocker_id": blocker_id,
        "striker_id": striker_id,
        "blocker_rel_x": round(blocker_rel_x, 3),
        "striker_rel_x": round(striker_rel_x, 3),
        "blocker_rel_lane": int(blocker.get("rel_lane_to_ego", 0)),
        "striker_rel_lane": int(striker.get("rel_lane_to_ego", 0)),
        "background_nearby_count": int(human_nearby),
    }


def _three_car_scene_gate_metrics(geometry, scene_mode):
    llm_entries = [payload for veh_id, payload in geometry.items() if "llm" in veh_id]
    ego_payload = geometry.get("ego_0")
    if ego_payload is None:
        return {
            "accepted": False,
            "reason": "missing_ego_vehicle",
            "scene_mode": scene_mode,
            "sampling_strategy": get_highway_sampling_strategy(scene_mode),
        }
    if len(llm_entries) < 2:
        return {
            "accepted": False,
            "reason": "missing_llm_agents",
            "llm_entries": llm_entries,
            "scene_mode": scene_mode,
            "sampling_strategy": get_highway_sampling_strategy(scene_mode),
        }

    role_map = assign_roles_from_geometry(geometry)
    blocker_id = next((veh_id for veh_id, role in role_map.items() if role == "Blocker"), "")
    striker_id = next((veh_id for veh_id, role in role_map.items() if role == "Striker"), "")
    blocker = geometry.get(blocker_id) or {}
    striker = geometry.get(striker_id) or {}
    if not blocker or not striker:
        return {
            "accepted": False,
            "reason": "role_assignment_failed",
            "role_map": role_map,
            "scene_mode": scene_mode,
            "sampling_strategy": get_highway_sampling_strategy(scene_mode),
        }

    if is_three_car_fixed_like_mode(scene_mode):
        blocker_range = SCENE_THREE_CAR_FIXED_BLOCKER_REL_X_RANGE
        striker_range = SCENE_THREE_CAR_FIXED_STRIKER_REL_X_RANGE
    else:
        blocker_range = SCENE_THREE_CAR_RANDOM_BLOCKER_REL_X_RANGE
        striker_range = SCENE_THREE_CAR_RANDOM_STRIKER_REL_X_RANGE

    blocker_rel_x = float(blocker["rel_x_to_ego"])
    striker_rel_x = float(striker["rel_x_to_ego"])
    blocker_rel_lane = int(blocker["rel_lane_to_ego"])
    striker_rel_lane = int(striker["rel_lane_to_ego"])
    striker_speed_adv = float(striker["speed"]) - float(ego_payload["speed"])
    projected_striker_rel_x = striker_rel_x + striker_speed_adv * SCENE_THREE_CAR_GATE_PROGRESS_FACTOR

    blocker_lane_ok = abs(blocker_rel_lane) == 1
    striker_lane_ok = abs(striker_rel_lane) == 1
    opposite_sides_ok = blocker_rel_lane * striker_rel_lane < 0
    blocker_x_ok = blocker_range[0] <= blocker_rel_x <= blocker_range[1]
    striker_x_ok = striker_range[0] <= striker_rel_x <= striker_range[1]
    striker_speed_adv_ok = striker_speed_adv >= SCENE_THREE_CAR_STRIKER_SPEED_ADV_RANGE[0]
    cut_in_window_ok = projected_striker_rel_x >= SCENE_THREE_CAR_STRIKER_GATE_REL_X_MAX
    key_spacing_ok = _key_vehicle_spacing_ok(
        geometry,
        ["ego_0", blocker_id, striker_id],
        min_gap=SCENE_THREE_CAR_MIN_KEY_SPACING,
    )

    accepted = all((
        blocker_lane_ok,
        striker_lane_ok,
        opposite_sides_ok,
        blocker_x_ok,
        striker_x_ok,
        striker_speed_adv_ok,
        cut_in_window_ok,
        key_spacing_ok,
    ))

    reason = "accepted"
    if not blocker_lane_ok:
        reason = "blocker_lane_not_adjacent"
    elif not striker_lane_ok:
        reason = "striker_lane_not_adjacent"
    elif not opposite_sides_ok:
        reason = "llm_not_on_opposite_sides"
    elif not blocker_x_ok:
        reason = "blocker_rel_x_out_of_band"
    elif not striker_x_ok:
        reason = "striker_rel_x_out_of_band"
    elif not striker_speed_adv_ok:
        reason = "striker_speed_adv_too_small"
    elif not cut_in_window_ok:
        reason = "striker_cut_in_window_too_far"
    elif not key_spacing_ok:
        reason = "critical_vehicles_too_aligned"

    return {
        "accepted": accepted,
        "reason": reason,
        "scene_mode": scene_mode,
        "sampling_strategy": get_highway_sampling_strategy(scene_mode),
        "role_map": dict(role_map),
        "blocker_id": blocker_id,
        "striker_id": striker_id,
        "blocker_rel_x": round(blocker_rel_x, 3),
        "striker_rel_x": round(striker_rel_x, 3),
        "blocker_rel_lane": blocker_rel_lane,
        "striker_rel_lane": striker_rel_lane,
        "striker_speed_adv": round(striker_speed_adv, 3),
        "projected_striker_rel_x": round(projected_striker_rel_x, 3),
        "background_nearby_count": 0,
    }


def scene_gate_metrics(geometry, scene_mode=None):
    resolved_mode = _resolve_scene_mode(scene_mode=scene_mode, geometry=geometry)
    if resolved_mode == SCENE_MODE_TRAFFIC:
        return _traffic_scene_gate_metrics(geometry)
    return _three_car_scene_gate_metrics(geometry, resolved_mode)


def _vehicle_catalog(env):
    catalog = {}
    for veh_id in env.initial_ids:
        if veh_id in env.initial_state:
            type_id, edge, lane, pos, speed = env.initial_state[veh_id]
        else:
            type_id = env.initial_vehicles.get_type(veh_id)
            edge, lane, pos, speed = SCENE_EDGE_ID, 0, 0.0, 0.0
        catalog[veh_id] = {
            "type_id": type_id,
            "edge": str(edge or SCENE_EDGE_ID),
            "lane": int(lane),
            "pos": float(pos),
            "speed": float(speed),
        }
    return catalog


def _random_speed(rng, low, high):
    return round(float(rng.uniform(low, high)), 3)


def _sample_human_slot(rng, base_x, ranges):
    low, high = rng.choice(ranges)
    return base_x + rng.uniform(low, high)


def _is_lane_slot_open(assigned, lane, pos, min_gap=SCENE_MIN_SAME_LANE_GAP):
    lane_load = 0
    for payload in assigned.values():
        if int(payload["lane"]) != int(lane):
            continue
        lane_load += 1
        if abs(float(payload["pos"]) - float(pos)) < min_gap:
            return False
    if lane_load >= SCENE_MAX_LANE_LOAD:
        return False
    return True


def _sample_vehicle_lane(rng, preferred_lane, lane_count):
    candidates = list(range(lane_count))
    rng.shuffle(candidates)
    ordered = [preferred_lane] + [lane for lane in candidates if lane != preferred_lane]
    return ordered


def _normalize_lane_list(lanes, lane_count):
    normalized = []
    for lane in lanes:
        if 0 <= int(lane) < lane_count and int(lane) not in normalized:
            normalized.append(int(lane))
    return normalized


def _lane_loads(assigned):
    loads = {}
    for payload in assigned.values():
        lane = int(payload["lane"])
        loads[lane] = loads.get(lane, 0) + 1
    return loads


def _ordered_lane_choices(preferred_lanes, lane_count, assigned):
    preferred = _normalize_lane_list(preferred_lanes, lane_count)
    lane_loads = _lane_loads(assigned)
    remaining = [lane for lane in range(lane_count) if lane not in preferred]
    remaining.sort(
        key=lambda lane: (
            lane_loads.get(lane, 0),
            min(abs(lane - pref) for pref in preferred) if preferred else lane,
            lane,
        )
    )
    return preferred + remaining


def _human_slot_specs(ego_lane, blocker_lane, striker_lane, lane_count):
    center_partner = 1 if ego_lane == 2 else 2
    outer_lanes = [lane for lane in range(lane_count) if lane not in (ego_lane, center_partner)]
    blocker_support = blocker_lane if blocker_lane != ego_lane else center_partner
    striker_support = striker_lane if striker_lane != ego_lane else center_partner
    lead_outer = outer_lanes[0] if outer_lanes else blocker_support
    trail_outer = outer_lanes[-1] if outer_lanes else striker_support
    return [
        {"rel_range": (-92.0, -70.0), "preferred_lanes": [trail_outer, center_partner]},
        {"rel_range": (-54.0, -34.0), "preferred_lanes": [striker_support, trail_outer]},
        {"rel_range": (18.0, 32.0), "preferred_lanes": [blocker_lane, lead_outer]},
        {"rel_range": (44.0, 62.0), "preferred_lanes": [lead_outer, blocker_support]},
        {"rel_range": (-124.0, -96.0), "preferred_lanes": [center_partner, trail_outer]},
        {"rel_range": (74.0, 96.0), "preferred_lanes": [center_partner, lead_outer]},
        {"rel_range": (-166.0, -136.0), "preferred_lanes": outer_lanes or [0]},
        {"rel_range": (126.0, 156.0), "preferred_lanes": list(reversed(outer_lanes)) or [lane_count - 1]},
    ]


def _build_layout_from_assigned(env, assigned):
    if any(veh_id not in assigned for veh_id in env.initial_ids):
        return None

    initial_state = {}
    start_positions = []
    start_lanes = []
    for veh_id in env.initial_ids:
        payload = assigned[veh_id]
        initial_state[veh_id] = (
            payload["type_id"],
            payload["edge"],
            int(payload["lane"]),
            float(payload["pos"]),
            float(payload["speed"]),
        )
        start_positions.append((payload["edge"], float(payload["pos"])))
        start_lanes.append(int(payload["lane"]))

    return {
        "initial_state": initial_state,
        "start_positions": start_positions,
        "start_lanes": start_lanes,
        "geometry": _geometry_from_initial_state(initial_state, env.initial_ids),
    }


def _assign_remaining_background(env, catalog, assigned, ego_pos, lane_count, road_length):
    remaining_ids = [veh_id for veh_id in env.initial_ids if veh_id not in assigned]
    if not remaining_ids:
        return True

    lane_cycle = [lane for lane in range(max(1, lane_count))]
    ego_speed = float(assigned["ego_0"]["speed"])
    for idx, veh_id in enumerate(remaining_ids):
        preferred_lane = lane_cycle[idx % len(lane_cycle)]
        lane_options = _ordered_lane_choices([preferred_lane], lane_count, assigned)
        selected = None
        for lane in lane_options:
            for extra in range(len(SCENE_THREE_CAR_BACKGROUND_OFFSETS)):
                rel_x = SCENE_THREE_CAR_BACKGROUND_OFFSETS[(idx + extra) % len(SCENE_THREE_CAR_BACKGROUND_OFFSETS)]
                pos = round(float(ego_pos) + float(rel_x), 3)
                if pos <= 24.0 or pos >= road_length - 24.0:
                    continue
                if _is_lane_slot_open(assigned, lane, pos, min_gap=max(SCENE_MIN_SAME_LANE_GAP, 22.0)):
                    selected = (lane, pos)
                    break
            if selected is not None:
                break

        if selected is None:
            return False

        lane, pos = selected
        assigned[veh_id] = {
            "type_id": catalog[veh_id]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(lane),
            "pos": float(pos),
            "speed": _random_speed(
                random.Random(idx + 17),
                max(14.0, ego_speed - 4.0),
                ego_speed + 0.2,
            ),
        }
    return True


def generate_custom_highway_opening(env, rng=None):
    rng = rng or random.Random()
    catalog = _vehicle_catalog(env)
    lane_count = int(env.k.network.num_lanes(SCENE_EDGE_ID))
    road_length = float(env.k.network.edge_length(SCENE_EDGE_ID))
    human_ids = [veh_id for veh_id in env.initial_ids if veh_id.startswith("human_")]

    for _ in range(24):
        base_x = float(rng.uniform(*SCENE_BASE_X_RANGE))
        ego_lane = int(rng.choice(SCENE_EGO_LANES))
        ego_pos = round(base_x, 3)
        assigned = {}
        ego_speed = _random_speed(rng, 19.5, 22.0)

        assigned["ego_0"] = {
            "type_id": catalog["ego_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": ego_lane,
            "pos": ego_pos,
            "speed": ego_speed,
        }

        blocker_lanes = [lane for lane in (ego_lane - 1, ego_lane, ego_lane + 1) if 0 <= lane < lane_count]
        if ego_lane in blocker_lanes and rng.random() < SCENE_BLOCKER_SAME_LANE_PROB:
            blocker_lane = int(ego_lane)
        else:
            alt_lanes = [lane for lane in blocker_lanes if lane != ego_lane]
            blocker_lane = int(rng.choice(alt_lanes or blocker_lanes))
        blocker_pos = round(ego_pos + rng.uniform(*SCENE_BLOCKER_REL_X_RANGE), 3)
        assigned["llm_0"] = {
            "type_id": catalog["llm_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": blocker_lane,
            "pos": blocker_pos,
            "speed": _random_speed(rng, max(17.5, ego_speed - 2.0), ego_speed + 0.1),
        }

        striker_lanes = [lane for lane in (ego_lane - 1, ego_lane, ego_lane + 1) if 0 <= lane < lane_count]
        rng.shuffle(striker_lanes)
        striker_pos = None
        striker_lane = ego_lane
        for lane in striker_lanes:
            pos = round(ego_pos + rng.uniform(*SCENE_STRIKER_REL_X_RANGE), 3)
            if abs(pos - ego_pos) < 3.5:
                pos = round(ego_pos - rng.uniform(4.0, 9.0), 3)
            if _is_lane_slot_open(assigned, lane, pos):
                striker_lane = int(lane)
                striker_pos = pos
                break
        if striker_pos is None:
            continue
        assigned["llm_1"] = {
            "type_id": catalog["llm_1"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": striker_lane,
            "pos": striker_pos,
            "speed": _random_speed(rng, max(18.0, ego_speed - 1.5), ego_speed + 1.0),
        }

        generation_failed = False
        slot_specs = _human_slot_specs(ego_lane, blocker_lane, striker_lane, lane_count)
        for human_id, slot_spec in zip(human_ids, slot_specs):
            lane_options = _ordered_lane_choices(slot_spec["preferred_lanes"], lane_count, assigned)
            selected = None
            for lane in lane_options:
                for _ in range(12):
                    rel_range = slot_spec["rel_range"]
                    pos = round(ego_pos + rng.uniform(rel_range[0], rel_range[1]), 3)
                    if pos <= 20.0 or pos >= road_length - 20.0:
                        continue
                    if _is_lane_slot_open(assigned, lane, pos):
                        selected = (lane, pos)
                        break
                if selected is not None:
                    break

            if selected is None:
                fallback_ranges = [
                    (-170.0, -138.0),
                    (-124.0, -96.0),
                    (-92.0, -64.0),
                    (-56.0, -34.0),
                    (20.0, 36.0),
                    (44.0, 68.0),
                    (76.0, 102.0),
                    (118.0, 154.0),
                ]
                for lane in _ordered_lane_choices(slot_spec["preferred_lanes"], lane_count, assigned):
                    fallback_pos = round(
                        _sample_human_slot(rng, ego_pos, fallback_ranges),
                        3,
                    )
                    if 20.0 < fallback_pos < road_length - 20.0 and _is_lane_slot_open(assigned, lane, fallback_pos):
                        selected = (lane, fallback_pos)
                        break

            if selected is None:
                generation_failed = True
                break

            lane, pos = selected
            assigned[human_id] = {
                "type_id": catalog[human_id]["type_id"],
                "edge": SCENE_EDGE_ID,
                "lane": int(lane),
                "pos": float(pos),
                "speed": _random_speed(rng, max(14.0, ego_speed - 5.0), ego_speed + 0.3),
            }

        if generation_failed:
            continue

        layout = _build_layout_from_assigned(env, assigned)
        if layout is not None and scene_gate_metrics(layout["geometry"], scene_mode=SCENE_MODE_TRAFFIC)["accepted"]:
            return layout

    return None


def generate_three_car_fixed_opening(env, rng=None):
    del rng
    catalog = _vehicle_catalog(env)
    lane_count = int(env.k.network.num_lanes(SCENE_EDGE_ID))
    road_length = float(env.k.network.edge_length(SCENE_EDGE_ID))
    if lane_count < 3 or "ego_0" not in catalog or "llm_0" not in catalog or "llm_1" not in catalog:
        return None

    ego_lane = 1 if lane_count > 2 else 0
    ego_pos = min(max(0.18 * road_length, 40.0), road_length - 180.0)
    ego_speed = 20.5
    blocker_lane = ego_lane + 1
    striker_lane = ego_lane - 1
    blocker_rel_x = sum(SCENE_THREE_CAR_FIXED_BLOCKER_REL_X_RANGE) / 2.0
    striker_rel_x = sum(SCENE_THREE_CAR_FIXED_STRIKER_REL_X_RANGE) / 2.0
    striker_speed_adv = sum(SCENE_THREE_CAR_FIXED_STRIKER_SPEED_ADV_RANGE) / 2.0

    assigned = {
        "ego_0": {
            "type_id": catalog["ego_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(ego_lane),
            "pos": float(ego_pos),
            "speed": float(ego_speed),
        },
        "llm_0": {
            "type_id": catalog["llm_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(blocker_lane),
            "pos": float(ego_pos + blocker_rel_x),
            "speed": float(ego_speed + 0.8),
        },
        "llm_1": {
            "type_id": catalog["llm_1"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(striker_lane),
            "pos": float(ego_pos + striker_rel_x),
            "speed": float(ego_speed + striker_speed_adv),
        },
    }
    if not _assign_remaining_background(env, catalog, assigned, ego_pos, lane_count, road_length):
        return None

    layout = _build_layout_from_assigned(env, assigned)
    if layout is None:
        return None
    gate = scene_gate_metrics(layout["geometry"], scene_mode=SCENE_MODE_THREE_CAR_FIXED)
    if gate["accepted"]:
        return layout
    return None


def generate_three_car_random_opening(env, rng=None):
    rng = rng or random.Random()
    catalog = _vehicle_catalog(env)
    lane_count = int(env.k.network.num_lanes(SCENE_EDGE_ID))
    road_length = float(env.k.network.edge_length(SCENE_EDGE_ID))
    if lane_count < 3 or "ego_0" not in catalog or "llm_0" not in catalog or "llm_1" not in catalog:
        return None

    for _ in range(24):
        ego_lane = int(rng.choice(SCENE_EGO_LANES))
        if ego_lane - 1 < 0 or ego_lane + 1 >= lane_count:
            continue
        ego_pos = round(float(rng.uniform(*SCENE_BASE_X_RANGE)), 3)
        ego_speed = _random_speed(rng, 19.5, 22.0)
        blocker_side = int(rng.choice((-1, 1)))
        striker_side = -blocker_side
        blocker_lane = ego_lane + blocker_side
        striker_lane = ego_lane + striker_side
        blocker_rel_x = rng.uniform(*SCENE_THREE_CAR_RANDOM_BLOCKER_REL_X_RANGE)
        striker_rel_x = rng.uniform(*SCENE_THREE_CAR_RANDOM_STRIKER_REL_X_RANGE)

        assigned = {
            "ego_0": {
                "type_id": catalog["ego_0"]["type_id"],
                "edge": SCENE_EDGE_ID,
                "lane": int(ego_lane),
                "pos": float(ego_pos),
                "speed": float(ego_speed),
            },
            "llm_0": {
                "type_id": catalog["llm_0"]["type_id"],
                "edge": SCENE_EDGE_ID,
                "lane": int(blocker_lane),
                "pos": round(float(ego_pos + blocker_rel_x), 3),
                "speed": _random_speed(
                    rng,
                    ego_speed + SCENE_THREE_CAR_BLOCKER_SPEED_ADV_RANGE[0],
                    ego_speed + SCENE_THREE_CAR_BLOCKER_SPEED_ADV_RANGE[1],
                ),
            },
            "llm_1": {
                "type_id": catalog["llm_1"]["type_id"],
                "edge": SCENE_EDGE_ID,
                "lane": int(striker_lane),
                "pos": round(float(ego_pos + striker_rel_x), 3),
                "speed": _random_speed(
                    rng,
                    ego_speed + SCENE_THREE_CAR_STRIKER_SPEED_ADV_RANGE[0],
                    ego_speed + SCENE_THREE_CAR_STRIKER_SPEED_ADV_RANGE[1],
                ),
            },
        }
        if not _assign_remaining_background(env, catalog, assigned, ego_pos, lane_count, road_length):
            continue

        layout = _build_layout_from_assigned(env, assigned)
        if layout is None:
            continue
        gate = scene_gate_metrics(layout["geometry"], scene_mode=SCENE_MODE_THREE_CAR_RANDOM)
        if gate["accepted"]:
            return layout
    return None


def generate_spawn_safe_fallback_opening(env):
    """Build a deterministic, spawn-stable tactical opening as final fallback."""
    catalog = _vehicle_catalog(env)
    lane_count = int(env.k.network.num_lanes(SCENE_EDGE_ID))
    road_length = float(env.k.network.edge_length(SCENE_EDGE_ID))
    human_ids = [veh_id for veh_id in env.initial_ids if veh_id.startswith("human_")]

    if lane_count <= 0 or "ego_0" not in catalog or "llm_0" not in catalog or "llm_1" not in catalog:
        return None

    ego_lane = 1 if lane_count > 1 else 0
    ego_pos = min(max(0.18 * road_length, 40.0), road_length - 180.0)
    ego_speed = 20.5

    blocker_lane = ego_lane
    striker_lane = ego_lane - 1 if ego_lane - 1 >= 0 else min(lane_count - 1, ego_lane + 1)

    assigned = {
        "ego_0": {
            "type_id": catalog["ego_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(ego_lane),
            "pos": float(ego_pos),
            "speed": float(ego_speed),
        },
        "llm_0": {
            "type_id": catalog["llm_0"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(blocker_lane),
            "pos": float(ego_pos + 24.0),
            "speed": 19.3,
        },
        "llm_1": {
            "type_id": catalog["llm_1"]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(striker_lane),
            "pos": float(ego_pos - 12.0),
            "speed": 21.2,
        },
    }

    lane_cycle = [lane for lane in range(lane_count)]
    rel_offsets = [-150.0, -118.0, -86.0, -54.0, 36.0, 72.0, 108.0, 146.0]
    for idx, human_id in enumerate(human_ids):
        lane = lane_cycle[idx % max(1, len(lane_cycle))]
        pos = float(ego_pos + rel_offsets[idx % len(rel_offsets)])
        if pos <= 24.0 or pos >= road_length - 24.0:
            continue
        if not _is_lane_slot_open(assigned, lane, pos, min_gap=max(SCENE_MIN_SAME_LANE_GAP, 22.0)):
            continue
        assigned[human_id] = {
            "type_id": catalog[human_id]["type_id"],
            "edge": SCENE_EDGE_ID,
            "lane": int(lane),
            "pos": float(pos),
            "speed": 18.5,
        }

    if len(assigned) != len(env.initial_ids):
        return None

    initial_state = {}
    start_positions = []
    start_lanes = []
    for veh_id in env.initial_ids:
        payload = assigned[veh_id]
        initial_state[veh_id] = (
            payload["type_id"],
            payload["edge"],
            int(payload["lane"]),
            float(payload["pos"]),
            float(payload["speed"]),
        )
        start_positions.append((payload["edge"], float(payload["pos"])))
        start_lanes.append(int(payload["lane"]))

    geometry = _geometry_from_initial_state(initial_state, env.initial_ids)
    if not scene_gate_metrics(geometry, scene_mode=SCENE_MODE_TRAFFIC).get("accepted", False):
        return None

    return {
        "initial_state": initial_state,
        "start_positions": start_positions,
        "start_lanes": start_lanes,
        "geometry": geometry,
    }


def _geometry_from_initial_state(initial_state, initial_ids):
    geometry = {}
    _, ego_edge, ego_lane, ego_pos, ego_speed = initial_state["ego_0"]
    ego_lane = int(ego_lane)
    ego_pos = float(ego_pos)
    ego_speed = float(ego_speed)

    for veh_id in initial_ids:
        type_id, edge, lane, pos, speed = initial_state[veh_id]
        geometry[veh_id] = {
            "veh_id": veh_id,
            "type_id": type_id,
            "edge": edge,
            "lane": int(lane),
            "pos": float(pos),
            "speed": float(speed),
            "ego_edge": ego_edge,
            "ego_lane": ego_lane,
            "ego_pos": ego_pos,
            "ego_speed": ego_speed,
            "rel_lane_to_ego": int(lane) - ego_lane,
            "rel_x_to_ego": float(pos) - ego_pos,
            "rel_speed_to_ego": float(speed) - ego_speed,
        }
    return geometry


def _apply_scene_layout(env, layout):
    env.initial_state = copy_initial_state(layout["initial_state"])
    for config in (env.initial_config, env.network.initial_config):
        config.shuffle = False
        config.spacing = "custom"
        config.additional_params["start_positions"] = list(layout["start_positions"])
        config.additional_params["start_lanes"] = list(layout["start_lanes"])


def copy_initial_state(initial_state):
    copied = {}
    for veh_id, payload in initial_state.items():
        type_id, edge, lane, pos, speed = payload
        copied[veh_id] = (type_id, edge, int(lane), float(pos), float(speed))
    return copied


def freeze_initial_scene(env, run_started_at, frozen_geometry, role_map, scene_gate_status):
    current_ids = list(env.initial_ids)
    frozen_initial_state = {}
    start_positions = []
    start_lanes = []
    signature_parts = []

    for veh_id in current_ids:
        type_id, edge, lane, pos, speed = env.initial_state[veh_id]
        lane = int(lane)
        pos = float(pos)
        speed = float(speed)

        frozen_initial_state[veh_id] = (type_id, edge, lane, pos, speed)
        start_positions.append((edge, pos))
        start_lanes.append(lane)
        signature_parts.append(
            "{}:{}:{}:{:.2f}:{:.2f}".format(veh_id, edge, lane, pos, speed)
        )

    env.initial_ids = list(current_ids)
    env.initial_state = frozen_initial_state

    for config in (env.initial_config, env.network.initial_config):
        config.shuffle = False
        config.spacing = "custom"
        config.additional_params["start_positions"] = list(start_positions)
        config.additional_params["start_lanes"] = list(start_lanes)

    scenario_id = "{}_{}".format(run_started_at, abs(hash("|".join(signature_parts))))
    scene_gate_snapshot = deepcopy(scene_gate_status or {})
    scene_gate_snapshot.pop("role_map", None)
    blocker_id = next((veh_id for veh_id, role in role_map.items() if role == "Blocker"), "")
    striker_id = next((veh_id for veh_id, role in role_map.items() if role == "Striker"), "")
    return {
        "scenario_id": scenario_id,
        "frozen_geometry": frozen_geometry,
        "role_map": {},
        "role_source": "",
        "geometry_role_hint": dict(role_map),
        "sampling_strategy": scene_gate_status.get(
            "sampling_strategy",
            get_highway_sampling_strategy(scene_gate_status.get("scene_mode", "")),
        ),
        "scene_gate_status": scene_gate_snapshot,
        "key_layout": {
            "ego_id": "ego_0",
            "blocker_id": blocker_id,
            "striker_id": striker_id,
            "blocker_rel_x": frozen_geometry.get(blocker_id, {}).get("rel_x_to_ego"),
            "striker_rel_x": frozen_geometry.get(striker_id, {}).get("rel_x_to_ego"),
        },
    }


def sample_and_freeze_scene(env, run_started_at):
    scene_mode = get_highway_scene_mode()
    sampling_strategy = get_highway_sampling_strategy(scene_mode)
    last_gate = None
    emission_path = env.sim_params.emission_path
    sim_emission_path = getattr(env.k.simulation, "emission_path", None)
    env.sim_params.emission_path = None
    if hasattr(env.k.simulation, "emission_path"):
        env.k.simulation.emission_path = None
    try:
        for sample_index in range(SCENE_GATE_SAMPLE_BUDGET):
            if scene_mode == SCENE_MODE_TRAFFIC:
                layout = generate_custom_highway_opening(env, rng=random.Random())
            elif is_three_car_fixed_like_mode(scene_mode):
                layout = generate_three_car_fixed_opening(env)
            else:
                layout = generate_three_car_random_opening(env, rng=random.Random())
            if layout is None:
                last_gate = {
                    "accepted": False,
                    "reason": "scene_layout_generation_failed",
                    "sample_index": sample_index + 1,
                    "scene_mode": scene_mode,
                    "sampling_strategy": sampling_strategy,
                }
                continue

            gate = scene_gate_metrics(layout["geometry"], scene_mode=scene_mode)
            gate["sample_index"] = sample_index + 1
            if not gate["accepted"]:
                last_gate = gate
                continue

            _apply_scene_layout(env, layout)
            try:
                env.reset()
            except FatalFlowError as exc:
                last_gate = {
                    "accepted": False,
                    "reason": "scene_reset_spawn_failed",
                    "sample_index": sample_index + 1,
                    "scene_mode": scene_mode,
                    "sampling_strategy": sampling_strategy,
                    "error": str(exc).strip(),
                    "layout_role_map": gate.get("role_map", {}),
                }
                env.k.vehicle = deepcopy(env.initial_vehicles)
                env.k.vehicle.master_kernel = env.k
                env.restart_simulation(env.sim_params)
                if hasattr(env.k.simulation, "emission_path"):
                    env.k.simulation.emission_path = None
                continue
            geometry = build_initial_geometry(env)
            frozen_gate = scene_gate_metrics(geometry, scene_mode=scene_mode)
            frozen_gate["sample_index"] = sample_index + 1
            if frozen_gate["accepted"]:
                role_map = frozen_gate.get("role_map", {}) or assign_roles_from_geometry(geometry)
                return freeze_initial_scene(
                    env,
                    run_started_at,
                    geometry,
                    role_map,
                    frozen_gate,
                )
            last_gate = frozen_gate
    finally:
        env.sim_params.emission_path = emission_path
        if hasattr(env.k.simulation, "emission_path"):
            env.k.simulation.emission_path = sim_emission_path

    if scene_mode == SCENE_MODE_TRAFFIC:
        fallback_layout = generate_spawn_safe_fallback_opening(env)
    else:
        fallback_layout = generate_three_car_fixed_opening(env)
    if fallback_layout is not None:
        fallback_gate = scene_gate_metrics(fallback_layout["geometry"], scene_mode=scene_mode)
        fallback_gate["sample_index"] = SCENE_GATE_SAMPLE_BUDGET + 1
        _apply_scene_layout(env, fallback_layout)
        try:
            env.reset()
            frozen_geometry = build_initial_geometry(env)
            frozen_gate = scene_gate_metrics(frozen_geometry, scene_mode=scene_mode)
            frozen_gate["sample_index"] = SCENE_GATE_SAMPLE_BUDGET + 1
            if frozen_gate["accepted"]:
                role_map = frozen_gate.get("role_map", {}) or assign_roles_from_geometry(frozen_geometry)
                frozen_gate["reason"] = "accepted_fallback_layout"
                return freeze_initial_scene(
                    env,
                    run_started_at,
                    frozen_geometry,
                    role_map,
                    frozen_gate,
                )
            last_gate = frozen_gate
        except FatalFlowError as exc:
            last_gate = {
                "accepted": False,
                "reason": "scene_reset_spawn_failed_fallback",
                "sample_index": SCENE_GATE_SAMPLE_BUDGET + 1,
                "scene_mode": scene_mode,
                "sampling_strategy": sampling_strategy,
                "error": str(exc).strip(),
            }

    return {
        "scenario_id": "",
        "frozen_geometry": {},
        "role_map": {},
        "role_source": "",
        "geometry_role_hint": {},
        "sampling_strategy": sampling_strategy,
        "scene_gate_status": {
            "accepted": False,
            "reason": "scene_sampling_failed",
            "sample_budget": SCENE_GATE_SAMPLE_BUDGET,
            "scene_mode": scene_mode,
            "sampling_strategy": sampling_strategy,
            "last_gate": last_gate,
        },
    }
