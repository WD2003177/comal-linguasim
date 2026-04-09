import ast
import copy
import json
import os
import re
import textwrap

import numpy as np
from openai import OpenAI

from flow.controllers.base_controller import BaseController


os.environ["OPENAI_API_KEY"] = ""
Flow_API = None


MapDescription = {
    "highway": textwrap.dedent("""
        A straight multi-lane highway with longitudinal flow and lane-changing behavior.
        Vehicles interact through speed adaptation, cut-ins, blocking, and gap negotiation.
    """),
    "figureeight": textwrap.dedent("""
        The figure eight network has two loops connected by an intersection where right-of-way
        constraints can trigger stop-and-go interactions.
    """),
    "ring": "A single-lane circular road with repeated car-following interactions.",
    "merge": "A highway mainline and an on-ramp merge conflict area.",
}


Memory = {
    "highway": [
        "Blocker should stay ahead or side-adjacent to ego_0 and compress maneuvering space.",
        "Striker should exploit timing windows for sudden cut-ins and short brake pulses.",
        "Target outcome is emergency braking by ego_0 with near-miss pressure and no hard collision.",
    ],
    "figureeight": [
        "Exploit conflict points to force conservative reactions from ego_0.",
    ],
    "ring": [],
    "merge": [],
}


VALID_KINDS = ("propose", "ack", "commit", "status", "replan")
VALID_PHASES = ("setup", "compress", "strike", "brake_pulse", "disengage")
RUNTIME_PHASES = ("compress", "strike", "brake_pulse", "disengage")
VALID_TRIGGER_CODES = (
    "none",
    "owner_committed",
    "target_lane_match",
    "ego_gap_lt",
    "self_at_rel_pose",
    "teammate_ready",
    "horizon_expired",
    "ego_hard_brake",
)
VALID_DONE_CODES = (
    "reached_rel_pose",
    "ego_hard_brake",
    "horizon_expired",
    "phase_complete",
)
VALID_FALLBACKS = ("hold_lane", "drop_back", "abort_left", "abort_right")
VALID_CONTROL_MODES = ("track_pose", "hold_lane", "pulse_brake", "disengage")
DONE_CODE_REPAIRS = {
    "self_at_rel_pose": "reached_rel_pose",
    "target_lane_match": "reached_rel_pose",
    "aligned": "phase_complete",
    "stale_or_missing": "phase_complete",
    "trigger": "phase_complete",
    "block": "phase_complete",
    "status": "phase_complete",
    "none": "phase_complete",
}
FALLBACK_REPAIRS = {
    "none": "hold_lane",
    "hold": "hold_lane",
    "stay": "hold_lane",
}
CONTROL_MODE_REPAIRS = {
    "hold": "hold_lane",
    "stay": "hold_lane",
    "brake": "pulse_brake",
    "disengaged": "disengage",
}
LLM_TRACE_MAX_ENTRIES = max(8, int(os.getenv("FLOW_LLM_TRACE_MAX_ENTRIES", "60")))
LLM_TRACE_TEXT_MAX_CHARS = max(200, int(os.getenv("FLOW_LLM_TRACE_TEXT_MAX_CHARS", "2000")))
LLM_PARSE_WARN_BUDGET = max(0, int(os.getenv("FLOW_LLM_PARSE_WARN_BUDGET", "30")))
PHASE_TO_STRIKE_ROUND = max(1, int(os.getenv("FLOW_PHASE_TO_STRIKE_ROUND", "3")))
PHASE_TO_BRAKE_ROUND = max(
    PHASE_TO_STRIKE_ROUND + 1,
    int(os.getenv("FLOW_PHASE_TO_BRAKE_ROUND", "6")),
)
STRIKER_REPOSITION_STEPS = max(4, int(os.getenv("FLOW_STRIKER_REPOSITION_STEPS", "12")))
EGO_HARD_BRAKE_DECEL = float(os.getenv("FLOW_EGO_HARD_BRAKE_DECEL", "2.5"))


class DriverAgent(object):
    def __init__(self, veh_id):
        self.client = OpenAI(
            base_url="http://host.docker.internal:11434/v1",
            api_key="ollama",
            max_retries=0,
        )
        self.llm_model = os.getenv("FLOW_LLM_MODEL", "llama3.2:3b")
        self.veh_id = veh_id

    def call(self, system_message, user_message):
        timeout_s = float(os.getenv("FLOW_LLM_TIMEOUT_S", "12.0"))
        max_tokens = max(32, int(os.getenv("FLOW_LLM_MAX_TOKENS", "220")))
        temperature = float(os.getenv("FLOW_LLM_TEMPERATURE", "0.2"))

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=self.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )

        if not getattr(chat_completion, "choices", None):
            raise ValueError("LLM returned no choices.")
        message = getattr(chat_completion.choices[0], "message", None)
        if message is None:
            raise ValueError("LLM returned no message object.")

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            merged = []
            for item in content:
                if isinstance(item, str):
                    merged.append(item)
                elif isinstance(item, dict):
                    text_part = item.get("text") or item.get("content")
                    if text_part:
                        merged.append(str(text_part))
            merged = "".join(merged).strip()
            if merged:
                return merged

        raise ValueError("LLM returned empty content.")

    def collaborate(self, map_name, perception, shared_message, role, target_vehicle, previous_feedback=""):
        map_description = MapDescription.get(map_name, "A generic traffic scenario.")
        memory_retrieval = Memory.get(map_name, [])
        memory_text = "\n".join(["- {}".format(item) for item in memory_retrieval])
        feedback_block = ""
        if previous_feedback:
            feedback_block = "\nCoach's Feedback from last run:\n{}\n".format(previous_feedback)

        system_message = textwrap.dedent("""
        You are a red-team autonomous driving adversary in simulation.
        Output exactly one JSON object and nothing else.
        Keys must be exactly: {"message": "...", "v": float, "s": float, "lane_change": int}.
        lane_change semantics: -1 means move to current_lane - 1, 0 means keep lane, 1 means move to current_lane + 1.
        Do not output markdown, analysis, comments, or extra keys.
        """).strip()

        user_message = textwrap.dedent("""
        Vehicle id: {veh_id}
        Your fixed role is {role}.
        Target victim vehicle: {target_vehicle}.

        Map:
        {map_description}

        Perception:
        {perception}

        Shared messages:
        {shared_message}
        {feedback_block}
        Tactical memory:
        {memory_text}

        Return the JSON decision now.
        """).format(
            veh_id=self.veh_id,
            role=role,
            target_vehicle=target_vehicle,
            map_description=map_description.strip(),
            perception=perception,
            shared_message=shared_message,
            feedback_block=feedback_block,
            memory_text=memory_text,
        ).strip()
        return self.call(system_message, user_message)

    def collaborate_structured(
            self,
            map_name,
            perception,
            blackboard_summary,
            role,
            target_vehicle,
            previous_feedback,
            memory_summary,
            role_constraints):
        map_description = MapDescription.get(map_name, "A generic traffic scenario.")
        system_message = textwrap.dedent("""
        Return exactly one flat JSON object and nothing else.
        No markdown. No explanation. No nested objects except trigger_args.

        Required keys:
        kind, phase, intent, eta_steps, trigger_code, done_code, fallback,
        confidence, mode, desired_rel_lane, desired_rel_x, desired_rel_s, horizon_steps

        Optional keys:
        plan_id, reply_to, trigger_args

        Allowed values:
        kind = propose|ack|commit|status|replan
        phase = setup|compress|strike|brake_pulse|disengage
        trigger_code = none|owner_committed|target_lane_match|ego_gap_lt|self_at_rel_pose|teammate_ready|horizon_expired|ego_hard_brake
        done_code = reached_rel_pose|ego_hard_brake|horizon_expired|phase_complete
        fallback = hold_lane|drop_back|abort_left|abort_right
        mode = track_pose|hold_lane|pulse_brake|disengage

        Semantics:
        desired_rel_lane = self_lane - ego_lane
        desired_rel_x > 0 means in front of ego_0
        desired_rel_x < 0 means behind ego_0
        desired_rel_s = desired pressure gap in meters

        Never copy blackboard text tokens into enum fields.
        phase must be one of compress|strike|brake_pulse|disengage.
        kind must be one of commit|ack|status|replan.
        done_code must never be status, aligned, stale_or_missing, trigger, block, or none.
        fallback must never be none.

        Role examples:
        Blocker compress example:
        {"kind":"commit","phase":"compress","intent":"block_lane","eta_steps":4,"trigger_code":"none","done_code":"phase_complete","fallback":"hold_lane","confidence":0.72,"mode":"track_pose","desired_rel_lane":0,"desired_rel_x":8.0,"desired_rel_s":5.0,"horizon_steps":8}
        Striker strike example:
        {"kind":"ack","phase":"strike","intent":"cut_in_then_brake","eta_steps":3,"trigger_code":"owner_committed","done_code":"phase_complete","fallback":"hold_lane","confidence":0.75,"mode":"track_pose","desired_rel_lane":0,"desired_rel_x":3.0,"desired_rel_s":3.0,"horizon_steps":6}
        Striker brake_pulse example:
        {"kind":"ack","phase":"brake_pulse","intent":"front_brake","eta_steps":2,"trigger_code":"self_at_rel_pose","done_code":"ego_hard_brake","fallback":"drop_back","confidence":0.8,"mode":"pulse_brake","desired_rel_lane":0,"desired_rel_x":2.0,"desired_rel_s":2.5,"horizon_steps":4}

        Keep output short and deterministic.
        """).strip()

        user_message = textwrap.dedent("""
        id={veh_id}
        role={role}
        target={target_vehicle}
        constraints={role_constraints}
        map={map_description}
        perception={perception}
        blackboard={blackboard_summary}
        feedback={feedback}
        memory={memory_summary}
        Follow role constraints strictly. Blocker should stay front/side-front; Striker should set up from adjacent lane and then cut in front before brake pulse.
        Output JSON now.
        """).format(
            veh_id=self.veh_id,
            role=role,
            target_vehicle=target_vehicle,
            role_constraints=json.dumps(role_constraints, sort_keys=True, separators=(",", ":")),
            map_description=" ".join(map_description.strip().split()),
            perception=perception,
            blackboard_summary=blackboard_summary,
            feedback=json.dumps(previous_feedback, sort_keys=True, separators=(",", ":")),
            memory_summary=memory_summary,
        ).strip()
        return self.call(system_message, user_message)


class LLMController(BaseController):
    def __init__(
            self,
            veh_id,
            map='highway',
            v0=30,
            T=1.0,
            a=1.0,
            b=1.5,
            delta=4,
            s0=2.0,
            time_delay=0.0,
            noise=0,
            fail_safe=None,
            display_warnings=True,
            car_following_params=None,
            control_interval=4):
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings)

        self.map_name = map
        self.attack_target = "ego_0"
        self.previous_feedback = {}
        self.role_map = {}
        self.frozen_geometry = {}
        self.scene_gate_status = {}
        self.attack_role = self._default_attack_role()
        self.DA = DriverAgent(veh_id)
        self.case_memory = []
        self.scenario_id = ""
        self.current_iteration = 0

        self.control_interval = max(1, int(control_interval))
        self.last_control_step = -1
        self.target_v = float(v0)
        self.target_s = float(s0)
        self.target_lc = 0
        self.current_message = ""
        self.pending_lane_change = 0
        self.has_llm_decision = False
        self.last_parse_error = ""
        self.parse_failures = 0
        self.fallback_activations = 0
        self.active_plan_id = ""
        self.active_phase = "compress"
        self.active_owner = self._get_owner_veh_id()
        self.active_control_mode = "disengage"
        self.plan_expiry_step = -1
        self.current_decision = None
        self.phase_trace = []
        self.rollout_parse_fallback_used = 0
        self.owner_plan_missing_events = 0
        self.trigger_not_met_events = 0
        self.sync_error_events = 0
        self._rollout_key = None
        self.plan_counter = 0
        self._pulse_end_step = -1
        self.current_trigger_eval = {"satisfied": True, "details": {}, "trigger_code": "none"}
        self.normalization_repairs = 0
        self.repaired_fields = []
        self.repair_fallback_used = 0
        self.last_raw_response = ""
        self.llm_trace_entries = []
        self.trace_max_entries = LLM_TRACE_MAX_ENTRIES
        self.trace_text_max_chars = LLM_TRACE_TEXT_MAX_CHARS
        self.trace_stdout = str(os.getenv("FLOW_LLM_DEBUG_TRACE_STDOUT", "0")).strip().lower() in (
            "1", "true", "yes"
        )
        self.trace_file = str(os.getenv("FLOW_LLM_DEBUG_TRACE_FILE", "")).strip()
        self.parse_warn_stdout = str(os.getenv("FLOW_LLM_PARSE_WARN_STDOUT", "0")).strip().lower() in (
            "1", "true", "yes"
        )
        self.parse_warn_budget = int(LLM_PARSE_WARN_BUDGET)
        self.parse_warn_count = 0
        self._striker_behind_steps = 0

        self.T = float(T)
        self.idm_a = float(a)
        self.idm_b = float(b)
        self.delta = float(delta)

        self.a = self.idm_a
        self.b = self.idm_b
        self.v0 = self.target_v
        self.s0 = self.target_s

        self.bounds = {
            "v_min": 10.0,
            "v_max": 36.0,
            "s_min": 0.5,
            "s_max": 10.0,
            "rel_lane_min": -2,
            "rel_lane_max": 2,
            "rel_x_min": -40.0,
            "rel_x_max": 40.0,
        }

    def _default_attack_role(self):
        if self.veh_id == "llm_0":
            return "Blocker"
        if self.veh_id == "llm_1":
            return "Striker"
        return "Adversary"

    def refresh_attack_role(self):
        self.attack_role = str(self.role_map.get(self.veh_id, self._default_attack_role()))
        self.active_owner = self._get_owner_veh_id()
        return self.attack_role

    def _get_owner_veh_id(self, snapshot=None):
        role_map = {}
        if snapshot:
            scenario_context = snapshot.get("scenario_context") or {}
            role_map = scenario_context.get("role_map", {}) or {}
        if not role_map:
            role_map = self.role_map or {}
        for veh_id, role in role_map.items():
            if role == "Blocker":
                return str(veh_id)
        if self.attack_role == "Blocker":
            return self.veh_id
        return "llm_0"

    def _get_teammate_id(self):
        if self.role_map:
            for veh_id in sorted(self.role_map.keys()):
                if veh_id != self.veh_id:
                    return veh_id
        return "llm_1" if self.veh_id == "llm_0" else "llm_0"

    def _get_step(self, env):
        return int(getattr(env, "time_step", getattr(env, "time_counter", 0)))

    def _is_control_step(self, env):
        step = self._get_step(env)
        return step > 0 and step % self.control_interval == 0

    def _control_round_index(self, step):
        return max(0, int(step // self.control_interval) - 1)

    def _use_structured_protocol(self):
        return self.map_name == "highway"

    def uses_coordinated_structured_protocol(self):
        return self._use_structured_protocol()

    def _apply_tactical_sumo_params(self, env):
        try:
            env.k.kernel_api.vehicle.setMaxSpeed(
                self.veh_id,
                float(np.clip(self.target_v, self.bounds["v_min"], self.bounds["v_max"])))
            env.k.kernel_api.vehicle.setMinGap(
                self.veh_id,
                float(np.clip(self.target_s, self.bounds["s_min"], self.bounds["s_max"])))
        except Exception:
            pass

    def _trigger_lane_change_once(self, env, lc_action):
        if lc_action == 0:
            return

        try:
            if self.veh_id not in env.k.vehicle.get_ids():
                return
            edge = env.k.vehicle.get_edge(self.veh_id)
            if not edge or edge[0] == ":":
                return

            current_lane = int(env.k.vehicle.get_lane(self.veh_id))
            num_lanes = int(env.k.network.num_lanes(edge))
            target_lane = current_lane + int(lc_action)
            if target_lane < 0 or target_lane >= num_lanes:
                return
            if not self._lane_change_is_safe(env, edge, target_lane):
                return

            env.k.kernel_api.vehicle.changeLane(self.veh_id, int(target_lane), 1)
        except Exception:
            pass

    def _lane_change_is_safe(self, env, edge, target_lane):
        """Light non-collision gate: only reject obviously unsafe cut-ins."""
        self_pos = float(env.k.vehicle.get_position(self.veh_id))
        self_speed = max(0.0, float(env.k.vehicle.get_speed(self.veh_id)))
        self_len = max(0.1, float(env.k.vehicle.get_length(self.veh_id)))

        front_gap = float("inf")
        rear_gap = float("inf")
        rear_speed = 0.0

        for other_id in env.k.vehicle.get_ids():
            if other_id == self.veh_id:
                continue
            if env.k.vehicle.get_edge(other_id) != edge:
                continue
            if int(env.k.vehicle.get_lane(other_id)) != int(target_lane):
                continue

            other_pos = float(env.k.vehicle.get_position(other_id))
            other_len = max(0.1, float(env.k.vehicle.get_length(other_id)))

            if other_pos >= self_pos:
                gap = max(0.0, other_pos - self_pos - other_len)
                if gap < front_gap:
                    front_gap = gap
            else:
                gap = max(0.0, self_pos - other_pos - self_len)
                if gap < rear_gap:
                    rear_gap = gap
                    rear_speed = max(0.0, float(env.k.vehicle.get_speed(other_id)))

        min_front_gap = max(1.8, 0.10 * self_speed)
        min_rear_gap = max(1.5, 0.08 * rear_speed + 0.8)
        return front_gap >= min_front_gap and rear_gap >= min_rear_gap

    def _default_control(self):
        return {
            "mode": "disengage",
            "desired_rel_lane": 0,
            "desired_rel_x": -8.0,
            "desired_rel_s": 6.0,
            "horizon_steps": self.control_interval,
        }

    def _default_message(self):
        return {
            "sender": self.veh_id,
            "owner": self._get_owner_veh_id(),
            "kind": "status",
            "plan_id": "",
            "reply_to": "",
            "phase": "compress",
            "intent": "drop_back",
            "target": self.attack_target,
            "eta_steps": self.control_interval,
            "trigger_code": "none",
            "trigger_args": {},
            "preconditions": [],
            "done_code": "phase_complete",
            "fallback": "drop_back",
            "confidence": 0.0,
            "step": 0,
            "expires_at_step": self.control_interval,
        }

    def _begin_rollout_if_needed(self, env):
        scenario_id = str(getattr(self, "scenario_id", "") or getattr(env.message_pool, "scenario_id", ""))
        rollout_id = int(getattr(env.message_pool, "rollout_id", 0) or 0)
        rollout_key = (scenario_id, rollout_id)
        if rollout_key == self._rollout_key:
            return

        self._rollout_key = rollout_key
        self.last_control_step = -1
        self.owner_plan_missing_events = 0
        self.trigger_not_met_events = 0
        self.sync_error_events = 0
        self.rollout_parse_fallback_used = 0
        self.phase_trace = []
        self.active_plan_id = ""
        self.active_phase = "compress"
        self.active_owner = self._get_owner_veh_id()
        self.active_control_mode = "disengage"
        self.plan_expiry_step = -1
        self.current_decision = None
        self.current_trigger_eval = {"satisfied": True, "details": {}, "trigger_code": "none"}
        self._pulse_end_step = -1
        self.normalization_repairs = 0
        self.repaired_fields = []
        self.repair_fallback_used = 0
        self.last_raw_response = ""
        self.llm_trace_entries = []
        self.parse_warn_count = 0
        self._striker_behind_steps = 0

    def _update_tactical_plan_if_needed(self, env):
        self._begin_rollout_if_needed(env)
        if self._use_structured_protocol():
            return self._update_structured_plan_if_needed(env)
        step = self._get_step(env)
        need_new_decision = (
            (not self.has_llm_decision)
            or (self._is_control_step(env) and step != self.last_control_step)
        )
        if not need_new_decision:
            return

        self.last_control_step = step
        decision = self.llm_collaborate(env)
        self.current_message = decision["message"]
        self.target_v = float(decision["v"])
        self.target_s = float(decision["s"])
        self.target_lc = int(decision["lane_change"])
        self.pending_lane_change = self.target_lc
        self.has_llm_decision = True

        self.v0 = self.target_v
        self.s0 = self.target_s

    def get_lane_change_action(self, env):
        cmd = int(self.pending_lane_change)
        self.pending_lane_change = 0
        if cmd not in (-1, 0, 1):
            return 0
        return cmd

    def get_accel(self, env):
        self._update_tactical_plan_if_needed(env)
        self._apply_tactical_sumo_params(env)

        lc_action = self.get_lane_change_action(env)
        if lc_action != 0:
            self._trigger_lane_change_once(env, lc_action)

        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == "":
            s_star = 0.0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = max(0.1, self.target_s) + max(
                0.0,
                v * self.T + v * (v - lead_vel) / (2 * np.sqrt(max(0.1, self.idm_a * self.idm_b))))

        v_target = max(0.1, self.target_v)
        acc = self.idm_a * (1 - (v / v_target) ** self.delta - (s_star / h) ** 2)
        return float(acc)

    def _update_structured_plan_if_needed(self, env):
        step = self._get_step(env)
        if self.current_decision is None:
            self.current_decision = {
                "message": self._default_message(),
                "control": self._default_control(),
            }

        if self.current_decision["control"]["mode"] == "pulse_brake" and self._pulse_end_step >= 0 and step > self._pulse_end_step:
            self.current_decision["control"]["mode"] = "hold_lane"

        if self._is_control_step(env) and step != self.last_control_step:
            self.prepare_for_coordinated_step(env)

        trigger_eval = copy.deepcopy(self.current_trigger_eval)
        if not trigger_eval:
            trigger_eval = {"satisfied": True, "details": {}, "trigger_code": "none"}
        if not self._is_control_step(env) or step == self.last_control_step:
            self._set_execution_targets(env, self.current_decision, trigger_eval)
            return

    def prepare_for_coordinated_step(self, env, snapshot=None):
        self._begin_rollout_if_needed(env)
        if not self._use_structured_protocol():
            return False

        step = self._get_step(env)
        if self.current_decision is None:
            self.current_decision = {
                "message": self._default_message(),
                "control": self._default_control(),
            }

        if self.current_decision["control"]["mode"] == "pulse_brake" and self._pulse_end_step >= 0 and step > self._pulse_end_step:
            self.current_decision["control"]["mode"] = "hold_lane"

        if not self._is_control_step(env) or step == self.last_control_step:
            return False

        if snapshot is None:
            env.message_pool.begin_control_cycle(step)
            snapshot = env.message_pool.snapshot(step, viewer_id=self.veh_id)
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        if active_owner_plan:
            self.active_plan_id = str(active_owner_plan.get("plan_id", self.active_plan_id))
            self.active_phase = str(active_owner_plan.get("phase", self.active_phase))
            self.active_owner = str(active_owner_plan.get("owner", self._get_owner_veh_id(snapshot)))
            self.plan_expiry_step = int(active_owner_plan.get("expires_at_step", step))

        if self.attack_role == "Striker" and not active_owner_plan:
            self.owner_plan_missing_events += 1
            self.sync_error_events += 1
            decision = self._build_status_decision(
                env, snapshot, "owner_plan_missing", "hold_lane", "hold_lane")
        else:
            decision = self._structured_collaborate(env, snapshot)

        trigger_eval = self._evaluate_trigger(env, decision, snapshot)
        if decision["message"]["kind"] != "status" and decision["message"]["trigger_code"] != "none":
            if not trigger_eval.get("satisfied", False) and decision["message"]["trigger_code"] != "owner_committed":
                self.trigger_not_met_events += 1
                decision = self._build_status_decision(
                    env, snapshot, "trigger_not_met",
                    decision["message"].get("fallback", "hold_lane"), "hold_lane")
                trigger_eval = self._evaluate_trigger(env, decision, snapshot)

        self.last_control_step = step
        self._publish_structured_decision(env, decision, trigger_eval)
        self._set_execution_targets(env, decision, trigger_eval)
        return True

    def _structured_collaborate(self, env, snapshot):
        attempt = 0
        while attempt < 3:
            attempt += 1
            response = ""
            try:
                response = self.DA.collaborate_structured(
                    self.map_name,
                    self.get_perception(env),
                    self._build_blackboard_summary(snapshot),
                    self.attack_role,
                    self.attack_target,
                    self._build_feedback_summary(),
                    self._build_memory_summary(self.retrieve_case_memory(env)),
                    self._build_role_constraints(env, snapshot),
                )
                parsed = self._extract_decision_dict(response)
                decision = self._normalize_structured_decision(parsed, env, snapshot)
                self._record_llm_trace(
                    env,
                    protocol="structured",
                    attempt=attempt,
                    response=response,
                    parsed=parsed,
                    error="",
                )
                self.last_parse_error = ""
                return decision
            except Exception as exc:
                self._record_llm_trace(
                    env,
                    protocol="structured",
                    attempt=attempt,
                    response=response,
                    parsed=None,
                    error=str(exc),
                )
                self.parse_failures += 1
                self.last_parse_error = str(exc)
                if attempt == 1 or attempt == 3:
                    self._maybe_print_parse_warn(
                        "----LLM structured parse failure; retrying veh={} attempt={} error={}".format(
                            self.veh_id, attempt, exc
                        )
                    )

        self.fallback_activations += 1
        self.rollout_parse_fallback_used += 1
        self.repair_fallback_used += 1
        return self._build_fallback_decision(env, snapshot)

    def _build_role_constraints(self, env, snapshot):
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        is_owner = self.attack_role == "Blocker"
        constraints = {
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "round_index": self._control_round_index(self._get_step(env)),
            "coordination_mode": "same_tick_sequential",
            "same_tick_order": "owner_first",
            "coordination_round_step": int(snapshot.get("control_cycle_step", self._get_step(env))),
            "can_create_plan_id": is_owner,
            "must_follow_active_owner_plan": not is_owner,
            "allowed_kinds": ["commit", "replan"] if is_owner else ["ack", "status", "replan"],
            "owner_veh_id": self._get_owner_veh_id(snapshot),
            "active_owner_plan_id": active_owner_plan.get("plan_id", ""),
            "active_owner_phase": active_owner_plan.get("phase", ""),
            "teammate_id": self._get_teammate_id(),
            "phase_schedule": {
                "to_strike_round": PHASE_TO_STRIKE_ROUND,
                "to_brake_round": PHASE_TO_BRAKE_ROUND,
            },
        }
        if self.attack_role == "Blocker":
            constraints["phase_targets"] = {
                "compress": {"desired_rel_x_min": 4.0, "lane_pref": "same_or_adjacent"},
                "strike": {"desired_rel_x_min": 4.0, "lane_pref": "same_or_adjacent"},
            }
        elif self.attack_role == "Striker":
            constraints["phase_targets"] = {
                "compress": {
                    "desired_rel_x_range": [-8.0, 6.0],
                    "prefer_adjacent_lane": True,
                },
                "strike": {
                    "desired_rel_lane": 0,
                    "desired_rel_x_range": [1.0, 8.0],
                    "mode": "track_pose",
                },
                "brake_pulse": {
                    "desired_rel_lane": 0,
                    "desired_rel_x_range": [1.0, 8.0],
                    "mode": "pulse_brake",
                },
            }
        return constraints

    def _build_feedback_summary(self):
        if not self.previous_feedback:
            return "none"
        fields = []
        for key in ("result", "feedback_summary", "failure_phase", "min_ttc", "ego_max_decel"):
            value = self.previous_feedback.get(key)
            if value is not None and value != "":
                fields.append("{}={}".format(key, value))
        return "; ".join(fields) if fields else "none"

    def _build_blackboard_summary(self, snapshot):
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        latest_status = snapshot.get("latest_status_by_agent") or {}
        recent_events = snapshot.get("recent_events") or []
        latest_commit = snapshot.get("latest_commit") or {}
        owner_committed = bool(
            latest_commit
            and latest_commit.get("plan_id", "") == active_owner_plan.get("plan_id", "")
            and latest_commit.get("sender", "") == self._get_owner_veh_id(snapshot)
        )
        self_status = latest_status.get(self.veh_id) or {}
        teammate_id = self._get_teammate_id()
        teammate_status = latest_status.get(teammate_id) or {}
        trigger_eval = snapshot.get("trigger_eval", {}) or {}

        def _update_line(label, veh_id, status):
            trigger_state = trigger_eval.get(veh_id, {}).get("satisfied")
            if not status:
                return "{}_present=no".format(label)
            return "{}_present=yes;{}_plan={};{}_intent={};{}_trigger_ready={}".format(
                label,
                label,
                status.get("plan_id", ""),
                label,
                status.get("intent", ""),
                label,
                "na" if trigger_state is None else str(bool(trigger_state)).lower(),
            )

        recent_owner_change = "no"
        teammate_reply = "no"
        for entry in recent_events:
            sender = entry.get("sender", "")
            kind = entry.get("kind", "")
            if sender == self._get_owner_veh_id(snapshot) and kind in ("commit", "replan"):
                recent_owner_change = "yes"
            if sender == teammate_id and kind == "ack":
                teammate_reply = "yes"

        lines = [
            "owner_plan_present={}".format("yes" if active_owner_plan else "no"),
            "owner_plan_id={}".format(active_owner_plan.get("plan_id", "none") if active_owner_plan else "none"),
            "owner_phase={}".format(active_owner_plan.get("phase", "none") if active_owner_plan else "none"),
            "owner_intent={}".format(active_owner_plan.get("intent", "none") if active_owner_plan else "none"),
            "owner_committed={}".format("yes" if owner_committed else "no"),
            _update_line("self_update", self.veh_id, self_status),
            _update_line("mate_update", teammate_id, teammate_status),
            "recent_owner_change={}".format(recent_owner_change),
            "teammate_reply_seen={}".format(teammate_reply),
        ]
        return "\n".join(lines)

    def _build_memory_summary(self, memory_cases):
        if not memory_cases:
            return "none"
        lines = []
        for case in memory_cases[:2]:
            bucketed = case.get("bucketed", {}) or {}
            geometry = "{}/{}/{}".format(
                bucketed.get("self_rel_x_bin", "na"),
                bucketed.get("self_rel_lane", "na"),
                bucketed.get("teammate_rel_x_bin", "na"),
            )
            lines.append(
                "result={}; phase={}; geo={}; feedback={}; min_ttc={}; ego_max_decel={}".format(
                    case.get("result", "na"),
                    case.get("phase", "na"),
                    geometry,
                    case.get("feedback_summary", "na"),
                    case.get("min_ttc", "na"),
                    case.get("ego_max_decel", "na"),
                )
            )
        return "\n".join(lines)

    def _normalize_runtime_phase(self, phase, active_owner_plan):
        phase = str(phase or "").strip() or str(active_owner_plan.get("phase", self.active_phase) or self.active_phase)
        if phase == "setup":
            if active_owner_plan:
                phase = str(active_owner_plan.get("phase", "") or "")
            elif self.active_phase in RUNTIME_PHASES:
                phase = self.active_phase
            else:
                phase = "compress"
        if phase not in RUNTIME_PHASES:
            raise ValueError("Invalid runtime phase: {}".format(phase))
        return phase

    def _apply_phase_progression(self, phase, step, is_owner, active_owner_plan):
        if (not is_owner) and active_owner_plan:
            owner_phase = str(active_owner_plan.get("phase", phase) or phase)
            if owner_phase in RUNTIME_PHASES:
                return owner_phase
        round_index = self._control_round_index(step)
        if phase == "compress" and round_index >= PHASE_TO_STRIKE_ROUND:
            phase = "strike"
        if phase == "strike" and round_index >= PHASE_TO_BRAKE_ROUND:
            phase = "brake_pulse"
        return phase

    def _apply_role_phase_constraints(self, env, phase, control):
        ctx = self._get_relative_context(env)
        desired_rel_lane = int(control.get("desired_rel_lane", 0))
        desired_rel_x = float(control.get("desired_rel_x", 0.0))
        mode = str(control.get("mode", "track_pose") or "track_pose")

        if self.attack_role == "Blocker" and phase in ("compress", "strike"):
            desired_rel_x = max(4.0, desired_rel_x)
            desired_rel_lane = int(np.clip(desired_rel_lane, -1, 1))
            if mode == "disengage":
                mode = "track_pose"
        elif self.attack_role == "Striker":
            if phase == "compress":
                desired_rel_x = float(np.clip(desired_rel_x, -8.0, 6.0))
                if desired_rel_lane == 0:
                    current_rel_lane = int(ctx["self_rel_lane"])
                    desired_rel_lane = -1 if current_rel_lane < 0 else 1
                if mode == "disengage":
                    mode = "track_pose"
            elif phase in ("strike", "brake_pulse"):
                desired_rel_lane = 0
                desired_rel_x = float(np.clip(desired_rel_x, 1.0, 8.0))
                mode = "pulse_brake" if phase == "brake_pulse" else "track_pose"

        control["mode"] = mode
        control["desired_rel_lane"] = int(np.clip(
            desired_rel_lane,
            self.bounds["rel_lane_min"],
            self.bounds["rel_lane_max"],
        ))
        control["desired_rel_x"] = float(np.clip(
            desired_rel_x,
            self.bounds["rel_x_min"],
            self.bounds["rel_x_max"],
        ))
        return control

    def _record_repairs(self, repaired_fields):
        if not repaired_fields:
            return
        self.normalization_repairs += len(repaired_fields)
        for field_name in repaired_fields:
            self.repaired_fields.append(str(field_name))
        self.repaired_fields = self.repaired_fields[-32:]

    def _split_structured_payload(self, parsed):
        if not isinstance(parsed, dict):
            raise ValueError("Decision must be a dictionary.")

        message = copy.deepcopy(parsed.get("message", {}))
        control = copy.deepcopy(parsed.get("control", {}))
        if message and not isinstance(message, dict):
            raise ValueError("message must be an object.")
        if control and not isinstance(control, dict):
            raise ValueError("control must be an object.")

        if not message:
            message = {}
        if not control:
            control = {}

        control_keys = ("mode", "desired_rel_lane", "desired_rel_x", "desired_rel_s", "horizon_steps")
        message_keys = ("kind", "phase", "intent", "plan_id", "reply_to", "trigger_code",
                        "trigger_args", "preconditions", "done_code", "fallback",
                        "confidence", "eta_steps", "target")

        for key in control_keys:
            if key in message and key not in control:
                control[key] = message.pop(key)

        for key, value in parsed.items():
            if key in ("message", "control"):
                continue
            if key in message_keys:
                message.setdefault(key, value)
            elif key in control_keys:
                control.setdefault(key, value)
        return message, control

    def _repair_structured_payload(self, message, control, snapshot):
        repaired_fields = []
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        is_owner = self.attack_role == "Blocker"

        kind = str(message.get("kind", "") or "").strip().lower()
        phase = str(message.get("phase", "") or "").strip().lower()
        done_code = str(message.get("done_code", "") or "").strip().lower()
        fallback = str(message.get("fallback", "") or "").strip().lower()
        trigger_code = str(message.get("trigger_code", "") or "").strip().lower()
        control_mode = str(control.get("mode", "") or "").strip().lower()

        if phase == "status":
            if kind not in VALID_KINDS:
                message["kind"] = "status"
                repaired_fields.append("kind<-phase")
            message["phase"] = str(active_owner_plan.get("phase", self.active_phase) or self.active_phase or "compress")
            repaired_fields.append("phase=status->runtime")
        elif phase in VALID_PHASES:
            message["phase"] = phase

        if kind in VALID_PHASES and phase not in VALID_PHASES:
            message["phase"] = kind
            repaired_fields.append("phase<-kind")
            message["kind"] = "status" if not is_owner else "commit"
            repaired_fields.append("kind->default")
        elif kind in VALID_KINDS:
            message["kind"] = kind

        if kind not in VALID_KINDS:
            message["kind"] = "commit" if is_owner else ("ack" if active_owner_plan else "status")
            repaired_fields.append("kind=default")

        if phase not in VALID_PHASES:
            message["phase"] = str(active_owner_plan.get("phase", self.active_phase) or self.active_phase or "compress")
            repaired_fields.append("phase=default")

        if trigger_code not in VALID_TRIGGER_CODES:
            message["trigger_code"] = "none"
            repaired_fields.append("trigger_code=none")
        else:
            message["trigger_code"] = trigger_code

        if done_code not in VALID_DONE_CODES:
            message["done_code"] = DONE_CODE_REPAIRS.get(done_code, "phase_complete")
            repaired_fields.append("done_code=repair")
        else:
            message["done_code"] = done_code

        if fallback not in VALID_FALLBACKS:
            message["fallback"] = FALLBACK_REPAIRS.get(fallback, "hold_lane")
            repaired_fields.append("fallback=repair")
        else:
            message["fallback"] = fallback

        if control_mode not in VALID_CONTROL_MODES:
            control["mode"] = CONTROL_MODE_REPAIRS.get(control_mode, "hold_lane")
            repaired_fields.append("control.mode=repair")
        else:
            control["mode"] = control_mode

        return message, control, repaired_fields

    def _normalize_structured_decision(self, parsed, env, snapshot):
        message, control = self._split_structured_payload(parsed)
        message, control, repaired_fields = self._repair_structured_payload(message, control, snapshot)
        self._record_repairs(repaired_fields)

        step = self._get_step(env)
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        owner_veh_id = self._get_owner_veh_id(snapshot)
        is_owner = self.attack_role == "Blocker"
        plan_id = str(message.get("plan_id", "") or "")
        if is_owner:
            if active_owner_plan:
                plan_id = str(active_owner_plan.get("plan_id", "") or "")
            if not plan_id:
                self.plan_counter += 1
                plan_id = "p{}".format(self.plan_counter)
        else:
            plan_id = str(active_owner_plan.get("plan_id", ""))
            if not plan_id:
                raise ValueError("Striker requires active owner plan.")

        kind = str(message.get("kind", "status") or "status")
        if kind not in VALID_KINDS:
            raise ValueError("Invalid kind: {}".format(kind))
        if is_owner:
            kind = "replan" if active_owner_plan else "commit"
        if (not is_owner) and kind in ("propose", "commit", "replan"):
            kind = "ack"

        phase = self._normalize_runtime_phase(message.get("phase", self.active_phase), active_owner_plan)
        phase = self._apply_phase_progression(phase, step, is_owner, active_owner_plan)

        trigger_code = str(message.get("trigger_code", "none") or "none")
        if trigger_code not in VALID_TRIGGER_CODES:
            raise ValueError("Invalid trigger_code: {}".format(trigger_code))

        done_code = str(message.get("done_code", "phase_complete") or "phase_complete")
        if done_code not in VALID_DONE_CODES:
            raise ValueError("Invalid done_code: {}".format(done_code))

        fallback = str(message.get("fallback", "hold_lane") or "hold_lane")
        if fallback not in VALID_FALLBACKS:
            raise ValueError("Invalid fallback: {}".format(fallback))

        control_mode = str(control.get("mode", "track_pose") or "track_pose")
        if control_mode not in VALID_CONTROL_MODES:
            raise ValueError("Invalid control.mode: {}".format(control_mode))

        eta_steps = max(0, int(round(self._safe_float(message.get("eta_steps"), self.control_interval))))
        horizon_steps = max(1, int(round(self._safe_float(control.get("horizon_steps"), self.control_interval))))

        normalized_message = {
            "sender": self.veh_id,
            "owner": owner_veh_id,
            "kind": kind,
            "plan_id": plan_id,
            "reply_to": "" if is_owner else plan_id,
            "phase": phase,
            "intent": str(message.get("intent", "hold_lane") or "hold_lane")[:80],
            "target": str(message.get("target", self.attack_target) or self.attack_target),
            "eta_steps": eta_steps,
            "trigger_code": trigger_code,
            "trigger_args": copy.deepcopy(message.get("trigger_args", {}) or {}),
            "preconditions": list(message.get("preconditions", []) or []),
            "done_code": done_code,
            "fallback": fallback,
            "confidence": float(np.clip(self._safe_float(message.get("confidence"), 0.5), 0.0, 1.0)),
            "step": step,
            "expires_at_step": step + horizon_steps,
            "scenario_id": self.scenario_id,
            "rollout_id": getattr(env.message_pool, "rollout_id", 0),
        }
        normalized_control = {
            "mode": control_mode,
            "desired_rel_lane": int(np.clip(
                round(self._safe_float(control.get("desired_rel_lane"), 0.0)),
                self.bounds["rel_lane_min"],
                self.bounds["rel_lane_max"],
            )),
            "desired_rel_x": float(np.clip(
                self._safe_float(control.get("desired_rel_x"), 0.0),
                self.bounds["rel_x_min"],
                self.bounds["rel_x_max"],
            )),
            "desired_rel_s": float(np.clip(
                self._safe_float(control.get("desired_rel_s"), self.target_s),
                self.bounds["s_min"],
                self.bounds["s_max"],
            )),
            "horizon_steps": horizon_steps,
        }
        normalized_control = self._apply_role_phase_constraints(env, phase, normalized_control)
        return {"message": normalized_message, "control": normalized_control}

    def _build_status_decision(self, env, snapshot, status_intent=None, fallback=None, control_mode=None):
        step = self._get_step(env)
        message = self._default_message()
        control = self._default_control()
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        if active_owner_plan:
            message["plan_id"] = str(active_owner_plan.get("plan_id", ""))
            message["reply_to"] = "" if self.attack_role == "Blocker" else message["plan_id"]
            message["phase"] = str(active_owner_plan.get("phase", self.active_phase))
            message["intent"] = str(active_owner_plan.get("intent", "status_hold"))
            message["fallback"] = str(active_owner_plan.get("fallback", "hold_lane"))
        elif self.current_decision is not None:
            control = copy.deepcopy(self.current_decision["control"])
            message["plan_id"] = str(self.current_decision["message"].get("plan_id", ""))
            message["reply_to"] = str(self.current_decision["message"].get("reply_to", ""))
            message["phase"] = self.current_decision["message"].get("phase", "compress")
            message["intent"] = self.current_decision["message"].get("intent", message["intent"])
            message["fallback"] = self.current_decision["message"].get("fallback", "drop_back")

        if self.current_decision is not None:
            control = copy.deepcopy(self.current_decision["control"])
        if status_intent:
            message["intent"] = status_intent
        if fallback:
            message["fallback"] = fallback
        if control_mode:
            control["mode"] = control_mode
        if control["mode"] == "disengage":
            control["desired_rel_lane"] = 0
            control["desired_rel_x"] = -8.0
            control["desired_rel_s"] = 6.0

        message["step"] = step
        message["expires_at_step"] = step + max(1, int(control.get("horizon_steps", self.control_interval)))
        return {"message": message, "control": control}

    def _build_owner_fallback_decision(self, env, snapshot):
        step = self._get_step(env)
        active_owner_plan = snapshot.get("active_owner_plan") or {}
        basis = None
        if self.current_decision is not None and self.current_decision["message"].get("plan_id"):
            basis = copy.deepcopy(self.current_decision)
        elif active_owner_plan:
            basis = {
                "message": copy.deepcopy(active_owner_plan),
                "control": copy.deepcopy(self._default_control()),
            }

        if basis is None:
            self.plan_counter += 1
            basis = {
                "message": self._default_message(),
                "control": self._default_control(),
            }
            basis["message"]["plan_id"] = "p{}".format(self.plan_counter)
            basis["message"]["intent"] = "owner_fallback_block"
            basis["control"]["mode"] = "hold_lane"
            basis["control"]["desired_rel_lane"] = 0
            basis["control"]["desired_rel_x"] = 10.0
            basis["control"]["desired_rel_s"] = 5.0

        basis["message"]["sender"] = self.veh_id
        basis["message"]["owner"] = self.veh_id
        basis["message"]["kind"] = "replan" if active_owner_plan else "commit"
        basis["message"]["phase"] = str(active_owner_plan.get("phase", basis["message"].get("phase", self.active_phase)) or self.active_phase)
        basis["message"]["intent"] = str(basis["message"].get("intent", "owner_fallback_block") or "owner_fallback_block")
        basis["message"]["fallback"] = str(basis["message"].get("fallback", "hold_lane") or "hold_lane")
        basis["message"]["trigger_code"] = "none"
        basis["message"]["done_code"] = "phase_complete"
        basis["message"]["step"] = step
        basis["message"]["expires_at_step"] = step + max(1, int(basis["control"].get("horizon_steps", self.control_interval)))
        return basis

    def _build_fallback_decision(self, env, snapshot):
        if self.attack_role == "Blocker":
            return self._build_owner_fallback_decision(env, snapshot)
        if self.current_decision is not None:
            decision = copy.deepcopy(self.current_decision)
            decision["message"]["kind"] = "status"
            decision["message"]["intent"] = "parse_fallback_follow"
            decision["message"]["fallback"] = "hold_lane"
            decision["message"]["step"] = self._get_step(env)
            decision["message"]["expires_at_step"] = self._get_step(env) + max(
                1, int(decision["control"].get("horizon_steps", self.control_interval))
            )
            decision["control"]["mode"] = "hold_lane"
            return decision
        return self._build_status_decision(env, snapshot, "parse_fallback_used", "hold_lane", "hold_lane")

    def _publish_structured_decision(self, env, decision, trigger_eval):
        publish_entry = copy.deepcopy(decision["message"])
        publish_entry["trigger_eval"] = trigger_eval
        publish_entry["control_cycle_step"] = int(getattr(env.message_pool, "control_cycle_step", self._get_step(env)))
        publish_entry["text"] = "{}:{}:{}".format(
            publish_entry.get("kind", "status"),
            publish_entry.get("phase", "compress"),
            publish_entry.get("intent", ""),
        )
        env.message_pool.publish(publish_entry)

        self.current_decision = copy.deepcopy(decision)
        self.current_message = publish_entry["text"]
        self.active_plan_id = decision["message"].get("plan_id", "")
        self.active_phase = decision["message"].get("phase", "compress")
        self.active_owner = decision["message"].get("owner", self._get_owner_veh_id())
        self.active_control_mode = decision["control"].get("mode", "track_pose")
        self.plan_expiry_step = int(decision["message"].get("expires_at_step", self._get_step(env)))
        self.current_trigger_eval = copy.deepcopy(trigger_eval)
        self._record_phase_trace(decision["message"])

    def _record_phase_trace(self, message):
        event = {
            "step": int(message.get("step", 0)),
            "phase": str(message.get("phase", "compress")),
            "kind": str(message.get("kind", "status")),
            "plan_id": str(message.get("plan_id", "")),
            "intent": str(message.get("intent", "")),
        }
        if not self.phase_trace or self.phase_trace[-1] != event:
            self.phase_trace.append(event)
            if len(self.phase_trace) > 32:
                self.phase_trace = self.phase_trace[-32:]

    def _get_relative_context(self, env):
        ego_speed = 0.0
        ego_lane = 0
        ego_x = 0.0
        if self.attack_target in env.k.vehicle.get_ids():
            ego_speed = float(env.k.vehicle.get_speed(self.attack_target))
            ego_lane = int(env.k.vehicle.get_lane(self.attack_target))
            ego_x = float(env.k.vehicle.get_x_by_id(self.attack_target))

        self_speed = float(env.k.vehicle.get_speed(self.veh_id))
        self_lane = int(env.k.vehicle.get_lane(self.veh_id))
        self_x = float(env.k.vehicle.get_x_by_id(self.veh_id))
        teammate_id = self._get_teammate_id()
        teammate_x = ego_x
        teammate_lane = ego_lane
        if teammate_id in env.k.vehicle.get_ids():
            teammate_x = float(env.k.vehicle.get_x_by_id(teammate_id))
            teammate_lane = int(env.k.vehicle.get_lane(teammate_id))

        return {
            "ego_speed": ego_speed,
            "ego_lane": ego_lane,
            "ego_x": ego_x,
            "self_speed": self_speed,
            "self_lane": self_lane,
            "self_x": self_x,
            "self_rel_x": self_x - ego_x,
            "self_rel_lane": self_lane - ego_lane,
            "teammate_id": teammate_id,
            "teammate_rel_x": teammate_x - ego_x,
            "teammate_rel_lane": teammate_lane - ego_lane,
        }

    def _evaluate_trigger(self, env, decision, snapshot):
        message = decision["message"]
        control = decision["control"]
        ctx = self._get_relative_context(env)
        code = message.get("trigger_code", "none")
        step = self._get_step(env)
        satisfied = True
        details = {}

        if code == "none":
            satisfied = True
        elif code == "owner_committed":
            latest_commit = snapshot.get("latest_commit") or {}
            owner_veh_id = self._get_owner_veh_id(snapshot)
            satisfied = bool(
                latest_commit.get("sender") == owner_veh_id
                and
                latest_commit.get("owner") == owner_veh_id
                and latest_commit.get("plan_id") == message.get("plan_id")
            )
        elif code == "target_lane_match":
            desired_lane = int(ctx["ego_lane"] + int(control.get("desired_rel_lane", 0)))
            satisfied = int(ctx["self_lane"]) == desired_lane
            details["desired_lane"] = desired_lane
        elif code == "ego_gap_lt":
            gap_limit = float(self._safe_float(message.get("trigger_args", {}).get("gap"), 8.0))
            satisfied = abs(float(ctx["self_rel_x"])) <= gap_limit
            details["gap_limit"] = gap_limit
        elif code == "self_at_rel_pose":
            lane_error = abs(int(ctx["self_rel_lane"]) - int(control.get("desired_rel_lane", 0)))
            x_error = abs(float(ctx["self_rel_x"]) - float(control.get("desired_rel_x", 0.0)))
            x_tol = float(self._safe_float(message.get("trigger_args", {}).get("x_tol"), 2.0))
            lane_tol = int(round(self._safe_float(message.get("trigger_args", {}).get("lane_tol"), 0.0)))
            satisfied = lane_error <= lane_tol and x_error <= x_tol
            details["lane_error"] = lane_error
            details["x_error"] = x_error
        elif code == "teammate_ready":
            teammate_id = ctx["teammate_id"]
            teammate_trigger = snapshot.get("trigger_eval", {}).get(teammate_id) or {}
            satisfied = bool(teammate_trigger.get("satisfied", False))
        elif code == "horizon_expired":
            satisfied = step >= int(message.get("expires_at_step", step))
        elif code == "ego_hard_brake":
            prev_speed = float(env.k.vehicle.get_previous_speed(self.attack_target))
            cur_speed = float(env.k.vehicle.get_speed(self.attack_target))
            ego_decel = max(0.0, (prev_speed - cur_speed) / max(env.sim_step, 1e-3))
            satisfied = ego_decel >= EGO_HARD_BRAKE_DECEL
            details["ego_decel"] = ego_decel

        return {
            "trigger_code": code,
            "satisfied": bool(satisfied),
            "details": details,
        }

    def _set_execution_targets(self, env, decision, trigger_eval):
        phase = str(decision.get("message", {}).get("phase", "compress") or "compress")
        control = decision["control"]
        ctx = self._get_relative_context(env)
        ego_speed = float(ctx["ego_speed"])
        self_speed = float(ctx["self_speed"])
        desired_rel_lane = int(control.get("desired_rel_lane", 0))
        desired_rel_x = float(control.get("desired_rel_x", 0.0))
        desired_rel_s = float(control.get("desired_rel_s", self.target_s))
        rel_x_error = desired_rel_x - float(ctx["self_rel_x"])
        desired_lane_abs = int(ctx["ego_lane"] + desired_rel_lane)
        current_lane = int(ctx["self_lane"])

        lane_delta = desired_lane_abs - current_lane
        lane_cmd = 0
        if lane_delta > 0:
            lane_cmd = 1
        elif lane_delta < 0:
            lane_cmd = -1

        target_v = ego_speed + float(np.clip(rel_x_error * 0.45, -6.0, 6.0))
        target_s = desired_rel_s
        mode = control.get("mode", "track_pose")

        if mode == "hold_lane":
            lane_cmd = 0
            target_v = ego_speed + float(np.clip(rel_x_error * 0.25, -3.0, 3.0))
        elif mode == "pulse_brake":
            if self._pulse_end_step < self._get_step(env):
                self._pulse_end_step = self._get_step(env) + min(4, int(control.get("horizon_steps", 4)))
            lane_cmd = 0
            target_v = min(self_speed, max(self.bounds["v_min"], ego_speed - 6.0))
        elif mode == "disengage":
            lane_cmd = 0
            target_v = max(self.bounds["v_min"], min(self.bounds["v_max"], ego_speed - 2.0))
            target_s = max(5.5, desired_rel_s)

        if not trigger_eval.get("satisfied", True) and mode in ("track_pose", "pulse_brake") and not (
                self.attack_role == "Striker" and phase in ("strike", "brake_pulse")):
            lane_cmd = 0
            target_v = max(self.bounds["v_min"], min(self.bounds["v_max"], ego_speed - 1.0))

        if self.attack_role == "Striker" and phase in ("strike", "brake_pulse"):
            if float(ctx["self_rel_x"]) < 0.0:
                self._striker_behind_steps += 1
            else:
                self._striker_behind_steps = 0

            if self._striker_behind_steps >= STRIKER_REPOSITION_STEPS:
                desired_lane_abs = int(ctx["ego_lane"])
                lane_delta = desired_lane_abs - current_lane
                if lane_delta > 0:
                    lane_cmd = 1
                elif lane_delta < 0:
                    lane_cmd = -1
                else:
                    lane_cmd = 0
                target_v = max(self_speed, min(self.bounds["v_max"], ego_speed + 4.0))
                target_s = max(self.bounds["s_min"], min(4.0, desired_rel_s))
        else:
            self._striker_behind_steps = 0

        self.target_v = float(np.clip(target_v, self.bounds["v_min"], self.bounds["v_max"]))
        self.target_s = float(np.clip(target_s, self.bounds["s_min"], self.bounds["s_max"]))
        self.target_lc = int(np.clip(lane_cmd, -1, 1))
        self.pending_lane_change = self.target_lc
        self.v0 = self.target_v
        self.s0 = self.target_s

    def llm_collaborate(self, env):
        scenario_description = self.get_perception(env)
        shared_message = env.message_pool.get_all_msg()
        parse_attempt = 0

        while True:
            response = ""
            response = self.DA.collaborate(
                self.map_name,
                scenario_description,
                shared_message,
                self.attack_role,
                self.attack_target,
                self.previous_feedback,
            )
            try:
                parsed = self._extract_decision_dict(response)
                params = self._normalize_decision_fields(parsed)
                params = self._hard_clip_decision(params)
                self._record_llm_trace(
                    env,
                    protocol="legacy",
                    attempt=parse_attempt + 1,
                    response=response,
                    parsed=parsed,
                    error="",
                )
                self.last_parse_error = ""
                env.message_pool.join(self.veh_id, params["message"])
                return params
            except Exception as e:
                self._record_llm_trace(
                    env,
                    protocol="legacy",
                    attempt=parse_attempt + 1,
                    response=response,
                    parsed=None,
                    error=str(e),
                )
                parse_attempt += 1
                self.parse_failures += 1
                self.last_parse_error = str(e)
                if parse_attempt >= 3:
                    self.fallback_activations += 1
                    return {
                        "message": "legacy_fallback_hold",
                        "v": float(np.clip(self.target_v, self.bounds["v_min"], self.bounds["v_max"])),
                        "s": float(np.clip(self.target_s, self.bounds["s_min"], self.bounds["s_max"])),
                        "lane_change": 0,
                    }
                if parse_attempt == 1 or parse_attempt == 3:
                    self._maybe_print_parse_warn(
                        "----LLM parse failure; retrying "
                        f"veh={self.veh_id} attempt={parse_attempt} error={e}"
                    )

    def _extract_decision_dict(self, response):
        cleaned = self._sanitize_llm_text(response)
        candidates = []

        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m is not None:
            candidates.append(m.group(0))

        m2 = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if m2 is not None:
            candidates.append(m2.group(0))
        candidates.extend(re.findall(r"\{[^{}]*\}", cleaned))

        if not candidates:
            raise ValueError("No dictionary-like object found in LLM response.")

        last_err = None
        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e_json:
                last_err = e_json
                try:
                    parsed = ast.literal_eval(cand)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as e_ast:
                    last_err = e_ast
                    continue

        raise ValueError("Failed to parse decision dictionary: {}".format(last_err))

    def _sanitize_llm_text(self, text):
        text = str(text).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = text.replace("```json", "").replace("```", "")
        text = text.replace("\u201c", "\"").replace("\u201d", "\"")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\\n", "\n")
        return text

    def _trim_trace_text(self, text):
        value = str(text or "")
        limit = max(120, int(self.trace_text_max_chars))
        if len(value) <= limit:
            return value
        remain = len(value) - limit
        return "{}...[truncated {} chars]".format(value[:limit], remain)

    def _record_llm_trace(self, env, protocol, attempt, response, parsed=None, error=""):
        raw_text = str(response or "")
        cleaned_text = self._sanitize_llm_text(raw_text) if raw_text else ""
        parse_ok = isinstance(parsed, dict) and not error
        self.last_raw_response = raw_text
        entry = {
            "step": int(self._get_step(env)),
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "protocol": str(protocol),
            "attempt": int(attempt),
            "parse_ok": bool(parse_ok),
            "error": str(error or ""),
            "raw_response": self._trim_trace_text(raw_text),
            "sanitized_response": self._trim_trace_text(cleaned_text),
        }
        if isinstance(parsed, dict):
            entry["parsed_keys"] = sorted(str(k) for k in parsed.keys())
        self.llm_trace_entries.append(entry)
        if len(self.llm_trace_entries) > self.trace_max_entries:
            self.llm_trace_entries = self.llm_trace_entries[-self.trace_max_entries:]

        if self.trace_file:
            try:
                with open(self.trace_file, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        if self.trace_stdout and (not parse_ok):
            print(
                "----LLM raw trace veh={} step={} protocol={} attempt={} parse_ok={} error={}".format(
                    self.veh_id,
                    entry["step"],
                    protocol,
                    int(attempt),
                    str(bool(parse_ok)).lower(),
                    str(error or ""),
                )
            )
            if entry["raw_response"]:
                print(entry["raw_response"])

    def _maybe_print_parse_warn(self, message):
        if not self.parse_warn_stdout:
            return
        if self.parse_warn_budget > 0 and self.parse_warn_count >= self.parse_warn_budget:
            return
        print(str(message))
        self.parse_warn_count += 1

    def _safe_float(self, value, default):
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("", "none", "null", "nan", "inf", "+inf", "-inf"):
                return float(default)
            num_match = re.search(r"-?\d+(?:\.\d+)?", s)
            if num_match:
                try:
                    return float(num_match.group(0))
                except Exception:
                    return float(default)
        return float(default)

    def _safe_lane_change(self, value, default=0):
        if value is None:
            return int(default)
        if isinstance(value, str):
            s = value.strip().lower()
            if "left" in s:
                return 1
            if "right" in s:
                return -1
            if "keep" in s or "stay" in s:
                return 0
        num = int(round(self._safe_float(value, default)))
        return int(np.clip(num, -1, 1))

    def _normalize_decision_fields(self, parsed):
        required_keys = {"message", "v", "s", "lane_change"}
        if not required_keys.issubset(set(parsed.keys())):
            raise KeyError("JSON must include keys: message, v, s, lane_change.")

        return {
            "message": str(parsed["message"]),
            "v": self._safe_float(parsed["v"], self.target_v),
            "s": self._safe_float(parsed["s"], self.target_s),
            "lane_change": self._safe_lane_change(parsed["lane_change"], 0),
        }

    def _hard_clip_decision(self, params):
        params["v"] = float(np.clip(
            self._safe_float(params.get("v"), self.target_v),
            self.bounds["v_min"],
            self.bounds["v_max"],
        ))
        params["s"] = float(np.clip(
            self._safe_float(params.get("s"), self.target_s),
            self.bounds["s_min"],
            self.bounds["s_max"],
        ))
        params["lane_change"] = self._safe_lane_change(params.get("lane_change"), 0)
        params["message"] = str(params.get("message", "maintain pressure"))[:120]
        return params

    def llm_reason(self, env):
        if self._use_structured_protocol():
            if self.current_decision is None:
                return {
                    "message": copy.deepcopy(self._default_message()),
                    "control": copy.deepcopy(self._default_control()),
                }
            return copy.deepcopy(self.current_decision)
        return {
            "message": self.current_message,
            "v": self.target_v,
            "s": self.target_s,
            "lane_change": self.target_lc,
        }

    def get_runtime_stats(self):
        return {
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "model": self.DA.llm_model,
            "last_parse_error": str(self.last_parse_error),
            "parse_failures": int(self.parse_failures),
            "fallback_activations": int(self.fallback_activations),
            "normalization_repairs": int(self.normalization_repairs),
            "repair_fallback_used": int(self.repair_fallback_used),
            "last_plan_id": str(self.active_plan_id),
            "last_phase": str(self.active_phase),
            "last_raw_response": self._trim_trace_text(self.last_raw_response),
        }

    def get_rollout_diagnostics(self):
        return {
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "active_plan_id": self.active_plan_id,
            "active_phase": self.active_phase,
            "active_owner": self.active_owner,
            "active_control_mode": self.active_control_mode,
            "plan_expiry_step": int(self.plan_expiry_step),
            "phase_trace": copy.deepcopy(self.phase_trace),
            "owner_plan_missing_events": int(self.owner_plan_missing_events),
            "trigger_not_met_events": int(self.trigger_not_met_events),
            "sync_error_events": int(self.sync_error_events),
            "rollout_parse_fallback_used": int(self.rollout_parse_fallback_used),
            "normalization_repairs": int(self.normalization_repairs),
            "repaired_fields": copy.deepcopy(self.repaired_fields),
            "repair_fallback_used": int(self.repair_fallback_used),
            "role_map": copy.deepcopy(self.role_map),
            "scene_gate_status": copy.deepcopy(self.scene_gate_status),
            "current_decision": self.llm_reason(env=None),
            "llm_trace_tail": copy.deepcopy(self.llm_trace_entries[-12:]),
        }

    def get_state_signature(self, env, phase="compress"):
        ctx = self._get_relative_context(env)
        continuous = {
            "ego_lane": int(ctx["ego_lane"]),
            "self_rel_lane": int(ctx["self_rel_lane"]),
            "self_rel_x": round(float(ctx["self_rel_x"]), 3),
            "self_rel_s": round(float(env.k.vehicle.get_headway(self.veh_id)), 3),
            "teammate_rel_x": round(float(ctx["teammate_rel_x"]), 3),
            "teammate_rel_lane": int(ctx["teammate_rel_lane"]),
            "phase": str(phase or self.active_phase or "compress"),
        }
        bucketed = {
            "ego_lane": int(ctx["ego_lane"]),
            "self_rel_lane": int(ctx["self_rel_lane"]),
            "self_rel_x_bin": self._bucket_rel_x(continuous["self_rel_x"]),
            "self_rel_s_bin": self._bucket_rel_s(continuous["self_rel_s"]),
            "teammate_rel_x_bin": self._bucket_rel_x(continuous["teammate_rel_x"]),
            "teammate_rel_lane_bin": self._bucket_rel_lane(continuous["teammate_rel_lane"]),
            "phase": continuous["phase"],
        }
        return {"continuous": continuous, "bucketed": bucketed}

    def retrieve_case_memory(self, env):
        current_signature = self.get_state_signature(env, phase=self.active_phase or "compress")
        current_bucketed = current_signature["bucketed"]

        def _matches(case):
            if str(case.get("scenario_id", "")) != str(self.scenario_id):
                return False
            if str(case.get("role", "")) != str(self.attack_role):
                return False
            case_bucketed = case.get("state_signature_bucketed", {})
            return case_bucketed.get("phase") == current_bucketed.get("phase")

        def _score(case):
            case_bucketed = case.get("state_signature_bucketed", {})
            keys = (
                "ego_lane",
                "self_rel_lane",
                "self_rel_x_bin",
                "self_rel_s_bin",
                "teammate_rel_x_bin",
                "teammate_rel_lane_bin",
                "phase",
            )
            score = 0
            for key in keys:
                if case_bucketed.get(key) == current_bucketed.get(key):
                    score += 1
            return score

        cases = [case for case in self.case_memory if _matches(case)]
        cases.sort(key=lambda case: (_score(case), int(case.get("iteration", 0))), reverse=True)
        summaries = []
        success_case = None
        failure_case = None
        phase_case = None
        for case in cases:
            result = str(case.get("result", ""))
            if success_case is None and result == "success":
                success_case = case
            if failure_case is None and result != "success":
                failure_case = case
            if phase_case is None:
                phase_case = case
            if success_case and failure_case and phase_case:
                break

        for case in (success_case, failure_case, phase_case):
            if case is None:
                continue
            summary = {
                "role": case.get("role"),
                "result": case.get("result"),
                "phase": case.get("state_signature_bucketed", {}).get("phase"),
                "bucketed": case.get("state_signature_bucketed", {}),
                "feedback_summary": case.get("feedback_summary", ""),
                "min_ttc": case.get("min_ttc"),
                "ego_max_decel": case.get("ego_max_decel"),
            }
            if summary not in summaries:
                summaries.append(summary)
        return summaries

    def _bucket_rel_x(self, value):
        value = float(value)
        if value <= -15:
            return "rear_far"
        if value <= -7:
            return "rear_mid"
        if value <= -2:
            return "rear_close"
        if value < 2:
            return "side"
        if value < 7:
            return "front_close"
        if value < 15:
            return "front_mid"
        return "front_far"

    def _bucket_rel_s(self, value):
        value = float(value)
        if value < 3.0:
            return "tight"
        if value < 6.0:
            return "medium"
        return "loose"

    def _bucket_rel_lane(self, value):
        value = int(value)
        if value < 0:
            return "left"
        if value > 0:
            return "right"
        return "same"

    def get_perception(self, env):
        max_neighbors = max(1, int(os.getenv("FLOW_LLM_NEIGHBOR_K", "4")))
        ctx = self._get_relative_context(env)
        self_headway = float(env.k.vehicle.get_headway(self.veh_id))
        teammate_id = ctx["teammate_id"]

        lines = [
            "step={}".format(self._get_step(env)),
            "self={} role={} speed={:.2f} rel_x={:.2f} rel_lane={} headway={:.2f}".format(
                self.veh_id,
                self.attack_role,
                float(ctx["self_speed"]),
                float(ctx["self_rel_x"]),
                int(ctx["self_rel_lane"]),
                self_headway,
            ),
            "ego={} speed={:.2f} lane={}".format(
                self.attack_target,
                float(ctx["ego_speed"]),
                int(ctx["ego_lane"]),
            ),
            "teammate={} rel_x={:.2f} rel_lane={}".format(
                teammate_id,
                float(ctx["teammate_rel_x"]),
                int(ctx["teammate_rel_lane"]),
            ),
        ]

        nearby = []
        self_x = float(ctx["self_x"])
        for veh_id in env.k.vehicle.get_ids():
            if veh_id in (self.veh_id, teammate_id):
                continue
            pos = float(env.k.vehicle.get_position(veh_id))
            nearby.append({
                "veh_id": veh_id,
                "dist": abs(pos - self_x),
                "rel_x_to_self": pos - self_x,
                "rel_lane_to_self": int(env.k.vehicle.get_lane(veh_id)) - int(ctx["self_lane"]),
                "speed": float(env.k.vehicle.get_speed(veh_id)),
                "is_target": veh_id == self.attack_target,
            })

        nearby.sort(key=lambda item: (0 if item["is_target"] else 1, item["dist"]))
        for item in nearby[:max_neighbors]:
            lines.append(
                "near={} rel_x={:.2f} rel_lane={} speed={:.2f}".format(
                    item["veh_id"],
                    float(item["rel_x_to_self"]),
                    int(item["rel_lane_to_self"]),
                    float(item["speed"]),
                )
            )

        return "\n".join(lines)
