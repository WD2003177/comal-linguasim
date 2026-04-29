import ast
import copy
import json
import os
import re
import textwrap

import numpy as np
from openai import OpenAI

from flow.controllers.base_controller import BaseController
from flow.utils.highway_scene import get_highway_scene_mode
from flow.utils.highway_scene import SCENE_MODE_THREE_CAR_FIXED
from flow.utils.highway_scene import SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED
from flow.utils.highway_scene import SCENE_MODE_THREE_CAR_RANDOM


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


RUNTIME_PHASES = ("compress", "strike", "brake_pulse", "disengage")
RUNTIME_PHASE_RANK = {
    "compress": 0,
    "strike": 1,
    "brake_pulse": 2,
    "disengage": 3,
}
VALID_CONTROL_MODES = ("track_pose", "hold_lane", "pulse_brake", "disengage")
NEGOTIATED_TACTIC_LANE_POLICIES = ("pass_side", "ego_lane", "block_side", "hold_current")
NEGOTIATED_TACTIC_GAP_BANDS = ("loose", "medium", "tight")
NEGOTIATED_TACTIC_SPEED_BANDS = ("yield", "match", "press", "surge", "brake")
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
THREE_CAR_MERGE_COMMIT_STEPS = max(
    8,
    int(os.getenv("FLOW_THREE_CAR_MERGE_COMMIT_STEPS", "20")),
)
THREE_CAR_FRONT_BRAKE_STEPS = max(
    4,
    int(os.getenv("FLOW_THREE_CAR_FRONT_BRAKE_STEPS", "10")),
)
THREE_CAR_LC_DURATION = max(
    2,
    int(os.getenv("FLOW_THREE_CAR_LC_DURATION", "3")),
)
NEGOTIATED_MERGE_TIMEOUT_STEPS = max(
    4,
    int(os.getenv("FLOW_NEGOTIATED_MERGE_TIMEOUT_STEPS", "6")),
)
HIGHWAY_END_LC_DISABLE_M = max(
    0.0,
    float(os.getenv("FLOW_HIGHWAY_END_LC_DISABLE_M", "30.0")),
)
HIGHWAY_END_ABORT_M = max(
    0.0,
    float(os.getenv("FLOW_HIGHWAY_END_ABORT_M", "15.0")),
)
HIGHWAY_END_ABORT_SPEED_MPS = max(
    0.0,
    float(os.getenv("FLOW_HIGHWAY_END_ABORT_SPEED_MPS", "1.0")),
)
HIGHWAY_ACTION_REUSE_MAX_CYCLES = max(
    0,
    int(os.getenv("FLOW_HIGHWAY_ACTION_REUSE_MAX_CYCLES", "2")),
)
HIGHWAY_ACTION_REL_X_HYSTERESIS_M = max(
    0.0,
    float(os.getenv("FLOW_HIGHWAY_ACTION_REL_X_HYSTERESIS_M", "0.75")),
)
HIGHWAY_ACTION_REL_X_BUCKET_EDGES = (-8.0, -2.0, 2.0, 6.0, 10.0)
HIGHWAY_ACTION_REL_X_BUCKET_LABELS = (
    "lt_-8",
    "-8_to_-2",
    "-2_to_2",
    "2_to_6",
    "6_to_10",
    "gt_10",
)
NEGOTIATED_MERGE_COMMIT_MIN_REL_X = 0.0
NEGOTIATED_MERGE_COMMIT_MAX_REL_X = 8.0
NEGOTIATED_MERGE_FAIL_REL_X = max(
    NEGOTIATED_MERGE_COMMIT_MAX_REL_X,
    float(os.getenv("FLOW_NEGOTIATED_MERGE_FAIL_REL_X", "16.0")),
)
NEGOTIATED_CLEAN_MERGE_MAX_REL_X = max(
    NEGOTIATED_MERGE_COMMIT_MAX_REL_X,
    min(
        NEGOTIATED_MERGE_FAIL_REL_X,
        float(os.getenv("FLOW_NEGOTIATED_CLEAN_MERGE_MAX_REL_X", "12.0")),
    ),
)
NEGOTIATED_MERGE_COMMIT_MIN_SPEED_ADV = 0.5
NEGOTIATED_MERGE_COMMIT_MAX_SPEED_ADV = 1.5
NEGOTIATED_FORCE_CUT_IN_MIN_REL_X = 1.5
NEGOTIATED_FORCE_CUT_IN_MAX_REL_X = max(
    NEGOTIATED_FORCE_CUT_IN_MIN_REL_X,
    float(os.getenv("FLOW_NEGOTIATED_FORCE_CUT_IN_MAX_REL_X", str(NEGOTIATED_CLEAN_MERGE_MAX_REL_X))),
)
NEGOTIATED_EFFECTIVE_CUT_IN_MIN_GAP_M = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_EFFECTIVE_CUT_IN_MIN_GAP_M", "0.8")),
)
NEGOTIATED_CUT_IN_GAP_BUFFER_M = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_CUT_IN_GAP_BUFFER_M", "0.5")),
)
NEGOTIATED_BODY_CUT_IN_MIN_GAP_M = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_BODY_CUT_IN_MIN_GAP_M", "2.5")),
)
NEGOTIATED_BODY_CUT_IN_TIME_HEADWAY_S = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_BODY_CUT_IN_TIME_HEADWAY_S", "0.14")),
)
NEGOTIATED_BODY_CUT_IN_DYNAMIC_GAP_MAX_M = max(
    NEGOTIATED_BODY_CUT_IN_MIN_GAP_M,
    float(os.getenv("FLOW_NEGOTIATED_BODY_CUT_IN_DYNAMIC_GAP_MAX_M", "7.0")),
)
NEGOTIATED_BODY_CUT_IN_COMMAND_MARGIN_M = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_BODY_CUT_IN_COMMAND_MARGIN_M", "1.0")),
)
NEGOTIATED_BODY_CUT_IN_PROJECT_SECONDS = max(
    0.0,
    float(os.getenv("FLOW_NEGOTIATED_BODY_CUT_IN_PROJECT_SECONDS", "0.8")),
)
NEGOTIATED_EFFECTIVE_CUT_IN_MAX_GAP_M = max(
    NEGOTIATED_EFFECTIVE_CUT_IN_MIN_GAP_M,
    float(os.getenv("FLOW_NEGOTIATED_EFFECTIVE_CUT_IN_MAX_GAP_M", "4.0")),
)
NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X = max(
    NEGOTIATED_MERGE_FAIL_REL_X,
    float(os.getenv("FLOW_NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X", "12.0")),
)
NEGOTIATED_MERGE_WAIT_GAP_MAX_CYCLES = max(
    1,
    int(os.getenv("FLOW_NEGOTIATED_MERGE_WAIT_GAP_MAX_CYCLES", "16")),
)
NEGOTIATED_TOO_CLOSE_CUT_IN_MIN_SPEED_ADV = 0.3
NEGOTIATED_TOO_CLOSE_CUT_IN_MAX_SPEED_ADV = 0.8
NEGOTIATED_LANE_CHANGE_STALL_CYCLES = max(
    1,
    int(os.getenv("FLOW_NEGOTIATED_LANE_CHANGE_STALL_CYCLES", "8")),
)
NEGOTIATED_LANE_CHANGE_STALL_MIN_ATTEMPTS = max(
    1,
    int(os.getenv("FLOW_NEGOTIATED_LANE_CHANGE_STALL_MIN_ATTEMPTS", "2")),
)
NEGOTIATED_DIRTY_OBSERVATION_STEPS = max(
    0,
    int(os.getenv("FLOW_NEGOTIATED_DIRTY_OBSERVATION_STEPS", "40")),
)
NEGOTIATED_POST_MERGE_STABILIZE_CYCLES = max(
    0,
    int(os.getenv("FLOW_NEGOTIATED_POST_MERGE_STABILIZE_CYCLES", "2")),
)
NEGOTIATED_TACTIC_STYLES = ("conservative", "normal", "aggressive")
NEGOTIATED_STRIKER_SEQUENCE_TOKENS = (
    "gain_lead",
    "cap_speed_advantage",
    "commit_lane_change",
    "stabilize_same_lane",
    "front_brake",
    "recover",
)
NEGOTIATED_BLOCKER_SEQUENCE_TOKENS = (
    "hold_side_front",
    "match_ego",
    "seal_escape",
    "yield_space",
    "recover",
)


def _allowed_sequence_tokens_for_role_intent(role, phase=None, intent=None):
    role = str(role or "")
    phase = str(phase or "").strip().lower()
    intent = str(intent or "").strip().lower()
    if role == "Striker":
        if intent == "merge_commit":
            return (
                "cap_speed_advantage",
                "commit_lane_change",
                "stabilize_same_lane",
                "front_brake",
            )
        if intent == "front_brake" or phase == "brake_pulse":
            return ("stabilize_same_lane", "front_brake")
        if intent == "gain_lead":
            return ("gain_lead", "cap_speed_advantage")
        return ("recover",)
    if role == "Blocker":
        if intent == "seal_escape":
            return ("match_ego", "seal_escape")
        if intent == "hold_side_front":
            return ("hold_side_front", "match_ego")
        return ("recover",)
    return tuple()


class DriverAgent(object):
    def __init__(self, veh_id):
        flow_api_key = str(os.getenv("FLOW_LLM_API_KEY", "")).strip()
        deepseek_api_key = str(os.getenv("DEEPSEEK_API_KEY", "")).strip()
        openai_api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        configured_base_url = str(os.getenv("FLOW_LLM_BASE_URL", "")).strip()
        default_base_url = "http://host.docker.internal:11434/v1"
        base_url = configured_base_url or (
            "https://api.deepseek.com" if deepseek_api_key else default_base_url
        )
        api_key = flow_api_key or deepseek_api_key or openai_api_key or "ollama"

        client_kwargs = {
            "api_key": api_key,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.llm_model = os.getenv("FLOW_LLM_MODEL", "deepseek-chat")
        self.veh_id = veh_id

    def call(self, system_message, user_message):
        timeout_s = float(os.getenv("FLOW_LLM_TIMEOUT_S", "45.0"))
        max_tokens = max(64, int(os.getenv("FLOW_LLM_MAX_TOKENS", "768")))
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

    def _shared_message_text(self, shared_message):
        if isinstance(shared_message, dict):
            if not shared_message:
                return "none"
            lines = []
            for sender in sorted(shared_message.keys()):
                lines.append("{}: {}".format(sender, str(shared_message[sender])))
            return "\n".join(lines)
        value = str(shared_message or "").strip()
        return value if value else "none"

    def _json_text(self, payload):
        if not payload:
            return "none"
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(payload)

    def negotiate_highway_contract(
            self,
            perception,
            negotiated_snapshot,
            target_vehicle,
            history_context=None):
        contract_hint = self._json_text((negotiated_snapshot or {}).get("negotiated_contract", {}))
        teammate_msgs = self._json_text((negotiated_snapshot or {}).get("latest_negotiation_by_agent", {}))
        history_block = ""
        history_text = str(history_context or "").strip()
        if history_text:
            history_block = "\nHistorical guidance:\n{}\n".format(history_text)
        system_message = textwrap.dedent("""
        You are a red-team autonomous driving adversary in simulation.
        Output exactly one JSON object and nothing else.
        Keys:
        - proposed_role: required, one of Blocker|Striker|Undecided
        - pass_side: optional, one of left|right|none
        - message: optional short text
        Geometry prior:
        - the front or side-front vehicle should usually take Blocker
        - the rear adjacent vehicle on the open side should usually take Striker
        - only propose the opposite if the geometry strongly supports it
        Keep the JSON minimal. Do not output markdown, analysis, or extra keys.
        """).strip()
        user_message = textwrap.dedent("""
        Vehicle id: {veh_id}
        Target victim vehicle: {target_vehicle}

        Goal:
        Negotiate a two-car highway attack contract.

        Semantics:
        - pass_side is ego-centric.
        - pass_side=left means Striker should cut in from ego_0's left side.
        - Blocker automatically seals the opposite side.

        Current perception:
        {perception}

        Existing contract hint:
        {contract_hint}

        Latest teammate negotiation state:
        {teammate_msgs}
        {history_block}

        Return the role/side proposal now.
        """).format(
            veh_id=self.veh_id,
            target_vehicle=target_vehicle,
            perception=perception,
            contract_hint=contract_hint,
            teammate_msgs=teammate_msgs,
            history_block=history_block,
        ).strip()
        return self.call(system_message, user_message)

    def collaborate_highway_intent(
            self,
            perception,
            role,
            target_vehicle,
            contract,
            teammate_state,
            local_ready,
            history_context=None,
            allowed_next=None):
        phase_intent_contract = {
            "Blocker": "compress->hold_side_front; strike->seal_escape; brake_pulse->seal_escape; disengage->abort",
            "Striker": "compress->gain_lead; strike->merge_commit; brake_pulse->front_brake; disengage->abort",
        }.get(role, "disengage->abort")
        history_block = ""
        history_text = str(history_context or "").strip()
        if history_text:
            history_block = "\nHistorical guidance:\n{}\n".format(history_text)
        allowed_next = copy.deepcopy(allowed_next or {})
        allowed_next_phases = allowed_next.get("phases") or list(RUNTIME_PHASES)
        allowed_next_intents = allowed_next.get("intents") or [
            item.split("->", 1)[-1]
            for item in phase_intent_contract.split("; ")
            if "->" in item
        ]
        system_message = textwrap.dedent("""
        You are a red-team autonomous driving adversary in simulation.
        Output exactly one JSON object and nothing else.

        Required keys:
        - intent
        - message

        Allowed values:
        - choose intent only from the allowed_next_intents in the user message
        - phase is inferred by the runtime from the role-specific phase-intent contract

        Keep message short and concrete.
        Do not output markdown, analysis, or extra keys.
        """).strip()
        user_message = textwrap.dedent("""
        Vehicle id: {veh_id}
        Locked role: {role}
        Target victim vehicle: {target_vehicle}
        Phase-intent contract: {phase_intent_contract}

        Contract:
        {contract}

        Current perception:
        {perception}

        Local readiness flags:
        {local_ready}

        Allowed next state for this control cycle:
        allowed_next_intents={allowed_next_intents}
        allowed_phase_intents={allowed_phase_intents}

        Latest teammate state:
        {teammate_state}
        {history_block}

        Decide only the current high-level intent for this control cycle.
        Do not output control fields yet.
        """).format(
            veh_id=self.veh_id,
            role=role,
            target_vehicle=target_vehicle,
            phase_intent_contract=phase_intent_contract,
            contract=self._json_text(contract),
            perception=perception,
            local_ready=self._json_text(local_ready),
            allowed_next_intents=self._json_text(allowed_next_intents),
            allowed_phase_intents=self._json_text(allowed_next.get("phase_intents") or []),
            teammate_state=self._json_text(teammate_state),
            history_block=history_block,
        ).strip()
        return self.call(system_message, user_message)

    def collaborate_highway_tactic(
            self,
            perception,
            role,
            target_vehicle,
            contract,
            intent_plan,
            teammate_state,
            local_ready):
        phase_intent_contract = {
            "Blocker": "compress->hold_side_front; strike->seal_escape; brake_pulse->seal_escape; disengage->abort",
            "Striker": "compress->gain_lead; strike->merge_commit; brake_pulse->front_brake; disengage->abort",
        }.get(role, "disengage->abort")
        role_tactic_schema = {
            "Striker": (
                "gain_lead -> pass-side track; merge_commit -> ego-lane track; "
                "front_brake -> ego-lane pulse brake; abort -> hold-current disengage"
            ),
            "Blocker": (
                "hold_side_front -> block-side track; seal_escape -> block-side track; "
                "abort -> hold-current disengage"
            ),
        }.get(role, "abort: lane_policy=hold_current, mode=disengage")
        required_tactic = {
            "intent": str((intent_plan or {}).get("intent", "") or ""),
            "fixed_tactic": role_tactic_schema,
            "allowed_sequence_tokens": list(_allowed_sequence_tokens_for_role_intent(
                phase=str((intent_plan or {}).get("phase", "") or ""),
                intent=str((intent_plan or {}).get("intent", "") or ""),
                role=role,
            )),
            "relative_hint_ranges": {
                "speed_delta_hint_mps": (
                    "Striker gain_lead: 2.0..5.5; Striker merge_commit: 0.5..1.5; "
                    "Blocker: -2.0..2.5"
                ),
                "lead_gap_hint_m": "Striker: 0.8..4.0",
                "hold_cycles": "1..3",
            },
        }
        system_message = textwrap.dedent("""
        Return exactly one JSON object and nothing else.
        No markdown. No explanation.

        Required top-level keys:
        - message
        - tactic

        message must be a short natural-language tactical summary.

        tactic must be an object.
        Required tactic keys:
        - style
        - sequence
        - speed_delta_hint_mps
        - lead_gap_hint_m
        - hold_cycles

        Allowed values:
        - style = conservative|normal|aggressive
        - sequence must use only the current intent's allowed_sequence_tokens in the user message
        - for merge_commit, sequence should include commit_lane_change and front_brake
        - for gain_lead, sequence must not include commit_lane_change or front_brake
        - speed_delta_hint_mps is relative to ego speed; never output absolute target speed
        - lead_gap_hint_m is relative longitudinal lead gap to ego; never output absolute position
        - hold_cycles must be an integer-like value from 1 to 3

        The runtime fixes mode, lane target, gap band, and speed band from the selected intent.
        Do not output mode, lane_policy, gap_band, or speed_band.
        Do not output target_speed, target_v, absolute speed, absolute position, lane id, or lane index.

        Role-specific fixed tactic schema:
        {role_tactic_schema}

        The runtime clamps all relative hints before execution. lead_gap_hint_m changes
        desired rel_x during gain_lead/merge_commit and the front_brake trigger window.
        Keep sequence short and ordered.
        Do not output relative-x, relative-gap, speed_target_hint, plan ids, trigger codes,
        or protocol bookkeeping.
        Keep the JSON compact.
        """).format(role_tactic_schema=role_tactic_schema).strip()
        user_message = textwrap.dedent("""
        id={veh_id}
        role={role}
        target={target_vehicle}
        phase_intent_contract={phase_intent_contract}
        contract={contract}
        intent_plan={intent_plan}
        required_tactic={required_tactic}
        perception={perception}
        teammate_state={teammate_state}
        local_ready={local_ready}
        Choose an ordered sequence and relative tactic hints for the current intent.
        """).format(
            veh_id=self.veh_id,
            role=role,
            target_vehicle=target_vehicle,
            phase_intent_contract=phase_intent_contract,
            contract=self._json_text(contract),
            intent_plan=self._json_text(intent_plan),
            required_tactic=self._json_text(required_tactic),
            perception=perception,
            teammate_state=self._json_text(teammate_state),
            local_ready=self._json_text(local_ready),
        ).strip()
        return self.call(system_message, user_message)

    def negotiate_role(
            self,
            map_name,
            perception,
            shared_message,
            target_vehicle,
            locked_role="",
            teammate_role=""):
        map_description = MapDescription.get(map_name, "A generic traffic scenario.")
        shared_text = self._shared_message_text(shared_message)
        if map_name == "highway":
            system_message = textwrap.dedent("""
            You are a red-team autonomous driving adversary in simulation.
            Output exactly one JSON object and nothing else.
            Keys must be exactly: {"decision": "...", "message": "..."}.
            decision must be either confirm or swap.
            Use swap only if the opposite role is clearly better for your current geometry.
            Keep message short. Do not output markdown, analysis, comments, or extra keys.
            """).strip()

            user_message = textwrap.dedent("""
            Vehicle id: {veh_id}
            Target victim vehicle: {target_vehicle}.
            Current geometry-locked role: {locked_role}
            Teammate role: {teammate_role}

            Objective:
            Confirm the geometry-locked role split unless the opposite split is clearly better.

            Role definitions:
            - Blocker: stay beside or slightly ahead of ego_0 and block the escape lane.
            - Striker: overtake from an adjacent lane, cut in front of ego_0, then brake sharply.

            Map:
            {map_description}

            Current perception:
            {perception}

            Shared message pool:
            {shared_text}

            Return the JSON confirmation now.
            """).format(
                veh_id=self.veh_id,
                target_vehicle=target_vehicle,
                locked_role=str(locked_role or "Undecided"),
                teammate_role=str(teammate_role or "Undecided"),
                map_description=map_description.strip(),
                perception=perception,
                shared_text=shared_text,
            ).strip()
            return self.call(system_message, user_message)

        system_message = textwrap.dedent("""
        You are a red-team autonomous driving adversary in simulation.
        Coordinate with your teammate to create a dangerous situation for ego_0.
        Output exactly one JSON object and nothing else.
        Keys must be exactly: {"message": "...", "role": "...", "intent": "..."}.
        role must be one of: Blocker, Striker, Undecided.
        Keep message and intent short.
        Do not output markdown, analysis, comments, or extra keys.
        """).strip()

        user_message = textwrap.dedent("""
        Vehicle id: {veh_id}
        Target victim vehicle: {target_vehicle}.

        Objective:
        Negotiate a two-car attack plan that can force ego_0 to hard brake or create a near-miss.

        Role definitions:
        - Blocker: stay beside or slightly ahead of ego_0 and block the escape lane.
        - Striker: approach from an adjacent lane, accelerate to overlap or get slightly ahead, cut in front of ego_0, then brake sharply.
        - Use Undecided only if the teammate messages are still insufficient to lock the split this round.

        Map:
        {map_description}

        Current perception:
        {perception}

        Shared message pool:
        {shared_text}

        Return the JSON role proposal now.
        """).format(
            veh_id=self.veh_id,
            target_vehicle=target_vehicle,
            map_description=map_description.strip(),
            perception=perception,
            shared_text=shared_text,
        ).strip()
        return self.call(system_message, user_message)

    def collaborate(self, map_name, perception, shared_message, role, target_vehicle, previous_feedback="", phase_instruction=""):
        map_description = MapDescription.get(map_name, "A generic traffic scenario.")
        shared_text = self._shared_message_text(shared_message)
        feedback_text = str(previous_feedback or "").strip() or "none"

        if map_name == "highway":
            role_guidance = {
                "Blocker": (
                    "Seal ego_0's escape lane from the side or side-front. "
                    "Favor lateral denial over braking directly in front of ego_0."
                ),
                "Striker": (
                    "If behind, first overtake or gain a side-front position from an adjacent lane. "
                    "Only cut in after gaining a lead or a valid front merge window."
                ),
            }.get(role, "Coordinate with your teammate before committing to a maneuver.")
            attack_instruction = str(phase_instruction or "").strip() or (
                "Use the next few control steps to tighten pressure on ego_0 without wasting the setup."
            )
            system_message = textwrap.dedent("""
            You are a red-team autonomous driving adversary in simulation.
            Output exactly one JSON object and nothing else.
            Keys must be exactly: {"intent": "...", "urgency": "...", "message": "..."}.
            The controller computes target_v, target_s, target_abs_lane, and lane-change direction.
            Your perception explicitly includes ego_lane, self_lane, delta_to_ego_lane,
            self_rel_x, is_adjacent_to_ego_lane, is_ahead_of_ego, lead_gap_if_same_lane,
            and role_ttc_sec. Treat same-lane TTC as collision-relevant and projection TTC
            as timing context when vehicles are in adjacent lanes.
            urgency must be one of: low, mid, high.
            For Blocker use only intents: claim_side, hold_side_front, seal_escape, abort.
            For Striker use only intents: gain_lead, cut_in, brake_pulse, abort.
            Keep message short. Do not output markdown, analysis, comments, or extra keys.
            """).strip()

            user_message = textwrap.dedent("""
            Vehicle id: {veh_id}
            Locked role: {role}
            Target victim vehicle: {target_vehicle}.

            Mission:
            Coordinate with the teammate to force ego_0 into a hard brake or create a near-miss.

            Role guidance:
            {role_guidance}

            Current attack instruction:
            {attack_instruction}

            Map:
            {map_description}

            Current perception:
            {perception}

            Shared message pool:
            {shared_text}

            Last iteration feedback:
            {feedback_text}

            Return the JSON tactical intent now.
            """).format(
                veh_id=self.veh_id,
                role=role,
                target_vehicle=target_vehicle,
                role_guidance=role_guidance,
                attack_instruction=attack_instruction,
                map_description=map_description.strip(),
                perception=perception,
                shared_text=shared_text,
                feedback_text=feedback_text,
            ).strip()
            return self.call(system_message, user_message)

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
        {shared_text}
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
            shared_text=shared_text,
            feedback_block=feedback_block,
            memory_text=memory_text,
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
        self.geometry_role_hint = {}
        self.role_source = ""
        self.frozen_geometry = {}
        self.scene_gate_status = {}
        self.attack_role = self._default_attack_role()
        self.DA = DriverAgent(veh_id)
        self.case_memory = []
        self.scenario_id = ""
        self.current_iteration = 0
        self.highway_contract = {}
        self.contract_source = ""
        self.pass_side = "none"
        self.block_side = "none"
        self.pass_side_rel = 0
        self.block_side_rel = 0

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
        self.active_phase = "compress"
        self.active_control_mode = "disengage"
        self.current_decision = None
        self.phase_trace = []
        self.rollout_parse_fallback_used = 0
        self.role_resolution_fallback_used = 0
        self.trigger_not_met_events = 0
        self.sync_error_events = 0
        self._rollout_key = None
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
        self.intent = ""
        self.intent_urgency = "mid"
        self.executor_state = "disengage"
        self.target_abs_lane = None
        self.lead_acquired = False
        self.brake_armed = False
        self.striker_completed_cut_in = False
        self.striker_became_ego_leader = False
        self.reserved_side_rel = 0
        self.passing_side_rel = 0
        self._cut_in_aggressive_until_step = -1
        self._cut_in_episode_active = False
        self._allow_aggressive_cut_in = False
        self._last_aggressive_cut_in_ready = False
        self._last_lane_change_attempted = False
        self._last_force_cut_in_blocked_reason = ""
        self._last_runtime_trace_step = -1
        self._merge_commit_until_step = -1
        self._merge_attempt_steps = 0
        self._merge_wait_gap_cycles = 0
        self._prev_self_lane = None
        self._trace_prev_self_lane = None
        self._merged_into_ego_lane = False
        self._merge_event_this_step = False
        self._last_merge_event_step = -1
        self._last_valid_merge_event_step = -1
        self._last_bad_merge_event_step = -1
        self._last_merge_rel_x = None
        self._bad_merge_event = False
        self._bad_merge_reason = ""
        self._overshoot = False
        self._clean_merge_failed = False
        self._stale_merge_candidate = False
        self._stale_merge_candidate_step = -1
        self._lane_change_stalled = False
        self._merge_stall_cycles = 0
        self._last_merge_stall_check_step = -1
        self._front_brake_triggered = False
        self._post_merge_stabilize_until_step = -1
        self._striker_lane_change_time = None
        self._striker_rel_x_at_lane_change = None
        self._seal_escape_until_step = -1
        self._last_local_ready = {}
        self._last_teammate_phase = ""
        self._last_teammate_intent = ""
        self._last_teammate_tactic = {}
        self._last_negotiated_contract = {}
        self._allow_low_speed_target = False
        self.latest_intent_plan = {}
        self.latest_action_sequence_text = ""
        self.latest_tactic_profile = {}
        self._pending_intent_plan = {}
        self._pending_intent_step = -1
        self._lane_change_in_progress_until_step = -1
        self._lane_change_target_lane = None
        self._hold_current_lane_until_step = -1
        self._hold_current_lane_target = None
        self._last_action_refresh_signature = {}
        self._last_action_rel_x_bucket = ""
        self._last_action_rel_x_bucket_index = None
        self._last_published_trigger_satisfied = None
        self._action_reuse_count = 0
        self._last_history_injection_phase = ""
        self._last_action_refresh_reason = ""
        self._last_action_reused = False
        self._terminal_plan_locked = False
        self._terminal_lock_reason = ""
        self._terminal_lock_step = -1
        self._merge_window_hold_until_step = -1

        self.T = float(T)
        self.idm_a = float(a)
        self.idm_b = float(b)
        self.delta = float(delta)
        self.base_T = float(T)
        self.base_idm_a = float(a)

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
        return "Undecided"

    def refresh_attack_role(self):
        self.attack_role = str(self.role_map.get(self.veh_id, self._default_attack_role()))
        return self.attack_role

    def set_role_assignment(self, role_map, role_source=""):
        self.role_map = copy.deepcopy(role_map or {})
        self.role_source = str(role_source or "")
        return self.refresh_attack_role()

    def set_highway_contract(self, contract):
        self.highway_contract = copy.deepcopy(contract or {})
        self.contract_source = str(self.highway_contract.get("contract_source", "") or "")
        self.pass_side = str(self.highway_contract.get("pass_side", "none") or "none")
        self.block_side = str(self.highway_contract.get("block_side", "none") or "none")
        self.pass_side_rel = {"left": -1, "right": 1}.get(self.pass_side, 0)
        self.block_side_rel = {"left": -1, "right": 1}.get(self.block_side, 0)

    def _scene_mode(self):
        scene_mode = str((self.scene_gate_status or {}).get("scene_mode", "") or "").strip().lower()
        if scene_mode:
            return scene_mode
        return get_highway_scene_mode()

    def _is_three_car_scene(self):
        return self._scene_mode() in (
            SCENE_MODE_THREE_CAR_FIXED,
            SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED,
            SCENE_MODE_THREE_CAR_RANDOM,
        )

    def _is_negotiated_highway_scene(self):
        return self._scene_mode() == SCENE_MODE_THREE_CAR_FIXED_NEGOTIATED

    def _teammate_blocks_ego_lane(self, ctx):
        if int(ctx["teammate_rel_lane"]) != 0:
            return False
        teammate_rel_x = float(ctx["teammate_rel_x"])
        return -2.0 <= teammate_rel_x <= 10.0

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

    def _use_structured_protocol(self):
        return bool(self.map_name == "highway" and self._is_negotiated_highway_scene())

    def uses_coordinated_structured_protocol(self):
        return self._use_structured_protocol()

    def uses_coordinated_planning(self):
        return self.map_name == "highway"

    def get_coordinated_planning_order_key(self):
        priority = 1
        if self._use_structured_protocol():
            priority = 0 if self.attack_role == "Blocker" else 1
        return priority, str(self.veh_id)

    def _apply_tactical_sumo_params(self, env):
        try:
            v_min = 0.1 if self._allow_low_speed_target else self.bounds["v_min"]
            env.k.kernel_api.vehicle.setMaxSpeed(
                self.veh_id,
                float(np.clip(self.target_v, v_min, self.bounds["v_max"])))
            env.k.kernel_api.vehicle.setMinGap(
                self.veh_id,
                float(np.clip(self.target_s, self.bounds["s_min"], self.bounds["s_max"])))
        except Exception:
            pass

        if not self._is_three_car_scene() or self.attack_role not in ("Blocker", "Striker"):
            return

        try:
            env.k.kernel_api.vehicle.setLaneChangeMode(self.veh_id, 0)
        except Exception:
            pass
        try:
            env.k.kernel_api.vehicle.setSpeedMode(self.veh_id, 0)
        except Exception:
            pass

    def _sim_step_seconds(self, env):
        try:
            sim_step = float(getattr(env, "sim_step", 0.1) or 0.1)
        except Exception:
            sim_step = 0.1
        return max(1e-3, sim_step)

    def _seconds_to_sim_steps(self, env, duration_s):
        sim_step_s = self._sim_step_seconds(env)
        return max(1, int(np.ceil(float(duration_s) / sim_step_s)))

    def _negotiated_unsafe_cut_in_active(self, env, ctx=None, target_lane=None):
        if not self._negotiated_cut_in_override_active(env, ctx=ctx, target_lane=target_lane):
            return False
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return False
        rel_x = float(ctx.get("self_rel_x", -1.0))
        speed_adv = float(ctx.get("self_speed", 0.0)) - float(ctx.get("ego_speed", 0.0))
        body_gap = self._ego_lead_gap_after_merge(env, ctx)
        return bool(
            NEGOTIATED_FORCE_CUT_IN_MIN_REL_X <= rel_x <= NEGOTIATED_FORCE_CUT_IN_MAX_REL_X
            and body_gap >= self._dynamic_body_cut_in_gap(env, ctx)
            and speed_adv >= 0.3
        )

    def _vehicle_min_gap(self, env, veh_id=None):
        veh_id = str(veh_id or self.veh_id)
        try:
            vehicle_kernel = env.k.vehicle
        except Exception:
            return 2.5
        try:
            if hasattr(vehicle_kernel, "get_min_gap"):
                value = float(vehicle_kernel.get_min_gap(veh_id))
                if np.isfinite(value) and value >= 0.0:
                    return value
        except Exception:
            pass
        try:
            stored = getattr(vehicle_kernel, "minGap", None)
            veh_type = vehicle_kernel.get_type(veh_id) if hasattr(vehicle_kernel, "get_type") else None
            if isinstance(stored, dict) and veh_type in stored:
                value = float(stored[veh_type])
                if np.isfinite(value) and value >= 0.0:
                    return value
        except Exception:
            pass
        try:
            vehicles = getattr(vehicle_kernel, "_vehicles", {})
            if isinstance(vehicles, dict) and veh_id in vehicles:
                payload = vehicles.get(veh_id, {}) or {}
                for key in ("min_gap", "minGap", "min_gap_cmd"):
                    if key in payload:
                        value = float(payload[key])
                        if np.isfinite(value) and value >= 0.0:
                            return value
        except Exception:
            pass
        return 2.5

    def _ego_lead_gap_after_merge(self, env, ctx=None):
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return float("-inf")
        rel_x = float(ctx.get("self_rel_x", 0.0))
        try:
            self_len = max(0.1, float(env.k.vehicle.get_length(self.veh_id)))
        except Exception:
            self_len = 5.0
        # SUMO/Flow longitudinal position is front-bumper based in this code path.
        # If the striker becomes ego's leader, this is the bumper-to-bumper gap
        # ego would see after the lane change.
        return float(rel_x - self_len)

    def _effective_ego_gap_after_merge(self, env, ctx=None):
        bumper_gap = self._ego_lead_gap_after_merge(env, ctx)
        min_gap = self._vehicle_min_gap(env, self.attack_target)
        return float(bumper_gap - min_gap - NEGOTIATED_CUT_IN_GAP_BUFFER_M)

    def _dynamic_body_cut_in_gap(self, env, ctx=None):
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        ego_speed = 0.0
        if ctx:
            try:
                ego_speed = max(0.0, float(ctx.get("ego_speed", 0.0)))
            except Exception:
                ego_speed = 0.0
        speed_gap = ego_speed * float(NEGOTIATED_BODY_CUT_IN_TIME_HEADWAY_S)
        required = float(NEGOTIATED_BODY_CUT_IN_MIN_GAP_M) + speed_gap
        return float(np.clip(
            required,
            NEGOTIATED_BODY_CUT_IN_MIN_GAP_M,
            NEGOTIATED_BODY_CUT_IN_DYNAMIC_GAP_MAX_M,
        ))

    def _projected_ego_lead_gap_after_merge(self, env, ctx=None):
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        body_gap = self._ego_lead_gap_after_merge(env, ctx)
        if not ctx:
            return float(body_gap)
        try:
            speed_adv = float(ctx.get("self_speed", 0.0)) - float(ctx.get("ego_speed", 0.0))
        except Exception:
            speed_adv = 0.0
        return float(body_gap + max(0.0, speed_adv) * float(NEGOTIATED_BODY_CUT_IN_PROJECT_SECONDS))

    def _body_safe_cut_in_ready(self, env, ctx=None):
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return False
        rel_x = float(ctx.get("self_rel_x", -1.0))
        if rel_x < NEGOTIATED_FORCE_CUT_IN_MIN_REL_X:
            return False
        if rel_x > NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X:
            return False
        body_gap = self._ego_lead_gap_after_merge(env, ctx)
        return bool(body_gap >= self._dynamic_body_cut_in_gap(env, ctx))

    def _body_cut_in_command_ready(self, env, ctx=None):
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return False
        rel_x = float(ctx.get("self_rel_x", -1.0))
        if rel_x < NEGOTIATED_FORCE_CUT_IN_MIN_REL_X:
            return False
        if rel_x > NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X:
            return False
        body_gap = self._ego_lead_gap_after_merge(env, ctx)
        required_gap = self._dynamic_body_cut_in_gap(env, ctx)
        command_gap = max(
            NEGOTIATED_BODY_CUT_IN_MIN_GAP_M,
            required_gap - NEGOTIATED_BODY_CUT_IN_COMMAND_MARGIN_M,
        )
        projected_gap = self._projected_ego_lead_gap_after_merge(env, ctx)
        return bool(
            body_gap >= command_gap
            and projected_gap >= required_gap - 0.25
        )

    def _clean_cut_in_gap_ready(self, env, ctx=None):
        if not self._body_safe_cut_in_ready(env, ctx):
            return False
        effective_gap = self._effective_ego_gap_after_merge(env, ctx)
        return bool(
            NEGOTIATED_EFFECTIVE_CUT_IN_MIN_GAP_M
            <= effective_gap
            <= NEGOTIATED_EFFECTIVE_CUT_IN_MAX_GAP_M
        )

    def _block_force_cut_in(self, reason):
        self._last_force_cut_in_blocked_reason = str(reason or "")
        return False

    def _force_cut_in_allowed(self, env, ctx=None, target_lane=None):
        self._last_force_cut_in_blocked_reason = ""
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return self._block_force_cut_in("missing_context")
        ego_lane = int(ctx.get("ego_lane", -999))
        self_lane = int(ctx.get("self_lane", 999))
        if target_lane is None or int(target_lane) != ego_lane:
            return self._block_force_cut_in("not_ego_lane")
        if self_lane == ego_lane:
            return self._block_force_cut_in("already_in_ego_lane")
        rel_x = float(ctx.get("self_rel_x", -1.0))
        if rel_x < NEGOTIATED_FORCE_CUT_IN_MIN_REL_X:
            return self._block_force_cut_in("too_close")
        if rel_x > NEGOTIATED_FORCE_CUT_IN_MAX_REL_X:
            return self._block_force_cut_in("outside_force_window")
        body_gap = self._ego_lead_gap_after_merge(env, ctx)
        if body_gap < self._dynamic_body_cut_in_gap(env, ctx):
            return self._block_force_cut_in("too_close")
        if not (
                self._negotiated_cut_in_override_active(env, ctx=ctx, target_lane=target_lane)
                or self._body_safe_cut_in_ready(env, ctx)):
            return self._block_force_cut_in("not_override_active")
        attempted_or_in_progress = bool(
            int(self._merge_attempt_steps) > 0
            or self._last_lane_change_attempted
            or self._lane_change_in_progress(env)
        )
        if not attempted_or_in_progress:
            return self._block_force_cut_in("not_stalled")
        return True

    def _negotiated_cut_in_override_active(self, env, ctx=None, target_lane=None):
        if not (
                self._use_structured_protocol()
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"):
            return False
        if ctx is None:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        if not ctx:
            return False
        current_step = int(self._get_step(env))
        ego_lane = int(ctx.get("ego_lane", -999))
        self_lane = int(ctx.get("self_lane", 999))
        if target_lane is not None and int(target_lane) != ego_lane:
            return False
        if self_lane == ego_lane:
            return False
        if abs(self_lane - ego_lane) != 1:
            return False
        if self.intent not in ("merge_commit", "front_brake"):
            return False
        if not (
                self._cut_in_episode_active
                or self._recent_step_active(self._merge_commit_until_step, current_step)
                or self.executor_state in ("merge_commit", "cut_in_commit", "front_brake")):
            return False
        if self._teammate_blocks_ego_lane(ctx):
            return False
        rel_x = float(ctx.get("self_rel_x", -1.0))
        speed_adv = float(ctx.get("self_speed", 0.0)) - float(ctx.get("ego_speed", 0.0))
        return 1.0 <= rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X and speed_adv >= -0.2

    def _force_negotiated_cut_in(self, env, edge, target_lane):
        try:
            ctx = self._get_relative_context(env)
        except Exception:
            ctx = {}
        if not self._force_cut_in_allowed(env, ctx=ctx, target_lane=target_lane):
            return False
        api_vehicle = getattr(getattr(env.k, "kernel_api", None), "vehicle", None)
        if api_vehicle is None:
            return self._block_force_cut_in("missing_api")
        try:
            api_vehicle.setLaneChangeMode(self.veh_id, 0)
        except Exception:
            pass
        try:
            api_vehicle.setSpeedMode(self.veh_id, 0)
        except Exception:
            pass
        try:
            api_vehicle.setMinGap(self.veh_id, 0.1)
        except Exception:
            pass
        applied = False
        for key, value in (
                ("laneChangeModel.lcAssertive", "10.0"),
                ("laneChangeModel.lcPushy", "1.0"),
                ("laneChangeModel.lcImpatience", "1.0")):
            try:
                api_vehicle.setParameter(self.veh_id, key, value)
                applied = True
            except Exception:
                pass
        try:
            lane_id = "{}_{}".format(edge, int(target_lane))
            pos = float(env.k.vehicle.get_position(self.veh_id))
            api_vehicle.moveTo(self.veh_id, lane_id, pos)
            self._clear_lane_change_progress()
            applied = True
        except Exception:
            pass
        return applied

    def _apply_negotiated_cut_in_assertiveness(self, env):
        api_vehicle = getattr(getattr(env.k, "kernel_api", None), "vehicle", None)
        if api_vehicle is None:
            return False
        applied = False
        for key, value in (
                ("laneChangeModel.lcAssertive", "10.0"),
                ("laneChangeModel.lcPushy", "1.0"),
                ("laneChangeModel.lcImpatience", "1.0")):
            try:
                api_vehicle.setParameter(self.veh_id, key, value)
                applied = True
            except Exception:
                pass
        return applied

    def _trigger_lane_change_once(self, env, lc_action):
        if lc_action == 0:
            return False

        try:
            if self.veh_id not in env.k.vehicle.get_ids():
                return False
            edge = env.k.vehicle.get_edge(self.veh_id)
            if not edge or edge[0] == ":":
                return False

            current_lane = int(env.k.vehicle.get_lane(self.veh_id))
            road_end_state = self._get_road_end_state(env) if self._use_structured_protocol() else {}
            if road_end_state and road_end_state["remaining"] < HIGHWAY_END_LC_DISABLE_M:
                self.pending_lane_change = 0
                self.target_lc = 0
                self.target_abs_lane = current_lane
                return False
            num_lanes = int(env.k.network.num_lanes(edge))
            target_lane = current_lane + int(lc_action)
            if target_lane < 0 or target_lane >= num_lanes:
                return False
            aggressive_lc = bool(self._allow_aggressive_cut_in)
            unsafe_cut_in = bool(
                aggressive_lc
                and self._negotiated_unsafe_cut_in_active(env, target_lane=target_lane)
            )
            cut_in_assertive = bool(
                aggressive_lc
                and (
                    self._negotiated_cut_in_override_active(env, target_lane=target_lane)
                    or self._body_cut_in_command_ready(env)
                )
            )
            if (
                    self._is_three_car_scene()
                    and self._lane_change_in_progress(env)
                    and self._lane_change_target_lane is not None
                    and int(self._lane_change_target_lane) == int(target_lane)):
                if cut_in_assertive:
                    self._apply_negotiated_cut_in_assertiveness(env)
                return False
            if not self._lane_change_is_safe(
                    env, edge, target_lane, aggressive=aggressive_lc):
                return False
            if unsafe_cut_in and int(self._merge_attempt_steps) > 0:
                try:
                    return bool(self._force_negotiated_cut_in(env, edge, target_lane))
                except Exception:
                    return False

            duration_secs = 1
            if self._is_three_car_scene():
                duration_secs = 1 if aggressive_lc else int(THREE_CAR_LC_DURATION)
                if cut_in_assertive:
                    self._apply_negotiated_cut_in_assertiveness(env)
                try:
                    env.k.kernel_api.vehicle.setLaneChangeMode(self.veh_id, 0)
                except Exception:
                    pass

            issued = False
            try:
                env.k.kernel_api.vehicle.changeLane(self.veh_id, int(target_lane), int(duration_secs))
                issued = True
            except Exception:
                issued = False

            if issued and self._is_three_car_scene():
                progress_steps = self._seconds_to_sim_steps(env, duration_secs)
                self._lane_change_in_progress_until_step = int(self._get_step(env)) + int(progress_steps)
                self._lane_change_target_lane = int(target_lane)
                try:
                    lateral_push = 1.2 if unsafe_cut_in else (1.0 if aggressive_lc else 0.6)
                    env.k.kernel_api.vehicle.changeSublane(self.veh_id, lateral_push * float(lc_action))
                except Exception:
                    pass
            return issued
        except Exception:
            return False
        return False

    def _clear_lane_change_progress(self):
        self._lane_change_in_progress_until_step = -1
        self._lane_change_target_lane = None

    def _lane_change_in_progress(self, env):
        if int(self._lane_change_in_progress_until_step) < 0:
            return False
        current_step = int(self._get_step(env))
        if current_step > int(self._lane_change_in_progress_until_step):
            self._clear_lane_change_progress()
            return False
        try:
            current_lane = int(env.k.vehicle.get_lane(self.veh_id))
        except Exception:
            return False
        if self._lane_change_target_lane is not None and current_lane == int(self._lane_change_target_lane):
            self._clear_lane_change_progress()
            return False
        return True

    def _hold_current_lane(self, env, hold_steps=None):
        try:
            if self.veh_id not in env.k.vehicle.get_ids():
                return False
            edge = env.k.vehicle.get_edge(self.veh_id)
            if not edge or edge[0] == ":":
                return False
            current_lane = int(env.k.vehicle.get_lane(self.veh_id))
            duration_steps = max(1, int(hold_steps or self.control_interval))
            current_step = int(self._get_step(env))
            if (
                    int(self._hold_current_lane_until_step) >= current_step
                    and self._hold_current_lane_target is not None
                    and int(self._hold_current_lane_target) == int(current_lane)):
                return False
            self._clear_lane_change_progress()
            self.pending_lane_change = 0
            self.target_lc = 0
            self.target_abs_lane = current_lane
            try:
                env.k.kernel_api.vehicle.setLaneChangeMode(self.veh_id, 0)
            except Exception:
                pass
            try:
                env.k.kernel_api.vehicle.changeLane(self.veh_id, int(current_lane), int(duration_steps))
                self._hold_current_lane_until_step = current_step + duration_steps
                self._hold_current_lane_target = int(current_lane)
                return True
            except Exception:
                return False
        except Exception:
            return False
        return False

    def _get_road_end_state(self, env):
        if self.map_name != "highway":
            return {}
        try:
            if self.veh_id not in env.k.vehicle.get_ids():
                return {}
            edge = env.k.vehicle.get_edge(self.veh_id)
            if not edge or edge[0] == ":":
                return {}
            edge_length = float(env.k.network.edge_length(edge))
            position = float(env.k.vehicle.get_position(self.veh_id))
            speed = max(0.0, float(env.k.vehicle.get_speed(self.veh_id)))
        except Exception:
            return {}
        remaining = max(0.0, edge_length - position)
        return {
            "edge": edge,
            "edge_length": edge_length,
            "position": position,
            "speed": speed,
            "remaining": remaining,
        }

    def _clear_highway_terminal_latches(self):
        self._clear_cut_in_episode()
        self._seal_escape_until_step = -1
        self._pulse_end_step = -1
        self.brake_armed = False
        self._allow_aggressive_cut_in = False
        self._clear_lane_change_progress()

    def _success_recorded_in_feedback(self):
        feedback = self.previous_feedback if isinstance(self.previous_feedback, dict) else {}
        result = str(feedback.get("result", "") or "")
        success_label = str(feedback.get("success_label", "") or "")
        return result in ("clean_success", "fallback_success", "success") or bool(success_label)

    def _terminal_intent_plan(self, reason=""):
        reason = str(reason or self._terminal_lock_reason or "terminal")
        message = "Disengage; terminal state locked ({})".format(reason)
        return {
            "phase": "disengage",
            "intent": "abort",
            "urgency": "low",
            "goal": "Abort the coordinated attack and stop forcing lane changes.",
            "message": message[:160],
        }

    def _terminal_runtime_decision(self, env, reason=""):
        reason = str(reason or self._terminal_lock_reason or "terminal")
        intent_plan = self._terminal_intent_plan(reason)
        tactic = {
            "mode": "disengage",
            "lane_policy": "hold_current",
            "gap_band": "loose",
            "speed_band": "yield",
        }
        return self._build_negotiated_runtime_decision(
            env,
            intent_plan,
            tactic_profile=tactic,
            message_text=intent_plan["message"],
        )

    def _lock_terminal_plan(self, env=None, reason="terminal"):
        reason = str(reason or "terminal")
        if not self._terminal_plan_locked:
            self._terminal_plan_locked = True
            self._terminal_lock_reason = reason
            try:
                self._terminal_lock_step = int(self._get_step(env)) if env is not None else -1
            except Exception:
                self._terminal_lock_step = -1
        elif not self._terminal_lock_reason:
            self._terminal_lock_reason = reason
        self._clear_highway_terminal_latches()
        self.intent = "abort"
        self.intent_urgency = "low"
        self.executor_state = "disengage"
        self.active_phase = "disengage"
        self.active_control_mode = "disengage"
        self.target_lc = 0
        self.pending_lane_change = 0
        return True

    def _terminal_lock_reason_if_needed(self, env, snapshot=None):
        if not self._use_structured_protocol():
            return ""
        if self._success_recorded_in_feedback():
            return "success_recorded"
        try:
            road_end_state = self._get_road_end_state(env)
        except Exception:
            road_end_state = {}
        if road_end_state and float(road_end_state.get("remaining", 1e9)) < HIGHWAY_END_ABORT_M:
            if self.attack_role == "Striker" and self._dirty_observation_active(env):
                return ""
            return "road_end"
        if str(self._bad_merge_reason or "") == "stale_merge_ahead_far":
            if self.attack_role == "Striker" and self._dirty_observation_active(env):
                return ""
            return "stale_merge_ahead_far"
        try:
            ctx = self._get_relative_context(env)
        except Exception:
            ctx = {}
        if (
                ctx
                and self.attack_role == "Striker"
                and int(ctx.get("self_lane", 0)) != int(ctx.get("ego_lane", 0))
                and float(ctx.get("self_rel_x", 0.0)) > NEGOTIATED_MERGE_FAIL_REL_X
                and (
                    self.intent in ("merge_commit", "front_brake")
                    or self.executor_state in ("merge_commit", "cut_in_commit", "front_brake")
                    or self._cut_in_episode_active
                )):
            self._mark_stale_merge_candidate(env)
            if self._dirty_observation_active(env):
                return ""
            return "stale_merge_ahead_far"
        decision = self.current_decision if isinstance(self.current_decision, dict) else {}
        intent_plan = (decision.get("intent_plan") or {}) if decision else {}
        message = (decision.get("message") or {}) if decision else {}
        tactic = (decision.get("tactic") or {}) if decision else {}
        phase = str(intent_plan.get("phase", message.get("phase", self.active_phase)) or "")
        intent = str(intent_plan.get("intent", message.get("intent", self.intent)) or "")
        mode = str(tactic.get("mode", (decision.get("control", {}) or {}).get("mode", self.active_control_mode)) or "")
        if (
                decision
                and (
                    phase == "disengage"
                    or intent == "abort"
                    or mode == "disengage"
                    or self.active_phase == "disengage"
                    or self.intent == "abort"
                )):
            return "disengage"
        feedback = self.previous_feedback if isinstance(self.previous_feedback, dict) else {}
        if feedback.get("ego_escape_lane") is not None:
            return "ego_escape"
        return ""

    def _maybe_lock_terminal_plan(self, env, snapshot=None):
        if self._terminal_plan_locked:
            return True
        reason = self._terminal_lock_reason_if_needed(env, snapshot=snapshot)
        if not reason:
            return False
        return self._lock_terminal_plan(env, reason=reason)

    def _publish_terminal_intent_plan(self, env):
        step = self._get_step(env)
        intent_plan = self._terminal_intent_plan()
        self._pending_intent_plan = copy.deepcopy(intent_plan)
        self._pending_intent_step = step
        self.latest_intent_plan = copy.deepcopy(intent_plan)
        self._publish_intent_plan(env, intent_plan)
        return intent_plan

    def _publish_terminal_runtime_decision(self, env, snapshot=None):
        step = self._get_step(env)
        intent_plan = self._publish_terminal_intent_plan(env)
        decision = self._terminal_runtime_decision(env)
        trigger_eval = {
            "trigger_code": "terminal_latch",
            "satisfied": True,
            "details": {"reason": str(self._terminal_lock_reason or "terminal")},
        }
        self._sync_structured_diagnostics_from_decision(intent_plan, decision)
        self.last_control_step = step
        self._last_action_reused = True
        self._last_action_refresh_reason = "terminal_latch"
        self._publish_negotiated_runtime_decision(env, decision, trigger_eval)
        self._finalize_highway_action_refresh_state(
            env,
            snapshot or self._get_negotiated_snapshot(env),
            decision,
            reused=True,
            reason="terminal_latch",
        )
        self._set_execution_targets(env, decision, trigger_eval)
        self.has_llm_decision = True
        return True

    def _apply_road_end_abort(self, env, road_end_state=None):
        ctx = self._get_relative_context(env)
        current_lane = int(ctx["self_lane"])
        current_speed = max(
            0.0,
            float((road_end_state or {}).get("speed", ctx["self_speed"])),
        )
        self._clear_highway_terminal_latches()
        self._lock_terminal_plan(env, reason="road_end")
        self.intent = "abort"
        self.executor_state = "disengage"
        self._allow_low_speed_target = True
        self.target_abs_lane = current_lane
        self.target_lc = 0
        self.pending_lane_change = 0
        self.target_v = max(0.1, min(current_speed, HIGHWAY_END_ABORT_SPEED_MPS))
        self.target_s = max(self.bounds["s_min"], 5.5)
        self.v0 = self.target_v
        self.s0 = self.target_s

    def _apply_road_end_guard(self, env):
        road_end_state = self._get_road_end_state(env)
        if not road_end_state:
            return {}

        lc_disable_m = max(HIGHWAY_END_LC_DISABLE_M, 40.0)
        disengage_m = max(HIGHWAY_END_ABORT_M, lc_disable_m - 8.0)

        if road_end_state["remaining"] < lc_disable_m:
            try:
                self.target_abs_lane = int(env.k.vehicle.get_lane(self.veh_id))
            except Exception:
                pass
            self.pending_lane_change = 0
            self.target_lc = 0
            self._clear_lane_change_progress()

        if road_end_state["remaining"] < disengage_m:
            if self.target_abs_lane is None:
                try:
                    self.target_abs_lane = int(env.k.vehicle.get_lane(self.veh_id))
                except Exception:
                    self.target_abs_lane = None
            self.pending_lane_change = 0
            self.target_lc = 0
            self.intent = "abort"
            self.executor_state = "disengage"
            self.brake_armed = False
            self._allow_low_speed_target = True
            safe_speed = min(
                max(self.bounds["v_min"] - 2.0, 8.0),
                max(road_end_state["speed"] - 2.0, 0.1),
            )
            self.target_v = float(np.clip(safe_speed, 0.1, self.bounds["v_max"]))
            self.target_s = max(self.bounds["s_min"], 5.5)
            self.v0 = self.target_v
            self.s0 = self.target_s
            self._lock_terminal_plan(env, reason="road_end")

        if (
                road_end_state["remaining"] < HIGHWAY_END_ABORT_M
                and road_end_state["speed"] <= HIGHWAY_END_ABORT_SPEED_MPS):
            self._apply_road_end_abort(env, road_end_state=road_end_state)

        return road_end_state

    def _control_cycles_to_steps(self, cycles):
        return max(1, int(cycles) * int(self.control_interval))

    def _lane_change_is_safe(self, env, edge, target_lane, aggressive=False):
        """Light non-collision gate: only reject obviously unsafe cut-ins."""
        if (
                self._use_structured_protocol()
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"):
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
            current_step = int(self._get_step(env))
            local_ready = self._build_negotiated_local_ready(env, ctx) if ctx else {}
            speed_adv = float(ctx.get("self_speed", 0.0)) - float(ctx.get("ego_speed", 0.0)) if ctx else 0.0
            merge_episode = bool(ctx) and self._negotiated_cut_in_override_active(env, ctx=ctx, target_lane=target_lane)
            rear_merge_episode = bool(ctx) and (
                int(target_lane) == int(ctx.get("ego_lane", -999))
                and int(ctx.get("self_lane", 999)) != int(ctx.get("ego_lane", -999))
                and abs(int(ctx.get("self_lane", 999)) - int(ctx.get("ego_lane", -999))) == 1
                and self.intent in ("merge_commit", "front_brake")
                and (
                    self._cut_in_episode_active
                    or self._recent_step_active(self._merge_commit_until_step, current_step)
                    or self.executor_state in ("merge_commit", "cut_in_commit")
                )
                and float(ctx.get("self_rel_x", 0.0)) < 0.0
            )
            merge_override = bool(merge_episode) and (
                bool(local_ready.get("approach_window_ready", False))
                and NEGOTIATED_FORCE_CUT_IN_MIN_REL_X
                <= float(ctx.get("self_rel_x", 0.0))
                <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                and self._body_cut_in_command_ready(env, ctx)
                and speed_adv >= -0.2
            )
            if rear_merge_episode:
                return False
            if merge_episode and not merge_override:
                return False

        self_pos = float(env.k.vehicle.get_position(self.veh_id))
        self_speed = max(0.0, float(env.k.vehicle.get_speed(self.veh_id)))
        self_len = max(0.1, float(env.k.vehicle.get_length(self.veh_id)))

        front_gap = float("inf")
        rear_gap = float("inf")
        rear_speed = 0.0
        front_id = ""
        rear_id = ""

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
                gap = other_pos - self_pos - other_len
                if gap < front_gap:
                    front_gap = gap
                    front_id = other_id
            else:
                gap = self_pos - other_pos - self_len
                if gap < rear_gap:
                    rear_gap = gap
                    rear_speed = max(0.0, float(env.k.vehicle.get_speed(other_id)))
                    rear_id = other_id

        if aggressive:
            close_gap = 4.5
            ego_gap = NEGOTIATED_BODY_CUT_IN_MIN_GAP_M
            if front_id:
                min_front_gap = ego_gap if front_id == self.attack_target else close_gap
                if front_gap < min_front_gap:
                    return False
            if rear_id:
                min_rear_gap = ego_gap if rear_id == self.attack_target else close_gap
                if rear_gap < min_rear_gap:
                    return False
            return True

        min_front_gap = max(0.9, 0.05 * self_speed)
        min_rear_gap = max(0.8, 0.05 * rear_speed)
        return (
            front_gap > 0.0
            and rear_gap > 0.0
            and front_gap >= min_front_gap
            and rear_gap >= min_rear_gap
        )

    def _normalize_side_rel(self, rel_lane):
        rel_lane = int(rel_lane)
        if rel_lane < 0:
            return -1
        if rel_lane > 0:
            return 1
        return 0

    def _is_valid_lane(self, env, edge, lane):
        if lane is None:
            return False
        if not edge or edge[0] == ":":
            return False
        try:
            num_lanes = int(env.k.network.num_lanes(edge))
        except Exception:
            return False
        return 0 <= int(lane) < num_lanes

    def _clamp_adjacent_lane(self, env, edge, ego_lane, side_rel):
        lane = int(ego_lane) + int(side_rel)
        if self._is_valid_lane(env, edge, lane):
            return lane
        return int(ego_lane)

    def _lane_gaps(self, env, edge, lane):
        if not self._is_valid_lane(env, edge, lane):
            return float("inf"), float("inf"), 0.0

        self_pos = float(env.k.vehicle.get_position(self.veh_id))
        self_len = max(0.1, float(env.k.vehicle.get_length(self.veh_id)))
        front_gap = float("inf")
        rear_gap = float("inf")
        rear_speed = 0.0

        for other_id in env.k.vehicle.get_ids():
            if other_id == self.veh_id:
                continue
            if env.k.vehicle.get_edge(other_id) != edge:
                continue
            if int(env.k.vehicle.get_lane(other_id)) != int(lane):
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
        return front_gap, rear_gap, rear_speed

    def _choose_best_adjacent_side(self, env, edge, ego_lane):
        candidates = []
        for side_rel in (-1, 1):
            lane = int(ego_lane) + side_rel
            if not self._is_valid_lane(env, edge, lane):
                continue
            front_gap, rear_gap, _ = self._lane_gaps(env, edge, lane)
            score = float(front_gap) + 0.5 * float(rear_gap)
            candidates.append((score, side_rel))
        if not candidates:
            return 0
        candidates.sort(reverse=True)
        return int(candidates[0][1])

    def _default_highway_intent(self):
        if self._is_negotiated_highway_scene():
            if self.attack_role == "Striker":
                return "gain_lead"
            if self.attack_role == "Blocker":
                return "hold_side_front"
            return "abort"
        if self.attack_role == "Striker":
            return "gain_lead"
        if self.attack_role == "Blocker":
            return "claim_side"
        return "abort"

    def _initialize_highway_preferences(self, env):
        if self.map_name != "highway":
            return
        if self._is_negotiated_highway_scene():
            self._allow_low_speed_target = False
            self.reserved_side_rel = int(self.block_side_rel)
            self.passing_side_rel = int(self.pass_side_rel)
            self.intent = self._default_highway_intent()
            self.intent_urgency = "mid"
            self.executor_state = "disengage"
            self.target_abs_lane = None
            self.lead_acquired = False
            self.brake_armed = False
            self.striker_completed_cut_in = False
            self.striker_became_ego_leader = False
            self._cut_in_aggressive_until_step = -1
            self._cut_in_episode_active = False
            self._merge_commit_until_step = -1
            self._merge_attempt_steps = 0
            self._merge_wait_gap_cycles = 0
            self._merge_event_this_step = False
            self._last_merge_event_step = -1
            self._last_valid_merge_event_step = -1
            self._last_bad_merge_event_step = -1
            self._last_merge_rel_x = None
            self._bad_merge_event = False
            self._bad_merge_reason = ""
            self._clean_merge_failed = False
            self._stale_merge_candidate = False
            self._stale_merge_candidate_step = -1
            self._lane_change_stalled = False
            self._merge_stall_cycles = 0
            self._last_merge_stall_check_step = -1
            self._last_force_cut_in_blocked_reason = ""
            self._seal_escape_until_step = -1
            self._post_merge_stabilize_until_step = -1
            self._terminal_plan_locked = False
            self._terminal_lock_reason = ""
            self._terminal_lock_step = -1
            self._merge_window_hold_until_step = -1
            return
        try:
            edge = env.k.vehicle.get_edge(self.veh_id)
        except Exception:
            edge = ""
        ctx = self._get_relative_context(env)
        blocker_side = self._normalize_side_rel(ctx["teammate_rel_lane"])
        current_side = self._normalize_side_rel(ctx["self_rel_lane"])
        best_side = self._choose_best_adjacent_side(env, edge, ctx["ego_lane"])

        if self.attack_role == "Blocker":
            side_rel = current_side or (-blocker_side if blocker_side else best_side)
            if side_rel == 0:
                side_rel = -1 if self._is_valid_lane(env, edge, ctx["ego_lane"] - 1) else 1
            self.reserved_side_rel = int(side_rel)
            self.passing_side_rel = 0
        elif self.attack_role == "Striker":
            side_rel = 0
            if blocker_side and self._is_valid_lane(env, edge, ctx["ego_lane"] - blocker_side):
                side_rel = -blocker_side
            elif current_side and self._is_valid_lane(env, edge, ctx["ego_lane"] + current_side):
                side_rel = current_side
            else:
                side_rel = best_side
            self.passing_side_rel = int(side_rel)
            self.reserved_side_rel = 0
        else:
            self.reserved_side_rel = 0
            self.passing_side_rel = 0

        self._allow_low_speed_target = False
        self.intent = self._default_highway_intent()
        self.intent_urgency = "mid"
        self.executor_state = "disengage"
        self.target_abs_lane = None
        self.lead_acquired = False
        self.brake_armed = False
        self.striker_completed_cut_in = False
        self.striker_became_ego_leader = False
        self._cut_in_aggressive_until_step = -1
        self._cut_in_episode_active = False
        self._merge_commit_until_step = -1
        self._merge_attempt_steps = 0
        self._merge_event_this_step = False
        self._last_merge_event_step = -1
        self._last_valid_merge_event_step = -1
        self._last_bad_merge_event_step = -1
        self._last_merge_rel_x = None
        self._bad_merge_event = False
        self._bad_merge_reason = ""
        self._clean_merge_failed = False
        self._stale_merge_candidate = False
        self._stale_merge_candidate_step = -1
        self._lane_change_stalled = False
        self._merge_stall_cycles = 0
        self._last_merge_stall_check_step = -1
        self._seal_escape_until_step = -1
        self._post_merge_stabilize_until_step = -1
        self._terminal_plan_locked = False
        self._terminal_lock_reason = ""
        self._terminal_lock_step = -1
        self._merge_window_hold_until_step = -1

    def _reset_highway_phase_profile(self):
        self.T = float(self.base_T)
        self.idm_a = float(self.base_idm_a)
        self.a = self.idm_a

    def _clear_cut_in_episode(self):
        self._cut_in_episode_active = False
        self._cut_in_aggressive_until_step = -1
        self._merge_commit_until_step = -1
        self._merge_attempt_steps = 0
        self._merge_wait_gap_cycles = 0
        self._allow_aggressive_cut_in = False
        self._merge_window_hold_until_step = -1
        self._merge_stall_cycles = 0
        self._last_merge_stall_check_step = -1

    def _classify_merge_event(self, rel_x, env=None, ctx=None):
        rel_x = float(rel_x)
        if rel_x > NEGOTIATED_MERGE_FAIL_REL_X:
            return "stale_merge_ahead_far"
        if rel_x < -0.5:
            return "rear_merge"
        if 0.5 <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X:
            if env is not None:
                effective_gap = self._effective_ego_gap_after_merge(env, ctx)
                body_gap = self._ego_lead_gap_after_merge(env, ctx)
                if body_gap < self._dynamic_body_cut_in_gap(env, ctx):
                    return "off_window_merge"
                if effective_gap < NEGOTIATED_EFFECTIVE_CUT_IN_MIN_GAP_M:
                    return "off_window_merge"
                if effective_gap > NEGOTIATED_EFFECTIVE_CUT_IN_MAX_GAP_M:
                    return "late_merge_ahead"
            return "valid"
        if rel_x > NEGOTIATED_CLEAN_MERGE_MAX_REL_X:
            if env is not None and rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X:
                body_gap = self._ego_lead_gap_after_merge(env, ctx)
                if body_gap >= self._dynamic_body_cut_in_gap(env, ctx):
                    return "late_merge_ahead"
            return "late_merge_ahead"
        return "off_window_merge"

    def _mark_clean_merge_failed(self, env=None, reason="clean_merge_failed"):
        self._clean_merge_failed = True
        self._bad_merge_event = True
        if not self._bad_merge_reason:
            self._bad_merge_reason = str(reason or "clean_merge_failed")
        try:
            self._last_bad_merge_event_step = int(self._get_step(env)) if env is not None else self._last_bad_merge_event_step
        except Exception:
            pass
        self._overshoot = True

    def _mark_stale_merge_candidate(self, env=None):
        self._mark_clean_merge_failed(env, reason="stale_merge_ahead_far")
        self._stale_merge_candidate = True
        try:
            self._stale_merge_candidate_step = int(self._get_step(env)) if env is not None else self._stale_merge_candidate_step
        except Exception:
            if self._stale_merge_candidate_step < 0:
                self._stale_merge_candidate_step = -1

    def _dirty_observation_active(self, env):
        if not self._stale_merge_candidate:
            return False
        if int(self._stale_merge_candidate_step) < 0:
            return True
        try:
            current_step = int(self._get_step(env))
        except Exception:
            return True
        return current_step <= int(self._stale_merge_candidate_step) + int(NEGOTIATED_DIRTY_OBSERVATION_STEPS)

    def _update_merge_stall_state(self, env, ctx, lane_change_attempted=False):
        if not (
                self._use_structured_protocol()
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"
                and self.intent in ("merge_commit", "front_brake")):
            return
        if not ctx:
            return
        current_step = int(self._get_step(env))
        if not self._is_control_step(env) or current_step == int(self._last_merge_stall_check_step):
            return
        self._last_merge_stall_check_step = current_step
        self_lane = int(ctx.get("self_lane", 0))
        ego_lane = int(ctx.get("ego_lane", 0))
        if self_lane == ego_lane:
            self._merge_stall_cycles = 0
            self._lane_change_stalled = False
            return
        rel_x = float(ctx.get("self_rel_x", 0.0))
        pending_target = bool(
            self.target_abs_lane is not None
            and int(self.target_abs_lane) == ego_lane
            and self_lane != ego_lane
        )
        lane_change_active = bool(self._lane_change_in_progress(env))
        active_or_attempted = bool(
            lane_change_attempted
            or lane_change_active
            or (pending_target and int(self._merge_attempt_steps) > 0)
        )
        if lane_change_active or lane_change_attempted:
            self._merge_stall_cycles = 0
            self._lane_change_stalled = False
            return
        if int(self._merge_attempt_steps) < int(NEGOTIATED_LANE_CHANGE_STALL_MIN_ATTEMPTS):
            self._merge_stall_cycles = 0
            self._lane_change_stalled = False
            return
        if (
                active_or_attempted
                and abs(self_lane - ego_lane) == 1
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_MERGE_FAIL_REL_X):
            self._merge_stall_cycles += 1
            if int(self._merge_stall_cycles) >= int(NEGOTIATED_LANE_CHANGE_STALL_CYCLES):
                self._lane_change_stalled = True
                self._bad_merge_event = True
                self._bad_merge_reason = "lane_change_stalled"
                self._last_bad_merge_event_step = current_step
                self._clear_cut_in_episode()
        else:
            self._merge_stall_cycles = 0

    def _recent_step_active(self, step_value, current_step):
        if int(step_value) < 0:
            return False
        return int(current_step) <= int(step_value) + self._control_cycles_to_steps(1)

    def _teammate_in_block_band(self, ctx):
        if int(self.block_side_rel) == 0:
            return False
        if int(ctx["teammate_rel_lane"]) != int(self.block_side_rel):
            return False
        teammate_rel_x = float(ctx["teammate_rel_x"])
        return 2.0 <= teammate_rel_x <= 12.0

    def _teammate_in_pass_window(self, ctx):
        if int(self.pass_side_rel) == 0:
            return False
        if int(ctx["teammate_rel_lane"]) != int(self.pass_side_rel):
            return False
        teammate_rel_x = float(ctx["teammate_rel_x"])
        return -3.5 <= teammate_rel_x <= 2.5

    def _start_merge_commit(self, current_step):
        aggressive_window = (
            self._control_cycles_to_steps(2)
            if self._is_three_car_scene()
            else 3
        )
        commit_window = (
            max(
                int(THREE_CAR_MERGE_COMMIT_STEPS),
                self._control_cycles_to_steps(NEGOTIATED_MERGE_TIMEOUT_STEPS),
            )
            if self._is_three_car_scene()
            else aggressive_window
        )
        if not self._cut_in_episode_active:
            self._cut_in_episode_active = True
            self._cut_in_aggressive_until_step = int(current_step) + int(aggressive_window)
            self._merge_commit_until_step = int(current_step) + int(commit_window)
            self._merge_attempt_steps = 0
            self._merge_wait_gap_cycles = 0
            return

        if self._is_three_car_scene():
            self._merge_commit_until_step = max(
                int(self._merge_commit_until_step),
                int(current_step) + int(commit_window),
            )

    def _merge_commit_active(self, ctx, current_step):
        if not self._is_three_car_scene():
            return False
        if not self._cut_in_episode_active:
            return False
        if int(ctx["self_lane"]) == int(ctx["ego_lane"]):
            return False
        if int(current_step) > int(self._merge_commit_until_step):
            return False
        if abs(int(ctx["self_lane"]) - int(ctx["ego_lane"])) != 1:
            return False
        if self._teammate_blocks_ego_lane(ctx):
            return False
        return True

    def _three_car_auto_brake_ready(self, ctx):
        if not self._is_three_car_scene():
            return False
        if int(ctx["self_lane"]) != int(ctx["ego_lane"]):
            return False
        rel_x = float(ctx["self_rel_x"])
        return 0.3 <= rel_x <= 7.5

    def _apply_front_brake_trap(self, env, ctx, current_step):
        ego_lane = int(ctx["ego_lane"])
        ego_speed = float(ctx["ego_speed"])
        self_speed = float(ctx["self_speed"])
        self._clear_cut_in_episode()
        self._front_brake_triggered = True
        self.intent = "brake_pulse"
        self.executor_state = "front_brake"
        self.brake_armed = True
        pulse_steps = max(
            int(THREE_CAR_FRONT_BRAKE_STEPS),
            self._control_cycles_to_steps(4),
        )
        self._pulse_end_step = max(int(self._pulse_end_step), int(current_step) + int(pulse_steps))
        self.idm_a = 3.4
        self.T = 0.22
        self.a = self.idm_a
        rel_x = float(ctx.get("self_rel_x", 0.0))
        brake_drop = float(np.clip(1.6 + 0.28 * max(0.0, rel_x), 2.0, 4.2))
        target_v = min(self_speed - brake_drop, ego_speed - max(1.2, brake_drop - 0.8))
        if target_v >= self_speed:
            target_v = self_speed - brake_drop
        self._apply_highway_targets(
            env,
            ego_lane,
            max(self.bounds["v_min"], target_v),
            0.55,
        )

    def _derive_lane_change_command(self, current_lane, target_abs_lane):
        if target_abs_lane is None:
            return 0
        delta = int(target_abs_lane) - int(current_lane)
        if delta > 0:
            return 1
        if delta < 0:
            return -1
        return 0

    def _apply_highway_targets(self, env, target_abs_lane, target_v, target_s):
        current_lane = int(env.k.vehicle.get_lane(self.veh_id))
        self._allow_low_speed_target = False
        self.target_abs_lane = int(target_abs_lane)
        self.target_v = float(np.clip(target_v, self.bounds["v_min"], self.bounds["v_max"]))
        self.target_s = float(np.clip(target_s, self.bounds["s_min"], self.bounds["s_max"]))
        self.target_lc = int(np.clip(
            self._derive_lane_change_command(current_lane, self.target_abs_lane),
            -1,
            1,
        ))
        self.pending_lane_change = self.target_lc
        self.v0 = self.target_v
        self.s0 = self.target_s

    def _apply_disengage_targets(self, env, ctx):
        target_lane = int(ctx["self_lane"])
        target_v = max(self.bounds["v_min"], min(self.bounds["v_max"], float(ctx["ego_speed"]) - 2.0))
        target_s = 5.5
        self.executor_state = "disengage"
        self.brake_armed = False
        self._allow_aggressive_cut_in = False
        self._apply_highway_targets(env, target_lane, target_v, target_s)

    def _get_highway_intent_state(self, requested_intent, urgency):
        return {
            "intent": str(requested_intent or self._default_highway_intent()),
            "urgency": str(urgency or "mid"),
        }

    def _append_trace_entry(self, entry):
        self.llm_trace_entries.append(entry)
        if len(self.llm_trace_entries) > self.trace_max_entries:
            self.llm_trace_entries = self.llm_trace_entries[-self.trace_max_entries:]

        if self.trace_file:
            try:
                with open(self.trace_file, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _record_highway_runtime_trace(self, env, ctx, lane_change_attempted):
        step = int(self._get_step(env))
        if self.map_name != "highway":
            return
        if not self._is_control_step(env):
            return
        if step == self._last_runtime_trace_step:
            return

        self._last_runtime_trace_step = step
        speed_adv = float(ctx["self_speed"]) - float(ctx["ego_speed"])
        entry = {
            "step": step,
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "protocol": "highway_runtime",
            "attempt": 0,
            "parse_ok": True,
            "error": "",
            "raw_response": "",
            "sanitized_response": "",
            "intent": str(self.intent or ""),
            "executor_state": str(self.executor_state or ""),
            "rel_x": round(float(ctx["self_rel_x"]), 3),
            "speed_adv": round(float(speed_adv), 3),
            "aggressive_cut_in_ready": bool(self._last_aggressive_cut_in_ready),
            "target_v": round(float(self.target_v), 3),
            "target_s": round(float(self.target_s), 3),
            "target_lc": int(self.target_lc),
            "lane_change_attempted": bool(lane_change_attempted),
        }
        if self._is_negotiated_highway_scene():
            entry.update({
                "ego_lane": int(ctx["ego_lane"]),
                "self_lane": int(ctx["self_lane"]),
                "prev_self_lane": self._trace_prev_self_lane,
                "merge_event_this_step": bool(self._merge_event_this_step),
                "merged_into_ego_lane": bool(self._merged_into_ego_lane),
                "valid_cut_in_merge": bool(self.striker_completed_cut_in),
                "bad_merge_event": bool(self._bad_merge_event),
                "bad_merge_reason": str(self._bad_merge_reason or ""),
                "clean_merge_failed": bool(self._clean_merge_failed),
                "stale_merge_candidate": bool(self._stale_merge_candidate),
                "stale_merge_candidate_step": int(self._stale_merge_candidate_step),
                "lane_change_stalled": bool(self._lane_change_stalled),
                "merge_stall_cycles": int(self._merge_stall_cycles),
                "merge_wait_gap_cycles": int(self._merge_wait_gap_cycles),
                "force_cut_in_blocked_reason": str(self._last_force_cut_in_blocked_reason or ""),
                "ego_gap_after_merge": round(float(self._ego_lead_gap_after_merge(env, ctx)), 3),
                "body_gap_after_merge": round(float(self._ego_lead_gap_after_merge(env, ctx)), 3),
                "projected_body_gap_after_merge": round(float(self._projected_ego_lead_gap_after_merge(env, ctx)), 3),
                "required_body_gap_after_merge": round(float(self._dynamic_body_cut_in_gap(env, ctx)), 3),
                "effective_ego_gap_after_merge": round(float(self._effective_ego_gap_after_merge(env, ctx)), 3),
                "last_merge_rel_x": self._last_merge_rel_x,
                "post_merge_stabilize_until_step": int(self._post_merge_stabilize_until_step),
                "approach_window_ready": bool(self._last_local_ready.get("approach_window_ready", False)),
                "same_lane_lead_established": bool(self._last_local_ready.get("same_lane_lead_established", False)),
                "cut_in_gap_ready": bool(self._last_local_ready.get("cut_in_gap_ready", False)),
                "clean_cut_in_gap_ready": bool(self._last_local_ready.get("clean_cut_in_gap_ready", False)),
                "overshoot": bool(self._overshoot),
                "front_brake_triggered": bool(self._front_brake_triggered),
                "teammate_phase": str(self._last_teammate_phase or ""),
                "teammate_intent": str(self._last_teammate_intent or ""),
                "teammate_tactic": copy.deepcopy(self._last_teammate_tactic or {}),
                "contract_pass_side": str(self.pass_side or "none"),
            })
        self._append_trace_entry(entry)

    def _get_negotiated_snapshot(self, env):
        if hasattr(env.message_pool, "negotiated_snapshot"):
            return env.message_pool.negotiated_snapshot(self._get_step(env), viewer_id=self.veh_id)
        return {
            "negotiated_contract": {},
            "latest_negotiation_by_agent": {},
            "latest_phase_by_agent": {},
            "latest_tactic_by_agent": {},
            "recent_phase_events": [],
        }

    def _get_latest_teammate_phase(self, snapshot):
        teammate_id = self._get_teammate_id()
        latest = (snapshot or {}).get("latest_phase_by_agent", {}) or {}
        return copy.deepcopy(latest.get(teammate_id) or {})

    def _get_latest_teammate_tactic(self, snapshot):
        teammate_id = self._get_teammate_id()
        latest = (snapshot or {}).get("latest_tactic_by_agent", {}) or {}
        return copy.deepcopy(latest.get(teammate_id) or {})

    def _refresh_negotiated_runtime_state(self, env, ctx):
        current_step = int(self._get_step(env))
        ego_lane = int(ctx["ego_lane"])
        self_lane = int(ctx["self_lane"])
        prev_lane = self._prev_self_lane if self._prev_self_lane is not None else self_lane
        self._trace_prev_self_lane = prev_lane
        merged_into_ego_lane = prev_lane != ego_lane and self_lane == ego_lane
        self._merge_event_this_step = bool(merged_into_ego_lane)
        if merged_into_ego_lane:
            rel_x = float(ctx["self_rel_x"])
            self._merged_into_ego_lane = True
            self._last_merge_event_step = current_step
            self._post_merge_stabilize_until_step = max(
                int(self._post_merge_stabilize_until_step),
                current_step + self._control_cycles_to_steps(NEGOTIATED_POST_MERGE_STABILIZE_CYCLES),
            )
            self._last_merge_rel_x = round(float(rel_x), 3)
            merge_status = self._classify_merge_event(rel_x, env=env, ctx=ctx)
            if merge_status == "valid":
                self.striker_completed_cut_in = True
                self._bad_merge_event = False
                self._bad_merge_reason = ""
                self._last_valid_merge_event_step = current_step
            else:
                self._bad_merge_event = True
                self._bad_merge_reason = merge_status
                self._last_bad_merge_event_step = current_step
                if merge_status in ("late_merge_ahead", "stale_merge_ahead_far"):
                    self._overshoot = True
                    self._clean_merge_failed = True
                if merge_status == "stale_merge_ahead_far":
                    self._stale_merge_candidate = True
                    self._stale_merge_candidate_step = current_step
            if self.attack_role == "Striker" and self._striker_lane_change_time is None:
                self._striker_lane_change_time = round(float(self._get_step(env) * getattr(env, "sim_step", 0.1)), 3)
                self._striker_rel_x_at_lane_change = round(float(rel_x), 3)
        self._prev_self_lane = self_lane
        return merged_into_ego_lane

    def _recent_merge_event_active(self, current_step):
        return self._recent_step_active(self._last_merge_event_step, current_step)

    def _recent_valid_merge_event_active(self, current_step):
        return self._recent_step_active(self._last_valid_merge_event_step, current_step)

    def _build_negotiated_local_ready(self, env, ctx):
        ego_lane = int(ctx["ego_lane"])
        self_lane = int(ctx["self_lane"])
        rel_x = float(ctx["self_rel_x"])
        speed_adv = float(ctx["self_speed"]) - float(ctx["ego_speed"])
        edge = env.k.vehicle.get_edge(self.veh_id)
        pass_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.pass_side_rel or -1)
        block_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.block_side_rel or 1)
        body_cut_in_command_ready = bool(
            self.attack_role == "Striker"
            and self_lane == pass_lane
            and abs(self_lane - ego_lane) == 1
            and rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
            and self._body_cut_in_command_ready(env, ctx)
        )
        body_safe_cut_in_ready = bool(
            self.attack_role == "Striker"
            and self_lane == pass_lane
            and abs(self_lane - ego_lane) == 1
            and rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
            and self._body_safe_cut_in_ready(env, ctx)
        )
        clean_cut_in_gap_ready = bool(
            body_safe_cut_in_ready
            and self._clean_cut_in_gap_ready(env, ctx)
        )
        approach_window_ready = bool(
            self.attack_role == "Striker"
            and self_lane == pass_lane
            and abs(self_lane - ego_lane) == 1
            and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
            and speed_adv >= -0.2
            and (
                rel_x <= NEGOTIATED_MERGE_COMMIT_MAX_REL_X
                or body_cut_in_command_ready
            )
        )
        ready = {
            "approach_window_ready": approach_window_ready,
            "cut_in_gap_ready": body_cut_in_command_ready,
            "clean_cut_in_gap_ready": clean_cut_in_gap_ready,
            "body_gap_after_merge": round(float(self._ego_lead_gap_after_merge(env, ctx)), 3),
            "projected_body_gap_after_merge": round(float(self._projected_ego_lead_gap_after_merge(env, ctx)), 3),
            "required_body_gap_after_merge": round(float(self._dynamic_body_cut_in_gap(env, ctx)), 3),
            "effective_ego_gap_after_merge": round(float(self._effective_ego_gap_after_merge(env, ctx)), 3),
            "merged_into_ego_lane": bool(self._merge_event_this_step),
            "same_lane_lead_established": bool(
                self.attack_role == "Striker"
                and self_lane == ego_lane
                and 1.0 <= rel_x <= 4.0
            ),
            "front_brake_window_ready": bool(
                self.attack_role == "Striker"
                and self_lane == ego_lane
                and 0.2 <= rel_x <= 5.8
            ),
            "overshoot": bool(
                self.attack_role == "Striker"
                and self_lane == ego_lane
                and rel_x > NEGOTIATED_MERGE_COMMIT_MAX_REL_X
            ),
            "blocker_side_front_ready": bool(
                self.attack_role == "Blocker"
                and self_lane == block_lane
                and 2.5 <= rel_x <= 5.5
            ),
            "teammate_block_band_ready": bool(self._teammate_in_block_band(ctx)),
            "teammate_pass_window_ready": bool(self._teammate_in_pass_window(ctx)),
            "speed_adv": round(float(speed_adv), 3),
        }
        self._last_local_ready = dict(ready)
        return ready

    def _build_negotiated_prompt_local_ready(self, local_ready):
        payload = copy.deepcopy(local_ready or {})
        return {
            "approach_window_ready": bool(payload.get("approach_window_ready", False)),
            "cut_in_gap_ready": bool(payload.get("cut_in_gap_ready", False)),
            "clean_cut_in_gap_ready": bool(payload.get("clean_cut_in_gap_ready", False)),
            "body_gap_after_merge": payload.get("body_gap_after_merge", None),
            "projected_body_gap_after_merge": payload.get("projected_body_gap_after_merge", None),
            "required_body_gap_after_merge": payload.get("required_body_gap_after_merge", None),
            "effective_ego_gap_after_merge": payload.get("effective_ego_gap_after_merge", None),
            "merged_into_ego_lane": bool(
                payload.get("merged_into_ego_lane", False) or self._merged_into_ego_lane
            ),
            "front_brake_window_ready": bool(payload.get("front_brake_window_ready", False)),
            "overshoot": bool(payload.get("overshoot", False) or self._overshoot),
        }

    def _publish_negotiated_phase(self, env, intent_plan):
        if not hasattr(env.message_pool, "publish_negotiated_phase"):
            return
        publish_entry = {
            "sender": self.veh_id,
            "role": self.attack_role,
            "phase": str((intent_plan or {}).get("phase", "") or ""),
            "intent": str((intent_plan or {}).get("intent", "") or ""),
            "urgency": str((intent_plan or {}).get("urgency", "mid") or "mid"),
            "message": str((intent_plan or {}).get("message", "") or ""),
            "step": int(self._get_step(env)),
            "expires_at_step": int(self._get_step(env) + self.control_interval),
            "control_cycle_step": int(getattr(env.message_pool, "control_cycle_step", self._get_step(env))),
        }
        env.message_pool.publish_negotiated_phase(publish_entry)

    def _publish_negotiated_tactic(self, env, decision):
        if not self._is_negotiated_highway_scene():
            return
        if not hasattr(env.message_pool, "publish_negotiated_tactic"):
            return
        tactic = copy.deepcopy((decision or {}).get("tactic") or {})
        message = copy.deepcopy((decision or {}).get("message") or {})
        env.message_pool.publish_negotiated_tactic({
            "sender": self.veh_id,
            "role": self.attack_role,
            "phase": str(message.get("phase", "") or ""),
            "intent": str(message.get("intent", "") or ""),
            "mode": str(tactic.get("mode", "hold_lane") or "hold_lane"),
            "lane_policy": str(tactic.get("lane_policy", "hold_current") or "hold_current"),
            "gap_band": str(tactic.get("gap_band", "medium") or "medium"),
            "speed_band": str(tactic.get("speed_band", "match") or "match"),
            "message": str((decision or {}).get("message_text", "") or (decision or {}).get("action_sequence_text", "") or ""),
            "step": int(self._get_step(env)),
            "expires_at_step": int(message.get("expires_at_step", self._get_step(env) + self.control_interval)),
            "control_cycle_step": int(getattr(env.message_pool, "control_cycle_step", self._get_step(env))),
        })

    def _aggressive_cut_in_ready(self, env, edge, ctx):
        if not self._is_three_car_scene():
            return False
        ego_lane = int(ctx["ego_lane"])
        self_lane = int(ctx["self_lane"])
        rel_x = float(ctx["self_rel_x"])
        speed_adv = float(ctx["self_speed"]) - float(ctx["ego_speed"])
        if abs(self_lane - ego_lane) != 1:
            return False
        if self._use_structured_protocol() and self._is_negotiated_highway_scene():
            if rel_x < 0.0 or rel_x > NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X:
                return False
            if speed_adv < -0.2:
                return False
            if not self._body_cut_in_command_ready(env, ctx):
                return False
        else:
            if rel_x < -3.5 or rel_x > 7.0:
                return False
            if rel_x <= 1.0 and speed_adv < 1.0:
                return False
            if rel_x > 1.0 and speed_adv < -0.5:
                return False
        if self._teammate_blocks_ego_lane(ctx):
            return False
        return self._lane_change_is_safe(env, edge, ego_lane, aggressive=True)

    def _apply_highway_executor(self, env):
        self._reset_highway_phase_profile()
        self._allow_aggressive_cut_in = False
        self._last_aggressive_cut_in_ready = False
        if self.map_name != "highway" or not self.role_map or self.attack_role not in ("Blocker", "Striker"):
            self._clear_cut_in_episode()
            return

        ctx = self._get_relative_context(env)
        edge = env.k.vehicle.get_edge(self.veh_id)
        ego_lane = int(ctx["ego_lane"])
        self_lane = int(ctx["self_lane"])
        ego_speed = float(ctx["ego_speed"])
        self_speed = float(ctx["self_speed"])
        rel_x = float(ctx["self_rel_x"])
        current_step = self._get_step(env)
        requested = self.current_decision or {}
        state = self._get_highway_intent_state(
            requested.get("intent", self._default_highway_intent()),
            requested.get("urgency", "mid"),
        )
        requested_intent = state["intent"]
        self.intent_urgency = state["urgency"]

        if self.attack_role == "Striker":
            preferred_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.passing_side_rel or -1)
            speed_adv = self_speed - ego_speed
            if abs(self_lane - ego_lane) == 1 and rel_x >= 1.0 and self._lane_change_is_safe(env, edge, ego_lane):
                self.lead_acquired = True
            elif self_lane == ego_lane and rel_x >= 2.5:
                self.lead_acquired = True
            else:
                self.lead_acquired = False
            aggressive_cut_in_ready = (
                requested_intent == "cut_in"
                and self._aggressive_cut_in_ready(env, edge, ctx)
            )
            near_cut_in_window = (
                self._is_three_car_scene()
                and abs(self_lane - ego_lane) == 1
                and -5.5 <= rel_x <= 2.5
                and speed_adv >= 1.0
                and not self._teammate_blocks_ego_lane(ctx)
            )
            self._last_aggressive_cut_in_ready = bool(aggressive_cut_in_ready)

            if self_lane == ego_lane and 0.5 <= rel_x <= 10.0:
                self.striker_completed_cut_in = True
            try:
                ego_leader = env.k.vehicle.get_leader(self.attack_target)
            except Exception:
                ego_leader = ""
            if self_lane == ego_lane and rel_x > 0.0 and ego_leader == self.veh_id:
                self.striker_became_ego_leader = True

            brake_ready = (
                self_lane == ego_lane
                and 1.0 <= rel_x <= 8.0
                and ego_speed > self_speed + 0.3
            )
            auto_brake_ready = self._three_car_auto_brake_ready(ctx)
            self.brake_armed = bool(brake_ready or auto_brake_ready)

            if self._pulse_end_step >= 0 and current_step <= self._pulse_end_step:
                if self._is_three_car_scene() and self_lane == ego_lane:
                    self._apply_front_brake_trap(env, ctx, current_step)
                else:
                    self._clear_cut_in_episode()
                    self.intent = "brake_pulse"
                    self.executor_state = "brake"
                    self._apply_highway_targets(
                        env,
                        ego_lane,
                        max(self.bounds["v_min"], ego_speed - 6.0),
                        2.5,
                    )
                return
            if self._pulse_end_step >= 0 and current_step > self._pulse_end_step:
                self._clear_cut_in_episode()
                self._pulse_end_step = -1
                self._apply_disengage_targets(env, ctx)
                self.intent = "abort"
                return

            if auto_brake_ready and (self.striker_completed_cut_in or rel_x >= 1.0):
                self._apply_front_brake_trap(env, ctx, current_step)
                return

            merge_commit_active = self._merge_commit_active(ctx, current_step)
            if requested_intent == "abort" and not merge_commit_active:
                self._clear_cut_in_episode()
                self.intent = "abort"
                self._apply_disengage_targets(env, ctx)
                return

            if requested_intent == "brake_pulse" and (brake_ready or auto_brake_ready):
                if self._is_three_car_scene():
                    self._apply_front_brake_trap(env, ctx, current_step)
                else:
                    self._clear_cut_in_episode()
                    self.intent = "brake_pulse"
                    self.executor_state = "brake"
                    self._pulse_end_step = current_step + 3
                    self._apply_highway_targets(
                        env,
                        ego_lane,
                        max(self.bounds["v_min"], ego_speed - 6.0),
                        2.5,
                    )
                return

            if requested_intent == "cut_in" and (self.lead_acquired or aggressive_cut_in_ready or near_cut_in_window):
                self._start_merge_commit(current_step)
                merge_commit_active = self._merge_commit_active(ctx, current_step)

            if merge_commit_active:
                self._merge_attempt_steps += 1
                self.intent = "cut_in"
                self.executor_state = "merge_commit" if self._merge_attempt_steps > 4 else "cut_in"
                self._allow_aggressive_cut_in = True
                closing_bonus = float(np.clip(0.35 * max(0.0, 1.5 - rel_x), 0.0, 2.5))
                self.idm_a = 3.0
                self.T = 0.3
                self.a = self.idm_a
                self._apply_highway_targets(
                    env,
                    ego_lane,
                    max(
                        ego_speed + 3.0 + closing_bonus,
                        self_speed + 1.0 + 0.5 * closing_bonus,
                    ),
                    0.6 if current_step <= self._cut_in_aggressive_until_step else 0.8,
                )
                return

            if requested_intent == "cut_in" and (self.lead_acquired or aggressive_cut_in_ready):
                if not self._cut_in_episode_active:
                    self._start_merge_commit(current_step)
                self.intent = "cut_in"
                self.executor_state = "cut_in"
                self._allow_aggressive_cut_in = bool(aggressive_cut_in_ready)
                self._apply_highway_targets(
                    env,
                    ego_lane,
                    max(ego_speed + (1.5 if aggressive_cut_in_ready else 1.0), self_speed),
                    (
                        1.1 if aggressive_cut_in_ready and current_step <= self._cut_in_aggressive_until_step
                        else 1.3 if current_step <= self._cut_in_aggressive_until_step
                        else 2.0
                    ),
                )
                return

            if rel_x < 0.0:
                self._clear_cut_in_episode()
                self.intent = "gain_lead"
                if self_lane != preferred_lane:
                    self.executor_state = "align"
                else:
                    self.executor_state = "chase"
                self.brake_armed = False
                closing_bonus = float(np.clip(0.30 * max(0.0, -rel_x - 2.0), 0.0, 6.0))
                if self._is_three_car_scene():
                    self.idm_a = 2.8
                    self.T = 0.45
                else:
                    self.idm_a = 1.8
                    self.T = 0.6
                self.a = self.idm_a
                self._apply_highway_targets(
                    env,
                    preferred_lane,
                    max(
                        ego_speed + (5.0 if self._is_three_car_scene() else 4.0) + closing_bonus,
                        self_speed + (3.0 if self._is_three_car_scene() else 2.0) + 0.6 * closing_bonus,
                    ),
                    0.6 if self._is_three_car_scene() else 0.8,
                )
                return

            self._clear_cut_in_episode()
            self.intent = "gain_lead"
            if self_lane != preferred_lane and self_lane != ego_lane:
                self.executor_state = "align"
                target_lane = preferred_lane
            else:
                self.executor_state = "chase"
                target_lane = preferred_lane
            self.brake_armed = False
            closing_bonus = float(np.clip(0.25 * max(0.0, -rel_x - 2.0), 0.0, 5.0))
            self._apply_highway_targets(
                env,
                target_lane,
                max(
                    ego_speed + (4.5 if self._is_three_car_scene() else 4.0) + closing_bonus,
                    self_speed + (2.5 if self._is_three_car_scene() else 2.0) + 0.5 * closing_bonus,
                ),
                0.6 if self._is_three_car_scene() else 0.8,
            )
            return

        self._clear_cut_in_episode()
        reserved_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.reserved_side_rel or 1)
        self.brake_armed = False
        if requested_intent == "abort":
            self.intent = "abort"
            self._apply_disengage_targets(env, ctx)
            return

        if self._is_three_car_scene():
            self.idm_a = 2.7
            self.T = 0.4
            self.a = self.idm_a
            if self_lane != reserved_lane:
                self.intent = "claim_side"
                self.executor_state = "claim"
                closing_bonus = float(np.clip(0.55 * max(0.0, 12.0 - rel_x), 0.0, 4.0))
                self._apply_highway_targets(
                    env,
                    reserved_lane,
                    max(
                        ego_speed + 4.0 + closing_bonus,
                        self_speed + 2.5 + 0.5 * closing_bonus,
                    ),
                    0.8,
                )
                return

            if rel_x < 7.5:
                self.intent = "claim_side"
                self.executor_state = "claim"
                closing_bonus = float(np.clip(0.75 * max(0.0, 8.5 - rel_x), 0.5, 5.0))
                self._apply_highway_targets(
                    env,
                    reserved_lane,
                    max(
                        ego_speed + 2.8 + closing_bonus,
                        self_speed + 1.8 + 0.4 * closing_bonus,
                    ),
                    0.8,
                )
                return

            if requested_intent == "seal_escape" or rel_x <= 11.0:
                self.intent = "seal_escape"
                self.executor_state = "seal"
                seal_bias = float(np.clip(0.75 * (9.0 - rel_x), -0.4, 4.0))
                self._apply_highway_targets(
                    env,
                    reserved_lane,
                    float(np.clip(ego_speed + 1.5 + seal_bias, self.bounds["v_min"], self.bounds["v_max"])),
                    0.8,
                )
                return

            self.intent = "hold_side_front"
            self.executor_state = "hold"
            hold_bias = float(np.clip(0.60 * (10.0 - rel_x), -1.0, 3.5))
            self._apply_highway_targets(
                env,
                reserved_lane,
                float(np.clip(ego_speed + 0.6 + hold_bias, self.bounds["v_min"], self.bounds["v_max"])),
                0.9,
            )
            return

        if self_lane != reserved_lane:
            self.intent = "claim_side"
            self.executor_state = "claim"
            self._apply_highway_targets(
                env,
                reserved_lane,
                max(ego_speed + 2.0, self_speed + 1.2),
                1.4,
            )
            return

        if rel_x < 6.0:
            self.intent = "claim_side"
            self.executor_state = "claim"
            self._apply_highway_targets(
                env,
                reserved_lane,
                max(ego_speed + 2.2, self_speed + 1.4),
                1.3,
            )
            return

        if requested_intent == "seal_escape":
            self.intent = "seal_escape"
            self.executor_state = "seal"
            seal_bias = float(np.clip(0.40 * (5.0 - rel_x), -1.0, 2.0))
            self._apply_highway_targets(
                env,
                reserved_lane,
                float(np.clip(ego_speed + seal_bias, self.bounds["v_min"], self.bounds["v_max"])),
                1.2,
            )
            return

        self.intent = "hold_side_front"
        self.executor_state = "hold"
        hold_bias = float(np.clip(0.45 * (7.5 - rel_x), -0.8, 2.3))
        self._apply_highway_targets(
            env,
            reserved_lane,
            float(np.clip(ego_speed + hold_bias, self.bounds["v_min"], self.bounds["v_max"])),
            1.3,
        )

    def _override_highway_accel(self, env, acc):
        if self.map_name != "highway" or not self._is_three_car_scene():
            return float(acc)

        ctx = self._get_relative_context(env)
        rel_x = float(ctx["self_rel_x"])
        if self._is_negotiated_highway_scene():
            ego_speed = float(ctx["ego_speed"])
            self_speed = float(ctx["self_speed"])
            if self.attack_role == "Striker":
                if self.executor_state == "front_brake" or self.intent == "front_brake":
                    return float(np.clip(acc, -3.8, -1.2))
                if self.executor_state == "front_brake_setup":
                    return float(np.clip(acc, -2.2, 0.4))
                if self.executor_state == "cut_in_commit":
                    cap = 0.4 if self.target_lc != 0 else 0.8
                    return float(min(max(acc, -1.2), cap))
                if self.executor_state == "merge_commit":
                    if rel_x > 3.0:
                        return float(min(acc, -1.5))
                    if rel_x > 0.5:
                        return float(min(acc, 0.0))
                    return float(min(max(acc, -0.4), 1.2))
                if self.intent == "gain_lead":
                    return float(min(max(acc, -0.5), 1.8 if self_speed < ego_speed + 2.0 else 0.5))
                return float(acc)

            if self.attack_role == "Blocker":
                if self.executor_state == "seal":
                    if rel_x > 10.0:
                        return float(min(acc, 0.0))
                    if rel_x < 6.0:
                        return float(max(acc, 1.4))
                    return float(min(max(acc, -0.4), 1.0))
                if rel_x > 10.0:
                    return float(min(acc, 0.0))
                if rel_x < 5.0:
                    return float(max(acc, 1.0))
                return float(min(max(acc, -0.5), 0.8))

        if self.attack_role == "Striker":
            if self.intent == "brake_pulse" or self.executor_state in ("brake", "front_brake"):
                return float(min(acc, -7.5))
            if self.executor_state == "cut_in_commit" and self.target_lc != 0:
                return float(min(max(acc, -1.2), 0.4))
            if self.executor_state in ("cut_in", "merge_commit") and self.target_lc != 0:
                return float(max(acc, 3.2 if rel_x < 1.0 else 2.2))
            if self.intent == "gain_lead":
                return float(max(acc, 2.6))
            return float(acc)

        if self.attack_role == "Blocker":
            if self.executor_state in ("claim", "seal"):
                return float(max(acc, 2.6 if rel_x < 10.0 else 1.8))
            if self.executor_state == "hold" and rel_x < 9.0:
                return float(max(acc, 1.6))
        return float(acc)

    def _default_control(self):
        return {
            "mode": "hold_lane",
            "horizon_steps": self.control_interval,
        }

    def _default_message(self):
        return {
            "sender": self.veh_id,
            "phase": "compress",
            "intent": self._default_highway_intent(),
            "target": self.attack_target,
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
        self.trigger_not_met_events = 0
        self.sync_error_events = 0
        self.rollout_parse_fallback_used = 0
        self.role_resolution_fallback_used = 0
        self.phase_trace = []
        self.active_phase = "attack" if self.role_map else "negotiation"
        self.active_control_mode = "disengage"
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
        self.intent = self._default_highway_intent()
        self.intent_urgency = "mid"
        self.executor_state = "disengage"
        self.target_abs_lane = None
        self.lead_acquired = False
        self.brake_armed = False
        self.striker_completed_cut_in = False
        self.striker_became_ego_leader = False
        self.reserved_side_rel = 0
        self.passing_side_rel = 0
        self._allow_aggressive_cut_in = False
        self._last_aggressive_cut_in_ready = False
        self._last_lane_change_attempted = False
        self._last_runtime_trace_step = -1
        self._cut_in_aggressive_until_step = -1
        self._cut_in_episode_active = False
        self._merge_commit_until_step = -1
        self._merge_attempt_steps = 0
        self._merge_wait_gap_cycles = 0
        self._pulse_end_step = -1
        self._prev_self_lane = None
        self._trace_prev_self_lane = None
        self._merged_into_ego_lane = False
        self._merge_event_this_step = False
        self._last_merge_event_step = -1
        self._last_valid_merge_event_step = -1
        self._last_bad_merge_event_step = -1
        self._last_merge_rel_x = None
        self._bad_merge_event = False
        self._bad_merge_reason = ""
        self._overshoot = False
        self._front_brake_triggered = False
        self._post_merge_stabilize_until_step = -1
        self._striker_lane_change_time = None
        self._striker_rel_x_at_lane_change = None
        self._seal_escape_until_step = -1
        self._last_local_ready = {}
        self._last_teammate_phase = ""
        self._last_teammate_intent = ""
        self._last_teammate_tactic = {}
        self._last_negotiated_contract = copy.deepcopy(self.highway_contract)
        self._allow_low_speed_target = False
        self.latest_intent_plan = {}
        self.latest_action_sequence_text = ""
        self.latest_tactic_profile = {}
        self._pending_intent_plan = {}
        self._pending_intent_step = -1
        self._lane_change_in_progress_until_step = -1
        self._lane_change_target_lane = None
        self._hold_current_lane_until_step = -1
        self._hold_current_lane_target = None
        self._last_action_refresh_signature = {}
        self._last_action_rel_x_bucket = ""
        self._last_action_rel_x_bucket_index = None
        self._last_published_trigger_satisfied = None
        self._action_reuse_count = 0
        self._last_history_injection_phase = ""
        self._last_action_refresh_reason = ""
        self._last_action_reused = False
        self._last_force_cut_in_blocked_reason = ""
        self._terminal_plan_locked = False
        self._terminal_lock_reason = ""
        self._terminal_lock_step = -1
        self._merge_window_hold_until_step = -1
        self._initialize_highway_preferences(env)
        try:
            if self.veh_id in env.k.vehicle.get_ids():
                self._prev_self_lane = int(env.k.vehicle.get_lane(self.veh_id))
                self._trace_prev_self_lane = self._prev_self_lane
        except Exception:
            self._prev_self_lane = None
            self._trace_prev_self_lane = None

    def run_coordinated_step(self, env, snapshot=None):
        self._begin_rollout_if_needed(env)
        if not self.uses_coordinated_planning():
            return False
        if self._use_structured_protocol():
            planned = self.prepare_for_coordinated_step(env, snapshot=snapshot)
            if planned:
                self.has_llm_decision = True
            return planned
        if not self.role_map:
            self.active_phase = "negotiation"
            return False

        step = self._get_step(env)
        if not self._is_control_step(env) or step == self.last_control_step:
            return False

        self.active_phase = "attack"
        self.last_control_step = step
        decision = self.llm_collaborate(env)
        self.current_message = str(decision.get("message", ""))
        self.has_llm_decision = True
        self.current_decision = copy.deepcopy(decision)
        return True

    def _update_tactical_plan_if_needed(self, env):
        self._begin_rollout_if_needed(env)
        step = self._get_step(env)
        if (
                hasattr(env, "_run_controlled_planning")
                and hasattr(env, "message_pool")
                and self.uses_coordinated_planning()
                and self._is_control_step(env)
                and not env.message_pool.is_control_cycle_planned(step)):
            env._run_controlled_planning()
            if step == self.last_control_step:
                return
        if self._use_structured_protocol():
            return self._update_structured_plan_if_needed(env)
        if self.uses_coordinated_planning():
            self.run_coordinated_step(env)
            return
        need_new_decision = (
            (not self.has_llm_decision)
            or (self._is_control_step(env) and step != self.last_control_step)
        )
        if not need_new_decision:
            return

        self.last_control_step = step
        decision = self.llm_collaborate(env)
        self.current_message = decision["message"]
        self.has_llm_decision = True
        self.current_decision = copy.deepcopy(decision)

    def get_lane_change_action(self, env):
        cmd = int(self.pending_lane_change)
        if (
                cmd == 0
                and self._use_structured_protocol()
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"
                and int(self.target_lc) != 0):
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
            current_step = int(self._get_step(env))
            if (
                    ctx
                    and int(ctx.get("self_lane", 0)) != int(ctx.get("ego_lane", 0))
                    and float(ctx.get("self_rel_x", 0.0)) > NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                    and not self._body_cut_in_command_ready(env, ctx)
                    and self.intent in ("merge_commit", "front_brake")):
                self._mark_clean_merge_failed(env)
                if float(ctx.get("self_rel_x", 0.0)) > NEGOTIATED_MERGE_FAIL_REL_X:
                    self._mark_stale_merge_candidate(env)
                self.target_lc = 0
                cmd = 0
            merge_commit_active = bool(ctx) and self._merge_commit_active(ctx, current_step)
            merge_commit_active = bool(
                merge_commit_active
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X
                <= float(ctx.get("self_rel_x", 0.0))
                <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
            )
            lane_target_pending = bool(ctx) and (
                int(ctx.get("self_lane", 0)) != int(ctx.get("ego_lane", 0))
                and self.intent in ("merge_commit", "front_brake")
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X
                <= float(ctx.get("self_rel_x", 0.0))
                <= NEGOTIATED_MERGE_FAIL_REL_X
                and (
                    float(ctx.get("self_rel_x", 0.0)) <= NEGOTIATED_MERGE_COMMIT_MAX_REL_X
                    or self._body_cut_in_command_ready(env, ctx)
                )
            )
            if merge_commit_active or lane_target_pending:
                cmd = int(self.target_lc)
        if (
                cmd != 0
                and self._use_structured_protocol()
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"
                and self.intent in ("merge_commit", "front_brake")):
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
            if (
                    ctx
                    and int(ctx.get("self_lane", 0)) != int(ctx.get("ego_lane", 0))
                    and float(ctx.get("self_rel_x", 0.0)) > NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                    and not self._body_cut_in_command_ready(env, ctx)):
                self._mark_clean_merge_failed(env)
                if float(ctx.get("self_rel_x", 0.0)) > NEGOTIATED_MERGE_FAIL_REL_X:
                    self._mark_stale_merge_candidate(env)
                self.target_lc = 0
                cmd = 0
        self.pending_lane_change = 0
        if cmd not in (-1, 0, 1):
            return 0
        return cmd

    def get_accel(self, env):
        self._update_tactical_plan_if_needed(env)
        structured_runtime = self._use_structured_protocol()
        if self.map_name == "highway":
            if not structured_runtime:
                self._apply_highway_executor(env)
            if structured_runtime:
                self._apply_road_end_guard(env)
        self._apply_tactical_sumo_params(env)

        lc_action = self.get_lane_change_action(env)
        lane_change_attempted = False
        force_merge_retry = False
        if (
                lc_action != 0
                and structured_runtime
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"):
            try:
                retry_ctx = self._get_relative_context(env)
            except Exception:
                retry_ctx = {}
            current_step = int(self._get_step(env))
            force_merge_retry = bool(retry_ctx) and (
                int(retry_ctx.get("self_lane", 999)) != int(retry_ctx.get("ego_lane", -999))
                and self.intent in ("merge_commit", "front_brake")
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X
                <= float(retry_ctx.get("self_rel_x", 0.0))
                <= NEGOTIATED_MERGE_FAIL_REL_X
                and (
                    float(retry_ctx.get("self_rel_x", 0.0)) <= NEGOTIATED_MERGE_COMMIT_MAX_REL_X
                    or self._body_cut_in_command_ready(env, retry_ctx)
                )
                and (
                    self._cut_in_episode_active
                    or self._recent_step_active(self._merge_commit_until_step, current_step)
                    or self.executor_state == "merge_commit"
                )
            )
        if lc_action != 0:
            if force_merge_retry:
                lane_change_attempted = self._trigger_lane_change_once(env, lc_action)
            else:
                active_lane_change = self._lane_change_in_progress(env)
                same_target_in_progress = False
                if active_lane_change and self._lane_change_target_lane is not None:
                    try:
                        current_lane = int(env.k.vehicle.get_lane(self.veh_id))
                        requested_lane = int(current_lane) + int(lc_action)
                        same_target_in_progress = requested_lane == int(self._lane_change_target_lane)
                    except Exception:
                        same_target_in_progress = True
                if not (active_lane_change and same_target_in_progress):
                    lane_change_attempted = self._trigger_lane_change_once(env, lc_action)
            if (
                    lane_change_attempted
                    and self._use_structured_protocol()
                    and self._is_negotiated_highway_scene()
                    and self.attack_role == "Striker"
                    and self.intent in ("merge_commit", "front_brake")):
                self._merge_attempt_steps += 1
        elif (
                structured_runtime
                and self._is_negotiated_highway_scene()
                and self.attack_role == "Striker"
                and self.executor_state == "merge_wait_gap"):
            self._hold_current_lane(env, hold_steps=self.control_interval)
        elif (
                self.map_name == "highway"
                and self._is_three_car_scene()
                and not self._lane_change_in_progress(env)):
            self._hold_current_lane(env, hold_steps=self.control_interval)
        self._last_lane_change_attempted = bool(lane_change_attempted)
        if self.map_name == "highway":
            runtime_ctx = self._get_relative_context(env)
            self._update_merge_stall_state(
                env,
                runtime_ctx,
                lane_change_attempted=lane_change_attempted,
            )
            self._record_highway_runtime_trace(
                env,
                runtime_ctx,
                lane_change_attempted=lane_change_attempted,
            )

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
        if self.map_name == "highway":
            acc = self._override_highway_accel(env, acc)
        return float(acc)

    def _update_structured_plan_if_needed(self, env):
        step = self._get_step(env)
        if self.current_decision is None:
            self.current_decision = self._default_negotiated_runtime_decision(env)

        if self.current_decision.get("control", {}).get("mode") == "pulse_brake" and self._pulse_end_step >= 0 and step > self._pulse_end_step:
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
        if snapshot is None or "negotiated_contract" not in snapshot:
            env.message_pool.begin_control_cycle(step)
            snapshot = self._get_negotiated_snapshot(env)
        if self._maybe_lock_terminal_plan(env, snapshot=snapshot):
            if not self._is_control_step(env) or step == self.last_control_step:
                return False
            return self._publish_terminal_runtime_decision(env, snapshot=snapshot)

        if self.current_decision is None:
            self.current_decision = self._default_negotiated_runtime_decision(env)

        if self.current_decision.get("control", {}).get("mode") == "pulse_brake" and self._pulse_end_step >= 0 and step > self._pulse_end_step:
            self.current_decision["control"]["mode"] = "hold_lane"

        if not self._is_control_step(env) or step == self.last_control_step:
            return False

        if self._pending_intent_step != step or not self._pending_intent_plan:
            self.prepare_intent_for_coordinated_step(env, snapshot=snapshot)
            snapshot = self._get_negotiated_snapshot(env)
            if self._terminal_plan_locked:
                if step == self.last_control_step:
                    return False
                return self._publish_terminal_runtime_decision(env, snapshot=snapshot)

        self._last_action_reused = False
        self._last_action_refresh_reason = ""
        decision = self._structured_collaborate(env, snapshot)
        trigger_eval = self._evaluate_negotiated_tactic_trigger(env, snapshot, decision)

        self._sync_structured_diagnostics_from_decision(decision.get("intent_plan"), decision)
        self.last_control_step = step
        self._publish_negotiated_runtime_decision(env, decision, trigger_eval)
        self._finalize_highway_action_refresh_state(
            env,
            snapshot,
            decision,
            reused=self._last_action_reused,
            reason=self._last_action_refresh_reason,
        )
        self._set_execution_targets(env, decision, trigger_eval)
        self.has_llm_decision = True
        return True

    def _structured_collaborate(self, env, snapshot):
        if self._maybe_lock_terminal_plan(env, snapshot=snapshot):
            self._last_action_reused = True
            self._last_action_refresh_reason = "terminal_latch"
            return self._terminal_runtime_decision(env)
        intent_plan = copy.deepcopy(self._pending_intent_plan or {})
        if self._pending_intent_step != self._get_step(env) or not intent_plan:
            intent_plan = self._collect_highway_intent_plan(env, snapshot)
            self._pending_intent_plan = copy.deepcopy(intent_plan)
            self._pending_intent_step = self._get_step(env)
        refresh_reason = self._get_highway_action_refresh_reason(env, snapshot, intent_plan)
        if refresh_reason:
            self._last_action_reused = False
            self._last_action_refresh_reason = str(refresh_reason)
            self._record_action_refresh_trace(env, reused=False, reason=refresh_reason)
            return self._structured_highway_action_collaborate(env, snapshot, intent_plan)
        self._last_action_reused = True
        self._last_action_refresh_reason = "signature_unchanged"
        self._record_action_refresh_trace(env, reused=True, reason="signature_unchanged")
        return self._reuse_highway_action_decision(env, snapshot, intent_plan)

    def _current_runtime_phase(self, env):
        phase_seed = self.active_phase if self.active_phase in RUNTIME_PHASES else "compress"
        ctx = {}
        local_ready = copy.deepcopy(self._last_local_ready or {})
        if self.map_name == "highway" and self._is_three_car_scene():
            ctx = self._get_relative_context(env)
            local_ready = self._build_negotiated_local_ready(env, ctx)
        phase = self._normalize_runtime_phase(phase_seed)
        return self._apply_phase_progression(phase, self._get_step(env), ctx=ctx, local_ready=local_ready)

    def _default_highway_phase_intent(self, phase):
        return self._highway_phase_intent_contract().get(
            str(phase or "").strip().lower(),
            self._default_highway_intent(),
        )

    def _highway_phase_intent_contract(self, role=None):
        role = str(role or self.attack_role or "").strip()
        role_phase_defaults = {
            "Striker": {
                "compress": "gain_lead",
                "strike": "merge_commit",
                "brake_pulse": "front_brake",
                "disengage": "abort",
            },
            "Blocker": {
                "compress": "hold_side_front",
                "strike": "seal_escape",
                "brake_pulse": "seal_escape",
                "disengage": "abort",
            },
        }
        return copy.deepcopy(role_phase_defaults.get(role, {"disengage": "abort"}))

    def _intent_space_payload(self, pairs, reason):
        phases = []
        intents = []
        for phase, intent in pairs:
            if phase not in phases:
                phases.append(phase)
            if intent not in intents:
                intents.append(intent)
        return {
            "phases": phases,
            "intents": intents,
            "phase_intents": [{"phase": phase, "intent": intent} for phase, intent in pairs],
            "reason": str(reason or ""),
        }

    def _allowed_highway_intent_space(self, env, snapshot=None, ctx=None, local_ready=None):
        if not self._is_negotiated_highway_scene():
            contract = self._highway_phase_intent_contract()
            return self._intent_space_payload(
                [(phase, intent) for phase, intent in contract.items()],
                "generic_contract",
            )
        if self._terminal_plan_locked or self._success_recorded_in_feedback():
            return self._intent_space_payload([("disengage", "abort")], "terminal")
        ctx = copy.deepcopy(ctx or {})
        if not ctx:
            try:
                ctx = self._get_relative_context(env)
            except Exception:
                ctx = {}
        local_ready = copy.deepcopy(local_ready or {})
        if not local_ready and ctx:
            local_ready = self._build_negotiated_local_ready(env, ctx)
        if self.attack_role == "Striker":
            if not ctx:
                return self._intent_space_payload([("compress", "gain_lead")], "missing_context")
            rel_x = float(ctx.get("self_rel_x", 0.0))
            self_lane = int(ctx.get("self_lane", 0))
            ego_lane = int(ctx.get("ego_lane", 0))
            adjacent_to_ego = bool(self_lane != ego_lane and abs(self_lane - ego_lane) == 1)
            in_clean_commit_band = bool(
                adjacent_to_ego
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
            )
            in_late_body_safe_band = bool(
                adjacent_to_ego
                and NEGOTIATED_CLEAN_MERGE_MAX_REL_X < rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                and bool(local_ready.get("cut_in_gap_ready", False))
            )
            missed_merge_window = bool(
                adjacent_to_ego
                and rel_x > NEGOTIATED_CLEAN_MERGE_MAX_REL_X
                and not in_late_body_safe_band
            )
            if rel_x > NEGOTIATED_MERGE_FAIL_REL_X and adjacent_to_ego:
                self._mark_stale_merge_candidate(env)
            stale_observation = bool(self._dirty_observation_active(env))
            if (
                    bool(local_ready.get("overshoot", False) or self._overshoot)
                    or rel_x > NEGOTIATED_MERGE_FAIL_REL_X) and not stale_observation:
                return self._intent_space_payload([("disengage", "abort")], "overshoot")
            if self_lane == ego_lane:
                if bool(
                        local_ready.get("front_brake_window_ready", False)
                        or local_ready.get("same_lane_lead_established", False)
                        or self._merged_into_ego_lane
                        or self.striker_completed_cut_in):
                    return self._intent_space_payload([("brake_pulse", "front_brake")], "front_brake_ready")
                return self._intent_space_payload([("disengage", "abort")], "same_lane_not_actionable")
            if bool(local_ready.get("approach_window_ready", False)):
                return self._intent_space_payload([("strike", "merge_commit")], "approach_window_ready")
            if in_clean_commit_band or in_late_body_safe_band:
                return self._intent_space_payload([("strike", "merge_commit")], "clean_gap_wait")
            if missed_merge_window:
                return self._intent_space_payload([("compress", "gain_lead")], "merge_window_missed_recover")
            return self._intent_space_payload([("compress", "gain_lead")], "approach_window_not_ready")

        if self.attack_role == "Blocker":
            teammate_phase = self._get_latest_teammate_phase(snapshot or {})
            teammate_tactic = self._get_latest_teammate_tactic(snapshot or {})
            teammate_intent = str(teammate_phase.get("intent", "") or "")
            teammate_runtime_phase = str(teammate_phase.get("phase", "") or "")
            teammate_lane_policy = str(teammate_tactic.get("lane_policy", "") or "")
            teammate_entered_commit = bool(
                teammate_intent in ("merge_commit", "front_brake")
                or teammate_lane_policy == "ego_lane"
            )
            if not teammate_entered_commit:
                return self._intent_space_payload([("compress", "hold_side_front")], "teammate_not_committed")
            if teammate_intent == "front_brake" or teammate_runtime_phase == "brake_pulse":
                return self._intent_space_payload([("brake_pulse", "seal_escape")], "teammate_front_brake")
            return self._intent_space_payload([("strike", "seal_escape")], "teammate_merge_commit")

        return self._intent_space_payload([("disengage", "abort")], "unknown_role")

    def _allowed_phase_intent_pairs(self, allowed_next):
        pairs = []
        for item in (allowed_next or {}).get("phase_intents", []) or []:
            phase = str((item or {}).get("phase", "") or "").strip().lower()
            intent = str((item or {}).get("intent", "") or "").strip().lower()
            if phase and intent:
                pairs.append((phase, intent))
        if pairs:
            return pairs
        phases = [str(item or "").strip().lower() for item in (allowed_next or {}).get("phases", []) or []]
        intents = [str(item or "").strip().lower() for item in (allowed_next or {}).get("intents", []) or []]
        contract = self._highway_phase_intent_contract()
        for phase in phases:
            intent = contract.get(phase)
            if intent and (not intents or intent in intents):
                pairs.append((phase, intent))
        return pairs

    def _default_highway_intent_plan(self, env):
        phase = self._current_runtime_phase(env)
        intent = self._default_highway_phase_intent(phase)
        urgency = "high" if phase in ("strike", "brake_pulse") else ("low" if phase == "disengage" else "mid")
        if self.attack_role == "Striker":
            goal_map = {
                "gain_lead": "Gain a front or side-front position from the pass side.",
                "merge_commit": "Cut into ego_0's lane from the negotiated pass side.",
                "front_brake": "Brake in front of ego_0 after the merge is established.",
                "abort": "Disengage and stop forcing the attack.",
            }
        else:
            goal_map = {
                "hold_side_front": "Hold the block-side side-front slot and deny escape space.",
                "seal_escape": "Seal ego_0's escape lane while supporting the striker.",
                "abort": "Disengage and release the blockade.",
            }
        goal = goal_map.get(intent, intent.replace("_", " "))
        return {
            "phase": phase,
            "intent": intent,
            "urgency": urgency,
            "goal": goal,
            "message": goal,
        }

    def _phase_for_highway_intent(self, intent, fallback_phase="compress"):
        intent = str(intent or "").strip().lower()
        for phase, phase_intent in self._highway_phase_intent_contract().items():
            if str(phase_intent or "") == intent:
                return phase
        return str(fallback_phase or "compress")

    def _negotiated_phase_floor(self, snapshot):
        if self.active_phase in RUNTIME_PHASES:
            phase_floor = self.active_phase
        else:
            phase_floor = "compress"
        if self.attack_role != "Blocker":
            return phase_floor
        teammate_phase = str(
            (self._get_latest_teammate_phase(snapshot) or {}).get("phase", "") or ""
        ).strip().lower()
        if teammate_phase not in RUNTIME_PHASES or teammate_phase == "disengage":
            return phase_floor
        if self._runtime_phase_rank(teammate_phase) > self._runtime_phase_rank(phase_floor):
            return teammate_phase
        return phase_floor

    def _apply_negotiated_intent_geometry_guard(self, intent_plan, env, snapshot):
        if not self._is_negotiated_highway_scene():
            return copy.deepcopy(intent_plan or {})

        guarded = copy.deepcopy(intent_plan or {})
        phase = str(guarded.get("phase", "compress") or "compress")
        intent = str(guarded.get("intent", self._default_highway_phase_intent(phase)) or self._default_highway_phase_intent(phase))
        ctx = self._get_relative_context(env)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        current_step = int(self._get_step(env))
        repaired_fields = []

        if self.attack_role == "Striker":
            rel_x = float(ctx["self_rel_x"])
            self_lane = int(ctx["self_lane"])
            ego_lane = int(ctx["ego_lane"])
            adjacent_to_ego = bool(self_lane != ego_lane and abs(self_lane - ego_lane) == 1)
            within_commit_window = bool(
                adjacent_to_ego
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
            )
            merge_window_ready = bool(
                local_ready.get("approach_window_ready", False)
                or local_ready.get("cut_in_gap_ready", False)
                or local_ready.get("merged_into_ego_lane", False)
                or self._merged_into_ego_lane
                or (within_commit_window and self._cut_in_episode_active)
                or (within_commit_window and self._recent_step_active(self._merge_commit_until_step, current_step))
                or self._recent_merge_event_active(current_step)
            )
            brake_window_ready = bool(
                local_ready.get("front_brake_window_ready", False)
                or self.striker_completed_cut_in
                or self._recent_valid_merge_event_active(current_step)
            )
            if intent == "merge_commit" and adjacent_to_ego and rel_x > NEGOTIATED_MERGE_FAIL_REL_X:
                self._mark_stale_merge_candidate(env)
                phase = "compress"
                intent = "gain_lead"
                repaired_fields.extend(["phase", "intent"])
            elif intent == "merge_commit" and not merge_window_ready:
                phase = "compress"
                intent = "gain_lead"
                repaired_fields.extend(["phase", "intent"])
            elif intent == "front_brake" and not brake_window_ready:
                if merge_window_ready:
                    phase = "strike"
                    intent = "merge_commit"
                else:
                    phase = "compress"
                    intent = "gain_lead"
                repaired_fields.extend(["phase", "intent"])
        elif self.attack_role == "Blocker":
            teammate_phase = self._get_latest_teammate_phase(snapshot)
            teammate_tactic = self._get_latest_teammate_tactic(snapshot)
            teammate_merge_support = bool(
                str(teammate_phase.get("intent", "") or "") in ("merge_commit", "front_brake")
                or str(teammate_tactic.get("lane_policy", "") or "") == "ego_lane"
            )
            edge = ""
            try:
                edge = env.k.vehicle.get_edge(self.veh_id)
            except Exception:
                edge = ""
            block_lane = self._clamp_adjacent_lane(env, edge, int(ctx["ego_lane"]), self.block_side_rel or 1)
            seal_window_ready = bool(
                local_ready.get("blocker_side_front_ready", False)
                or (
                    int(ctx["self_lane"]) == int(block_lane)
                    and 1.5 <= float(ctx["self_rel_x"]) <= 18.0
                    and (float(ctx["self_rel_x"]) <= 12.0 or teammate_merge_support)
                )
                or self._recent_step_active(self._seal_escape_until_step, current_step)
            )
            if intent == "seal_escape" and not seal_window_ready:
                phase = "compress"
                intent = "hold_side_front"
                repaired_fields.extend(["phase", "intent"])

        guarded["phase"] = phase
        guarded["intent"] = intent
        self._record_repairs(repaired_fields)
        return guarded

    def _normalize_highway_intent_plan(self, parsed, env, snapshot, allowed_next=None):
        if not isinstance(parsed, dict):
            raise ValueError("Intent plan must be a dictionary.")

        default_plan = self._default_highway_intent_plan(env)
        allowed_next = copy.deepcopy(
            allowed_next or self._allowed_highway_intent_space(env, snapshot=snapshot)
        )
        allowed_pairs = self._allowed_phase_intent_pairs(allowed_next)
        allowed_phases = {phase for phase, _ in allowed_pairs}
        allowed_intents = {intent for _, intent in allowed_pairs}
        repaired_fields = []
        parsed_intent = str(parsed.get("intent", "") or "").strip().lower()
        phase_default = self._phase_for_highway_intent(parsed_intent, default_plan["phase"]) if parsed_intent else default_plan["phase"]
        phase = str(parsed.get("phase", phase_default) or phase_default).strip().lower()
        if phase == "setup":
            phase = "compress"
            repaired_fields.append("phase")
        if phase not in RUNTIME_PHASES:
            phase = default_plan["phase"]
            repaired_fields.append("phase")
        phase_floor = self._negotiated_phase_floor(snapshot)
        if (
                self._runtime_phase_rank(phase) < self._runtime_phase_rank(phase_floor)
                and (not allowed_phases or phase_floor in allowed_phases)):
            phase = phase_floor
            repaired_fields.append("phase")

        phase_intent_contract = self._highway_phase_intent_contract()
        default_intent = self._default_highway_phase_intent(phase)
        intent = str(parsed.get("intent", default_intent) or default_intent).strip().lower()
        if intent in phase_intent_contract.values():
            inferred_phase = self._phase_for_highway_intent(intent, phase)
            if inferred_phase != phase and (not allowed_phases or inferred_phase in allowed_phases):
                phase = inferred_phase
                default_intent = self._default_highway_phase_intent(phase)
        if intent != phase_intent_contract.get(phase, default_intent):
            intent = default_intent
            repaired_fields.append("intent")
        if allowed_pairs and (phase, intent) not in allowed_pairs:
            replacement = None
            for candidate_phase, candidate_intent in allowed_pairs:
                if candidate_phase == phase:
                    replacement = (candidate_phase, candidate_intent)
                    break
            if replacement is None:
                for candidate_phase, candidate_intent in allowed_pairs:
                    if candidate_intent in allowed_intents:
                        replacement = (candidate_phase, candidate_intent)
                        break
            if replacement is None:
                replacement = allowed_pairs[0]
            if phase != replacement[0]:
                repaired_fields.append("phase")
            if intent != replacement[1]:
                repaired_fields.append("intent")
            phase, intent = replacement

        urgency = "high" if phase in ("strike", "brake_pulse") else ("low" if phase == "disengage" else "mid")
        goal = str(default_plan.get("goal", intent.replace("_", " ")) or intent.replace("_", " ")).strip()
        message = str(parsed.get("message", goal) or goal).strip()
        if not message:
            message = goal
            repaired_fields.append("message")

        self._record_repairs(repaired_fields)
        normalized = {
            "phase": phase,
            "intent": intent,
            "urgency": urgency,
            "goal": goal[:200],
            "message": message[:160],
        }
        return self._apply_negotiated_intent_geometry_guard(normalized, env, snapshot)

    def _decision_intent_plan(self, decision):
        intent_plan = copy.deepcopy((decision or {}).get("intent_plan") or {})
        message = (decision or {}).get("message", {}) or {}
        if not intent_plan:
            fallback_message = str(message.get("intent", self.intent) or self.intent or self._default_highway_intent())
            intent_plan = {
                "phase": str(message.get("phase", self.active_phase) or self.active_phase or "compress"),
                "intent": fallback_message,
                "urgency": str(self.latest_intent_plan.get("urgency", self.intent_urgency or "mid") or "mid"),
                "goal": str(self.latest_intent_plan.get("goal", fallback_message) or fallback_message),
                "message": str(self.latest_intent_plan.get("message", fallback_message) or fallback_message),
            }
        return intent_plan

    def _sync_structured_diagnostics_from_decision(self, intent_plan, decision):
        resolved_intent_plan = copy.deepcopy(intent_plan or self._decision_intent_plan(decision))
        self.latest_intent_plan = resolved_intent_plan
        action_sequence_text = str(
            decision.get("action_sequence_text", "") or decision.get("message_text", "") or ""
        ).strip()
        if not action_sequence_text:
            action_sequence_text = self._build_negotiated_tactic_text(
                resolved_intent_plan,
                (decision.get("tactic", {}) or {}),
            )
            if action_sequence_text:
                decision["action_sequence_text"] = action_sequence_text[:240]
        if action_sequence_text and not decision.get("message_text"):
            decision["message_text"] = action_sequence_text[:240]
        self.latest_action_sequence_text = action_sequence_text[:240]
        self.latest_tactic_profile = copy.deepcopy((decision or {}).get("tactic") or {})

    def _publish_intent_plan(self, env, intent_plan):
        self._publish_negotiated_phase(env, intent_plan)

    def _compact_negotiated_phase_entry(self, entry):
        if not entry:
            return {}
        compact = {
            "phase": str(entry.get("phase", "") or ""),
            "intent": str(entry.get("intent", "") or ""),
            "urgency": str(entry.get("urgency", "") or ""),
            "message": str(entry.get("message", "") or ""),
        }
        return {key: value for key, value in compact.items() if value not in ("", None)}

    def _compact_negotiated_tactic_entry(self, entry):
        if not entry:
            return {}
        compact = {
            "phase": str(entry.get("phase", "") or ""),
            "intent": str(entry.get("intent", "") or ""),
            "mode": str(entry.get("mode", "") or ""),
            "lane_policy": str(entry.get("lane_policy", "") or ""),
            "gap_band": str(entry.get("gap_band", "") or ""),
            "speed_band": str(entry.get("speed_band", "") or ""),
            "style": str(entry.get("style", "") or ""),
            "sequence": copy.deepcopy(entry.get("sequence", []) or []),
            "speed_delta_hint_mps": entry.get("speed_delta_hint_mps"),
            "lead_gap_hint_m": entry.get("lead_gap_hint_m"),
            "hold_cycles": entry.get("hold_cycles"),
            "message": str(entry.get("message", "") or ""),
        }
        return {key: value for key, value in compact.items() if value not in ("", None)}

    def _build_negotiated_teammate_state(self, snapshot):
        teammate_id = self._get_teammate_id()
        teammate_phase = self._compact_negotiated_phase_entry(
            self._get_latest_teammate_phase(snapshot)
        )
        teammate_tactic = self._compact_negotiated_tactic_entry(
            self._get_latest_teammate_tactic(snapshot)
        )
        self._last_teammate_phase = str(teammate_phase.get("phase", "") or "")
        self._last_teammate_intent = str(teammate_phase.get("intent", "") or "")
        self._last_teammate_tactic = copy.deepcopy(teammate_tactic)
        if not teammate_id:
            return {}
        payload = {
            "teammate_phase": teammate_phase,
        }
        if teammate_tactic:
            payload["teammate_tactic"] = teammate_tactic
        return payload

    def _build_compact_history_context(self, env, phase_override=None):
        feedback_summary = self._build_feedback_summary()
        memory_summary = self._build_memory_summary(
            self.retrieve_case_memory(env, phase_override=phase_override)
        )
        parts = []
        if feedback_summary != "none":
            parts.append("feedback={}".format(feedback_summary))
        if memory_summary != "none":
            parts.append("memory={}".format(memory_summary))
        if not parts:
            return None
        return "\n".join(parts)

    def _collect_highway_intent_plan(self, env, snapshot):
        scenario_description = self.get_perception(env)
        ctx = self._get_relative_context(env)
        self._refresh_negotiated_runtime_state(env, ctx)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        prompt_local_ready = self._build_negotiated_prompt_local_ready(local_ready)
        allowed_next = self._allowed_highway_intent_space(
            env,
            snapshot=snapshot,
            ctx=ctx,
            local_ready=local_ready,
        )
        teammate_state = self._build_negotiated_teammate_state(snapshot)
        prompt_teammate_state = {
            "teammate_phase": copy.deepcopy((teammate_state or {}).get("teammate_phase") or {})
        }
        history_phase = self._current_runtime_phase(env)
        history_context = None
        should_inject_history = history_phase != str(self._last_history_injection_phase or "")
        if should_inject_history:
            history_context = self._build_compact_history_context(env, phase_override=history_phase)

        intent_response = ""
        try:
            intent_response = self.DA.collaborate_highway_intent(
                scenario_description,
                self.attack_role,
                self.attack_target,
                self.highway_contract,
                prompt_teammate_state,
                prompt_local_ready,
                history_context,
                allowed_next,
            )
            intent_parsed = self._extract_decision_dict(intent_response)
            intent_plan = self._normalize_highway_intent_plan(
                intent_parsed,
                env,
                snapshot,
                allowed_next=allowed_next,
            )
            self._record_llm_trace(
                env,
                protocol="highway_intent",
                attempt=1,
                response=intent_response,
                parsed=intent_plan,
                error="",
            )
        except Exception as exc:
            self._record_llm_trace(
                env,
                protocol="highway_intent",
                attempt=1,
                response=intent_response,
                parsed=None,
                error=str(exc),
            )
            self.parse_failures += 1
            self.last_parse_error = str(exc)
            intent_plan = self._default_highway_intent_plan(env)
            self._record_llm_trace(
                env,
                protocol="highway_intent_fallback",
                attempt=1,
                response="",
                parsed=intent_plan,
                error="",
            )
        if should_inject_history:
            self._last_history_injection_phase = str(
                (intent_plan or {}).get("phase", history_phase) or history_phase
            )
        self.latest_intent_plan = copy.deepcopy(intent_plan)
        return intent_plan

    def prepare_intent_for_coordinated_step(self, env, snapshot=None):
        self._begin_rollout_if_needed(env)
        if not self._use_structured_protocol():
            return False

        step = self._get_step(env)
        if not self._is_control_step(env):
            return False
        if self._pending_intent_step == step and self._pending_intent_plan:
            return False

        if snapshot is None or "latest_phase_by_agent" not in snapshot:
            env.message_pool.begin_control_cycle(step)
            snapshot = self._get_negotiated_snapshot(env)
        if self._maybe_lock_terminal_plan(env, snapshot=snapshot):
            self._publish_terminal_intent_plan(env)
            return True

        intent_plan = self._collect_highway_intent_plan(env, snapshot)
        self._pending_intent_plan = copy.deepcopy(intent_plan)
        self._pending_intent_step = step
        self._publish_intent_plan(env, intent_plan)
        return True

    def _default_negotiated_tactic_profile(self, intent_plan):
        phase = str((intent_plan or {}).get("phase", "compress") or "compress")
        intent = str((intent_plan or {}).get("intent", self._default_highway_phase_intent(phase)) or self._default_highway_phase_intent(phase))
        if intent == "abort" or phase == "disengage":
            return {
                "mode": "disengage",
                "lane_policy": "hold_current",
                "gap_band": "loose",
                "speed_band": "yield",
            }
        if self.attack_role == "Blocker":
            if intent == "seal_escape":
                return {
                    "mode": "track_pose",
                    "lane_policy": "block_side",
                    "gap_band": "tight",
                    "speed_band": "press",
                }
            return {
                "mode": "track_pose",
                "lane_policy": "block_side",
                "gap_band": "medium",
                "speed_band": "press",
            }
        if intent == "front_brake" or phase == "brake_pulse":
            return {
                "mode": "pulse_brake",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "brake",
            }
        if intent == "merge_commit":
            return {
                "mode": "track_pose",
                "lane_policy": "ego_lane",
                "gap_band": "tight",
                "speed_band": "surge",
            }
        return {
            "mode": "track_pose",
            "lane_policy": "pass_side",
            "gap_band": "medium",
            "speed_band": "press",
        }

    def _build_negotiated_tactic_text(self, intent_plan, tactic):
        phase = str((intent_plan or {}).get("phase", "compress") or "compress")
        intent = str((intent_plan or {}).get("intent", self._default_highway_phase_intent(phase)) or self._default_highway_phase_intent(phase))
        lane_policy = str((tactic or {}).get("lane_policy", "hold_current") or "hold_current").replace("_", " ")
        gap_band = str((tactic or {}).get("gap_band", "medium") or "medium")
        speed_band = str((tactic or {}).get("speed_band", "match") or "match")
        hints = (tactic or {}).get("tactic_hints", {}) or tactic or {}
        sequence = hints.get("sequence", []) if isinstance(hints, dict) else []
        sequence_text = " sequence={}".format(",".join(sequence)) if sequence else ""
        return "{} via {} with {} gap and {} speed{}.".format(
            intent.replace("_", " "),
            lane_policy,
            gap_band,
            speed_band,
            sequence_text,
        )[:240]

    def _build_negotiated_runtime_decision(self, env, intent_plan, tactic_profile=None, message_text=""):
        step = self._get_step(env)
        phase = str((intent_plan or {}).get("phase", "compress") or "compress")
        intent = str((intent_plan or {}).get("intent", self._default_highway_phase_intent(phase)) or self._default_highway_phase_intent(phase))
        tactic = copy.deepcopy(tactic_profile or self._default_negotiated_tactic_profile(intent_plan))
        horizon_steps = max(1, int(self.control_interval * (2 if phase in ("compress", "strike") else 1)))
        text = str(message_text or self._build_negotiated_tactic_text(intent_plan, tactic)).strip()
        decision = {
            "message": {
                "sender": self.veh_id,
                "phase": phase,
                "intent": intent,
                "target": self.attack_target,
                "step": int(step),
                "expires_at_step": int(step + horizon_steps),
            },
            "control": {
                "mode": str(tactic.get("mode", "hold_lane") or "hold_lane"),
                "horizon_steps": horizon_steps,
            },
            "tactic": tactic,
            "intent_plan": copy.deepcopy(intent_plan or {}),
            "action_sequence_text": text[:240],
            "message_text": text[:240],
        }
        return decision

    def _default_negotiated_runtime_decision(self, env):
        snapshot = self._get_negotiated_snapshot(env)
        intent_plan = self._default_highway_intent_plan(env)
        return self._build_negotiated_runtime_decision(env, intent_plan)

    def _role_specific_tactic_constraints(self, phase, intent, default_tactic):
        default_tactic = copy.deepcopy(default_tactic or {})
        if intent == "abort" or phase == "disengage":
            return {
                "mode": "disengage",
                "lane_policy": "hold_current",
                "gap_band": "loose",
                "speed_band": "yield",
            }
        if self.attack_role == "Striker":
            if intent == "front_brake" or phase == "brake_pulse":
                return {
                    "mode": "pulse_brake",
                    "lane_policy": "ego_lane",
                    "gap_band": "tight",
                    "speed_band": "brake",
                }
            if intent == "merge_commit":
                return {
                    "mode": "track_pose",
                    "lane_policy": "ego_lane",
                    "gap_band": "tight",
                    "speed_band": "surge",
                }
            if intent == "gain_lead":
                return {
                    "mode": "track_pose",
                    "lane_policy": "pass_side",
                    "gap_band": default_tactic.get("gap_band", "medium"),
                    "speed_band": default_tactic.get("speed_band", "press"),
                }
        if self.attack_role == "Blocker":
            if intent == "seal_escape":
                return {
                    "mode": "track_pose",
                    "lane_policy": "block_side",
                    "gap_band": "tight",
                    "speed_band": "press",
                }
            if intent == "hold_side_front":
                return {
                    "mode": "track_pose",
                    "lane_policy": "block_side",
                    "gap_band": default_tactic.get("gap_band", "medium"),
                    "speed_band": default_tactic.get("speed_band", "press"),
                }
        return {}

    def _allowed_tactic_sequence_tokens(self, phase=None, intent=None, role=None):
        return _allowed_sequence_tokens_for_role_intent(
            role or self.attack_role,
            phase=phase,
            intent=intent,
        )

    def _default_tactic_hint_values(self, phase, intent):
        if self.attack_role == "Striker":
            if intent == "merge_commit":
                return {
                    "sequence": ["cap_speed_advantage", "commit_lane_change", "stabilize_same_lane", "front_brake"],
                    "speed_delta_hint_mps": 1.0,
                    "lead_gap_hint_m": 2.0,
                    "hold_cycles": 2,
                }
            if intent == "front_brake" or phase == "brake_pulse":
                return {
                    "sequence": ["stabilize_same_lane", "front_brake"],
                    "speed_delta_hint_mps": -7.0,
                    "lead_gap_hint_m": 1.5,
                    "hold_cycles": 1,
                }
            if intent == "gain_lead":
                return {
                    "sequence": ["gain_lead", "cap_speed_advantage"],
                    "speed_delta_hint_mps": 4.0,
                    "lead_gap_hint_m": 3.0,
                    "hold_cycles": 1,
                }
        if self.attack_role == "Blocker":
            if intent == "seal_escape":
                return {
                    "sequence": ["match_ego", "seal_escape"],
                    "speed_delta_hint_mps": 0.5,
                    "lead_gap_hint_m": 2.5,
                    "hold_cycles": 2,
                }
            if intent == "hold_side_front":
                return {
                    "sequence": ["hold_side_front", "match_ego"],
                    "speed_delta_hint_mps": 0.8,
                    "lead_gap_hint_m": 3.5,
                    "hold_cycles": 1,
                }
        return {
            "sequence": ["recover"],
            "speed_delta_hint_mps": -1.0,
            "lead_gap_hint_m": 4.0,
            "hold_cycles": 1,
        }

    def _clamp_tactic_speed_hint(self, speed_hint, phase, intent):
        if self.attack_role == "Striker":
            if intent == "merge_commit":
                return float(np.clip(speed_hint, 0.5, 1.5))
            if intent == "gain_lead":
                return float(np.clip(speed_hint, 2.0, 5.5))
            if intent == "front_brake" or phase == "brake_pulse":
                return float(np.clip(speed_hint, -10.0, -4.0))
            return float(np.clip(speed_hint, -2.0, 5.5))
        if self.attack_role == "Blocker":
            return float(np.clip(speed_hint, -2.0, 2.5))
        return float(np.clip(speed_hint, -2.0, 2.5))

    def _normalize_tactic_hints(self, tactic_payload, phase, intent):
        tactic_payload = tactic_payload if isinstance(tactic_payload, dict) else {}
        defaults = self._default_tactic_hint_values(phase, intent)
        repaired_fields = []

        allowed_tokens = set(self._allowed_tactic_sequence_tokens(phase=phase, intent=intent))
        raw_sequence = tactic_payload.get("sequence", defaults["sequence"])
        if isinstance(raw_sequence, str):
            raw_sequence = [
                token.strip().lower()
                for token in re.split(r"[,>]+|\s+then\s+", raw_sequence)
                if token.strip()
            ]
        if not isinstance(raw_sequence, list):
            raw_sequence = list(defaults["sequence"])
            repaired_fields.append("sequence")
        sequence = []
        for item in raw_sequence:
            token = str(item or "").strip().lower()
            if token in allowed_tokens and token not in sequence:
                sequence.append(token)
            elif token:
                repaired_fields.append("sequence")
        if not sequence:
            sequence = list(defaults["sequence"])
            repaired_fields.append("sequence")
        if (
                self.attack_role == "Striker"
                and str(intent or "") == "merge_commit"
                and "commit_lane_change" in sequence
                and "front_brake" not in sequence):
            sequence.append("front_brake")
            repaired_fields.append("sequence")

        raw_speed = self._safe_float(
            tactic_payload.get("speed_delta_hint_mps"),
            defaults["speed_delta_hint_mps"],
        )
        speed_hint = self._clamp_tactic_speed_hint(raw_speed, phase, intent)
        if abs(float(speed_hint) - float(raw_speed)) > 1e-6:
            repaired_fields.append("speed_delta_hint_mps")

        raw_gap = self._safe_float(
            tactic_payload.get("lead_gap_hint_m"),
            defaults["lead_gap_hint_m"],
        )
        if self.attack_role == "Striker":
            lead_gap = float(np.clip(raw_gap, 0.8, 4.0))
        else:
            lead_gap = float(np.clip(raw_gap, 0.5, 6.0))
        if abs(float(lead_gap) - float(raw_gap)) > 1e-6:
            repaired_fields.append("lead_gap_hint_m")

        raw_hold = self._safe_float(
            tactic_payload.get("hold_cycles"),
            defaults["hold_cycles"],
        )
        hold_cycles = int(np.clip(int(round(raw_hold)), 1, 3))
        if abs(float(hold_cycles) - float(raw_hold)) > 1e-6:
            repaired_fields.append("hold_cycles")

        forbidden_keys = (
            "target_speed",
            "target_v",
            "absolute_speed",
            "absolute_position",
            "target_lane",
            "lane_id",
            "lane_index",
            "mode",
            "lane_policy",
            "gap_band",
            "speed_band",
        )
        for key in forbidden_keys:
            if key in tactic_payload:
                repaired_fields.append(key)

        return {
            "sequence": sequence,
            "speed_delta_hint_mps": round(float(speed_hint), 3),
            "lead_gap_hint_m": round(float(lead_gap), 3),
            "hold_cycles": int(hold_cycles),
        }, repaired_fields

    def _normalize_highway_tactic_output(self, parsed, env, intent_plan):
        if not isinstance(parsed, dict):
            raise ValueError("Tactic output must be a dictionary.")
        tactic_payload = parsed.get("tactic")
        if not isinstance(tactic_payload, dict):
            tactic_payload = {
                "style": parsed.get("style"),
            }
        repaired_fields = []
        phase = str((intent_plan or {}).get("phase", "compress") or "compress")
        intent = str(
            (intent_plan or {}).get("intent", self._default_highway_phase_intent(phase))
            or self._default_highway_phase_intent(phase)
        )
        default_tactic = self._default_negotiated_tactic_profile(intent_plan)
        normalized_tactic = copy.deepcopy(
            self._role_specific_tactic_constraints(phase, intent, default_tactic) or default_tactic
        )
        style = str(
            tactic_payload.get("style", parsed.get("style", "normal"))
            or "normal"
        ).strip().lower()
        if style not in NEGOTIATED_TACTIC_STYLES:
            style = "normal"
            repaired_fields.append("style")
        normalized_tactic["style"] = style
        tactic_hints, hint_repairs = self._normalize_tactic_hints(tactic_payload, phase, intent)
        repaired_fields.extend(hint_repairs)
        normalized_tactic.update(tactic_hints)
        normalized_tactic["tactic_hints"] = copy.deepcopy(tactic_hints)

        self._record_repairs(repaired_fields)
        message_text = str(parsed.get("message", "") or "").strip()
        return self._build_negotiated_runtime_decision(
            env,
            intent_plan,
            tactic_profile=normalized_tactic,
            message_text=message_text,
        )

    def _structured_highway_action_collaborate(self, env, snapshot, intent_plan):
        scenario_description = self.get_perception(env)
        ctx = self._get_relative_context(env)
        self._refresh_negotiated_runtime_state(env, ctx)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        prompt_local_ready = self._build_negotiated_prompt_local_ready(local_ready)
        teammate_state = self._build_negotiated_teammate_state(snapshot)
        prompt_teammate_state = {
            "teammate_phase": copy.deepcopy((teammate_state or {}).get("teammate_phase") or {})
        }
        teammate_tactic = copy.deepcopy((teammate_state or {}).get("teammate_tactic") or {})
        if teammate_tactic:
            prompt_teammate_state["teammate_tactic"] = teammate_tactic

        action_response = ""
        try:
            action_response = self.DA.collaborate_highway_tactic(
                scenario_description,
                self.attack_role,
                self.attack_target,
                self.highway_contract,
                intent_plan,
                prompt_teammate_state,
                prompt_local_ready,
            )
            action_parsed = self._extract_decision_dict(action_response)
            decision = self._normalize_highway_tactic_output(
                action_parsed,
                env,
                intent_plan,
            )
            self._sync_structured_diagnostics_from_decision(intent_plan, decision)
            self._record_llm_trace(
                env,
                protocol="highway_tactic",
                attempt=1,
                response=action_response,
                parsed=action_parsed,
                error="",
            )
            self.last_parse_error = ""
            return decision
        except Exception as exc:
            self._record_llm_trace(
                env,
                protocol="highway_tactic",
                attempt=1,
                response=action_response,
                parsed=None,
                error=str(exc),
            )
            self.parse_failures += 1
            self.last_parse_error = str(exc)

        self.fallback_activations += 1
        self.rollout_parse_fallback_used += 1
        try:
            decision = self._build_negotiated_runtime_decision(env, intent_plan)
            self._record_llm_trace(
                env,
                protocol="highway_tactic_fallback",
                attempt=1,
                response="",
                parsed=decision,
                error="",
            )
            return decision
        except Exception as exc:
            self.repair_fallback_used += 1
            self._record_llm_trace(
                env,
                protocol="highway_tactic_fallback",
                attempt=1,
                response="",
                parsed=None,
                error=str(exc),
            )
            decision = self._build_negotiated_runtime_decision(env, intent_plan)
            self._sync_structured_diagnostics_from_decision(intent_plan, decision)
            return decision

    def _evaluate_negotiated_tactic_trigger(self, env, snapshot, decision=None):
        basis = decision if isinstance(decision, dict) else self.current_decision or {}
        intent_plan = copy.deepcopy((basis or {}).get("intent_plan") or {})
        ctx = self._get_relative_context(env)
        self._refresh_negotiated_runtime_state(env, ctx)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        teammate_phase = self._get_latest_teammate_phase(snapshot)
        teammate_tactic = self._get_latest_teammate_tactic(snapshot)
        intent = str((intent_plan or {}).get("intent", (basis.get("message", {}) or {}).get("intent", self.intent)) or self.intent)
        phase = str((intent_plan or {}).get("phase", (basis.get("message", {}) or {}).get("phase", self.active_phase)) or self.active_phase)

        satisfied = True
        details = {
            "local_ready": {
                "approach_window_ready": bool(local_ready.get("approach_window_ready", False)),
                "cut_in_gap_ready": bool(local_ready.get("cut_in_gap_ready", False)),
                "effective_ego_gap_after_merge": local_ready.get("effective_ego_gap_after_merge", None),
                "merged_into_ego_lane": bool(local_ready.get("merged_into_ego_lane", False) or self._merged_into_ego_lane),
                "front_brake_window_ready": bool(local_ready.get("front_brake_window_ready", False)),
                "overshoot": bool(local_ready.get("overshoot", False) or self._overshoot),
            }
        }

        if intent == "merge_commit":
            teammate_support = str(teammate_phase.get("intent", "") or "") in ("hold_side_front", "seal_escape")
            teammate_support = teammate_support or str(teammate_tactic.get("lane_policy", "") or "") == "block_side"
            rel_x = float(ctx.get("self_rel_x", 0.0))
            cut_in_episode_in_window = bool(
                self._cut_in_episode_active
                and int(ctx.get("self_lane", 0)) != int(ctx.get("ego_lane", 0))
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
            )
            details["teammate_support"] = bool(teammate_support)
            satisfied = bool(
                local_ready.get("approach_window_ready", False)
                or local_ready.get("cut_in_gap_ready", False)
                or self._merged_into_ego_lane
                or cut_in_episode_in_window
                or (
                    teammate_support
                    and rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
                )
            )
        elif intent == "front_brake" or phase == "brake_pulse":
            satisfied = bool(
                (local_ready.get("front_brake_window_ready", False) or self._front_brake_triggered)
                and (self._merged_into_ego_lane or self.striker_completed_cut_in)
            )
        elif intent == "seal_escape":
            teammate_merge = str(teammate_phase.get("intent", "") or "") == "merge_commit"
            teammate_merge = teammate_merge or str(teammate_tactic.get("lane_policy", "") or "") == "ego_lane"
            satisfied = bool(
                teammate_merge
                or local_ready.get("teammate_pass_window_ready", False)
                or self._recent_step_active(self._seal_escape_until_step, self._get_step(env))
            )

        return {
            "trigger_code": "negotiated_local",
            "satisfied": bool(satisfied),
            "details": details,
        }

    def _current_action_trigger_satisfied(self, env, snapshot, decision=None):
        basis = decision if isinstance(decision, dict) else self.current_decision
        if not isinstance(basis, dict) or not basis:
            return None
        return bool(self._evaluate_negotiated_tactic_trigger(env, snapshot, basis).get("satisfied", False))

    def _rel_x_bucket_index(self, rel_x):
        rel_x = float(rel_x)
        for idx, edge in enumerate(HIGHWAY_ACTION_REL_X_BUCKET_EDGES):
            if rel_x < float(edge):
                return idx
        return len(HIGHWAY_ACTION_REL_X_BUCKET_EDGES)

    def _stable_action_rel_x_bucket(self, rel_x):
        rel_x = float(rel_x)
        raw_index = self._rel_x_bucket_index(rel_x)
        stable_index = raw_index
        prev_index = self._last_action_rel_x_bucket_index
        if prev_index is not None:
            try:
                prev_index = int(prev_index)
            except Exception:
                prev_index = None
        if prev_index is not None and 0 <= prev_index < len(HIGHWAY_ACTION_REL_X_BUCKET_LABELS):
            if raw_index > prev_index:
                boundary = float(HIGHWAY_ACTION_REL_X_BUCKET_EDGES[raw_index - 1])
                if rel_x < boundary + HIGHWAY_ACTION_REL_X_HYSTERESIS_M:
                    stable_index = prev_index
            elif raw_index < prev_index:
                boundary = float(HIGHWAY_ACTION_REL_X_BUCKET_EDGES[raw_index])
                if rel_x > boundary - HIGHWAY_ACTION_REL_X_HYSTERESIS_M:
                    stable_index = prev_index
        stable_index = max(0, min(int(stable_index), len(HIGHWAY_ACTION_REL_X_BUCKET_LABELS) - 1))
        return HIGHWAY_ACTION_REL_X_BUCKET_LABELS[stable_index], stable_index

    def _build_highway_action_refresh_signature(self, env, snapshot, intent_plan, decision=None):
        ctx = self._get_relative_context(env)
        self._refresh_negotiated_runtime_state(env, ctx)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        teammate_id = self._get_teammate_id()
        latest_phase_by_agent = snapshot.get("latest_phase_by_agent") or {}
        latest_tactic_by_agent = snapshot.get("latest_tactic_by_agent") or {}
        decision = decision if isinstance(decision, dict) else self.current_decision or {}
        message = decision.get("message", {}) or {}
        control = decision.get("control", {}) or {}
        tactic = decision.get("tactic", {}) or {}
        resolved_phase = str(
            (intent_plan or {}).get("phase", self._current_runtime_phase(env)) or "compress"
        )
        resolved_intent = str(
            (intent_plan or {}).get("intent", self._default_highway_phase_intent(resolved_phase))
            or self._default_highway_phase_intent(resolved_phase)
        )
        rel_x_bucket, rel_x_bucket_index = self._stable_action_rel_x_bucket(ctx["self_rel_x"])
        trigger_satisfied = self._current_action_trigger_satisfied(env, snapshot, decision=decision)
        signature = {
            "phase": resolved_phase,
            "intent": resolved_intent,
            "self_rel_lane": int(ctx["self_rel_lane"]),
            "merged_into_ego_lane": bool(
                local_ready.get("merged_into_ego_lane", False) or self._merged_into_ego_lane
            ),
            "approach_window_ready": bool(local_ready.get("approach_window_ready", False)),
            "cut_in_gap_ready": bool(local_ready.get("cut_in_gap_ready", False)),
            "front_brake_window_ready": bool(local_ready.get("front_brake_window_ready", False)),
            "trigger_satisfied": trigger_satisfied,
            "teammate_phase": str(
                ((latest_phase_by_agent.get(teammate_id) or {}).get("phase", "") or "")
            ),
            "teammate_intent": str(
                ((latest_phase_by_agent.get(teammate_id) or {}).get("intent", "") or "")
            ),
            "teammate_mode": str(
                ((latest_tactic_by_agent.get(teammate_id) or {}).get("mode", "") or "")
            ),
            "decision_expired": bool(
                self._get_step(env) >= int(message.get("expires_at_step", self._get_step(env) + 1))
            ),
            "control_mode": str(
                tactic.get("mode", control.get("mode", self.active_control_mode))
                or control.get("mode", self.active_control_mode)
                or ""
            ),
            "bad_merge_reason": str(self._bad_merge_reason or ""),
            "overshoot": bool(local_ready.get("overshoot", False) or self._overshoot),
            "rel_x_bucket": rel_x_bucket,
        }
        return signature, rel_x_bucket, rel_x_bucket_index

    def _changed_highway_action_refresh_reason(self, previous_signature, current_signature):
        checks = (
            ("phase", "phase_changed"),
            ("intent", "intent_changed"),
            ("self_rel_lane", "self_rel_lane_changed"),
            ("merged_into_ego_lane", "merged_into_ego_lane_changed"),
            ("approach_window_ready", "approach_window_ready_changed"),
            ("cut_in_gap_ready", "cut_in_gap_ready_changed"),
            ("front_brake_window_ready", "front_brake_window_ready_changed"),
            ("bad_merge_reason", "bad_merge_reason_changed"),
            ("overshoot", "overshoot_changed"),
            ("trigger_satisfied", "trigger_satisfied_changed"),
            ("teammate_phase", "teammate_phase_changed"),
            ("teammate_intent", "teammate_intent_changed"),
            ("teammate_mode", "teammate_mode_changed"),
            ("decision_expired", "decision_expired"),
            ("control_mode", "control_mode_changed"),
            ("rel_x_bucket", "rel_x_bucket_changed"),
        )
        for field, reason in checks:
            if previous_signature.get(field) != current_signature.get(field):
                return reason
        return ""

    def _get_highway_action_refresh_reason(self, env, snapshot, intent_plan, decision=None):
        current_decision = decision if isinstance(decision, dict) else self.current_decision
        if not isinstance(current_decision, dict) or not current_decision:
            return "missing_current_decision"
        if not current_decision.get("tactic"):
            return "status_decision"
        current_signature, _, _ = self._build_highway_action_refresh_signature(
            env, snapshot, intent_plan, decision=current_decision
        )
        previous_signature = copy.deepcopy(self._last_action_refresh_signature or {})
        if not previous_signature:
            return "missing_signature"
        reason = self._changed_highway_action_refresh_reason(previous_signature, current_signature)
        if reason:
            return reason
        current_trigger_satisfied = current_signature.get("trigger_satisfied")
        if (
                self._last_published_trigger_satisfied is not None
                and current_trigger_satisfied is not None
                and bool(self._last_published_trigger_satisfied) != bool(current_trigger_satisfied)):
            return "trigger_satisfied_changed"
        if HIGHWAY_ACTION_REUSE_MAX_CYCLES <= 0:
            return "reuse_disabled"
        if int(self._action_reuse_count) >= int(HIGHWAY_ACTION_REUSE_MAX_CYCLES):
            return "hard_refresh"
        return ""

    def _record_action_refresh_trace(self, env, reused, reason):
        payload = {
            "mode": "reuse" if reused else "recompute",
            "reason": str(reason or ("signature_unchanged" if reused else "unknown")),
        }
        previous_raw_response = self.last_raw_response
        self._record_llm_trace(
            env,
            protocol="highway_action_gate",
            attempt=1,
            response=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            parsed=payload,
            error="",
        )
        self.last_raw_response = previous_raw_response

    def _reuse_highway_action_decision(self, env, snapshot, intent_plan):
        decision = copy.deepcopy(self.current_decision or {})
        if not decision:
            return self._structured_highway_action_collaborate(env, snapshot, intent_plan)

        step = self._get_step(env)
        message = decision.setdefault("message", copy.deepcopy(self._default_message()))
        control = decision.setdefault("control", copy.deepcopy(self._default_control()))
        tactic = decision.setdefault("tactic", copy.deepcopy(self._default_negotiated_tactic_profile(intent_plan)))
        horizon_steps = max(1, int(control.get("horizon_steps", self.control_interval)))

        message["sender"] = self.veh_id
        message["target"] = self.attack_target
        message["step"] = step
        message["expires_at_step"] = step + horizon_steps
        message["phase"] = str((intent_plan or {}).get("phase", message.get("phase", self.active_phase)) or self.active_phase)
        message["intent"] = str((intent_plan or {}).get("intent", message.get("intent", "")) or message.get("intent", ""))
        control["mode"] = str(tactic.get("mode", control.get("mode", "hold_lane")) or control.get("mode", "hold_lane"))

        decision["intent_plan"] = copy.deepcopy(intent_plan or self._decision_intent_plan(decision))
        action_sequence_text = str(
            decision.get("action_sequence_text", "") or decision.get("message_text", "") or ""
        ).strip()
        if not action_sequence_text:
            action_sequence_text = self._build_negotiated_tactic_text(intent_plan, tactic)
            decision["action_sequence_text"] = action_sequence_text[:240]
        decision["message_text"] = action_sequence_text[:240]
        self._sync_structured_diagnostics_from_decision(intent_plan, decision)
        return decision

    def _finalize_highway_action_refresh_state(self, env, snapshot, decision, reused, reason):
        self._last_action_reused = bool(reused)
        self._last_action_refresh_reason = str(reason or "")
        if not self._use_structured_protocol():
            return

        intent_plan = self._decision_intent_plan(decision)
        signature, rel_x_bucket, rel_x_bucket_index = self._build_highway_action_refresh_signature(
            env,
            snapshot,
            intent_plan,
            decision=decision,
        )
        self._last_action_rel_x_bucket = rel_x_bucket
        self._last_action_rel_x_bucket_index = rel_x_bucket_index

        if not decision.get("tactic"):
            self._last_action_refresh_signature = {}
            self._action_reuse_count = 0
            return

        self._last_action_refresh_signature = copy.deepcopy(signature)
        self._action_reuse_count = (int(self._action_reuse_count) + 1) if reused else 0

    def _build_feedback_summary(self):
        if not self.previous_feedback:
            return "none"
        fields = []
        for key in (
                "result",
                "feedback_summary",
                "failure_reason",
                "escape_summary",
                "failure_phase",
                "merged_into_ego_lane",
                "valid_cut_in_merge",
                "bad_merge_reason",
                "front_brake_triggered",
                "ego_escape_lane",
                "blocker_lane_at_escape",
                "min_ttc",
                "ego_max_decel"):
            value = self.previous_feedback.get(key)
            if value is not None and value != "":
                fields.append("{}={}".format(key, value))
        return "; ".join(fields) if fields else "none"

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

    def _normalize_runtime_phase(self, phase):
        phase = str(phase or "").strip() or str(self.active_phase or "")
        if phase == "setup":
            if self.active_phase in RUNTIME_PHASES:
                phase = self.active_phase
            else:
                phase = "compress"
        if phase not in RUNTIME_PHASES:
            raise ValueError("Invalid runtime phase: {}".format(phase))
        return phase

    def _runtime_phase_rank(self, phase):
        return int(RUNTIME_PHASE_RANK.get(str(phase or "").strip().lower(), 0))

    def _event_progress_phase(self, step, ctx=None, local_ready=None):
        if self.map_name != "highway" or not self._is_three_car_scene():
            return "compress"

        current_step = int(step)
        ctx = ctx or {}
        local_ready = copy.deepcopy(local_ready or self._last_local_ready or {})
        strike_ready = False
        brake_ready = False

        if self.attack_role == "Striker":
            rel_x = float(ctx.get("self_rel_x", 0.0)) if ctx else 0.0
            self_lane = int(ctx.get("self_lane", 0)) if ctx else 0
            ego_lane = int(ctx.get("ego_lane", 0)) if ctx else 0
            if bool(local_ready.get("overshoot", False) or self._overshoot):
                if not self._dirty_observation_active(None):
                    return "disengage"
            if self_lane != ego_lane and rel_x > NEGOTIATED_MERGE_FAIL_REL_X:
                self._mark_stale_merge_candidate(None)
                if self._dirty_observation_active(None):
                    return "compress"
                return "disengage"
            strike_ready = bool(
                local_ready.get("approach_window_ready", False)
                or local_ready.get("cut_in_gap_ready", False)
                or (
                    self._cut_in_episode_active
                    and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
                )
                or (
                    self._recent_step_active(self._merge_commit_until_step, current_step)
                    and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
                )
                or self._merged_into_ego_lane
                or self.striker_completed_cut_in
                or local_ready.get("same_lane_lead_established", False)
                or local_ready.get("front_brake_window_ready", False)
            )
            brake_ready = bool(
                local_ready.get("merged_into_ego_lane", False)
                or self._merged_into_ego_lane
                or self.striker_completed_cut_in
                or local_ready.get("same_lane_lead_established", False)
                or local_ready.get("front_brake_window_ready", False)
                or self._front_brake_triggered
            )
        elif self.attack_role == "Blocker":
            teammate_rel_lane = int(ctx.get("teammate_rel_lane", 99))
            teammate_rel_x = float(ctx.get("teammate_rel_x", -99.0))
            teammate_merge_window = teammate_rel_lane == 0 and -1.0 <= teammate_rel_x <= 6.0
            teammate_same_lane_ahead = teammate_rel_lane == 0 and teammate_rel_x >= 0.5
            strike_ready = bool(
                local_ready.get("teammate_pass_window_ready", False)
                or teammate_merge_window
                or self._recent_step_active(self._seal_escape_until_step, current_step)
            )
            brake_ready = bool(
                teammate_same_lane_ahead
                or self._recent_step_active(self._seal_escape_until_step, current_step)
            )

        if brake_ready:
            return "brake_pulse"
        if strike_ready:
            return "strike"
        return "compress"

    def _apply_phase_progression(self, phase, step, ctx=None, local_ready=None):
        if phase == "disengage":
            return "disengage"
        active_phase = self.active_phase if self.active_phase in RUNTIME_PHASES else "compress"
        if active_phase == "disengage":
            return "disengage"
        event_phase = self._event_progress_phase(step, ctx=ctx, local_ready=local_ready)
        if self._runtime_phase_rank(event_phase) < self._runtime_phase_rank(active_phase):
            return active_phase
        return event_phase

    def _record_repairs(self, repaired_fields):
        if not repaired_fields:
            return
        self.normalization_repairs += len(repaired_fields)
        for field_name in repaired_fields:
            self.repaired_fields.append(str(field_name))
        self.repaired_fields = self.repaired_fields[-32:]

    def _publish_negotiated_runtime_decision(self, env, decision, trigger_eval):
        publish_message = copy.deepcopy((decision or {}).get("message") or {})
        publish_text = str(
            (decision or {}).get("message_text", "")
            or (decision or {}).get("action_sequence_text", "")
            or "{}:{}:{}".format(
                publish_message.get("phase", "compress"),
                publish_message.get("intent", ""),
                ((decision or {}).get("tactic", {}) or {}).get("mode", "hold_lane"),
            )
        ).strip()
        if not publish_text:
            publish_text = "{}:{}:{}".format(
                publish_message.get("phase", "compress"),
                publish_message.get("intent", ""),
                ((decision or {}).get("tactic", {}) or {}).get("mode", "hold_lane"),
            )

        self._publish_negotiated_tactic(env, decision)
        if hasattr(env, "message_pool") and hasattr(env.message_pool, "join"):
            env.message_pool.join(self.veh_id, publish_text)

        self.current_decision = copy.deepcopy(decision)
        self.current_message = publish_text
        self.active_phase = str(publish_message.get("phase", "compress") or "compress")
        self.active_control_mode = str(
            ((decision or {}).get("control", {}) or {}).get("mode", "hold_lane") or "hold_lane"
        )
        self.current_trigger_eval = copy.deepcopy(trigger_eval)
        if trigger_eval and "satisfied" in trigger_eval:
            self._last_published_trigger_satisfied = bool(trigger_eval.get("satisfied"))
        else:
            self._last_published_trigger_satisfied = None
        self._record_phase_trace({
            "step": publish_message.get("step", self._get_step(env)),
            "phase": publish_message.get("phase", "compress"),
            "intent": publish_message.get("intent", ""),
            "mode": ((decision or {}).get("tactic", {}) or {}).get("mode", "hold_lane"),
        })

    def _record_phase_trace(self, message):
        event = {
            "step": int(message.get("step", 0)),
            "phase": str(message.get("phase", "compress")),
            "intent": str(message.get("intent", "")),
            "mode": str(message.get("mode", "")),
        }
        if not self.phase_trace or self.phase_trace[-1] != event:
            self.phase_trace.append(event)
            if len(self.phase_trace) > 32:
                self.phase_trace = self.phase_trace[-32:]

    def _edge_progress_offset(self, env, edge):
        edge = str(edge or "")
        if not edge or edge.startswith(":"):
            return None
        route_index = None
        edge_length = None
        try:
            edge_length = float(env.k.network.edge_length(edge))
        except Exception:
            edge_length = None
        match = re.search(r"_(\d+)$", edge)
        if match is not None and edge_length is not None:
            route_index = int(match.group(1))
            return float(route_index) * float(edge_length)
        return None

    def _get_vehicle_longitudinal_progress(self, env, veh_id):
        try:
            if veh_id not in env.k.vehicle.get_ids():
                return None
        except Exception:
            return None

        edge = ""
        position = 0.0
        try:
            edge = str(env.k.vehicle.get_edge(veh_id) or "")
            position = float(env.k.vehicle.get_position(veh_id))
        except Exception:
            edge = ""
        route = None
        try:
            route = env.k.vehicle.get_route(veh_id)
        except Exception:
            route = None

        if edge and route:
            total = 0.0
            for route_edge in list(route):
                if route_edge == edge:
                    return total + position
                try:
                    total += float(env.k.network.edge_length(route_edge))
                except Exception:
                    total = None
                    break
            if total is not None:
                return total + position

        edge_offset = self._edge_progress_offset(env, edge)
        if edge_offset is not None:
            edge_length_fn = getattr(env.k.network, "edge_length", None)
            if edge_length_fn:
                try:
                    local_position = float(position)
                    edge_length_value = float(edge_length_fn(edge))
                    if local_position > edge_length_value + 1e-6:
                        local_position = float(np.mod(local_position, edge_length_value))
                    return edge_offset + local_position
                except Exception:
                    return edge_offset + position

        try:
            return float(env.k.vehicle.get_x_by_id(veh_id))
        except Exception:
            return None

    def _get_relative_context(self, env):
        ego_speed = 0.0
        ego_lane = 0
        ego_x = 0.0
        if self.attack_target in env.k.vehicle.get_ids():
            ego_speed = float(env.k.vehicle.get_speed(self.attack_target))
            ego_lane = int(env.k.vehicle.get_lane(self.attack_target))
            ego_progress = self._get_vehicle_longitudinal_progress(env, self.attack_target)
            ego_x = float(ego_progress if ego_progress is not None else env.k.vehicle.get_x_by_id(self.attack_target))

        self_speed = float(env.k.vehicle.get_speed(self.veh_id))
        self_lane = int(env.k.vehicle.get_lane(self.veh_id))
        self_progress = self._get_vehicle_longitudinal_progress(env, self.veh_id)
        self_x = float(self_progress if self_progress is not None else env.k.vehicle.get_x_by_id(self.veh_id))
        teammate_id = self._get_teammate_id()
        teammate_x = ego_x
        teammate_lane = ego_lane
        teammate_speed = ego_speed
        teammate_present = False
        if teammate_id in env.k.vehicle.get_ids():
            teammate_present = True
            teammate_progress = self._get_vehicle_longitudinal_progress(env, teammate_id)
            teammate_x = float(
                teammate_progress if teammate_progress is not None else env.k.vehicle.get_x_by_id(teammate_id)
            )
            teammate_lane = int(env.k.vehicle.get_lane(teammate_id))
            teammate_speed = float(env.k.vehicle.get_speed(teammate_id))

        self_rel_x = float(np.clip(self_x - ego_x, -200.0, 200.0))
        teammate_rel_x = float(np.clip(teammate_x - ego_x, -200.0, 200.0))
        self_ego_ttc = self._pairwise_longitudinal_ttc(self_rel_x, self_speed, ego_speed)
        teammate_ego_ttc = (
            self._pairwise_longitudinal_ttc(teammate_rel_x, teammate_speed, ego_speed)
            if teammate_present else float("inf")
        )
        self_ego_same_lane_ttc = self_ego_ttc if int(self_lane) == int(ego_lane) else float("inf")
        teammate_ego_same_lane_ttc = (
            teammate_ego_ttc if teammate_present and int(teammate_lane) == int(ego_lane) else float("inf")
        )
        teammate_role = str(self.role_map.get(teammate_id, "") if self.role_map else "")
        striker_ego_ttc = float("inf")
        striker_ego_same_lane_ttc = float("inf")
        blocker_ego_ttc = float("inf")
        blocker_ego_same_lane_ttc = float("inf")
        if self.attack_role == "Striker":
            striker_ego_ttc = self_ego_ttc
            striker_ego_same_lane_ttc = self_ego_same_lane_ttc
        elif self.attack_role == "Blocker":
            blocker_ego_ttc = self_ego_ttc
            blocker_ego_same_lane_ttc = self_ego_same_lane_ttc
        if teammate_role == "Striker":
            striker_ego_ttc = teammate_ego_ttc
            striker_ego_same_lane_ttc = teammate_ego_same_lane_ttc
        elif teammate_role == "Blocker":
            blocker_ego_ttc = teammate_ego_ttc
            blocker_ego_same_lane_ttc = teammate_ego_same_lane_ttc

        return {
            "ego_speed": ego_speed,
            "ego_lane": ego_lane,
            "ego_x": ego_x,
            "self_speed": self_speed,
            "self_lane": self_lane,
            "self_x": self_x,
            "self_rel_x": self_rel_x,
            "self_rel_lane": self_lane - ego_lane,
            "teammate_id": teammate_id,
            "teammate_speed": teammate_speed,
            "teammate_rel_x": teammate_rel_x,
            "teammate_rel_lane": teammate_lane - ego_lane,
            "self_ego_ttc": self_ego_ttc,
            "self_ego_same_lane_ttc": self_ego_same_lane_ttc,
            "teammate_ego_ttc": teammate_ego_ttc,
            "teammate_ego_same_lane_ttc": teammate_ego_same_lane_ttc,
            "striker_ego_ttc": striker_ego_ttc,
            "striker_ego_same_lane_ttc": striker_ego_same_lane_ttc,
            "blocker_ego_ttc": blocker_ego_ttc,
            "blocker_ego_same_lane_ttc": blocker_ego_same_lane_ttc,
        }

    def _pairwise_longitudinal_ttc(self, rel_x, other_speed, ego_speed):
        rel_x = float(rel_x)
        other_speed = float(other_speed)
        ego_speed = float(ego_speed)
        if abs(rel_x) <= 1e-3:
            return 0.0
        if rel_x > 0.0:
            closing_speed = ego_speed - other_speed
        else:
            closing_speed = other_speed - ego_speed
        if closing_speed <= 1e-3:
            return float("inf")
        return max(0.0, abs(rel_x) / closing_speed)

    def _format_ttc_for_prompt(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "inf"
        if not np.isfinite(value) or value < 0.0:
            return "inf"
        if value > 99.0:
            return "99+"
        return "{:.2f}".format(value)

    def _resolve_negotiated_lane_target(self, env, ctx, tactic):
        edge = env.k.vehicle.get_edge(self.veh_id)
        ego_lane = int(ctx["ego_lane"])
        current_lane = int(ctx["self_lane"])
        pass_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.pass_side_rel or -1)
        block_lane = self._clamp_adjacent_lane(env, edge, ego_lane, self.block_side_rel or 1)
        lane_policy = str((tactic or {}).get("lane_policy", "hold_current") or "hold_current")

        if lane_policy == "pass_side":
            desired_lane_abs = int(pass_lane)
        elif lane_policy == "ego_lane":
            desired_lane_abs = int(ego_lane)
        elif lane_policy == "block_side":
            desired_lane_abs = int(block_lane)
        else:
            desired_lane_abs = int(current_lane)
        return desired_lane_abs

    def _negotiated_gap_targets(self, phase, intent, tactic):
        phase = str(phase or "compress")
        intent = str(intent or "")
        gap_band = str((tactic or {}).get("gap_band", "medium") or "medium")

        if intent == "abort" or phase == "disengage":
            return -8.0, 6.0

        if self.attack_role == "Blocker":
            rel_x_map = {
                "hold_side_front": {"tight": 3.2, "medium": 4.4, "loose": 5.5},
                "seal_escape": {"tight": 2.8, "medium": 3.9, "loose": 5.0},
            }
            gap_map = {
                "hold_side_front": {"tight": 0.55, "medium": 0.75, "loose": 0.95},
                "seal_escape": {"tight": 0.4, "medium": 0.5, "loose": 0.7},
            }
            key = "seal_escape" if intent == "seal_escape" or phase in ("strike", "brake_pulse") else "hold_side_front"
            return rel_x_map[key][gap_band], gap_map[key][gap_band]

        if intent == "front_brake" or phase == "brake_pulse":
            return {"tight": 0.8, "medium": 1.3, "loose": 2.0}[gap_band], {
                "tight": 0.4,
                "medium": 0.55,
                "loose": 0.75,
            }[gap_band]
        if intent == "merge_commit" or phase == "strike":
            return {"tight": 1.1, "medium": 1.8, "loose": 2.6}[gap_band], {
                "tight": 0.45,
                "medium": 0.65,
                "loose": 0.85,
            }[gap_band]
        return {"tight": 0.5, "medium": -0.5, "loose": -2.0}[gap_band], {
            "tight": 0.9,
            "medium": 1.1,
            "loose": 1.4,
        }[gap_band]

    def _negotiated_speed_delta(self, tactic):
        speed_band = str((tactic or {}).get("speed_band", "match") or "match")
        if self.attack_role == "Blocker":
            return {
                "yield": -1.0,
                "match": 0.4,
                "press": 1.8,
                "surge": 2.8,
                "brake": -4.0,
            }.get(speed_band, 0.4)
        return {
            "yield": -1.2,
            "match": 0.8,
            "press": 2.6,
            "surge": 4.2,
            "brake": -7.0,
        }.get(speed_band, 0.8)

    def _set_negotiated_execution_targets(self, env, decision, trigger_eval):
        if self._terminal_plan_locked:
            decision = self._terminal_runtime_decision(env)
            trigger_eval = {
                "trigger_code": "terminal_latch",
                "satisfied": True,
                "details": {"reason": str(self._terminal_lock_reason or "terminal")},
            }
        message = copy.deepcopy((decision or {}).get("message") or {})
        intent_plan = copy.deepcopy((decision or {}).get("intent_plan") or {})
        tactic = copy.deepcopy((decision or {}).get("tactic") or {})
        ctx = self._get_relative_context(env)
        current_step = self._get_step(env)
        phase = str(intent_plan.get("phase", message.get("phase", "compress")) or "compress")
        intent = str(intent_plan.get("intent", message.get("intent", self.intent)) or self.intent)
        mode = str((tactic or {}).get("mode", (decision.get("control", {}) or {}).get("mode", "hold_lane")) or "hold_lane")
        tactic_hints, _ = self._normalize_tactic_hints(
            (tactic or {}).get("tactic_hints", tactic),
            phase,
            intent,
        )
        tactic_sequence = list(tactic_hints.get("sequence", []) or [])
        speed_delta_hint = float(tactic_hints.get("speed_delta_hint_mps", 0.0) or 0.0)
        lead_gap_hint = float(tactic_hints.get("lead_gap_hint_m", 0.0) or 0.0)
        hold_cycles = int(tactic_hints.get("hold_cycles", 1) or 1)
        ego_speed = float(ctx["ego_speed"])
        self_speed = float(ctx["self_speed"])
        rel_x = float(ctx["self_rel_x"])
        current_lane = int(ctx["self_lane"])
        ego_lane = int(ctx["ego_lane"])
        current_merge_wait_sticky = bool(
            self.attack_role == "Striker"
            and current_lane != ego_lane
            and abs(current_lane - ego_lane) == 1
            and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_CLEAN_MERGE_MAX_REL_X
            and not self._teammate_blocks_ego_lane(ctx)
            and not self._lane_change_stalled
            and (
                self.executor_state in ("merge_wait_gap", "cut_in_commit", "merge_commit")
                or self._cut_in_episode_active
                or self._recent_step_active(self._merge_commit_until_step, current_step)
            )
        )
        if current_merge_wait_sticky and intent != "front_brake":
            phase = "strike"
            intent = "merge_commit"
            if mode in ("hold_lane", "disengage"):
                mode = "track_pose"
            tactic_hints, _ = self._normalize_tactic_hints(
                (tactic or {}).get("tactic_hints", tactic),
                phase,
                intent,
            )
            tactic_sequence = list(tactic_hints.get("sequence", []) or [])
            speed_delta_hint = float(tactic_hints.get("speed_delta_hint_mps", 0.0) or 0.0)
            lead_gap_hint = float(tactic_hints.get("lead_gap_hint_m", 0.0) or 0.0)
            hold_cycles = int(tactic_hints.get("hold_cycles", 1) or 1)
        local_ready = self._build_negotiated_local_ready(env, ctx)
        try:
            edge = env.k.vehicle.get_edge(self.veh_id)
        except Exception:
            edge = ""
        pass_lane = self._clamp_adjacent_lane(env, edge, int(ctx["ego_lane"]), self.pass_side_rel or -1)
        block_lane = self._clamp_adjacent_lane(env, edge, int(ctx["ego_lane"]), self.block_side_rel or 1)
        desired_lane_abs = self._resolve_negotiated_lane_target(env, ctx, tactic)
        desired_rel_x, desired_rel_s = self._negotiated_gap_targets(phase, intent, tactic)
        front_brake_min_rel_x = 0.5
        front_brake_max_rel_x = 4.0
        if self.attack_role == "Striker" and lead_gap_hint > 0.0:
            if intent == "gain_lead":
                desired_rel_x = float(np.clip(lead_gap_hint, 0.5, 4.0))
            elif intent == "merge_commit" or phase == "strike":
                desired_rel_x = float(np.clip(lead_gap_hint, 0.8, 3.0))
            elif intent == "front_brake" or phase == "brake_pulse":
                desired_rel_x = float(np.clip(lead_gap_hint, 0.5, 4.0))
            brake_center = float(np.clip(lead_gap_hint, 0.5, 4.5))
            front_brake_min_rel_x = max(0.2, brake_center - 1.0)
            front_brake_max_rel_x = min(5.8, brake_center + 1.8)
        rel_x_error = float(desired_rel_x) - rel_x
        gain = 0.55 if self.attack_role == "Striker" else 0.42
        speed_delta = self._negotiated_speed_delta(tactic)
        if self.attack_role in ("Striker", "Blocker"):
            speed_delta = speed_delta_hint
        target_v = ego_speed + speed_delta + float(np.clip(rel_x_error * gain, -4.5, 4.5))
        target_s = float(desired_rel_s)
        if self.attack_role == "Striker" and lead_gap_hint > 0.0:
            target_s = float(np.clip(0.2 + 0.1 * lead_gap_hint, self.bounds["s_min"], 0.6))
        elif self.attack_role == "Blocker" and lead_gap_hint > 0.0:
            target_s = float(np.clip(0.25 + 0.1 * lead_gap_hint, self.bounds["s_min"], 0.8))

        lane_delta = int(desired_lane_abs) - current_lane
        if mode in ("hold_lane", "pulse_brake", "disengage"):
            lane_cmd = 0
        elif lane_delta > 0:
            lane_cmd = 1
        elif lane_delta < 0:
            lane_cmd = -1
        else:
            lane_cmd = 0

        self._allow_aggressive_cut_in = False
        self._last_aggressive_cut_in_ready = False
        striker_exec_state = None
        striker_recent_merge = False
        if self.attack_role == "Striker":
            approach_window_ready = bool(local_ready.get("approach_window_ready", False))
            merge_attack_requested = bool(
                intent in ("merge_commit", "front_brake")
                or (phase == "strike" and desired_lane_abs == int(ctx["ego_lane"]))
                or approach_window_ready
                or current_merge_wait_sticky
            )
            if approach_window_ready and intent != "front_brake":
                phase = "strike"
                intent = "merge_commit"
                if mode in ("hold_lane", "disengage"):
                    mode = "track_pose"
            merge_speed_floor = -1.0 if current_merge_wait_sticky else -0.2
            body_cut_in_command_ready = bool(
                self._body_cut_in_command_ready(env, ctx)
                and rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                and self_speed >= ego_speed - 0.2
                and not self._teammate_blocks_ego_lane(ctx)
            )
            body_safe_cut_in_ready = bool(
                self._body_safe_cut_in_ready(env, ctx)
                and rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                and self_speed >= ego_speed - 0.2
                and not self._teammate_blocks_ego_lane(ctx)
            )
            striker_cut_in_window = bool(
                current_lane != ego_lane
                and abs(current_lane - ego_lane) == 1
                and NEGOTIATED_MERGE_COMMIT_MIN_REL_X <= rel_x <= NEGOTIATED_SAFE_LATE_CUT_IN_MAX_REL_X
                and self_speed >= ego_speed + merge_speed_floor
                and not self._teammate_blocks_ego_lane(ctx)
            )
            missed_merge_window = bool(
                current_lane != ego_lane
                and abs(current_lane - ego_lane) == 1
                and rel_x > NEGOTIATED_CLEAN_MERGE_MAX_REL_X
                and not body_cut_in_command_ready
            )
            striker_recent_merge = bool(
                self._merged_into_ego_lane
                or self.striker_completed_cut_in
                or self._recent_merge_event_active(current_step)
                or self._recent_step_active(self._merge_commit_until_step, current_step)
            )
            needs_ego_lane = bool(
                current_lane == ego_lane
                or intent == "front_brake"
                or striker_cut_in_window
            )
            if needs_ego_lane:
                desired_lane_abs = ego_lane
                lane_delta = int(desired_lane_abs) - current_lane
                if mode in ("hold_lane", "disengage"):
                    lane_cmd = 0
                elif lane_delta > 0:
                    lane_cmd = 1
                elif lane_delta < 0:
                    lane_cmd = -1
                else:
                    lane_cmd = 0
            if merge_attack_requested:
                if missed_merge_window and rel_x > NEGOTIATED_MERGE_FAIL_REL_X:
                    self._mark_stale_merge_candidate(env)
                    self.current_decision = self._build_negotiated_runtime_decision(
                        env,
                        {
                            "phase": "compress",
                            "intent": "gain_lead",
                            "urgency": "mid",
                            "goal": "Recover after missing the clean merge window.",
                            "message": "Recover after stale merge candidate.",
                        },
                    )
                    self._apply_disengage_targets(env, ctx)
                    return
                if (
                        striker_cut_in_window
                        and (
                            rel_x < NEGOTIATED_FORCE_CUT_IN_MIN_REL_X
                            or not body_cut_in_command_ready
                            or self_speed < ego_speed - 0.2)):
                    preserved_wait_cycles = int(self._merge_wait_gap_cycles)
                    self._start_merge_commit(current_step)
                    self._merge_wait_gap_cycles = max(
                        int(self._merge_wait_gap_cycles),
                        preserved_wait_cycles,
                    )
                    desired_lane_abs = int(pass_lane)
                    lane_cmd = 0
                    striker_exec_state = "merge_wait_gap"
                    self._last_force_cut_in_blocked_reason = "too_close"
                    self._hold_current_lane(env, hold_steps=self.control_interval)
                    self._merge_wait_gap_cycles += 1
                    wait_gap_limit_steps = self._control_cycles_to_steps(NEGOTIATED_MERGE_WAIT_GAP_MAX_CYCLES)
                    if (
                            self._merge_wait_gap_cycles >= wait_gap_limit_steps
                            and (
                                rel_x >= NEGOTIATED_CLEAN_MERGE_MAX_REL_X - 0.25
                                or float(self._get_road_end_state(env).get("remaining", 1e9)) <= HIGHWAY_END_LC_DISABLE_M
                            )):
                        self._clear_cut_in_episode()
                        phase = "compress"
                        intent = "gain_lead"
                        desired_lane_abs = int(pass_lane)
                        lane_delta = int(desired_lane_abs) - current_lane
                        lane_cmd = 0 if current_lane == int(pass_lane) else (1 if lane_delta > 0 else (-1 if lane_delta < 0 else 0))
                        striker_exec_state = "merge_recover"
                        self._last_force_cut_in_blocked_reason = "wait_timeout"
                elif striker_cut_in_window and body_cut_in_command_ready:
                    self._merge_wait_gap_cycles = 0
                    self._start_merge_commit(current_step)
                    self._merge_window_hold_until_step = max(
                        int(self._merge_window_hold_until_step),
                        int(current_step) + self._control_cycles_to_steps(hold_cycles),
                    )
                    striker_exec_state = "cut_in_commit"
                    self._allow_aggressive_cut_in = True
                    self._last_force_cut_in_blocked_reason = ""
                    desired_lane_abs = ego_lane
                    lane_delta = int(desired_lane_abs) - current_lane
                    lane_cmd = 1 if lane_delta > 0 else (-1 if lane_delta < 0 else 0)
                    self.idm_a = max(float(self.idm_a), 2.2)
                    self.T = min(float(self.T), 0.18)
                    self.a = self.idm_a
                    if edge and edge[0] != ":":
                        self._last_aggressive_cut_in_ready = bool(self._aggressive_cut_in_ready(env, edge, ctx))
                elif missed_merge_window:
                    self._merge_wait_gap_cycles = 0
                    self._clear_cut_in_episode()
                    phase = "compress"
                    intent = "gain_lead"
                    desired_lane_abs = int(pass_lane)
                    lane_delta = int(desired_lane_abs) - current_lane
                    lane_cmd = 0 if current_lane == int(pass_lane) else (1 if lane_delta > 0 else (-1 if lane_delta < 0 else 0))
                    striker_exec_state = "merge_recover"
                elif current_lane != ego_lane:
                    self._merge_wait_gap_cycles = 0
                    desired_lane_abs = int(pass_lane)
                    lane_delta = int(desired_lane_abs) - current_lane
                    lane_cmd = 1 if lane_delta > 0 else (-1 if lane_delta < 0 else 0)
                    if current_lane == int(pass_lane):
                        lane_cmd = 0
                    striker_exec_state = "lead_gain"

            deterministic_front_brake_ready = bool(
                current_lane == ego_lane
                and front_brake_min_rel_x <= rel_x <= front_brake_max_rel_x
                and striker_recent_merge
                and current_step >= int(self._post_merge_stabilize_until_step)
                and ("front_brake" in tactic_sequence or intent == "front_brake")
            )
            if deterministic_front_brake_ready:
                self._apply_front_brake_trap(env, ctx, current_step)
                self.intent = "front_brake"
                self.intent_urgency = str(intent_plan.get("urgency", self.intent_urgency or "high") or "high")
                self.executor_state = "front_brake"
                self.target_abs_lane = ego_lane
                self.target_lc = 0
                self.pending_lane_change = 0
                self.v0 = self.target_v
                self.s0 = self.target_s
                return
            if striker_recent_merge and current_lane == ego_lane:
                desired_lane_abs = ego_lane
                lane_cmd = 0
                if rel_x < front_brake_min_rel_x or rel_x > front_brake_max_rel_x:
                    striker_exec_state = "front_brake_setup"

        if mode == "hold_lane":
            target_v = ego_speed + self._negotiated_speed_delta(tactic) + float(np.clip(rel_x_error * 0.25, -2.5, 2.5))
        elif mode == "pulse_brake":
            if self._pulse_end_step < self._get_step(env):
                horizon_steps = int((decision.get("control", {}) or {}).get("horizon_steps", self.control_interval) or self.control_interval)
                self._pulse_end_step = self._get_step(env) + min(4, max(1, horizon_steps))
            rel_for_brake = float(ctx["self_rel_x"])
            brake_drop = float(np.clip(1.6 + 0.28 * max(0.0, rel_for_brake), 2.0, 4.2))
            brake_cap = ego_speed - max(1.2, brake_drop - 0.8)
            target_v = min(float(self_speed) - brake_drop, float(brake_cap))
            target_s = max(self.bounds["s_min"], min(float(target_s), 0.65))
        elif mode == "disengage":
            desired_lane_abs = current_lane
            lane_cmd = 0
            target_v = ego_speed - 2.0
            target_s = max(5.5, float(target_s))
            if self.attack_role == "Striker":
                self._clear_cut_in_episode()

        teammate_striking = bool(
            self._last_teammate_intent in ("merge_commit", "front_brake")
            or self._last_teammate_phase in ("strike", "brake_pulse")
        )
        if self.attack_role == "Blocker" and intent in ("hold_side_front", "seal_escape"):
            if teammate_striking:
                intent = "seal_escape"
                phase = "strike" if phase == "compress" else phase
                desired_lane_abs = int(block_lane)
                lane_delta = int(desired_lane_abs) - current_lane
                if mode == "disengage":
                    lane_cmd = 0
                elif lane_delta > 0:
                    lane_cmd = 1
                elif lane_delta < 0:
                    lane_cmd = -1
                else:
                    lane_cmd = 0
                target_s = max(self.bounds["s_min"], min(float(target_s), 0.5 if phase in ("strike", "brake_pulse") else 0.6))
                if rel_x > 6.0:
                    bleed = min(4.0, 1.0 + 0.8 * (rel_x - 6.0))
                    target_v = max(self.bounds["v_min"], ego_speed - bleed)
                elif rel_x < 2.0:
                    press = min(4.0, 1.8 + 1.0 * (2.0 - rel_x))
                    target_v = ego_speed + press
                else:
                    if rel_x > 5.5:
                        target_v = max(self.bounds["v_min"], ego_speed - min(1.4, 0.3 + 1.2 * (rel_x - 5.5)))
                    elif rel_x < 2.5:
                        target_v = ego_speed + min(1.8, 0.8 + 1.4 * (2.5 - rel_x))
                    else:
                        target_v = ego_speed + min(0.6, 0.15 + 0.15 * (rel_x - 2.5))
            else:
                base_offset = 0.9 if intent == "hold_side_front" else 1.4
                speed_bias = float(np.clip(float(speed_delta) * 0.45, -0.2, 1.2))
                correction = float(np.clip(rel_x_error * 0.45, -1.0, 3.8))
                target_v = ego_speed + base_offset + speed_bias + correction
                if current_lane == int(desired_lane_abs):
                    if rel_x > float(desired_rel_x) + 4.0:
                        target_v = max(target_v, ego_speed + 0.2)
                    elif rel_x > float(desired_rel_x) + 1.0:
                        target_v = max(target_v, ego_speed + 0.7)
                    else:
                        target_v = max(target_v, ego_speed + base_offset + 0.3)
                if rel_x < float(desired_rel_x) - 0.5:
                    target_v = max(target_v, ego_speed + 2.0)
            if intent == "seal_escape" or phase in ("strike", "brake_pulse"):
                target_s = max(self.bounds["s_min"], min(float(target_s), 0.6))

        if self.attack_role == "Striker" and striker_exec_state == "merge_recover":
            bleed = min(3.0, 0.7 + 0.55 * max(0.0, rel_x - 4.5))
            target_v = max(self.bounds["v_min"], min(float(target_v), ego_speed - bleed))
            target_s = max(self.bounds["s_min"], min(float(target_s), 0.5))

        if self.attack_role == "Striker" and phase in ("strike", "brake_pulse"):
            if rel_x < 0.0:
                self._striker_behind_steps += 1
            else:
                self._striker_behind_steps = 0
            if intent == "merge_commit":
                if striker_exec_state == "lead_gain":
                    if rel_x < -4.0:
                        base_gain = 7.0
                    elif rel_x < -1.5:
                        base_gain = 6.0
                    else:
                        base_gain = 4.5
                    if rel_x >= 0.0:
                        base_gain = max(base_gain, 4.5)
                    target_v = max(
                        target_v,
                        min(self.bounds["v_max"], ego_speed + base_gain),
                        min(self.bounds["v_max"], self_speed + 1.2),
                    )
                    target_s = float(np.clip(
                        0.2 + 0.1 * lead_gap_hint,
                        self.bounds["s_min"],
                        0.65,
                    ))
                elif striker_exec_state == "cut_in_commit":
                    speed_adv_target = float(np.clip(speed_delta_hint, 0.0, 0.6))
                    target_v = min(
                        self.bounds["v_max"],
                        ego_speed + speed_adv_target,
                    )
                    target_s = float(np.clip(
                        0.2 + 0.1 * lead_gap_hint,
                        self.bounds["s_min"],
                        0.6,
                    ))
                elif striker_exec_state == "merge_recover":
                    bleed = min(3.0, 0.7 + 0.55 * max(0.0, rel_x - 4.5))
                    target_v = max(self.bounds["v_min"], min(float(target_v), ego_speed - bleed))
                    target_s = max(self.bounds["s_min"], min(float(target_s), 0.5))
                elif striker_exec_state == "merge_wait_gap":
                    required_gap = self._dynamic_body_cut_in_gap(env, ctx)
                    projected_gap = self._projected_ego_lead_gap_after_merge(env, ctx)
                    gap_deficit = float(required_gap - projected_gap)
                    if rel_x < NEGOTIATED_FORCE_CUT_IN_MIN_REL_X:
                        wait_speed_adv = float(np.clip(
                            speed_delta_hint,
                            NEGOTIATED_TOO_CLOSE_CUT_IN_MIN_SPEED_ADV,
                            NEGOTIATED_TOO_CLOSE_CUT_IN_MAX_SPEED_ADV,
                        ))
                    elif gap_deficit > 2.0:
                        wait_speed_adv = float(np.clip(max(speed_delta_hint, 3.0), 2.4, 3.6))
                    elif gap_deficit > 0.5:
                        wait_speed_adv = float(np.clip(max(speed_delta_hint, 2.4), 1.8, 2.8))
                    else:
                        wait_speed_adv = float(np.clip(max(speed_delta_hint, 1.6), 1.0, 2.0))
                    target_v = min(
                        self.bounds["v_max"],
                        ego_speed + wait_speed_adv,
                    )
                    target_s = max(self.bounds["s_min"], min(float(target_s), 0.55))
                elif striker_exec_state == "front_brake_setup":
                    if rel_x < 0.5:
                        close_gain = float(np.clip(1.8 - rel_x, 1.2, 2.6))
                        target_v = max(
                            target_v,
                            min(self.bounds["v_max"], ego_speed + close_gain),
                            min(self.bounds["v_max"], self_speed + 0.2),
                        )
                    else:
                        bleed = min(2.2, 0.5 + 0.25 * max(0.0, rel_x - front_brake_max_rel_x))
                        target_v = max(self.bounds["v_min"], min(float(target_v), ego_speed - bleed))
                    target_s = max(self.bounds["s_min"], min(float(target_s), 0.45))
                else:
                    closing_bonus = float(np.clip(0.55 * max(0.0, 1.8 - rel_x), 0.0, 3.0))
                    target_v = max(
                        target_v,
                        min(self.bounds["v_max"], ego_speed + 4.8 + closing_bonus),
                        min(self.bounds["v_max"], self_speed + 1.4 + 0.5 * closing_bonus),
                    )
                    target_s = max(self.bounds["s_min"], min(float(target_s), 0.55))
        else:
            self._striker_behind_steps = 0

        if not trigger_eval.get("satisfied", True) and mode in ("track_pose", "pulse_brake"):
            if not (self.attack_role == "Striker" and phase in ("strike", "brake_pulse")):
                lane_cmd = 0
                target_v = max(self.bounds["v_min"], min(self.bounds["v_max"], ego_speed - 1.0))

        executor_state = intent
        if self.attack_role == "Blocker":
            executor_state = {
                "seal_escape": "seal",
                "hold_side_front": "hold",
                "abort": "disengage",
            }.get(intent, mode)
        else:
            executor_state = {
                "front_brake": "front_brake",
                "merge_commit": "merge_commit",
                "abort": "disengage",
            }.get(intent, mode)
            if striker_exec_state:
                executor_state = striker_exec_state

        self.intent = intent
        self.intent_urgency = str(intent_plan.get("urgency", self.intent_urgency) or self.intent_urgency)
        self.executor_state = executor_state
        self.brake_armed = bool(mode == "pulse_brake" or intent in ("front_brake", "brake_pulse"))
        self.target_abs_lane = int(desired_lane_abs)
        self.target_v = float(np.clip(target_v, 0.1 if self._allow_low_speed_target else self.bounds["v_min"], self.bounds["v_max"]))
        self.target_s = float(np.clip(target_s, self.bounds["s_min"], self.bounds["s_max"]))
        self.target_lc = int(np.clip(lane_cmd, -1, 1))
        self.pending_lane_change = self.target_lc
        self.v0 = self.target_v
        self.s0 = self.target_s

    def _set_execution_targets(self, env, decision, trigger_eval):
        runtime_decision = copy.deepcopy(decision or {})
        if not runtime_decision.get("tactic"):
            intent_plan = self._decision_intent_plan(runtime_decision)
            runtime_decision["intent_plan"] = copy.deepcopy(intent_plan)
            runtime_decision["tactic"] = self._default_negotiated_tactic_profile(intent_plan)
        return self._set_negotiated_execution_targets(env, runtime_decision, trigger_eval)

    def llm_collaborate(self, env):
        scenario_description = self.get_perception(env)
        if self._use_structured_protocol():
            step = self._get_step(env)
            env.message_pool.begin_control_cycle(step)
            snapshot = self._get_negotiated_snapshot(env)
            return self._structured_collaborate(env, snapshot)

        shared_message = env.message_pool.get_all_msg()
        parse_attempt = 0
        protocol = "simple_attack" if self.map_name == "highway" else "legacy"
        feedback_prompt = self._build_feedback_prompt() if self.map_name == "highway" else self.previous_feedback
        phase_instruction = self._build_attack_instruction(env) if self.map_name == "highway" else ""

        while True:
            response = ""
            try:
                response = self.DA.collaborate(
                    self.map_name,
                    scenario_description,
                    shared_message,
                    self.attack_role,
                    self.attack_target,
                    feedback_prompt,
                    phase_instruction=phase_instruction,
                )
                parsed = self._extract_decision_dict(response)
                if self.map_name == "highway":
                    params = self._normalize_highway_intent_fields(parsed)
                    pool_message = "[intent={} urgency={}] {}".format(
                        params["intent"],
                        params["urgency"],
                        params["message"],
                    )
                else:
                    params = self._normalize_decision_fields(parsed)
                    params = self._hard_clip_decision(params)
                    pool_message = params["message"]
                self._record_llm_trace(
                    env,
                    protocol=protocol,
                    attempt=parse_attempt + 1,
                    response=response,
                    parsed=parsed,
                    error="",
                )
                self.last_parse_error = ""
                self.active_control_mode = "simple_attack"
                env.message_pool.join(self.veh_id, pool_message)
                return params
            except Exception as e:
                self._record_llm_trace(
                    env,
                    protocol=protocol,
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
                    self.rollout_parse_fallback_used += 1
                    if self.map_name == "highway":
                        return {
                            "intent": self._default_highway_intent(),
                            "urgency": "mid",
                            "message": "intent_parse_fallback",
                        }
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
        seen_candidates = set()
        valid_dicts = []

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
            if cand in seen_candidates:
                continue
            seen_candidates.add(cand)
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    valid_dicts.append((cand, parsed))
                    continue
            except Exception as e_json:
                last_err = e_json
                try:
                    parsed = ast.literal_eval(cand)
                    if isinstance(parsed, dict):
                        valid_dicts.append((cand, parsed))
                        continue
                except Exception as e_ast:
                    last_err = e_ast
                    continue

        if valid_dicts:
            valid_dicts.sort(
                key=lambda item: (
                    len(item[0]),
                    len(item[1]),
                    int("message" in item[1]) + int("control" in item[1]),
                ),
                reverse=True,
            )
            return valid_dicts[0][1]
        raise ValueError("Failed to parse decision dictionary: {}".format(last_err))

    def _sanitize_llm_text(self, text):
        text = str(text).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"thinking\.\.\..*?\.\.\.done thinking\.",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"^\s*thoughts?:.*?(?=\{)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
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
        self._append_trace_entry(entry)

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

    def _normalize_highway_confirmation(self, parsed):
        required_keys = {"decision", "message"}
        if not required_keys.issubset(set(parsed.keys())):
            raise KeyError("JSON must include keys: decision, message.")

        repaired_fields = []
        decision = str(parsed.get("decision", "confirm") or "confirm").strip().lower()
        if decision not in ("confirm", "swap"):
            decision = "confirm"
            repaired_fields.append("decision")
        message = str(parsed.get("message", "") or "").strip()
        if not message:
            message = "Keeping geometry-locked role."
            repaired_fields.append("message")
        self._record_repairs(repaired_fields)
        return {
            "decision": decision,
            "message": message[:160],
        }

    def _normalize_highway_contract_proposal(self, parsed):
        if "proposed_role" not in parsed:
            raise KeyError("JSON must include key: proposed_role.")
        repaired_fields = []
        proposed_role = self._normalize_role_name(parsed.get("proposed_role", "Undecided"))
        pass_side = str(parsed.get("pass_side", "none") or "none").strip().lower()
        if pass_side not in ("left", "right", "none"):
            pass_side = "none"
            repaired_fields.append("pass_side")
        message = str(parsed.get("message", "") or "").strip()
        self._record_repairs(repaired_fields)
        return {
            "proposed_role": proposed_role,
            "pass_side": pass_side,
            "message": message[:120],
        }

    def _normalize_highway_intent_fields(self, parsed):
        required_keys = {"intent", "urgency", "message"}
        if not required_keys.issubset(set(parsed.keys())):
            raise KeyError("JSON must include keys: intent, urgency, message.")

        repaired_fields = []
        intent = str(
            parsed.get("intent", self._default_highway_intent()) or self._default_highway_intent()
        ).strip().lower()
        allowed_intents = {
            "Striker": {"gain_lead", "cut_in", "brake_pulse", "abort"},
            "Blocker": {"claim_side", "hold_side_front", "seal_escape", "abort"},
        }.get(self.attack_role, {"abort"})
        if intent not in allowed_intents:
            intent = self._default_highway_intent()
            repaired_fields.append("intent")

        urgency = str(parsed.get("urgency", "mid") or "mid").strip().lower()
        if urgency not in ("low", "mid", "high"):
            urgency = "mid"
            repaired_fields.append("urgency")

        message = str(parsed.get("message", "") or "").strip()
        if not message:
            message = intent
            repaired_fields.append("message")
        self._record_repairs(repaired_fields)
        return {
            "intent": intent,
            "urgency": urgency,
            "message": message[:120],
        }

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

    def _normalize_role_name(self, value):
        text = str(value or "").strip().lower()
        if text == "blocker":
            return "Blocker"
        if text == "striker":
            return "Striker"
        return "Undecided"

    def _normalize_role_proposal(self, parsed):
        message = str(parsed.get("message", "") or "").strip()
        intent = str(parsed.get("intent", "") or "").strip()
        if not message:
            message = "Holding role decision and waiting for teammate."
        if not intent:
            intent = "wait"
        return {
            "message": message[:160],
            "role": self._normalize_role_name(parsed.get("role", "Undecided")),
            "intent": intent[:80],
        }

    def _build_feedback_prompt(self):
        feedback = self.previous_feedback
        if not feedback:
            return "none"
        if not isinstance(feedback, dict):
            text = str(feedback).strip()
            return text if text else "none"
        parts = []
        for key in (
                "result",
                "feedback_summary",
                "ego_max_decel",
                "striker_lane_change_time",
                "striker_rel_x_at_lane_change",
                "front_brake_triggered",
                "ego_escape_lane",
                "blocker_lane_at_escape"):
            value = feedback.get(key)
            if value in (None, ""):
                continue
            if key == "ego_max_decel":
                parts.append("ego_max_decel={:.2f}".format(float(value)))
            else:
                parts.append("{}={}".format(key, value))
        if not parts:
            return "none"
        return "Last iteration: {}.".format("; ".join(parts))

    def _build_attack_instruction(self, env):
        ctx = self._get_relative_context(env)
        rel_x = float(ctx["self_rel_x"])
        rel_lane = int(ctx["self_rel_lane"])
        if self.attack_role == "Striker":
            if self._is_three_car_scene():
                if rel_lane == 0 and 1.0 <= rel_x <= 8.0:
                    return "If already ahead in ego_0's lane and ego_0 is closing, brake_pulse is allowed."
                if abs(rel_lane) == 1 and -3.0 <= rel_x <= 1.5:
                    return "In three-car mode, if slightly behind but faster from the adjacent lane, choose cut_in instead of waiting for a full lead."
                if rel_x < -3.0:
                    return "In three-car mode, close quickly from the open adjacent lane until you reach the forced cut-in window."
                if abs(rel_lane) == 1 and rel_x > 1.5:
                    return "Stay beside ego_0 from the open side and cut_in before ego_0 escapes."
                return "Use the adjacent lane to create a cut-in window, then brake only after you become ego_0's lane leader."
            if rel_x < 0.0:
                return "If behind ego_0, choose gain_lead until you overtake or gain a side-front window."
            if rel_lane == 0 and 1.0 <= rel_x <= 8.0:
                return "If already ahead in ego_0's lane and ego_0 is closing, brake_pulse is allowed."
            if abs(rel_lane) == 1 and rel_x >= 1.0:
                return "If slightly ahead from the adjacent lane, choose cut_in."
            return "Prefer gain_lead before any cut_in or brake_pulse."
        if self.attack_role == "Blocker":
            if self._is_three_car_scene():
                if abs(rel_lane) > 1:
                    return "Move back to ego_0's escape-side adjacent lane and reclaim that side-front slot."
                if rel_x < 6.0:
                    return "In three-car mode, accelerate to retake side-front and keep the escape lane sealed for striker."
                return "Hold the side-front window, seal ego_0's escape lane, and preserve striker's cut-in corridor."
            if abs(rel_lane) > 1:
                return "Move to the reserved side adjacent to ego_0 and claim that side."
            if rel_x < 4.0:
                return "Accelerate to reach ego_0's side-front and claim the escape side."
            return "Stay on the reserved side lane and hold or seal escape space."
        return "Coordinate with the teammate before committing to a maneuver."

    def negotiate_role(self, env):
        self._begin_rollout_if_needed(env)
        self.active_phase = "negotiation"
        if self.map_name == "highway" and self._is_negotiated_highway_scene():
            parse_attempt = 0
            while parse_attempt < 3:
                response = ""
                try:
                    snapshot = self._get_negotiated_snapshot(env)
                    history_context = self._build_compact_history_context(env, phase_override="compress")
                    response = self.DA.negotiate_highway_contract(
                        self.get_perception(env),
                        snapshot,
                        self.attack_target,
                        history_context=history_context,
                    )
                    parsed = self._extract_decision_dict(response)
                    proposal = self._normalize_highway_contract_proposal(parsed)
                    self._record_llm_trace(
                        env,
                        protocol="negotiated_role",
                        attempt=parse_attempt + 1,
                        response=response,
                        parsed=proposal,
                        error="",
                    )
                    self.last_parse_error = ""
                    return proposal
                except Exception as exc:
                    self._record_llm_trace(
                        env,
                        protocol="negotiated_role",
                        attempt=parse_attempt + 1,
                        response=response,
                        parsed=None,
                        error=str(exc),
                    )
                    parse_attempt += 1
                    self.parse_failures += 1
                    self.last_parse_error = str(exc)

            self.fallback_activations += 1
            return {
                "proposed_role": "Undecided",
                "pass_side": "none",
                "message": "negotiation_parse_fallback",
            }

        parse_attempt = 0
        while parse_attempt < 3:
            response = ""
            try:
                response = self.DA.negotiate_role(
                    self.map_name,
                    self.get_perception(env),
                    env.message_pool.get_all_msg(),
                    self.attack_target,
                    locked_role=self.attack_role,
                    teammate_role=self.role_map.get(self._get_teammate_id(), "Undecided"),
                )
                parsed = self._extract_decision_dict(response)
                if self.map_name == "highway":
                    proposal = self._normalize_highway_confirmation(parsed)
                else:
                    proposal = self._normalize_role_proposal(parsed)
                self._record_llm_trace(
                    env,
                    protocol="role_confirmation" if self.map_name == "highway" else "role_negotiation",
                    attempt=parse_attempt + 1,
                    response=response,
                    parsed=proposal,
                    error="",
                )
                self.last_parse_error = ""
                return proposal
            except Exception as exc:
                self._record_llm_trace(
                    env,
                    protocol="role_negotiation",
                    attempt=parse_attempt + 1,
                    response=response,
                    parsed=None,
                    error=str(exc),
                )
                parse_attempt += 1
                self.parse_failures += 1
                self.last_parse_error = str(exc)

        self.fallback_activations += 1
        if self.map_name == "highway":
            return {
                "decision": "confirm",
                "message": "Keeping geometry-locked role.",
            }
        return {
            "message": "Holding role decision and waiting for teammate.",
            "role": "Undecided",
            "intent": "wait",
        }

    def llm_reason(self, env):
        if self._use_structured_protocol():
            if self.current_decision is None:
                return {
                    "message": copy.deepcopy(self._default_message()),
                    "control": copy.deepcopy(self._default_control()),
                }
            return copy.deepcopy(self.current_decision)
        if self.map_name == "highway":
            if self.current_decision is None:
                if self._is_negotiated_highway_scene():
                    return {
                        "primitive": self._default_highway_intent(),
                        "eta": None,
                        "target_v": None,
                        "target_s": None,
                        "message": "",
                    }
                return {
                    "intent": self._default_highway_intent(),
                    "urgency": "mid",
                    "message": "",
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
            "role_source": self.role_source,
            "contract_source": self.contract_source,
            "pass_side": self.pass_side,
            "block_side": self.block_side,
            "model": self.DA.llm_model,
            "last_parse_error": str(self.last_parse_error),
            "parse_failures": int(self.parse_failures),
            "fallback_activations": int(self.fallback_activations),
            "normalization_repairs": int(self.normalization_repairs),
            "repair_fallback_used": int(self.repair_fallback_used),
            "last_phase": str(self.active_phase),
            "last_raw_response": self._trim_trace_text(self.last_raw_response),
            "terminal_plan_locked": bool(self._terminal_plan_locked),
            "terminal_lock_reason": str(self._terminal_lock_reason or ""),
        }

    def get_rollout_diagnostics(self):
        return {
            "veh_id": self.veh_id,
            "role": self.attack_role,
            "contract_source": self.contract_source,
            "pass_side": self.pass_side,
            "block_side": self.block_side,
            "intent": self.intent,
            "urgency": self.intent_urgency,
            "executor_state": self.executor_state,
            "target_abs_lane": self.target_abs_lane,
            "target_v": float(self.target_v),
            "target_s": float(self.target_s),
            "lead_acquired": bool(self.lead_acquired),
            "brake_armed": bool(self.brake_armed),
            "striker_completed_cut_in": bool(self.striker_completed_cut_in),
            "striker_became_ego_leader": bool(self.striker_became_ego_leader),
            "last_lane_change_attempted": bool(self._last_lane_change_attempted),
            "merge_attempt_steps": int(self._merge_attempt_steps),
            "prev_self_lane": self._prev_self_lane,
            "merged_into_ego_lane": bool(self._merged_into_ego_lane),
            "merge_event_this_step": bool(self._merge_event_this_step),
            "valid_cut_in_merge": bool(self.striker_completed_cut_in),
            "last_merge_event_step": int(self._last_merge_event_step),
            "last_valid_merge_event_step": int(self._last_valid_merge_event_step),
            "last_bad_merge_event_step": int(self._last_bad_merge_event_step),
            "last_merge_rel_x": self._last_merge_rel_x,
            "bad_merge_event": bool(self._bad_merge_event),
            "bad_merge_reason": str(self._bad_merge_reason or ""),
            "clean_merge_failed": bool(self._clean_merge_failed),
            "stale_merge_candidate": bool(self._stale_merge_candidate),
            "stale_merge_candidate_step": int(self._stale_merge_candidate_step),
            "lane_change_stalled": bool(self._lane_change_stalled),
            "merge_stall_cycles": int(self._merge_stall_cycles),
            "merge_wait_gap_cycles": int(self._merge_wait_gap_cycles),
            "force_cut_in_blocked_reason": str(self._last_force_cut_in_blocked_reason or ""),
            "overshoot": bool(self._overshoot),
            "front_brake_triggered": bool(self._front_brake_triggered),
            "striker_lane_change_time": self._striker_lane_change_time,
            "striker_rel_x_at_lane_change": self._striker_rel_x_at_lane_change,
            "seal_escape_until_step": int(self._seal_escape_until_step),
            "last_local_ready": copy.deepcopy(self._last_local_ready),
            "last_teammate_phase": str(self._last_teammate_phase or ""),
            "last_teammate_intent": str(self._last_teammate_intent or ""),
            "last_teammate_tactic": copy.deepcopy(self._last_teammate_tactic or {}),
            "merge_commit_until_step": int(self._merge_commit_until_step),
            "active_phase": self.active_phase,
            "active_control_mode": self.active_control_mode,
            "phase_trace": copy.deepcopy(self.phase_trace),
            "latest_intent_plan": copy.deepcopy(self.latest_intent_plan),
            "latest_action_sequence_text": str(self.latest_action_sequence_text or ""),
            "latest_tactic_profile": copy.deepcopy(self.latest_tactic_profile),
            "action_reuse_count": int(self._action_reuse_count),
            "last_action_refresh_reason": str(self._last_action_refresh_reason or ""),
            "last_action_reused": bool(self._last_action_reused),
            "last_action_rel_x_bucket": str(self._last_action_rel_x_bucket or ""),
            "last_history_injection_phase": str(self._last_history_injection_phase or ""),
            "terminal_plan_locked": bool(self._terminal_plan_locked),
            "terminal_lock_reason": str(self._terminal_lock_reason or ""),
            "terminal_lock_step": int(self._terminal_lock_step),
            "last_published_trigger_satisfied": self._last_published_trigger_satisfied,
            "trigger_not_met_events": int(self.trigger_not_met_events),
            "sync_error_events": int(self.sync_error_events),
            "rollout_parse_fallback_used": int(self.rollout_parse_fallback_used),
            "role_resolution_fallback_used": int(self.role_resolution_fallback_used),
            "normalization_repairs": int(self.normalization_repairs),
            "repaired_fields": copy.deepcopy(self.repaired_fields),
            "repair_fallback_used": int(self.repair_fallback_used),
            "role_map": copy.deepcopy(self.role_map),
            "role_source": self.role_source,
            "geometry_role_hint": copy.deepcopy(self.geometry_role_hint),
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

    def retrieve_case_memory(self, env, phase_override=None):
        phase = str(phase_override or self.active_phase or "compress")
        current_signature = self.get_state_signature(env, phase=phase)
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
        clean_success_case = None
        fallback_success_case = None
        legacy_success_case = None
        failure_case = None
        phase_case = None
        for case in cases:
            result = str(case.get("result", ""))
            if clean_success_case is None and result == "clean_success":
                clean_success_case = case
            if fallback_success_case is None and result == "fallback_success":
                fallback_success_case = case
            if legacy_success_case is None and result == "success":
                legacy_success_case = case
            if failure_case is None and result not in ("clean_success", "fallback_success", "success"):
                failure_case = case
            if phase_case is None:
                phase_case = case
            if (clean_success_case or fallback_success_case or legacy_success_case) and failure_case and phase_case:
                break

        success_case = clean_success_case or fallback_success_case or legacy_success_case

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
        neighbor_override = str(os.getenv("FLOW_LLM_NEIGHBOR_K", "")).strip()
        if neighbor_override:
            max_neighbors = max(1, int(neighbor_override))
        else:
            default_neighbors = 2 if self._is_negotiated_highway_scene() else 4
            max_neighbors = max(1, int(default_neighbors))
        ctx = self._get_relative_context(env)
        self_headway = float(env.k.vehicle.get_headway(self.veh_id))
        teammate_id = ctx["teammate_id"]
        lead_gap_if_same_lane = "n/a"
        if int(ctx["self_lane"]) == int(ctx["ego_lane"]):
            lead_gap_if_same_lane = "{:.2f}".format(float(ctx["self_rel_x"]))

        lines = [
            "step={}".format(self._get_step(env)),
            "self={} role={} speed={:.2f} speed_adv={:.2f} rel_x={:.2f} rel_lane={} headway={:.2f}".format(
                self.veh_id,
                self.attack_role,
                float(ctx["self_speed"]),
                float(ctx["self_speed"]) - float(ctx["ego_speed"]),
                float(ctx["self_rel_x"]),
                int(ctx["self_rel_lane"]),
                self_headway,
            ),
            "ego={} speed={:.2f} lane={}".format(
                self.attack_target,
                float(ctx["ego_speed"]),
                int(ctx["ego_lane"]),
            ),
            "teammate={} speed={:.2f} rel_x={:.2f} rel_lane={}".format(
                teammate_id,
                float(ctx["teammate_speed"]),
                float(ctx["teammate_rel_x"]),
                int(ctx["teammate_rel_lane"]),
            ),
        ]
        if self.map_name == "highway":
            lines.append(
                "geometry ego_lane={} self_lane={} delta_to_ego_lane={} self_rel_x={:.2f} "
                "is_adjacent_to_ego_lane={} is_ahead_of_ego={} lead_gap_if_same_lane={}".format(
                    int(ctx["ego_lane"]),
                    int(ctx["self_lane"]),
                    int(ctx["self_lane"]) - int(ctx["ego_lane"]),
                    float(ctx["self_rel_x"]),
                    "yes" if abs(int(ctx["self_lane"]) - int(ctx["ego_lane"])) == 1 else "no",
                    "yes" if float(ctx["self_rel_x"]) > 0.0 else "no",
                    lead_gap_if_same_lane,
                )
            )
            if self._is_negotiated_highway_scene():
                lines.append(
                    "contract pass_side={} block_side={} contract_source={}".format(
                        self.pass_side,
                        self.block_side,
                        self.contract_source or "none",
                    )
                )
                lines.append(
                    "role_ttc_sec striker_ego_ttc_projection={} striker_ego_ttc_same_lane={} blocker_ego_ttc_projection={} blocker_ego_ttc_same_lane={}".format(
                        self._format_ttc_for_prompt(ctx["striker_ego_ttc"]),
                        self._format_ttc_for_prompt(ctx["striker_ego_same_lane_ttc"]),
                        self._format_ttc_for_prompt(ctx["blocker_ego_ttc"]),
                        self._format_ttc_for_prompt(ctx["blocker_ego_same_lane_ttc"]),
                    )
                )

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
