"""
Shared communication substrate for LLM agents.
"""

from copy import deepcopy


class message_pool:
    def __init__(self):
        self.state = {"merge": 0, "merge_pos": 0, "merge_t": 0}
        self.scenario_id = ""
        self.rollout_id = 0
        self.scenario_context = {}
        self.reset_rollout("", 0)

    def reset_rollout(self, scenario_id, rollout_id):
        self.scenario_id = str(scenario_id or "")
        self.rollout_id = int(rollout_id or 0)
        self.msg = {}
        self.control_cycle_step = -1
        self.planned_control_cycle_step = -1
        self.negotiated_contract = {}
        self.negotiated_history = []
        self.negotiated_cycle_events = []
        self.latest_negotiation_by_agent = {}
        self.latest_phase_by_agent = {}
        self.latest_tactic_by_agent = {}

    def set_scenario_context(self, context):
        self.scenario_context = deepcopy(context or {})

    def get_all_msg(self):
        return dict(self.msg)

    def join(self, veh_id, message):
        self.msg[str(veh_id)] = str(message)

    def begin_control_cycle(self, step):
        step = int(step or 0)
        if step != self.control_cycle_step:
            self.control_cycle_step = step
            self.planned_control_cycle_step = -1
            self.negotiated_cycle_events = []

    def mark_control_cycle_planned(self, step):
        self.planned_control_cycle_step = int(step or 0)

    def is_control_cycle_planned(self, step):
        return int(self.planned_control_cycle_step) == int(step or 0)

    def set_negotiated_contract(self, contract):
        self.negotiated_contract = deepcopy(contract or {})

    def publish_negotiated_negotiation(self, entry):
        item = deepcopy(entry or {})
        sender = str(item.get("sender", "")).strip()
        if not sender:
            raise ValueError("Negotiation entry requires sender.")
        item = {
            "namespace": "negotiated",
            "type": "negotiation",
            "sender": sender,
            "proposed_role": str(item.get("proposed_role", "Undecided") or "Undecided"),
            "pass_side": str(item.get("pass_side", "none") or "none"),
            "message": str(item.get("message", "") or ""),
            "step": int(item.get("step", 0) or 0),
            "scenario_id": str(item.get("scenario_id", self.scenario_id) or self.scenario_id),
            "rollout_id": int(item.get("rollout_id", self.rollout_id) or self.rollout_id),
            "control_cycle_step": int(item.get("control_cycle_step", self.control_cycle_step) or self.control_cycle_step),
        }
        self.latest_negotiation_by_agent[sender] = item
        self.negotiated_history.append(item)
        if item["control_cycle_step"] == self.control_cycle_step:
            self.negotiated_cycle_events.append(item)

    def publish_negotiated_phase(self, entry):
        item = deepcopy(entry or {})
        sender = str(item.get("sender", "")).strip()
        if not sender:
            raise ValueError("Phase entry requires sender.")
        expires_at_step = item.get("expires_at_step", 0)
        item = {
            "namespace": "negotiated",
            "type": "phase",
            "sender": sender,
            "role": str(item.get("role", "") or ""),
            "phase": str(item.get("phase", "") or ""),
            "intent": str(item.get("intent", "") or ""),
            "urgency": str(item.get("urgency", "mid") or "mid"),
            "message": str(item.get("message", "") or ""),
            "step": int(item.get("step", 0) or 0),
            "expires_at_step": int(expires_at_step or 0),
            "scenario_id": str(item.get("scenario_id", self.scenario_id) or self.scenario_id),
            "rollout_id": int(item.get("rollout_id", self.rollout_id) or self.rollout_id),
            "control_cycle_step": int(
                item.get("control_cycle_step", self.control_cycle_step) or self.control_cycle_step
            ),
        }
        self.latest_phase_by_agent[sender] = item
        self.negotiated_history.append(item)
        if item["control_cycle_step"] == self.control_cycle_step:
            self.negotiated_cycle_events.append(item)

    def publish_negotiated_tactic(self, entry):
        item = deepcopy(entry or {})
        sender = str(item.get("sender", "")).strip()
        if not sender:
            raise ValueError("Tactic entry requires sender.")
        expires_at_step = item.get("expires_at_step", 0)
        item = {
            "namespace": "negotiated",
            "type": "tactic",
            "sender": sender,
            "role": str(item.get("role", "") or ""),
            "phase": str(item.get("phase", "") or ""),
            "intent": str(item.get("intent", "") or ""),
            "mode": str(item.get("mode", "hold_lane") or "hold_lane"),
            "lane_policy": str(item.get("lane_policy", "hold_current") or "hold_current"),
            "gap_band": str(item.get("gap_band", "medium") or "medium"),
            "speed_band": str(item.get("speed_band", "match") or "match"),
            "message": str(item.get("message", "") or ""),
            "step": int(item.get("step", 0) or 0),
            "expires_at_step": int(expires_at_step or 0),
            "scenario_id": str(item.get("scenario_id", self.scenario_id) or self.scenario_id),
            "rollout_id": int(item.get("rollout_id", self.rollout_id) or self.rollout_id),
            "control_cycle_step": int(
                item.get("control_cycle_step", self.control_cycle_step) or self.control_cycle_step
            ),
        }
        self.latest_tactic_by_agent[sender] = item
        self.negotiated_history.append(item)
        if item["control_cycle_step"] == self.control_cycle_step:
            self.negotiated_cycle_events.append(item)

    def negotiated_snapshot(self, step, viewer_id=None):
        step = int(step or 0)
        latest_negotiation = {}
        for sender, entry in self.latest_negotiation_by_agent.items():
            latest_negotiation[sender] = self._sanitize_entry(entry)

        latest_phase = {}
        for sender, entry in self.latest_phase_by_agent.items():
            if self._is_negotiated_active(entry, step):
                latest_phase[sender] = self._sanitize_entry(entry)

        latest_tactic = {}
        for sender, entry in self.latest_tactic_by_agent.items():
            if self._is_negotiated_active(entry, step):
                latest_tactic[sender] = self._sanitize_entry(entry)

        recent_events = []
        for entry in self.negotiated_history[-8:]:
            if entry.get("type") == "negotiation" or self._is_negotiated_active(entry, step):
                recent_events.append(self._sanitize_entry(entry))

        return {
            "scenario_id": self.scenario_id,
            "rollout_id": self.rollout_id,
            "viewer_id": viewer_id,
            "control_cycle_step": int(self.control_cycle_step),
            "scenario_context": deepcopy(self.scenario_context),
            "negotiated_contract": deepcopy(self.negotiated_contract),
            "latest_negotiation_by_agent": latest_negotiation,
            "latest_phase_by_agent": latest_phase,
            "latest_tactic_by_agent": latest_tactic,
            "recent_phase_events": recent_events[-2:],
            "negotiated_cycle_events": [self._sanitize_entry(entry) for entry in self.negotiated_cycle_events],
        }

    def _sanitize_entry(self, entry):
        if not entry:
            return None
        return deepcopy(entry)

    def _is_negotiated_active(self, entry, step):
        if not entry:
            return False
        expires_at_step = int(entry.get("expires_at_step", 0) or 0)
        if expires_at_step <= 0:
            return True
        return step <= expires_at_step
