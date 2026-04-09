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
        self.history = []
        self.control_cycle_step = -1
        self.cycle_entries = []
        self.active_owner_plan = None
        self.latest_commit = None
        self.latest_status_by_agent = {}
        self.latest_trigger_eval = {}
        self.latest_entry_by_sender = {}

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
            self.cycle_entries = []

    def publish(self, entry):
        if not isinstance(entry, dict):
            raise TypeError("Blackboard entry must be a dict.")

        item = deepcopy(entry)
        sender = str(item.get("sender", "")).strip()
        if not sender:
            raise ValueError("Blackboard entry requires sender.")

        item["sender"] = sender
        item["owner"] = str(item.get("owner", "") or "")
        item["kind"] = str(item.get("kind", "status") or "status")
        item["plan_id"] = str(item.get("plan_id", "") or "")
        item["reply_to"] = str(item.get("reply_to", "") or "")
        item["phase"] = str(item.get("phase", "setup") or "setup")
        item["intent"] = str(item.get("intent", "") or "")
        item["target"] = str(item.get("target", "") or "")
        item["eta_steps"] = int(item.get("eta_steps", 0) or 0)
        item["trigger_code"] = str(item.get("trigger_code", "none") or "none")
        item["trigger_args"] = deepcopy(item.get("trigger_args", {}) or {})
        item["preconditions"] = list(item.get("preconditions", []) or [])
        item["done_code"] = str(item.get("done_code", "phase_complete") or "phase_complete")
        item["fallback"] = str(item.get("fallback", "hold_lane") or "hold_lane")
        item["confidence"] = float(item.get("confidence", 0.0) or 0.0)
        item["step"] = int(item.get("step", 0) or 0)
        item["expires_at_step"] = int(item.get("expires_at_step", 0) or 0)
        item["scenario_id"] = str(item.get("scenario_id", self.scenario_id) or self.scenario_id)
        item["rollout_id"] = int(item.get("rollout_id", self.rollout_id) or self.rollout_id)
        item["control_cycle_step"] = int(item.get("control_cycle_step", self.control_cycle_step) or self.control_cycle_step)

        self.history.append(item)
        if item["control_cycle_step"] == self.control_cycle_step:
            self.cycle_entries.append(item)
        self.latest_entry_by_sender[sender] = item

        text_message = item.get("text")
        if not text_message:
            text_message = item["intent"] or item["kind"]
        self.join(sender, text_message)

        if self._is_owner_plan_entry(item) and item["kind"] in ("commit", "replan"):
            self.active_owner_plan = item
        if self._is_owner_plan_entry(item) and item["kind"] == "commit":
            self.latest_commit = item
        if item["kind"] in ("status", "ack", "commit", "replan"):
            self.latest_status_by_agent[sender] = item
        if "trigger_eval" in item:
            self.latest_trigger_eval[sender] = deepcopy(item["trigger_eval"])

    def snapshot(self, step, viewer_id=None):
        step = int(step or 0)
        active_owner_plan = self._sanitize_entry(self._resolve_active_owner_plan(step))
        latest_commit = self._sanitize_entry(self._resolve_latest_commit(step))
        latest_status = {}
        for sender, entry in self.latest_status_by_agent.items():
            if self._is_active(entry, step):
                latest_status[sender] = self._sanitize_entry(entry)

        trigger_eval = {}
        for sender, payload in self.latest_trigger_eval.items():
            if sender in latest_status:
                trigger_eval[sender] = deepcopy(payload)

        recent_events = []
        for entry in self.history[-6:]:
            if self._is_active(entry, step) or entry["kind"] != "status":
                recent_events.append(self._sanitize_entry(entry))

        return {
            "scenario_id": self.scenario_id,
            "rollout_id": self.rollout_id,
            "viewer_id": viewer_id,
            "control_cycle_step": int(self.control_cycle_step),
            "scenario_context": deepcopy(self.scenario_context),
            "active_owner_plan": active_owner_plan,
            "latest_commit": latest_commit,
            "latest_status_by_agent": latest_status,
            "trigger_eval": trigger_eval,
            "recent_events": recent_events,
            "cycle_entries": [self._sanitize_entry(entry) for entry in self.cycle_entries],
        }

    def _is_active(self, entry, step):
        if not entry:
            return False
        if self._is_plan_terminal(entry):
            return False
        expires_at_step = int(entry.get("expires_at_step", 0) or 0)
        if expires_at_step <= 0:
            return True
        return step <= expires_at_step

    def _is_owner_plan_entry(self, entry):
        if not entry:
            return False
        owner = str(entry.get("owner", "") or "")
        sender = str(entry.get("sender", "") or "")
        return bool(owner) and owner == sender

    def _is_plan_terminal(self, entry):
        if not entry:
            return False
        return str(entry.get("phase", "") or "") == "disengage"

    def _resolve_active_owner_plan(self, step):
        if self._is_active(self.active_owner_plan, step):
            return self.active_owner_plan
        for entry in reversed(self.history):
            if self._is_owner_plan_entry(entry) and entry.get("kind") in ("commit", "replan"):
                if self._is_active(entry, step):
                    self.active_owner_plan = entry
                    return entry
                self.active_owner_plan = None
                return None
        self.active_owner_plan = None
        return None

    def _resolve_latest_commit(self, step):
        if self._is_active(self.latest_commit, step):
            return self.latest_commit
        for entry in reversed(self.history):
            if entry.get("kind") == "commit" and self._is_owner_plan_entry(entry):
                if self._is_active(entry, step):
                    self.latest_commit = entry
                    return entry
                self.latest_commit = None
                return None
        self.latest_commit = None
        return None

    def _sanitize_entry(self, entry):
        if not entry:
            return None
        return deepcopy(entry)
