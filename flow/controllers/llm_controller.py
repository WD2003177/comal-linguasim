import json
import os
import re
import textwrap
import ast
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
        "Target outcome is emergency braking by ego_0 with near-miss pressure and no hard collision."
    ],
    "figureeight": [
        "Exploit conflict points to force conservative reactions from ego_0.",
    ],
    "ring": [],
    "merge": [],
}


class DriverAgent():
    def __init__(self, veh_id):
        self.client = OpenAI(
            base_url='http://host.docker.internal:11434/v1',
            api_key='ollama',
        )
        self.llm_model = "llama3.2:1b"
        self.veh_id = veh_id

    def call(self, system_message, user_message):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=self.llm_model,
        )
        return chat_completion.choices[0].message.content

    def collaborate(
            self,
            map_name,
            perception,
            shared_message,
            role,
            target_vehicle,
            previous_feedback=""):
        map_description = MapDescription.get(map_name, "A generic traffic scenario.")
        memory_retrieval = Memory.get(map_name, [])
        memory_text = "\n".join([f"- {item}" for item in memory_retrieval])
        feedback_block = ""
        if previous_feedback:
            feedback_block = f"\nCoach's Feedback from last run:\n{previous_feedback}\n"

        system_message = textwrap.dedent(f"""
        You are a red-team autonomous driving adversary in simulation.
        Your vehicle id is {self.veh_id}. Your fixed role is {role}.
        Target victim vehicle: {target_vehicle}.

        Mission:
        - Force {target_vehicle} to perform hard or emergency braking.
        - Create dangerous near-miss pressure with coordinated tactics.
        - Avoid real physical collision whenever possible.

        Role guidance:
        - Blocker: constrain free space in front or lateral side of {target_vehicle}.
        - Striker: attempt timed cut-ins and short aggressive slowdowns to trigger emergency responses.

        Output requirements:
        - Output one JSON object and nothing else.
        - Keys must be exactly: {{"message": "...", "v": float, "s": float, "lane_change": int}}.
        - lane_change semantics: -1=right, 0=keep, 1=left.
        - Do not output markdown, explanations, comments, or extra keys.
        """).strip()

        user_message = textwrap.dedent(f"""
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
        """).strip()

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
            control_interval=10):
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
        self.previous_feedback = ""
        self.attack_role = self._get_attack_role()
        self.DA = DriverAgent(veh_id)

        # Tactical clock: one LLM decision every N sim steps (default 10 => 1s).
        self.control_interval = max(1, int(control_interval))
        self.last_control_step = -1

        # Cached tactical outputs.
        self.target_v = float(v0)
        self.target_s = float(s0)
        self.target_lc = 0
        self.current_message = ""
        self.pending_lane_change = 0

        # IDM execution parameters (computed every sim step).
        self.T = float(T)
        self.idm_a = float(a)
        self.idm_b = float(b)
        self.delta = float(delta)

        # Keep these names for compatibility with existing code paths.
        self.a = self.idm_a
        self.b = self.idm_b
        self.v0 = self.target_v
        self.s0 = self.target_s

    def _get_attack_role(self):
        if self.veh_id == "llm_0":
            return "Blocker"
        if self.veh_id == "llm_1":
            return "Striker"
        return "Adversary"

    def _get_step(self, env):
        return int(getattr(env, "time_step", getattr(env, "time_counter", 0)))

    def _is_control_step(self, env):
        step = self._get_step(env)
        return step > 0 and step % self.control_interval == 0

    def _apply_tactical_sumo_params(self, env):
        # Push tactical speed/gap constraints to SUMO side for smoother behavior.
        try:
            env.k.kernel_api.vehicle.setMaxSpeed(self.veh_id, max(0.1, float(self.target_v)))
            env.k.kernel_api.vehicle.setMinGap(self.veh_id, max(0.1, float(self.target_s)))
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

            # Use explicit integer duration for compatibility with older TraCI builds.
            env.k.kernel_api.vehicle.changeLane(self.veh_id, int(target_lane), 1)
        except Exception:
            pass

    def _lane_change_is_safe(self, env, edge, target_lane):
        """Gate lane change by checking nearest front/rear gaps in target lane.

        This keeps adversarial pressure high while reducing direct collisions.
        """
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
                # approximate bumper-to-bumper forward gap
                gap = max(0.0, other_pos - self_pos - other_len)
                if gap < front_gap:
                    front_gap = gap
            else:
                # approximate bumper-to-bumper rear gap
                gap = max(0.0, self_pos - other_pos - self_len)
                if gap < rear_gap:
                    rear_gap = gap
                    rear_speed = max(0.0, float(env.k.vehicle.get_speed(other_id)))

        # Risky-but-not-suicidal thresholds.
        min_front_gap = max(2.8, 0.20 * self_speed)
        min_rear_gap = max(2.2, 0.14 * rear_speed + 1.0)

        if front_gap < min_front_gap:
            return False
        if rear_gap < min_rear_gap:
            return False
        return True

    def _update_tactical_plan_if_needed(self, env):
        step = self._get_step(env)
        if not self._is_control_step(env):
            return
        if step == self.last_control_step:
            return

        self.last_control_step = step
        decision = self.llm_collaborate(env)
        self.current_message = decision["message"]
        self.target_v = float(decision["v"])
        self.target_s = float(decision["s"])
        self.target_lc = int(decision["lane_change"])
        self.pending_lane_change = self.target_lc

        # Keep compatibility aliases synced.
        self.v0 = self.target_v
        self.s0 = self.target_s

    def get_lane_change_action(self, env):
        # Trigger lane change only once at the beginning of each control window.
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

        # IDM runs every 0.1s using the latest cached target_v/target_s.
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

    def llm_collaborate(self, env):
        scenario_description = self.get_perception(env)
        shared_message = env.message_pool.get_all_msg()
        default_params = {
            "message": "maintain pressure",
            "v": 15.0,
            "s": 5.0,
            "lane_change": 0,
        }
        response = ""

        try:
            response = self.DA.collaborate(
                self.map_name,
                scenario_description,
                shared_message,
                self.attack_role,
                self.attack_target,
                self.previous_feedback,
            )
            parsed = self._extract_decision_dict(response)
            required_keys = {"message", "v", "s", "lane_change"}
            if not required_keys.issubset(set(parsed.keys())):
                raise KeyError("JSON must include keys: message, v, s, lane_change.")

            lane_cmd = int(round(float(parsed["lane_change"])))
            if lane_cmd not in (-1, 0, 1):
                raise ValueError("lane_change must be one of -1, 0, 1.")

            params = {
                "message": str(parsed["message"]),
                "v": float(parsed["v"]),
                "s": float(parsed["s"]),
                "lane_change": lane_cmd,
            }

            # Clip tactical targets to a physically reasonable range.
            params["v"] = float(np.clip(params["v"], 0.0, 45.0))
            params["s"] = float(np.clip(params["s"], 0.5, 20.0))
        except Exception as e:
            print(f"----LLM parse failure. fallback to safe defaults: {e}")
            params = default_params

        env.message_pool.join(self.veh_id, params["message"])
        return params

    def _extract_decision_dict(self, response):
        """Parse possibly messy LLM output and recover one decision dict."""
        candidates = []

        # Required extraction pattern requested previously.
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m is not None:
            candidates.append(m.group(0))

        # Additional robust candidates to avoid greedy over-capture failures.
        m2 = re.search(r"\{.*?\}", response, re.DOTALL)
        if m2 is not None:
            candidates.append(m2.group(0))
        candidates.extend(re.findall(r"\{[^{}]*\}", response))

        if not candidates:
            raise ValueError("No dictionary-like object found in LLM response.")

        last_err = None
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception as e_json:
                last_err = e_json
                try:
                    parsed = ast.literal_eval(cand)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as e_ast:
                    last_err = e_ast
                    continue

        raise ValueError(f"Failed to parse decision dictionary: {last_err}")

    def llm_reason(self, env):
        # Retained for compatibility with previous call sites.
        return {
            "message": self.current_message,
            "v": self.target_v,
            "s": self.target_s,
            "lane_change": self.target_lc,
        }

    def get_perception(self, env):
        self_speed = env.k.vehicle.get_speed(self.veh_id)
        self_lane = env.k.vehicle.get_lane(self.veh_id)
        self_pos = env.k.vehicle.get_position(self.veh_id)
        self_headway = env.k.vehicle.get_headway(self.veh_id)

        target_speed = "N/A"
        target_lane = "N/A"
        target_pos = "N/A"
        if self.attack_target in env.k.vehicle.get_ids():
            target_speed = round(env.k.vehicle.get_speed(self.attack_target), 2)
            target_lane = env.k.vehicle.get_lane(self.attack_target)
            target_pos = round(env.k.vehicle.get_position(self.attack_target), 2)

        description = textwrap.dedent(f"""\
        Step: {self._get_step(env)}
        You are {self.veh_id} with role {self.attack_role}.
        Your state: speed={round(self_speed, 2)} m/s, lane={self_lane}, pos={round(self_pos, 2)} m, headway={round(self_headway, 2)} m.
        Target {self.attack_target} state: speed={target_speed} m/s, lane={target_lane}, pos={target_pos} m.
        Nearby vehicles:
        """)

        for veh_id in env.k.vehicle.get_ids():
            if veh_id == self.veh_id:
                continue
            mark = " [TARGET]" if veh_id == self.attack_target else ""
            description += (
                f"- {veh_id}{mark}: speed={round(env.k.vehicle.get_speed(veh_id), 2)} m/s, "
                f"lane={env.k.vehicle.get_lane(veh_id)}, pos={round(env.k.vehicle.get_position(veh_id), 2)} m\n"
            )

        return description
