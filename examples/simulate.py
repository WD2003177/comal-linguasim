"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import json
import os
import sys
from datetime import datetime

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams
from flow.utils.highway_scene import SCENE_GATE_SAMPLE_BUDGET
from flow.utils.highway_scene import sample_and_freeze_scene
from flow.utils.rllib import FlowParamsEncoder


MAX_ITERATIONS = 10
HARD_BRAKE_DECEL = 2.5
SUCCESS_TTC = 3.0
PROGRESS_EVERY_STEPS = max(0, int(os.getenv("FLOW_PROGRESS_EVERY_STEPS", "200")))


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


def collect_llm_stats(env_instance):
    stats = {}
    for veh_id, controller in iter_llm_controllers(env_instance).items():
        if hasattr(controller, "get_runtime_stats"):
            stats[veh_id] = controller.get_runtime_stats()
    return stats


def collect_llm_diagnostics(env_instance):
    diagnostics = {}
    for veh_id, controller in iter_llm_controllers(env_instance).items():
        if hasattr(controller, "get_rollout_diagnostics"):
            diagnostics[veh_id] = controller.get_rollout_diagnostics()
    return diagnostics


def inject_rollout_context(env, rolling_feedback, case_memory, scenario_context, iteration):
    controllers = iter_llm_controllers(env)
    initial_signatures = {}
    env.message_pool.set_scenario_context(scenario_context)
    for veh_id, controller in controllers.items():
        controller.previous_feedback = rolling_feedback
        controller.case_memory = case_memory
        controller.scenario_id = scenario_context.get("scenario_id", "")
        controller.current_iteration = iteration
        controller.frozen_geometry = scenario_context.get("frozen_geometry", {})
        controller.role_map = scenario_context.get("role_map", {})
        controller.scene_gate_status = scenario_context.get("scene_gate_status", {})
        if hasattr(controller, "refresh_attack_role"):
            controller.refresh_attack_role()
        if hasattr(controller, "get_state_signature"):
            initial_signatures[veh_id] = controller.get_state_signature(env, phase="compress")
    return controllers, initial_signatures


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


def choose_failure_phase(diagnostics):
    for veh_id in ("llm_0", "llm_1"):
        diag = diagnostics.get(veh_id)
        if diag and diag.get("active_phase"):
            return diag.get("active_phase")
    for diag in diagnostics.values():
        if diag.get("active_phase"):
            return diag.get("active_phase")
    return "setup"


def determine_feedback_summary(crashed, success, diagnostics, failure_phase):
    if crashed:
        return "collision_on_strike"
    if any(diag.get("rollout_parse_fallback_used", 0) > 0 for diag in diagnostics.values()):
        return "parse_fallback_used"
    if any(diag.get("owner_plan_missing_events", 0) > 0 for diag in diagnostics.values()):
        return "owner_plan_missing"
    if any(diag.get("trigger_not_met_events", 0) > 0 for diag in diagnostics.values()):
        return "trigger_not_met"
    if success:
        return "success"
    if failure_phase in ("setup", "compress"):
        return "blocker_late"
    if failure_phase in ("strike", "brake_pulse"):
        return "striker_early"
    return "ego_escaped_right"


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
        "model": os.getenv("FLOW_LLM_MODEL", "llama3.2:3b"),
        "llm_timeout_s": os.getenv("FLOW_LLM_TIMEOUT_S", "8.0"),
        "llm_max_retries": "3",
        "llm_neighbor_k": os.getenv("FLOW_LLM_NEIGHBOR_K", "6"),
        "role_map": scenario_context.get("role_map", {}),
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
        "Scene frozen: scenario_id={} sample_index={} role_map={}".format(
            scenario_id,
            (scenario_context.get("scene_gate_status") or {}).get("sample_index", ""),
            scenario_context.get("role_map", {}),
        ),
        flush=True,
    )

    interrupted = False
    current_iteration = 0
    current_step_in_iteration = 0
    current_min_ttc = float("inf")
    current_ego_max_decel = 0.0
    current_crashed = False
    try:
        for iteration in range(MAX_ITERATIONS):
            current_iteration = iteration + 1
            current_step_in_iteration = 0
            crashed = False
            ego_max_decel = 0.0
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

            prev_ego_speed = None
            if "ego_0" in env.k.vehicle.get_ids():
                prev_ego_speed = float(env.k.vehicle.get_speed("ego_0"))

            for _ in range(env.env_params.horizon):
                action = rl_actions(state)
                state, reward, done, _ = env.step(action)
                current_step_in_iteration += 1
                current_crashed = bool(crashed)

                if "ego_0" in env.k.vehicle.get_ids():
                    ego_speed = float(env.k.vehicle.get_speed("ego_0"))
                    if prev_ego_speed is not None:
                        ego_decel = max(0.0, (prev_ego_speed - ego_speed) / max(env.sim_step, 1e-3))
                        ego_max_decel = max(ego_max_decel, ego_decel)
                    prev_ego_speed = ego_speed

                min_ttc = min(min_ttc, compute_min_ttc(env))
                current_min_ttc = min_ttc
                current_ego_max_decel = ego_max_decel

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

            diagnostics = collect_llm_diagnostics(env)
            llm_stats = collect_llm_stats(env)
            failure_phase = choose_failure_phase(diagnostics)
            sync_error = any(
                diag.get("owner_plan_missing_events", 0) > 0 or diag.get("sync_error_events", 0) > 0
                for diag in diagnostics.values()
            )
            feedback_summary = determine_feedback_summary(crashed, success, diagnostics, failure_phase)

            feedback = {
                "scenario_id": scenario_id,
                "iteration": iteration + 1,
                "result": "success" if success else ("collision" if crashed else "too_safe" if too_safe else "failed"),
                "feedback_summary": feedback_summary,
                "min_ttc": None if min_ttc == float("inf") else round(float(min_ttc), 3),
                "ego_max_decel": round(float(ego_max_decel), 3),
                "hard_brake_event": bool(hard_brake_event),
                "failure_phase": failure_phase,
                "sync_error": bool(sync_error),
                "collision": bool(crashed),
                "scene_gate_status": scenario_context.get("scene_gate_status", {}),
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
                    "hard_brake_event": feedback["hard_brake_event"],
                    "failure_phase": feedback["failure_phase"],
                    "sync_error": feedback["sync_error"],
                    "collision": feedback["collision"],
                    "feedback_summary": feedback["feedback_summary"],
                })

            case_memory = case_memory[-120:]
            rolling_feedback = feedback

            print(
                "Iteration {}/{} | min_ttc={} | ego_max_decel={:.3f} | hard_brake={} | feedback={}".format(
                    iteration + 1,
                    MAX_ITERATIONS,
                    "inf" if min_ttc == float("inf") else "{:.3f}".format(min_ttc),
                    ego_max_decel,
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
                "scene_gate_status": scenario_context.get("scene_gate_status", {}),
                "sampling_strategy": scenario_context.get("sampling_strategy", ""),
                "crashed": bool(crashed),
                "result": feedback["result"],
                "feedback_summary": feedback_summary,
                "min_ttc": feedback["min_ttc"],
                "ego_max_decel": feedback["ego_max_decel"],
                "hard_brake_event": feedback["hard_brake_event"],
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
                env.k.simulation.save_emission(run_id=iteration)

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
                print("危险场景生成成功")
                break
    except KeyboardInterrupt:
        interrupted = True
        if current_iteration > 0 and (
                (not iteration_logs) or iteration_logs[-1].get("iteration") != current_iteration):
            try:
                interrupted_diagnostics = collect_llm_diagnostics(env)
                interrupted_stats = collect_llm_stats(env)
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
