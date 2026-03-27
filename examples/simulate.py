"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import sys
import json
import os
from flow.core.experiment import Experiment

from flow.core.params import AimsunParams
from flow.utils.rllib import FlowParamsEncoder


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')

    # optional input parameters
    parser.add_argument(
        '--num_runs', type=int, default=1,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--aimsun',
        action='store_true',
        help='Specifies whether to run the simulation using the simulator '
             'Aimsun. If not specified, the simulator used is SUMO.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # Get the flow_params object.
    module = __import__("exp_configs.non_rl", fromlist=[flags.exp_config])
    flow_params = getattr(module, flags.exp_config).flow_params

    # Get the custom callables for the runner.
    if hasattr(getattr(module, flags.exp_config), "custom_callables"):
        callables = getattr(module, flags.exp_config).custom_callables
    else:
        callables = None

    flow_params['sim'].render = not flags.no_render
    flow_params['simulator'] = 'aimsun' if flags.aimsun else 'traci'

    # If Aimsun is being called, replace SumoParams with AimsunParams.
    if flags.aimsun:
        sim_params = AimsunParams()
        sim_params.__dict__.update(flow_params['sim'].__dict__)
        flow_params['sim'] = sim_params

    # Specify an emission path if they are meant to be generated.
    if flags.gen_emission:
        flow_params['sim'].emission_path = "./data"

        # Create the flow_params object
        fp_ = flow_params['exp_tag']
        dir_ = flow_params['sim'].emission_path
        with open(os.path.join(dir_, "{}.json".format(fp_)), 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Create the experiment object.
    exp = Experiment(flow_params, callables)

    # LinguaSim closed-loop red-team refinement on highway.
    MAX_ITERATIONS = 10
    env = exp.env
    exp_tag = str(flow_params.get("exp_tag", "")).lower()
    network_name = getattr(flow_params.get("network"), "__name__", "").lower()
    if "highway" not in exp_tag and "highway" not in network_name:
        print("Warning: LinguaSim closed-loop is intended for highway, not figure eight.")

    # Optional hook if exp config provides a custom rl_actions function.
    if hasattr(getattr(module, flags.exp_config), "rl_actions"):
        rl_actions = getattr(module, flags.exp_config).rl_actions
    else:
        def rl_actions(*_):
            return None

    for iteration in range(MAX_ITERATIONS):
        crashed = False
        min_ttc = float("inf")

        state = env.reset()
        for step in range(env.env_params.horizon):
            action = rl_actions(state)
            state, reward, done, _ = env.step(action)

            veh_ids = env.k.vehicle.get_ids()
            if "ego_0" in veh_ids:
                ego_speed = env.k.vehicle.get_speed("ego_0")
                ego_leader = env.k.vehicle.get_leader("ego_0")
                llm_ids = [veh_id for veh_id in veh_ids if "llm" in veh_id]

                for llm_id in llm_ids:
                    llm_speed = env.k.vehicle.get_speed(llm_id)
                    llm_leader = env.k.vehicle.get_leader(llm_id)

                    # LLM is behind ego and catching up.
                    if llm_leader == "ego_0":
                        rel_speed = llm_speed - ego_speed
                        if rel_speed > 0:
                            headway = env.k.vehicle.get_headway(llm_id)
                            if headway > 0:
                                min_ttc = min(min_ttc, headway / rel_speed)

                    # Ego is behind LLM and catching up.
                    elif ego_leader == llm_id:
                        rel_speed = ego_speed - llm_speed
                        if rel_speed > 0:
                            headway = env.k.vehicle.get_headway("ego_0")
                            if headway > 0:
                                min_ttc = min(min_ttc, headway / rel_speed)

            if env.k.simulation.check_collision():
                crashed = True
                break

            if done:
                break

        if crashed:
            feedback = (
                "FAILED: You caused a physical collision! This is unacceptable. "
                "Increase your safety distance 's' parameter immediately."
            )
        elif 0 < min_ttc <= 3.0:
            feedback = "SUCCESS"
            print("危险场景生成成功")
        else:
            feedback = (
                f"FAILED: Too safe. Min TTC was {min_ttc:.2f}s. The ego vehicle "
                "easily dodged your attack. Decrease 's' to press closer, or "
                "execute the cut-in more abruptly."
            )

        for veh_id in env.k.vehicle.get_rl_ids() + env.k.vehicle.get_human_ids():
            if "llm" in veh_id:
                controller = env.k.vehicle.get_acc_controller(veh_id)
                if controller is not None:
                    controller.previous_feedback = feedback

        print(
            f"Iteration {iteration + 1}/{MAX_ITERATIONS} | "
            f"min_ttc={min_ttc:.3f} | feedback={feedback}"
        )

        if feedback == "SUCCESS":
            if flags.gen_emission and env.simulator == "traci":
                env.k.simulation.save_emission(run_id=iteration)
            break

    env.terminate()
