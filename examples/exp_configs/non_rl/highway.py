"""Highway red-team setting with ego victim and LLM attackers."""

from flow.controllers import IDMController, LLMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import VehicleParams, SumoLaneChangeParams, SumoCarFollowingParams
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv

vehicles = VehicleParams()
vehicles.add(
    veh_id="ego",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
        model="SL2015",
        lc_sublane=1.0,
    ),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=3.0,
    ),
    num_vehicles=1,
    color="blue")

vehicles.add(
    veh_id="llm",
    acceleration_controller=(LLMController, {"map": "highway"}),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="no_lc_safe",
        model="SL2015",
        lc_sublane=1.0,
    ),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=4.5,
    ),
    num_vehicles=2,
    color="yellow")

vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
        model="SL2015",
        lc_sublane=1.0,
    ),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=3.0,
    ),
    num_vehicles=8,
    color="white")

env_additional_params = ADDITIONAL_ENV_PARAMS.copy()
net_additional_params = ADDITIONAL_NET_PARAMS.copy()
net_additional_params["lanes"] = max(3, net_additional_params["lanes"])
net_additional_params["length"] = 1800


flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
        lateral_resolution=1.0,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=500,
        additional_params=env_additional_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=net_additional_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        shuffle=True,
    ),
)
