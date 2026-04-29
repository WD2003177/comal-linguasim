"""Highway red-team setting with ego victim and LLM attackers."""

from flow.controllers import IDMController, LLMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import VehicleParams, SumoLaneChangeParams, SumoCarFollowingParams
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv
from flow.utils.highway_scene import default_custom_start_params
from flow.utils.highway_scene import get_highway_scene_mode


SCENE_MODE = get_highway_scene_mode()
NUM_HUMANS = 8 if SCENE_MODE == "traffic" else 0
LLM_LANE_CHANGE_MODE = "sumo_default" if SCENE_MODE == "traffic" else "no_lc_aggressive"
LLM_SPEED_MODE = "obey_safe_speed" if SCENE_MODE == "traffic" else "aggressive"
LLM_LC_PUSHY = 0.0 if SCENE_MODE == "traffic" else 1.0
LLM_LC_ASSERTIVE = 1.0 if SCENE_MODE == "traffic" else 10.0
LLM_LC_SPEED_GAIN = 1.0 if SCENE_MODE == "traffic" else 5.0
LLM_LC_COOPERATIVE = 1.0 if SCENE_MODE == "traffic" else 0.0
LLM_LC_KEEP_RIGHT = 1.0 if SCENE_MODE == "traffic" else 0.0
LLM_DECEL = 4.5 if SCENE_MODE == "traffic" else 7.5

vehicles = VehicleParams()
vehicles.add(
    veh_id="ego",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
        model="SL2015",
        lc_sublane=0.2,
        lc_pushy=0.0,
        lc_assertive=1.0,
        lc_accel_lat=2.0,
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
        lane_change_mode=LLM_LANE_CHANGE_MODE,
        model="SL2015",
        lc_sublane=1.0,
        lc_pushy=LLM_LC_PUSHY,
        lc_assertive=LLM_LC_ASSERTIVE,
        lc_speed_gain=LLM_LC_SPEED_GAIN,
        lc_cooperative=LLM_LC_COOPERATIVE,
        lc_keep_right=LLM_LC_KEEP_RIGHT,
    ),
    car_following_params=SumoCarFollowingParams(
        speed_mode=LLM_SPEED_MODE,
        decel=LLM_DECEL,
    ),
    num_vehicles=2,
    color="yellow")

if NUM_HUMANS > 0:
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
        num_vehicles=NUM_HUMANS,
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
        horizon=1500,
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
        spacing="custom",
        shuffle=False,
        additional_params=default_custom_start_params(scene_mode=SCENE_MODE),
    ),
)
