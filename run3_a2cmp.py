import os
import sys
from datetime import datetime
import gymnasium as gym


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import NewEnv
import traci

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == '__main__':


    experiment_time = str(datetime.now()).replace(":", "-").split('.')[0]

    # multiprocess environment
    n_cpu = 1
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: NewEnv(
        net_file='nets/2way-single-intersection/111.net.xml',
        route_file='nets/2way-single-intersection/222.rou.xml',
        out_csv_name='outputs/a2c_2way-single-intersection_{}'.format(experiment_time),
        single_agent=True,
        use_gui=True,
        num_seconds=3600,
        min_green=10,
        max_depart_delay=0)])

    model = A2C('MlpPolicy', env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=14400)

