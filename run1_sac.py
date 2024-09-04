import os
import sys
from datetime import datetime
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import OldEnv

def make_env(env_config):
    return OldEnv(
        net_file='nets/2way-single-intersection/111.net.xml',
        route_file='nets/2way-single-intersection/111.rou.xml',
        out_csv_name='outputs/sac_2way-single-intersection_{}'.format(env_config['experiment_time']),
        use_gui=False,
        num_seconds=3600,
        min_green=10,
        max_depart_delay=0,
        single_agent=True
    )

if __name__ == '__main__':
    experiment_time = str(datetime.now()).replace(":", "-").split('.')[0]

    # Register the environment with Ray
    register_env("sumo_env", lambda config: make_env(config))

    # Define SAC configuration for single-agent
    config = {
        "env": "sumo_env",
        "num_workers": 1,  # SAC usually works better with fewer workers
        "framework": "torch",  # Use "torch" or "tf" based on your preference
        "env_config": {
            "experiment_time": experiment_time
        },
        "gamma": 0.99,  # Discount factor
        "lr": 0.0003,  # Learning rate
        "tau": 0.005,  # Target network update rate
        "batch_size": 256,  # Batch size for experience replay
        "train_batch_size": 1024,  # Training batch size
        "target_entropy": "auto",  # Automatic entropy tuning
        "num_gpus": 0,  # Number of GPUs to use
    }

    # Train using Ray's SAC algorithm
    tune.run("SAC", config=config, stop={"timesteps_total": 14400})
