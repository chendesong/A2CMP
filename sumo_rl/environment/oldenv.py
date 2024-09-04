import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from .oldsignal import OldSignal
from gymnasium import spaces


class OldEnv(MultiAgentEnv):
    def __init__(self, net_file, route_file, out_csv_name=None, use_gui=False, num_seconds=20000,
                 max_depart_delay=100000, time_to_teleport=-1, delta_time=5, yellow_time=2,
                 min_green=5, max_green=50, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.sim_max_time = num_seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        self.sim_step = 0  # 初始化sim_step

        traci.start([self._sumo_binary, '-n', self._net])  # 开始TraCI连接以获得初始信息
        self.ts_ids = traci.trafficlight.getIDList()
        self.traffic_signals = {
            ts: OldSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green)
            for ts in self.ts_ids
        }
        self.vehicles = dict()

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        # 定义观测空间和动作空间
        self._define_spaces()

    def _define_spaces(self):
        example_observation = self.traffic_signals[self.ts_ids[0]].compute_observation()
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=example_observation.shape,
                                             dtype=np.float32)

        # 假设交通信号灯的动作空间是离散的
        self._action_space = spaces.Discrete(len(self.traffic_signals[self.ts_ids[0]].phases))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        try:
            traci.close()  # 关闭之前的连接，确保重新启动
        except traci.exceptions.FatalTraCIError:
            pass

        self.run += 1
        self.metrics = []
        self.sim_step = 0  # 每次重置时初始化仿真步骤

        sumo_cmd = [
            self._sumo_binary,
            '-n', self._net,
            '-r', self._route,
            '--max-depart-delay', str(self.max_depart_delay),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', str(self.time_to_teleport),
            '--random'
        ]
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        self.traffic_signals = {
            ts: OldSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green)
            for ts in self.ts_ids
        }
        self.vehicles = dict()

        return {ts: self._compute_observations()[ts] for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}, {}

    def step(self, action):
        try:
            if not action:
                for _ in range(self.delta_time):
                    self._sumo_step()
                    if self.sim_step % 5 == 0:
                        info = self._compute_step_info()
                        self.metrics.append(info)
            else:
                self._apply_actions(action)

                time_to_act = False
                while not time_to_act:
                    self._sumo_step()
                    for ts in self.ts_ids:
                        self.traffic_signals[ts].update()
                        if self.traffic_signals[ts].time_to_act:
                            time_to_act = True

                    if self.sim_step % 5 == 0:
                        info = self._compute_step_info()
                        self.metrics.append(info)

            observations = self._compute_observations()
            rewards = self._compute_rewards()
            done = {'__all__': self.sim_step > self.sim_max_time}
            terminated = {'__all__': self.sim_step > self.sim_max_time}
            truncated = {'__all__': False}

            return observations, rewards, terminated, truncated, {}

        finally:
            if self.sim_step >= self.sim_max_time or done['__all__']:
                print("Simulation finished, saving CSV...")
                self.save_csv(self.out_csv_name, self.run)

    def _apply_actions(self, actions):
        for ts, action in actions.items():
            self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if
                self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if
                self.traffic_signals[ts].time_to_act}

    def _sumo_step(self):
        traci.simulationStep()
        self.sim_step += 1  # 更新sim_step

    def _compute_step_info(self):
        total_vehicle_wait_time = sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        total_pedestrian_wait_time = sum(self.traffic_signals[ts].get_pedestrian_waiting_time() for ts in self.ts_ids)
        total_wait_time = 0.6 * total_vehicle_wait_time + 0.4 * total_pedestrian_wait_time
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': total_wait_time,
            'total_co2_emission': sum(self.traffic_signals[ts].get_total_emission() for ts in self.ts_ids),
            'mean_co2_emission': sum(self.traffic_signals[ts].get_mean_emission() for ts in self.ts_ids),
            'total_travel_time': sum(self.traffic_signals[ts].get_total_travel_time() for ts in self.ts_ids)
        }

    def close(self):
        traci.close()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0]
        # elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases:]]
        return self.radix_encode([phase] + density_queue, ts_id)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green // self.delta_time):
            if elapsed <= self.delta_time + i * self.delta_time:
                return i
        return self.max_green // self.delta_time - 1

    def radix_encode(self, values, ts_id):
        res = 0
        self.radix_factors = [s.n for s in self.traffic_signals[ts_id].discrete_observation_space.spaces]
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    """ def radix_decode(self, value):
        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res """


