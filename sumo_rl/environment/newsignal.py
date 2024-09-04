import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gymnasium import spaces



class NewSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        self.last_measure = 0.0
        self.last_reward = None
        self.phases = traci.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.num_green_phases = len(
            self.phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order only in lane
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))  # only out lane
        self.vehicle_base_co2 = 2500  # add by me
        self.HDV_count = 2
        self.LDV_count = 0.05
        self.Bus_count = 1

        #添加的
        self.reduced_phases = [1, 2, 4, 5] #2 way intersection
        #self.reduced_phases = [0, 1, 2, 3, 4] #london intersection
        #self.reduced_phases = [0, 1, 2, 3, 4, 5]  # test improvement
        self.full_phases = list(range(len(self.phases)))    # 假设 full phase 使用所有 green phases
        self.current_phases = self.full_phases  # 初始默认使用 full phases
        self.use_reduced_phases = False  # 初始设置为不使用 reduced phases
        self.reference_phase = None  # 添加的属性，用于存储参考相位

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        # change by me
        # self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2*len(self.lanes)), high=np.ones(self.num_green_phases + 2*len(self.lanes)))
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 3 * len(self.lanes)),
                                            high=np.ones(self.num_green_phases + 3 * len(self.lanes)))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),  # Green Phase
            # spaces.Discrete(self.max_green//self.delta_time),            # Elapsed time of phase
            *(spaces.Discrete(10) for _ in range(2 * len(self.lanes)))  # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)


        logic = traci.trafficlight.Logic("new-program" + self.id, 0, 0, phases=self.phases)
        traci.trafficlight.setProgramLogic(self.id, logic)





    def calculate_vehicle_pressure(self):
        """
        Calculate the pressure for each controlled lane.
        Returns a list of pressures for each lane.
        """
        pressures = []
        for lane in self.lanes:
            inflow = traci.lane.getLastStepVehicleNumber(lane)  # Incoming vehicles for the lane
            outflow = 0
            for link in traci.lane.getLinks(lane):
                outflow += traci.lane.getLastStepVehicleNumber(link[0])
            pressure = inflow - outflow
            pressures.append(pressure)

        # 获取压力最大的车道及其压力值
        max_pressure_index = np.argmax(pressures)
        max_pressure_lane = self.lanes[max_pressure_index]
        max_pressure_value = pressures[max_pressure_index]


        return max_pressure_lane



    def calculate_pedestrian_pressure(self):
        pedestrian_pressures = {lane: 0 for lane in self.lanes}
        walking_areas = self.get_walking_areas()
        heng_pressure = 0
        shu_pressure = 0

        for area in walking_areas:
            pedestrian_ids = traci.edge.getLastStepPersonIDs(area)
            for pedestrian_id in pedestrian_ids:
                destination_lane = self.determine_destination_lane(pedestrian_id)
                if destination_lane and destination_lane in pedestrian_pressures:
                    pedestrian_pressures[destination_lane] += 1
                    # Debugging output for tracing pedestrian pressure calculations


                # 获取下一个想去的边缘
                next_edge = traci.person.getNextEdge(pedestrian_id)
                if next_edge:
                    # 如果是E5_0或E7_0或-E5_0或-E7_0
                    if next_edge in ['E5_0', 'E7_0', '-E5_0', '-E7_0']:
                        # 计算竖向压力
                        shu_pressure += 1
                    else:
                        # 计算横向压力
                        heng_pressure += 1

        # 判断并打印压力更大的方向
        if shu_pressure > heng_pressure:
            return shu_pressure, "shu"
        else:
            return heng_pressure, "heng"



    def determine_destination_lane(self, pedestrian_id):
        """
        Determine the target lane for a given pedestrian based on their current and next edge.
        """
        try:
            # 获取行人当前所在的边缘
            current_edge = traci.person.getRoadID(pedestrian_id)

            # 获取行人的下一个边缘（假设行人有目标）
            next_edge = traci.person.getNextEdge(pedestrian_id)

            if not next_edge:
                # print(f"Warning: Pedestrian {pedestrian_id} does not have a next edge.")
                return None

            # 匹配当前 traffic signal 控制的车道
            for lane in self.lanes:
                lane_edge = lane.split('_')[0]  # 获取车道的边缘ID
                if lane_edge == next_edge:
                    return lane

        except traci.TraCIException as e:
             print(f"Error determining destination lane for pedestrian {pedestrian_id}: {e}")

        # 如果无法找到合适的车道，返回 None（或默认车道）
        return None

    def determine_phase_based_on_vehicle_pressure(self):
        # 计算行人压力
        pedestrian_pressure, pedestrian_direction = self.calculate_pedestrian_pressure()
        max_pressure_lane = self.calculate_vehicle_pressure()

        # 优先检查行人压力
        if pedestrian_pressure > 0:
            if pedestrian_direction == "shu" and max_pressure_lane in ['E7_1', 'E7_2', 'E5_1', 'E5_2']:
                # 行人压力是竖向且车道压力也在特定车道上
                self.reference_phase = self.full_phases[3]  # 更新参考相位为 full phases 的第 4 个相位
                print(f"Reference phase set based on pedestrian shu pressure: {self.reference_phase}")
            elif pedestrian_direction == "heng"and max_pressure_lane in ['E4_1', 'E4_2', 'E6_1', 'E6_2']:
                # 行人压力是横向
                self.reference_phase = self.full_phases[0]  # 更新参考相位为 full phases 的第 1 个相位
                print(f"Reference phase set based on pedestrian heng pressure: {self.reference_phase}")
        else:
            self.reference_phase = None  # 或者使用 self.reference_phase = '' 根据需要

            print("No significant pedestrian or vehicle pressure detected; no reference phase set.")



    def phase(self):
        return traci.trafficlight.getPhase(self.id)



    def check_pedestrians_waiting(self):
        WALKINGAREAS = [f':{self.id}_w{i}' for i in range(4)]  # 假设行人区域命名为 w0, w1, w2, w3
        pedestrian_waiting = False

        for area in WALKINGAREAS:
            pedestrians = traci.edge.getLastStepPersonIDs(area)
            for person in pedestrians:
                waiting_time = traci.person.getWaitingTime(person)
                if waiting_time > 0:  # 如果有行人等待，切换到 full phases
                    pedestrian_waiting = True
                    break
            if pedestrian_waiting:
                break

        # 根据行人等待状态切换 phase
        if pedestrian_waiting:
            self.current_phases = self.full_phases
            self.use_reduced_phases = False
        else:
            self.current_phases = self.reduced_phases
            self.use_reduced_phases = True

        #print(f"Pedestrian waiting: {pedestrian_waiting}, Using phases: {self.current_phases}")

    @property
    def time_to_act(self):
        # 打印调试信息
        print(f"Checking if it's time to act: {self.next_action_time == self.env.sim_step}, Current sim step: {self.env.sim_step}, Next action time: {self.next_action_time}")
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        self.check_pedestrians_waiting()  # 检查行人状态

        # 调用以更新参考相位，但不改变实际相位设置逻辑
        self.determine_phase_based_on_vehicle_pressure()

        # 使用现有的 current_phases 和 green_phase 来设置下一相位
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            new_phase = self.current_phases[self.green_phase % len(self.current_phases)]
            traci.trafficlight.setPhase(self.id, new_phase)
            self.is_yellow = False
            print(f"Switched to phase: {new_phase} (after yellow)")

    def set_next_phase(self, new_phase_index):
        self.check_pedestrians_waiting()

        phase_set = self.current_phases
        phase_idx = new_phase_index % len(phase_set)
        new_phase = phase_set[phase_idx]

        # 更新 next_action_time
        self.next_action_time = self.env.sim_step + self.delta_time

        # 打印调试信息
        #print(
           # f"Setting next phase to: {new_phase} for traffic signal {self.id}, next action at {self.next_action_time}")

        traci.trafficlight.setPhase(self.id, new_phase)

    def compute_observation(self):
        phase_id = [1 if self.phase() // 2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
        density = self.get_lanes_density()
        ##co2 = self.get_lanes_emission()
        queue = self.get_lanes_queue()
        # add by me
        type = self.get_lanes_emission_norm(self.lanes)  # average type is between 0 (PC) - 1 (LDV) for each lane
        observation = np.array(phase_id + density + queue + type)
        ##observation = np.array(phase_id + co2 + queue)
        return observation

    def get_walking_areas(self):
        """
        Returns a list of walking areas (edges) associated with this traffic signal.
        Assumes walking areas are named as `:{ts_id}_w0`, `:{ts_id}_w1`, etc.
        """
        return [f':{self.id}_w{i}' for i in range(4)]  # 假设有四个行人区域 w0, w1, w2, w3

    def compute_reward(self):
        # 获取当前的车辆和行人等待时间，并进行必要的缩放
        base_reward = self.co2_queue_reward()  # 基础奖励可以是现有的奖励机制

        # 检查当前相位与参考相位的一致性
        if self.reference_phase is not None and self.current_phases[
            self.green_phase % len(self.current_phases)] == self.reference_phase:
            consistency_bonus = 2.0  # 您可以根据需要调整这个奖励值
            print(f"Consistency reward given for phase match: {consistency_bonus}")
        else:
            consistency_bonus = 0.0

        self.last_reward = base_reward + consistency_bonus  # 总奖励是基础奖励加上一致性奖励
        return self.last_reward

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num())) ** 2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        # ts_wait = sum(self.get_waiting_time())
        # change by me
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0 / ts_wait
        return reward

    def _waiting_time_reward3(self):
        # ts_wait = sum(self.get_waiting_time())
        # change by me
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
                for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
                for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
                for lane in self.lanes]

    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    ## Added By me
    def change_veh_class(self):
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                traci.vehicle.setEmissionClass(veh, "HBEFA3/PC_G_EU4")  # HBEFA3/HDV_D_EU4
                a = traci.vehicle.getCO2Emission(veh)

    def get_lane_weight(self, lanes):
        weights = []
        for lane in lanes:
            count = 0
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                if traci.vehicle.getEmissionClass(veh) == "HBEFA3/HDV":
                    count += self.HDV_count
                elif traci.vehicle.getEmissionClass(veh) == "HBEFA3/LDV":
                    count += self.LDV_count
                elif traci.vehicle.getEmissionClass(veh) == "HBEFA3/Bus":
                    count += self.Bus_count
            weights.append((len(veh_list) + count) / max(1, len(veh_list)))
        return weights

    def _weighted_waiting_time_reward(self):
        weighted_wait = [a * b for a, b in zip(self.get_waiting_time_per_lane(), self.get_lane_weight(self.lanes))]
        ts_wait = sum(weighted_wait) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def get_stopped_vehicles_num(self):
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]

    def get_lanes_emission_norm(self, lanes):
        vehicle_base_max = self.vehicle_base_co2 + 500  # base value
        return [max(0, min(1, (traci.lane.getCO2Emission(lane) - self.vehicle_base_co2) / vehicle_base_max /
                           max(1, traci.lane.getLastStepVehicleNumber(lane)))) for lane in lanes]

    def get_lanes_emission(self):
        return [traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 for
                lane in self.lanes]

    def get_out_lanes_emission(self):
        return [traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 for
                lane in self.out_lanes]

    def pressure_co2_reward(self):
        # return abs(sum(self.get_lanes_emission())-sum(self.get_out_lanes_emission()))
        in_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        in_weighted_pressure = [a * (b + 1) for a, b in zip(in_pressure, self.get_lanes_emission_norm(self.lanes))]
        out_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
        out_weighted_pressure = [a * (b + 1) for a, b in
                                 zip(out_pressure, self.get_lanes_emission_norm(self.out_lanes))]
        return abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))

    def pressure_co2_reward2(self):
        new_average = abs(sum(self.get_lanes_emission()) - sum(self.get_out_lanes_emission()))
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def pressure_co2_reward3(self):
        return abs(sum(traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 for lane in self.lanes) -
                   sum(traci.lane.getLastStepVehicleNumber(lane) / self.vehicle_base_co2 for lane in self.out_lanes))

    def pressure_co2_reward4(self):
        in_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        in_weighted_pressure = [a * b for a, b in zip(in_pressure, self.get_lane_weight(self.lanes))]
        out_pressure = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
        out_weighted_pressure = [a * b for a, b in zip(out_pressure, self.get_lane_weight(self.out_lanes))]
        return abs(sum(in_weighted_pressure) - sum(out_weighted_pressure))

    def co2_reward(self):
        alpha = 0  # weighting
        new_co2 = self.get_total_emission() / self.vehicle_base_co2
        reward = self.last_measure - new_co2
        self.last_measure = new_co2
        return reward - alpha * self.get_pressure()

    def co2_queue_reward(self):
        # 获取车辆队列长度
        queue = self.get_stopped_vehicles_num()

        # 计算每条车道的CO2排放权重
        weight = [(traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 /
                   max(1, traci.lane.getLastStepVehicleNumber(lane))) for lane in self.lanes]

        # 计算加权的车辆队列长度
        weighted_queue = [a * b for a, b in zip(queue, weight)]

        # 获取行人等待时间
        pedestrian_waiting_time = self.get_pedestrian_waiting_time()

        # 将行人等待时间加入奖励的计算中
        # 这里我们可以使用一个系数 alpha 来控制行人等待时间的惩罚力度
        alpha = 1.0  # 你可以根据需求调整这个系数

        # 计算综合惩罚
        reward = -(sum(weighted_queue) ** 2) - alpha * pedestrian_waiting_time

        return reward

    def get_total_emission(self):
        return sum([traci.lane.getCO2Emission(lane) for lane in self.lanes])

    def get_mean_emission(self):
        a = sum([traci.lane.getCO2Emission(lane) for lane in self.lanes]) / 1000
        n_t = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        return a / n_t if n_t else 0

    def get_pedestrian_waiting_time(self):
        pedestrian_waiting_time = 0.0

        # Use walking areas associated with this traffic signal
        walking_areas = self.get_walking_areas()

        for area in walking_areas:
            pedestrian_ids = traci.edge.getLastStepPersonIDs(area)
            for pedestrian_id in pedestrian_ids:
                pedestrian_waiting_time += traci.person.getWaitingTime(pedestrian_id)

        return pedestrian_waiting_time

    def get_total_travel_time(self):
        # return sum([traci.lane.getTraveltime (lane) for lane in self.lanes])
        v_wa = sum(
            [traci.lane.getLastStepMeanSpeed(lane) * traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        n_t = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])
        l_t = [traci.lane.getLength(lane) for lane in self.lanes][0]  # only the ingoing lane
        return n_t * l_t / v_wa if v_wa > 1 else 0