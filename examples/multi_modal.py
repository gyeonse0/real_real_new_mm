import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from FileReader import *
from SolutionPlotter import *
from RouteInitializer import *
from RouteGenerator import *
from Destroy import *
from Repair import *
from MultiModalState import *

SEED = 1234
rnd_state = np.random.RandomState(None)

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\User\OneDrive\바탕 화면\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

Rep = Repair()
Des = Destroy()
plotter = SolutionPlotter(data)

initializer = RouteInitializer(data)
initial_truck = initializer.init_truck()
plotter.plot_current_solution(initial_truck,name="Init Solution(Truck NN)")

#destroy_operators = [Des.random_removal, Des.can_drone_removal,Des.high_cost_removal] #여기에 오퍼레이터 추가
#repair_operators = [Rep.drone_first_truck_second, Rep.truck_first_drone_second, Rep.heavy_truck_repair] #여기에 오퍼레이터 추가
destroy_operators = [Des.random_removal, Des.can_drone_removal,Des.high_cost_removal] #여기에 오퍼레이터 추가
repair_operators = [Rep.drone_first_truck_second, Rep.truck_first_drone_second, Rep.heavy_truck_repair] #여기에 오퍼레이터 추가
destroy_counts = {destroyer.__name__: 0 for destroyer in destroy_operators}
repair_counts = {repairer.__name__: 0 for repairer in repair_operators}
destroy_scores = {destroyer.__name__: [] for destroyer in destroy_operators}
repair_scores = {repairer.__name__: [] for repairer in repair_operators}
drone_k_opt_count=0

destroy_probabilities = [0.33, 0.34, 0.33]  # 각각의 파괴 연산자에 대한 확률(성능기반의 score가 아니라, 확률만 고려함으로써 더욱 랜덤성 부여)
repair_probabilities = [0.6, 0.1, 0.3]  # 각각의 수리 연산자에 대한 확률
init = initial_truck
iteration_num=50000

# 초기 설정
start_temperature = 100
end_temperature = 0.01
step = 0.1

current_states = []  # 상태를 저장할 리스트
objectives = []  # 목적 함수 값을 저장할 리스트

# 초기 온도 설정
temperature = start_temperature
current_num=0

start_time = time.time()

while current_num <= iteration_num:
    if current_num==0:
        #처음 에는 init을 기반으로 수행
        selected_destroy_operator = np.random.choice(destroy_operators, p=destroy_probabilities)
        selected_repair_operator = np.random.choice(repair_operators, p=repair_probabilities)

        destroyed_state = selected_destroy_operator(init, rnd_state)
        repaired_state = selected_repair_operator(destroyed_state, rnd_state)
        
        current_states.append(repaired_state)
        objective_value = MultiModalState(repaired_state).all_time()
        objectives.append(objective_value)

        d_idx = destroy_operators.index(selected_destroy_operator)
        r_idx = repair_operators.index(selected_repair_operator)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1

    elif current_num % 100 == 0 and current_num != 0:
        
        k_opt_state = Rep.drone_k_opt(current_states[-1],rnd_state)

        if MultiModalState(current_states[-1]).all_time() > MultiModalState(k_opt_state).all_time():
            current_states.append(k_opt_state)
            objective_value = MultiModalState(k_opt_state).all_time()
            objectives.append(objective_value)
        else:
            # 이전 상태를 그대로 유지
            current_states.append(current_states[-1])
            objectives.append(MultiModalState(current_states[-1]).all_time())

        # 온도 갱신
        temperature = max(end_temperature, temperature - step)
        current_num+=1
        drone_k_opt_count+=1

    else:
        # 파괴 및 수리 연산자 확률에 따라 랜덤 선택(select)
        selected_destroy_operator = np.random.choice(destroy_operators, p=destroy_probabilities)
        selected_repair_operator = np.random.choice(repair_operators, p=repair_probabilities)
        # 선택된 연산자를 사용하여 상태 업데이트
        destroyed_state = selected_destroy_operator(current_states[-1].copy(), rnd_state)
        repaired_state = selected_repair_operator(destroyed_state, rnd_state)
        
        # 이전 objective 값과 비교하여 수락 여부 결정(accept)
        if np.exp((MultiModalState(current_states[-1]).all_time() - MultiModalState(repaired_state).all_time()) / temperature) >= rnd.random():
            current_states.append(repaired_state)
            objective_value = MultiModalState(repaired_state).all_time()
            objectives.append(objective_value)
        else:
            # 이전 상태를 그대로 유지
            current_states.append(current_states[-1])
            objectives.append(MultiModalState(current_states[-1]).all_time())

        # 온도 갱신
        temperature = max(end_temperature, temperature - step)

        #오퍼레이터 수 계산
        d_idx = destroy_operators.index(selected_destroy_operator)
        r_idx = repair_operators.index(selected_repair_operator)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1


min_objective = min(objectives)
min_index = objectives.index(min_objective)
end_time = time.time()
execution_time = end_time - start_time

print("\nBest Objective Value:",MultiModalState(current_states[min_index]).all_time())
print("Best Solution:",MultiModalState(current_states[min_index]).routes)
print("Iteration #:",min_index)
pct_diff = 100 * (MultiModalState(current_states[min_index]).all_time() - init.all_time()) / init.all_time()
print(f"This is {-(pct_diff):.1f}% better than the initial solution, which is {init.all_time()}.")

plotter.plot_current_solution(current_states[min_index])

plt.figure(figsize=(10, 6))
plt.plot(objectives, label='Current Objective')
plt.plot(np.minimum.accumulate(objectives), color='orange', linestyle='-', label='Best Objective')

plt.title('Progress of Objective Value')
plt.xlabel('Iteration(#)')
plt.ylabel('Objective Value(Kwh)')
plt.grid(True)
plt.legend()
plt.show()

print("\nDestroy Operator Counts(#):")
for name, count in destroy_counts.items():
    print(f"{name}: {count}")
print("\nRepair Operator Counts(#):")
for name, count in repair_counts.items():
    print(f"{name}: {count}")
print("\nDrone k_opt Counts(#):",drone_k_opt_count)

print("Execution time:", execution_time, "seconds")

truck_soc, drone_soc = MultiModalState(current_states[min_index]).soc()
truck_time_arrival, drone_time_arrival = MultiModalState(current_states[min_index]).new_time_arrival()

total_routes = MultiModalState(current_states[min_index]).routes.routes
truck_current_kwh = data["battery_kwh_t"]
drone_current_kwh = data["battery_kwh_d"]
for i, route in enumerate(total_routes):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # TRUCK_PATH
    truck_path = [x if route[idx][1] != ONLY_DRONE else None for idx, x in enumerate(truck_soc[i])]
    # 특정 조건을 만족하는 값의 인덱스를 필터링
    excluded_indices_truck = [i for i, value in enumerate(truck_path) if value is None]
    # None 값을 이전 값과 다음 값의 중간 값으로 대체
    for j in range(1, len(truck_path) - 1):
        if truck_path[j] is None:
            left_index = j - 1
            right_index = j + 1
            left_value = None
            right_value = None
            
            # 이전 값이 None이면서 인덱스가 리스트 범위를 벗어나지 않을 때까지 이동
            while left_index >= 0 and truck_path[left_index] is None:
                left_index -= 1
            if left_index >= 0:
                left_value = truck_path[left_index]
            
            # 다음 값이 None이면서 인덱스가 리스트 범위를 벗어나지 않을 때까지 이동
            while right_index < len(truck_path) and truck_path[right_index] is None:
                right_index += 1
            if right_index < len(truck_path):
                right_value = truck_path[right_index]
            
            # 이전 값과 다음 값이 모두 None이 아닌 경우에만 중간 값으로 대체
            if left_value is not None and right_value is not None:
                truck_path[j] = (left_value + right_value) / 2

    # Plot truck data
    ax1.plot(range(len(route)), truck_path, marker='', linestyle='-', label='Truck', color='blue')
    for iter in range(len(truck_path)):
        if iter in excluded_indices_truck:
            ax1.plot(iter, truck_path[iter], marker='', linestyle='', color='blue')
        else:
            ax1.plot(iter, truck_path[iter], marker='.', color='blue')
    
    
    # DRONE_PATH
    drone_path = [x if route[idx][1] != ONLY_TRUCK else None for idx, x in enumerate(drone_soc[i])]
    # 특정 조건을 만족하는 값의 인덱스를 필터링
    excluded_indices_drone = [i for i, value in enumerate(drone_path) if value is None]
    # None 값을 이전 값과 다음 값의 중간 값으로 대체
    for j in range(1, len(drone_path) - 1):
        if drone_path[j] is None:
            left_index = j - 1
            right_index = j + 1
            left_value = None
            right_value = None
            
            # 이전 값이 None이면서 인덱스가 리스트 범위를 벗어나지 않을 때까지 이동
            while left_index >= 0 and drone_path[left_index] is None:
                left_index -= 1
            if left_index >= 0:
                left_value = drone_path[left_index]
            
            # 다음 값이 None이면서 인덱스가 리스트 범위를 벗어나지 않을 때까지 이동
            while right_index < len(drone_path) and drone_path[right_index] is None:
                right_index += 1
            if right_index < len(drone_path):
                right_value = drone_path[right_index]
            
            # 이전 값과 다음 값이 모두 None이 아닌 경우에만 중간 값으로 대체
            if left_value is not None and right_value is not None:
                drone_path[j] = (left_value + right_value) / 2
           
    # Plot drone data
    ax1.plot(range(len(route)), drone_path, marker='', linestyle='-', label='Drone', color='red')
    for iter in range(len(drone_path)):
        if iter in excluded_indices_drone:
            ax1.plot(iter, drone_path[iter], marker='', linestyle='', color='red')
        else:
            ax1.plot(iter, drone_path[iter], marker='.', color='red')



    # 설정을 변경하여 오른쪽에 새로운 y 축
    ax2 = ax1.twinx()

    def fill_zero_values(arr):
        filled_arr = arr.copy()  # 복사하여 수정할 배열 생성
        none_indices = []  # None 값을 가진 인덱스를 저장할 배열

        # 배열의 두 번째 원소부터 끝까지 반복
        for i in range(1, len(filled_arr)):
            if filled_arr[i] == None:
                left_index = i - 1
                right_index = i + 1
                left_value = None
                right_value = None

                # 이전 값이 0인 경우에 대해 앞으로 이동하면서 0이 아닌 값을 찾음
                while left_index >= 0 and filled_arr[left_index] == None:
                    left_index -= 1
                if left_index >= 0:
                    left_value = filled_arr[left_index]

                # 다음 값이 0인 경우에 대해 뒤로 이동하면서 0이 아닌 값을 찾음
                while right_index < len(filled_arr) and filled_arr[right_index] == None:
                    right_index += 1
                if right_index < len(filled_arr):
                    right_value = filled_arr[right_index]

                # 앞뒤 값이 모두 유효한 경우에만 평균으로 대체
                if left_value is not None and right_value is not None:
                    # 앞과 뒤의 차이를 구하여 스텝을 계산
                    step = (right_value - left_value) / (right_index - left_index)
                    # 평균 값으로 대체
                    filled_arr[i] = left_value + step * (i - left_index)
                else:
                    none_indices.append(i)

        return filled_arr
    
    filled_truck_time_arrival = [fill_zero_values(arrival) for arrival in truck_time_arrival]
    filled_drone_time_arrival = [fill_zero_values(arrival) for arrival in drone_time_arrival]


    ax2.plot(range(len(route)), truck_time_arrival[i],  marker='.', linestyle='-', label='Truck Time', color='green')
    ax2.plot(range(len(route)), drone_time_arrival[i],  marker='.', linestyle='-', label='Drone Time', color='orange')
    ax2.plot(range(len(route)), filled_truck_time_arrival[i], linestyle='-', color='green')
    ax2.plot(range(len(route)), filled_drone_time_arrival[i], linestyle='-', color='orange')


    ax1.set_xlabel('Customer')
    ax1.set_ylabel('SOC (%)')
    ax2.set_ylabel('Time (m)')

    ax1.text(-0.02, -0.13, f'Truck Energy Consumption: {MultiModalState(current_states[min_index]).truck_consumtion()}kwh\nDrone Energy Consumption: {MultiModalState(current_states[min_index]).drone_consumption()}kwh\nTruck Charging to Drone: {0.0}kwh', fontsize=12, ha='left', va='bottom', bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round'), transform=ax1.transAxes)

    plt.title(f"Route {i+1} OFV Graph")
    plt.grid(True)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')

    # Set x ticks to show all customers
    plt.xticks(range(len(route)), [customer[0] for customer in route])
    
    plt.show()