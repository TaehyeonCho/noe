"""
ev_pack_thermal_liionpack.py

- PyBaMM + liionpack 기반 EV 유사 배터리 모듈(4p12s, 48셀) 열-전기 시뮬레이션
- Chen2020 LG M50 21700 5Ah 셀 파라미터 사용
- 비균일 냉각(측면 냉각판 부근 셀의 열전달계수 증가) 반영
- 결과를 CSV 데이터셋(ev_module_thermal_dataset.csv)으로 저장
"""

import os
import numpy as np
import pandas as pd
import pybamm
import liionpack as lp


# ============================================================
# 1. 팩 구성: 4p12s → 총 48셀
# ============================================================
# Np: 병렬 개수(같은 전압 노드에 붙어 있는 셀 개수)
# Ns: 직렬 개수(전기차 모듈에서 직렬로 쌓인 그룹 개수)
Np = 4   # parallel cells per group
Ns = 12  # series groups
Ncells = Np * Ns

print(f"Pack configuration: {Np}p{Ns}s, total cells = {Ncells}")

# ============================================================
# 2. 회로(netlist) 생성
# ============================================================
# liionpack 기본 예제와 비슷한 수준의 버스바/내부 저항 값 사용
#   Rb: busbar resistance
#   Ri: internal cell resistance
netlist = lp.setup_circuit(
    Np=Np,
    Ns=Ns,
    Rb=1e-5,   # [Ω]
    Ri=3e-2,   # [Ω]
)
print("Netlist created.")


# ============================================================
# 3. 실험 조건: EV 주행을 단순화한 전류 패턴
# ============================================================
# - 단위: 전류는 [A], 시간은 [seconds]
# - period는 liionpack이 결과를 샘플링하는 간격(고정 간격)입니다.
period_s = 10.0  # 10초마다 샘플링

experiment = pybamm.Experiment(
    [
        # 1) 항속 주행(중간 부하)
        "Discharge at 60 A for 600 seconds",   # 10분

        # 2) 가속/언덕 구간(고부하)
        "Discharge at 120 A for 60 seconds",   # 1분

        # 3) 정차 (신호 대기 등)
        "Rest for 120 seconds",               # 2분

        # 4) 회생 제동 (충전)
        "Charge at 40 A for 30 seconds",      # 30초

        # 5) 다시 정차
        "Rest for 120 seconds",               # 2분
    ]
    * 2,  # 위 패턴을 2회 반복
    period=f"{int(period_s)} seconds",
)

print("Experiment (drive-like current profile) defined.")


# ============================================================
# 4. 파라미터: Chen2020 + 열전달계수 입력으로 사용
# ============================================================
# Chen2020: LG M50 21700 5Ah 셀에 대한 검증된 파라미터셋
parameter_values = pybamm.ParameterValues("Chen2020")

# 열전달계수(총 열전달 계수)를 입력값으로 바꾸어,
# 각 셀마다 다른 냉각 조건을 줄 수 있게 설정
parameter_values.update(
    {"Total heat transfer coefficient [W.m-2.K-1]": "[input]"}
)

print("Parameter values (Chen2020) loaded and HTC set as input.")


# ============================================================
# 5. 냉각 조건: 측면 냉각판이 있는 2D 모듈을 가정한 HTC 분포
# ============================================================
# 셀 배치: Ns x Np 2차원 배열로 생각 (행: 직렬 방향, 열: 병렬 방향)
# 여기서는 모듈 양쪽 측면(열 방향의 첫/마지막 열)에 냉각판이 붙어 있다고 가정
# → 가장 바깥 열의 열전달계수(htc)가 더 크게 설정됨.

# 기본적으로 전체는 15 W/m^2K 정도로 설정(공랭 + 약한 접촉)
htc_matrix = np.ones((Ns, Np)) * 15.0

# 양측 열(열 0, 열 Np-1)은 냉각판과 직접 접촉 → 강한 냉각
htc_matrix[:, 0] = 30.0   # 왼쪽 열
htc_matrix[:, -1] = 30.0  # 오른쪽 열

# 필요하면 가운데 열은 조금 약하게 (예: 10 W/m^2K)
if Np > 2:
    htc_matrix[:, 1:-1] = 15.0

# liionpack 입력 형식에 맞게 1D로 펼치기 (Ns*Np 길이)
htc = htc_matrix.ravel()
inputs = {"Total heat transfer coefficient [W.m-2.K-1]": htc}

print("HTC distribution per cell defined (edge-cooled module).")


# ============================================================
# 6. 출력으로 받을 변수 목록 정의
# ============================================================
# - 각 셀 단자 전압
# - 각 셀 체적 평균 온도
output_variables = [
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]

# CPU 코어 수에 맞춰 병렬 계산
nproc = os.cpu_count()
print(f"Using up to {nproc} processes.")


# ============================================================
# 7. 팩 해석: 열-전기 일체 시뮬레이션
# ============================================================
output = lp.solve(
    netlist=netlist,
    sim_func=lp.thermal_simulation,   # SPMe + lumped thermal model 사용
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    inputs=inputs,
    initial_soc=0.9,                  # 초기 SOC 90% 가정
    nproc=nproc,
)

print("Simulation finished.")


# ============================================================
# 8. 결과 배열 추출
#    - V_cells: (Nt, Ncells) 각 시간, 각 셀의 전압
#    - T_cells: (Nt, Ncells) 각 시간, 각 셀의 온도
# ============================================================
V_cells = output["Terminal voltage [V]"]              # [Nt, Ncells]
T_cells = output["Volume-averaged cell temperature [K]"]  # [Nt, Ncells]

Nt = V_cells.shape[0]
print(f"Number of time steps: {Nt}")

# liionpack은 Experiment의 period에 맞추어 값을 저장하므로,
# 시간 벡터는 아래와 같이 계산해도 일관성이 있습니다.
time_s = np.arange(Nt) * period_s


# ============================================================
# 9. 논문용 데이터셋(DataFrame) 구성 및 CSV 저장
#    컬럼 구조:
#    - time_s
#    - T_cell_1_K, ..., T_cell_N_K
#    - V_cell_1_V, ..., V_cell_N_V
# ============================================================
data_dict = {"time_s": time_s}

# 온도 컬럼들 추가
for idx in range(Ncells):
    col_name = f"T_cell_{idx+1}_K"
    data_dict[col_name] = T_cells[:, idx]

# 전압 컬럼들 추가
for idx in range(Ncells):
    col_name = f"V_cell_{idx+1}_V"
    data_dict[col_name] = V_cells[:, idx]

df = pd.DataFrame(data_dict)
csv_name = "ev_module_thermal_dataset.csv"
df.to_csv(csv_name, index=False, float_format="%.6f")

print(f"Saved dataset to {csv_name}, shape = {df.shape}")
