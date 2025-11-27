"""
ev_pack_liionpack_literature.py

- PyBaMM + liionpack 기반 EV 배터리 팩(기본 4p12s, 48셀) 전기-열 시뮬레이션
- Tranter et al. (2022, JOSS) / liionpack 공식 예제 06, 07 스타일의 방법론 적용
- 단일 셀: PyBaMM "Chen2020" 파라미터 (LG M50 21700 5Ah)
- 모델: SPMe + lumped thermal ("thermal": "lumped")
- 냉각 조건: 측면 냉각판을 가정한 heat transfer coefficient(HTC) 분포
- 실험: 일정 전류 방전(예: 100 A, 600초) + 휴지
- 결과: time, 각 셀 온도/전압을 CSV로 저장
"""

import os
import numpy as np
import pandas as pd
import pybamm
import liionpack as lp

# ==============================
# 1. 기본 시뮬레이션 설정값
# ==============================

# (1) 팩 구성: Np 병렬, Ns 직렬 (기본 4p12s = 48셀)
Np = 4   # parallel cells per group (팩 용량에 영향)
Ns = 12  # series groups (팩 정격 전압에 영향)
Ncells = Np * Ns

# (2) 방전 전류와 시간 (팩 단 전류 [A], 시간 [s])
I_discharge = 100.0   # 예: 100 A 방전
t_discharge = 600.0   # 600초(10분) 방전
t_rest = 300.0        # 방전 후 300초(5분) 휴지

# (3) 출력 샘플링 간격
period_s = 10.0       # 10초마다 데이터 저장

# ==============================
# 2. netlist 생성 (팩 회로)
# ==============================

netlist = lp.setup_circuit(
    Np=Np,
    Ns=Ns,
    Rb=1e-5,   # busbar resistance [Ω]
    Ri=3e-2,   # cell internal resistance [Ω]
)
print(f"Netlist created for {Np}p{Ns}s pack, total cells = {Ncells}")

# ==============================
# 3. 실험(Experiment) 정의
# ==============================

experiment = pybamm.Experiment(
    [
        f"Discharge at {I_discharge} A for {int(t_discharge)} seconds",
        f"Rest for {int(t_rest)} seconds",
    ],
    period=f"{int(period_s)} seconds",
)

print("Experiment defined:")
print(" ", experiment)  # 간단히 전체 실험 설정만 출력

# ==============================
# 4. 파라미터: Chen2020 + HTC 입력
# ==============================

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": "[input]",
    }
)

print("Parameter values loaded: Chen2020 with HTC as input.")

# ==============================
# 5. 셀별 열전달계수(HTC) 분포
# ==============================

htc_center = 10.0   # 가운데 셀들
htc_edge = 30.0     # 측면 냉각판에 접한 셀들

htc_matrix = np.ones((Ns, Np)) * htc_center
htc_matrix[:, 0] = htc_edge
htc_matrix[:, -1] = htc_edge

htc_vector = htc_matrix.ravel()
inputs = {
    "Total heat transfer coefficient [W.m-2.K-1]": htc_vector
}

print("HTC distribution defined (edge-cooled pack).")
print(" - center HTC =", htc_center, "W/m^2K")
print(" - edge HTC   =", htc_edge, "W/m^2K")

# ==============================
# 6. 출력 변수 선택
# ==============================

output_variables = [
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]

nproc = os.cpu_count()
print(f"Using up to {nproc} processes for pack simulation.")

# ==============================
# 7. 팩 전기-열 시뮬레이션 실행
# ==============================

output = lp.solve(
    netlist=netlist,
    sim_func=lp.thermal_simulation,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    inputs=inputs,
    initial_soc=0.9,
    nproc=nproc,
)

print("Simulation finished.")

# ==============================
# 8. 결과 배열 추출
# ==============================

V_cells = output["Terminal voltage [V]"]                 # [Nt, Ncells]
T_cells = output["Volume-averaged cell temperature [K]"] # [Nt, Ncells]

Nt = V_cells.shape[0]
print(f"Number of time steps = {Nt}")

time_s = np.arange(Nt) * period_s

# ==============================
# 9. DataFrame & CSV 저장
# ==============================

data_dict = {"time_s": time_s}

for idx in range(Ncells):
    data_dict[f"T_cell_{idx+1}_K"] = T_cells[:, idx]

for idx in range(Ncells):
    data_dict[f"V_cell_{idx+1}_V"] = V_cells[:, idx]

df = pd.DataFrame(data_dict)

csv_name = "ev_pack_literature_sim_dataset.csv"
df.to_csv(csv_name, index=False, float_format="%.6f")

print(f"Saved dataset to {csv_name}, shape = {df.shape}")
