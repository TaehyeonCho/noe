"""
evPackCases.py

- PyBaMM + liionpack을 이용해 EV 배터리 팩(4p12s, 48셀) 전기-열 시뮬레이션
- Tranter et al. (2022, JOSS) / liionpack 공식 예제 스타일
- Chen2020 파라미터(LG M50 21700 5Ah) + SPMe + lumped thermal
- 세 가지 케이스를 한 번에 계산해서 CSV로 저장
  1) base_100A_normalcool   : 100A, 보통 냉각
  2) caseA_150A_weakcool    : 150A, 약한 냉각 (위험 케이스)
  3) caseB_100A_strongcool  : 100A, 강한 냉각 (안전 케이스)
"""

import os
import numpy as np
import pandas as pd
import pybamm
import liionpack as lp

# ------------------------------
# 공통 팩 설정
# ------------------------------
Np = 4   # 병렬
Ns = 12  # 직렬
Ncells = Np * Ns
period_s = 10.0  # 결과 샘플링 간격 [s]

print(f"Pack config: {Np}p{Ns}s, total cells = {Ncells}")

# 공통 netlist 생성
netlist = lp.setup_circuit(
    Np=Np,
    Ns=Ns,
    Rb=1e-5,   # busbar resistance [Ω]
    Ri=3e-2,   # cell internal resistance [Ω]
)

# 공통 파라미터 (Chen2020 + HTC를 input으로 바꾸기)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {"Total heat transfer coefficient [W.m-2.K-1]": "[input]"}
)

# 출력 변수 (전압 + 온도)
output_variables = [
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]

nproc = os.cpu_count()
print(f"Using up to {nproc} processes.")

# ------------------------------
# 시뮬레이션 케이스 목록 정의
# ------------------------------
cases = [
    {
        "name": "base_100A_normalcool",
        "I_discharge": 100.0,
        "t_discharge": 600.0,
        "t_rest": 300.0,
        "htc_center": 10.0,
        "htc_edge": 30.0,
    },
    {
        "name": "caseA_150A_weakcool",   # 위험 케이스: 고전류 + 약한 냉각
        "I_discharge": 150.0,
        "t_discharge": 600.0,
        "t_rest": 300.0,
        "htc_center": 5.0,
        "htc_edge": 10.0,
    },
    {
        "name": "caseB_100A_strongcool", # 안전 케이스: 강한 냉각
        "I_discharge": 100.0,
        "t_discharge": 600.0,
        "t_rest": 300.0,
        "htc_center": 20.0,
        "htc_edge": 40.0,
    },
]

# ------------------------------
# 각 케이스별 시뮬레이션 실행 함수
# ------------------------------
def run_case(case):
    name = case["name"]
    I_dis = case["I_discharge"]
    t_dis = case["t_discharge"]
    t_rest = case["t_rest"]
    htc_center = case["htc_center"]
    htc_edge = case["htc_edge"]

    print("\n====================================")
    print(f"Running case: {name}")
    print(f"  I_discharge = {I_dis} A")
    print(f"  t_discharge = {t_dis} s, t_rest = {t_rest} s")
    print(f"  HTC center = {htc_center} W/m^2K, edge = {htc_edge} W/m^2K")
    print("====================================")

    # 1) Experiment 정의
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_dis} A for {int(t_dis)} seconds",
            f"Rest for {int(t_rest)} seconds",
        ],
        period=f"{int(period_s)} seconds",
    )

    # 2) HTC 분포 (Ns x Np 매트릭스 → ravel)
    htc_matrix = np.ones((Ns, Np)) * htc_center
    htc_matrix[:, 0] = htc_edge
    htc_matrix[:, -1] = htc_edge
    htc_vector = htc_matrix.ravel()

    inputs = {
        "Total heat transfer coefficient [W.m-2.K-1]": htc_vector
    }

    # 3) 팩 전기-열 시뮬레이션 실행
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

    print("  Simulation finished.")

    # 4) 결과 추출
    V_cells = output["Terminal voltage [V]"]                 # [Nt, Ncells]
    T_cells = output["Volume-averaged cell temperature [K]"] # [Nt, Ncells]

    Nt = V_cells.shape[0]
    time_s = np.arange(Nt) * period_s

    # 5) DataFrame & CSV 저장
    data = {"time_s": time_s}
    for idx in range(Ncells):
        data[f"T_cell_{idx+1}_K"] = T_cells[:, idx]
    for idx in range(Ncells):
        data[f"V_cell_{idx+1}_V"] = V_cells[:, idx]

    df = pd.DataFrame(data)
    csv_name = f"ev_pack_{name}.csv"
    df.to_csv(csv_name, index=False, float_format="%.6f")

    print(f"  Saved: {csv_name}, shape = {df.shape}")


# ------------------------------
# 메인: 모든 케이스 실행
# ------------------------------
if __name__ == "__main__":
    for case in cases:
        run_case(case)

    print("\nAll cases finished.")
0