import os
import numpy as np
import pandas as pd
import pybamm
import liionpack as lp

# ===== 1. 팩 구성 =====
Np = 4
Ns = 12
Ncells = Np * Ns

# 안전 시나리오: 전류는 100A 그대로, 냉각 강화
I_discharge = 100.0
t_discharge = 600.0
t_rest = 300.0
period_s = 10.0

netlist = lp.setup_circuit(
    Np=Np,
    Ns=Ns,
    Rb=1e-5,
    Ri=3e-2,
)
print(f"[Case B] Netlist: {Np}p{Ns}s, {Ncells} cells, 100A discharge")

experiment = pybamm.Experiment(
    [
        f"Discharge at {I_discharge} A for {int(t_discharge)} seconds",
        f"Rest for {int(t_rest)} seconds",
    ],
    period=f"{int(period_s)} seconds",
)

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {"Total heat transfer coefficient [W.m-2.K-1]": "[input]"}
)

# 냉각 강화: 전체 HTC 크게
htc_center = 20.0
htc_edge = 40.0

htc_matrix = np.ones((Ns, Np)) * htc_center
htc_matrix[:, 0] = htc_edge
htc_matrix[:, -1] = htc_edge

htc_vector = htc_matrix.ravel()
inputs = {"Total heat transfer coefficient [W.m-2.K-1]": htc_vector}

output_variables = [
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]

nproc = os.cpu_count()
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

V_cells = output["Terminal voltage [V]"]
T_cells = output["Volume-averaged cell temperature [K]"]
Nt = V_cells.shape[0]
time_s = np.arange(Nt) * period_s

data = {"time_s": time_s}
for i in range(Ncells):
    data[f"T_cell_{i+1}_K"] = T_cells[:, i]
for i in range(Ncells):
    data[f"V_cell_{i+1}_V"] = V_cells[:, i]

df = pd.DataFrame(data)
csv_name = "ev_pack_caseB_strongcool_100A.csv"
df.to_csv(csv_name, index=False, float_format="%.6f")
print(f"[Case B] Saved to {csv_name}, shape = {df.shape}")
