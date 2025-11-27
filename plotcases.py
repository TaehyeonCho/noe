import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cases = [
    ("ev_pack_base_100A_normalcool.csv",  "Base (100A, normal cooling)"),
    ("ev_pack_caseA_150A_weakcool.csv",   "Case A (150A, weak cooling)"),
    ("ev_pack_caseB_100A_strongcool.csv", "Case B (100A, strong cooling)"),
]

Tcrit_C = 90.0

# 1) 최대 온도 vs 시간
plt.figure()
for fname, label in cases:
    df = pd.read_csv(fname)
    time = df["time_s"].values

    temp_cols = [c for c in df.columns if c.startswith("T_cell_")]
    temps_K = df[temp_cols].values
    T_max = temps_K.max(axis=1)
    T_max_C = T_max - 273.15

    plt.plot(time, T_max_C, label=label)

plt.axhline(Tcrit_C, linestyle="--")  # 임계 온도 선 (예: 90℃)
plt.xlabel("Time [s]")
plt.ylabel("Max cell temperature [°C]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_max_temp_vs_time.png", dpi=300)
print("Saved fig_max_temp_vs_time.png")

# 2) 셀 간 온도차 ΔT vs 시간
plt.figure()
for fname, label in cases:
    df = pd.read_csv(fname)
    time = df["time_s"].values

    temp_cols = [c for c in df.columns if c.startswith("T_cell_")]
    temps_K = df[temp_cols].values
    T_max = temps_K.max(axis=1)
    T_min = temps_K.min(axis=1)
    delta_T = T_max - T_min

    plt.plot(time, delta_T, label=label)

plt.xlabel("Time [s]")
plt.ylabel("Temperature difference ΔT [K]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_deltaT_vs_time.png", dpi=300)
print("Saved fig_deltaT_vs_time.png")
