import pandas as pd
import numpy as np

df = pd.read_csv("ev_pack_literature_sim_dataset.csv")

time = df["time_s"].values

# 온도/전압 컬럼만 따로 추출
temp_cols = [c for c in df.columns if c.startswith("T_cell_")]
volt_cols = [c for c in df.columns if c.startswith("V_cell_")]

temps_K = df[temp_cols].values    # shape (Nt, Ncells)
volts_V = df[volt_cols].values    # shape (Nt, Ncells)

# 1) 각 시간에서 최대/최소 온도, 온도차
T_max = temps_K.max(axis=1)
T_min = temps_K.min(axis=1)
delta_T = T_max - T_min

# 2) 최대 온도가 언제, 몇 ℃까지 올라갔는지
idx_peak = T_max.argmax()
t_peak = time[idx_peak]
T_peak_K = T_max[idx_peak]
T_peak_C = T_peak_K - 273.15

print(f"최대 온도: {T_peak_C:.2f} ℃ (t = {t_peak} s)")

print(f"마지막 시점 온도 범위: "
      f"{T_min[-1]-273.15:.2f} ℃ ~ {T_max[-1]-273.15:.2f} ℃")
print(f"마지막 시점 셀 간 온도차: {delta_T[-1]:.2f} K")

# 3) 가장 뜨거운 셀 번호 (마지막 시점 기준)
last_temps = temps_K[-1, :]
hottest_cell_idx = last_temps.argmax()  # 0-based index
print(f"마지막 시점 기준 hotspot 셀 번호(1부터): {hottest_cell_idx+1}")
