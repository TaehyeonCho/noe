import pandas as pd
import numpy as np

def summarize_case(csv_path, name):
    df = pd.read_csv(csv_path)
    time = df["time_s"].values
    temp_cols = [c for c in df.columns if c.startswith("T_cell_")]
    temps = df[temp_cols].values

    T_max = temps.max(axis=1)
    T_min = temps.min(axis=1)
    delta_T = T_max - T_min

    idx_peak = T_max.argmax()
    T_peak_C = T_max[idx_peak] - 273.15
    t_peak = time[idx_peak]

    print(f"[{name}] 최대 온도: {T_peak_C:.2f} ℃ (t = {t_peak} s)")
    print(f"[{name}] 마지막 시점 온도 범위: "
          f"{T_min[-1]-273.15:.2f} ℃ ~ {T_max[-1]-273.15:.2f} ℃")
    print(f"[{name}] 마지막 시점 셀 간 온도차: {delta_T[-1]:.2f} K")
    print("-"*50)

summarize_case("ev_pack_literature_sim_dataset.csv", "Base")
summarize_case("ev_pack_caseA_weakcool_150A.csv", "CaseA_WeakCool_150A")
summarize_case("ev_pack_caseB_strongcool_100A.csv", "CaseB_StrongCool_100A")
