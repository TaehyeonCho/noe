"""
makeEvPackHeatmap.py

- ev_pack_*.csv 3개를 읽어서
  1) 마지막 시점 / 특정 시점의 셀 온도를 Ns x Np 행렬로 reshape
  2) heatmap 이미지를 PNG로 저장
  3) 케이스별 Tmax, ΔT(최대 온도차) 등을 계산해서 요약 출력

※ 전제:
  - CSV 컬럼 구조: time_s, T_cell_1_K ... T_cell_48_K, V_cell_1_V ... V_cell_48_V
  - 팩 구성: 4p12s → Np = 4, Ns = 12, Ncells = 48
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 공통 설정 ----------
Np = 4
Ns = 12
Ncells = Np * Ns

# TR 온셋 임계 온도 (문헌 참고값, 예: 90 ℃)
TR_ONSET_C = 90.0
TR_ONSET_K = TR_ONSET_C + 273.15

# 그림 저장 폴더
OUT_DIR = "fig_evpack"
os.makedirs(OUT_DIR, exist_ok=True)

# 케이스 정보 (evPackCases.py에서 썼던 이름과 맞추기)
CASE_FILES = {
    "base_100A_normalcool": "ev_pack_base_100A_normalcool.csv",
    "caseA_150A_weakcool": "ev_pack_caseA_150A_weakcool.csv",
    "caseB_100A_strongcool": "ev_pack_caseB_100A_strongcool.csv",
}

# heatmap을 그리고 싶은 시간(sec) 리스트
SNAP_TIMES = [0.0, 30.0, 150.0, 600.0]  # 필요하면 바꿔도 됨


def load_case_df(fname: str) -> pd.DataFrame:
    """CSV 하나 읽어오기"""
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} 파일을 찾을 수 없습니다.")
    df = pd.read_csv(fname)
    return df


def get_temperature_matrix(df: pd.DataFrame, idx: int) -> np.ndarray:
    """
    특정 time index의 셀 온도(48개)를 Ns x Np 행렬로 reshape
    idx: df 행 인덱스 (0 ~ Nt-1)
    """
    # 온도 컬럼만 추출
    t_cols = [f"T_cell_{i+1}_K" for i in range(Ncells)]
    T_vec = df.loc[idx, t_cols].values  # shape: (48,)
    T_mat = T_vec.reshape(Ns, Np)       # shape: (12,4)
    return T_mat


def find_nearest_index(df: pd.DataFrame, target_time: float) -> int:
    """target_time과 가장 가까운 time_s 인덱스를 찾는다."""
    times = df["time_s"].values
    idx = int(np.argmin(np.abs(times - target_time)))
    return idx


def plot_heatmap(T_mat: np.ndarray, time_s: float, case_name: str, fname: str):
    """Ns x Np 온도 행렬을 heatmap으로 저장"""
    # K → ℃
    T_c = T_mat - 273.15

    plt.figure(figsize=(4, 6))  # 세로 길게 (Ns=12, Np=4)
    im = plt.imshow(T_c, origin="lower", aspect="auto",
                    cmap="inferno")  # colormap은 취향대로 변경 가능
    plt.colorbar(im, label="Temperature [°C]")
    plt.title(f"{case_name}  -  t = {time_s:.0f} s")
    plt.xlabel("Parallel index (Np)")
    plt.ylabel("Series index (Ns)")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Heatmap saved: {fname}")


def analyze_case(case_name: str, csv_name: str):
    """한 케이스에 대해 heatmap + Tmax/ΔT/마진 계산"""
    print(f"\n===== Case: {case_name} =====")
    df = load_case_df(csv_name)

    times = df["time_s"].values
    Nt = len(times)

    # 모든 시점의 온도 배열 (Nt, Ncells)
    t_cols = [f"T_cell_{i+1}_K" for i in range(Ncells)]
    T_all = df[t_cols].values  # (Nt, 48)

    # Tmax(t), Tmin(t), ΔT(t) 계산
    Tmax_t = T_all.max(axis=1)   # (Nt,)
    Tmin_t = T_all.min(axis=1)
    dT_t = Tmax_t - Tmin_t

    # 전체 시뮬 동안의 최대값들
    Tmax_max_K = float(Tmax_t.max())
    Tmax_max_C = Tmax_max_K - 273.15
    dT_max_K = float(dT_t.max())

    # TR 마진 (Tonset - Tmax_max)
    margin_C = TR_ONSET_K - Tmax_max_K  # 수치상 °C와 동일
    # Tmax가 언제 발생했는지도 한번 저장
    idx_Tmax = int(np.argmax(Tmax_t))
    t_at_Tmax = float(times[idx_Tmax])

    print(f"  Tmax_max = {Tmax_max_C:.2f} °C (t = {t_at_Tmax:.1f} s)")
    print(f"  ΔT_max  = {dT_max_K:.2f} K")
    print(f"  Margin to TR onset (90°C 기준) = {margin_C:.2f} °C")

    # --- heatmap 그리기 ---
    # 1) 마지막 시점
    last_idx = Nt - 1
    T_last = get_temperature_matrix(df, last_idx)
    fname_last = os.path.join(
        OUT_DIR, f"heatmap_{case_name}_t{int(times[last_idx])}s.png"
    )
    plot_heatmap(T_last, times[last_idx], case_name, fname_last)

    # 2) SNAP_TIMES 리스트에 있는 시점들
    for t_target in SNAP_TIMES:
        idx = find_nearest_index(df, t_target)
        T_mat = get_temperature_matrix(df, idx)
        fname = os.path.join(
            OUT_DIR,
            f"heatmap_{case_name}_t{int(times[idx])}s.png",
        )
        plot_heatmap(T_mat, times[idx], case_name, fname)

    # 요약값 리턴 (나중에 표 만들 때 사용)
    summary = {
        "case": case_name,
        "Tmax_max_C": Tmax_max_C,
        "dT_max_K": dT_max_K,
        "Margin_to_TR_onset_C": margin_C,
        "t_at_Tmax_s": t_at_Tmax,
    }
    return summary


if __name__ == "__main__":
    summaries = []

    for case_name, csv_name in CASE_FILES.items():
        summary = analyze_case(case_name, csv_name)
        summaries.append(summary)

    # 케이스 요약 표를 CSV로 저장
    df_sum = pd.DataFrame(summaries)
    out_csv = "ev_pack_case_summary.csv"
    df_sum.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"\nSummary table saved: {out_csv}")
    print(df_sum)
