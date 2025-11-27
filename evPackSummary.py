"""
evPackSummary.py

- ev_pack_base_100A_normalcool.csv
- ev_pack_caseA_150A_weakcool.csv
- ev_pack_caseB_100A_strongcool.csv

세 개 케이스에 대해
  * Tmax (최대 셀 온도, ℃)
  * Tmax 발생 시간
  * Tmax 발생 셀 번호 / (Ns, Np) 위치
  * ΔT_last (마지막 시점 셀 간 온도차)
  * ΔT_max  (시뮬레이션 전체에서의 최대 셀 간 온도차)
  * 안전 한계 온도 대비 마진 (예: T_limit = 90 ℃ 기준)

을 계산해서 summary CSV로 저장.
"""

import numpy as np
import pandas as pd

# -------------------------
# 팩 구성 (우리 시뮬레이션 설정과 동일)
# -------------------------
Ns = 12  # 직렬 그룹 개수 (heatmap 세로축)
Np = 4   # 병렬 셀 개수 (heatmap 가로축)
Ncells = Ns * Np

# -------------------------
# 케이스별 파일 이름 정의
# -------------------------
cases = [
    {
        "name": "base_100A_normalcool",
        "csv": "ev_pack_base_100A_normalcool.csv",
    },
    {
        "name": "caseA_150A_weakcool",
        "csv": "ev_pack_caseA_150A_weakcool.csv",
    },
    {
        "name": "caseB_100A_strongcool",
        "csv": "ev_pack_caseB_100A_strongcool.csv",
    },
]

# TR/안전 한계 온도 (임시로 90 ℃ 사용)
# -> 나중에 열폭주/안전 기준 논문에서 가져온 실제 값으로 바꿔 넣으시면 됩니다.
T_limit_C = 90.0

summary_list = []

for case in cases:
    name = case["name"]
    csv_path = case["csv"]
    print(f"\n=== {name} ===")

    # -------------------------
    # 1) 데이터 로드
    # -------------------------
    df = pd.read_csv(csv_path)

    # 시간 벡터 [s]
    time_s = df["time_s"].values  # shape (Nt,)

    # 온도 컬럼만 뽑기 (Kelvin -> Celsius)
    T_cols = [c for c in df.columns if c.startswith("T_cell_")]
    T_K = df[T_cols].values       # shape (Nt, Ncells)
    T_C = T_K - 273.15            # ℃로 변환

    # -------------------------
    # 2) Tmax 및 그 시점/위치
    # -------------------------
    Tmax_C = T_C.max()                      # 전체 셀·전체 시간 중 최대 온도
    flat_index = T_C.argmax()               # 0 ~ Nt*Ncells-1
    t_index, cell_index = divmod(flat_index, Ncells)

    t_Tmax_s = time_s[t_index]             # Tmax 발생 시간 [s]

    # 셀 번호(1부터)와 (Ns, Np) 좌표
    cell_number = cell_index + 1           # 1 ~ 48
    ns_index = cell_index // Np            # 0 ~ Ns-1
    np_index = cell_index % Np             # 0 ~ Np-1

    # -------------------------
    # 3) ΔT_last: 마지막 시점 셀간 온도차
    # -------------------------
    T_last = T_C[-1, :]                     # 마지막 시점 모든 셀
    dT_last = T_last.max() - T_last.min()   # [K = ℃]

    # -------------------------
    # 4) ΔT_max: 전체 시간 중 최대 셀간 온도차
    # -------------------------
    dT_all = T_C.max(axis=1) - T_C.min(axis=1)  # 각 시간별 (max - min)
    dT_max = dT_all.max()

    # -------------------------
    # 5) 안전 한계 대비 마진
    # -------------------------
    margin_C = T_limit_C - Tmax_C  # 양수면 여유, 음수면 초과

    # -------------------------
    # 6) 요약 리스트에 저장
    # -------------------------
    summary_list.append(
        {
            "case": name,
            "Tmax_C": Tmax_C,
            "t_Tmax_s": t_Tmax_s,
            "T_limit_C": T_limit_C,
            "margin_to_limit_C": margin_C,
            "cell_number": cell_number,
            "ns_index(0-based)": ns_index,
            "np_index(0-based)": np_index,
            "dT_last_K": dT_last,
            "dT_max_K": dT_max,
        }
    )

    # 터미널에 간단 출력
    print(f"  Tmax = {Tmax_C:.2f} ℃ at t = {t_Tmax_s:.1f} s")
    print(f"  Hotspot cell = #{cell_number} (Ns={ns_index}, Np={np_index})")
    print(f"  ΔT_last = {dT_last:.2f} K, ΔT_max = {dT_max:.2f} K")
    print(f"  Margin to {T_limit_C:.1f} ℃ = {margin_C:.2f} ℃")

# -------------------------
# 최종 요약표 CSV로 저장
# -------------------------
summary_df = pd.DataFrame(summary_list)
summary_csv = "ev_pack_case_summary.csv"
summary_df.to_csv(summary_csv, index=False, float_format="%.3f")

print(f"\nSaved summary table -> {summary_csv}")
print(summary_df)
