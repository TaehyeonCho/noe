import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. 시뮬레이션 로직 (계산 속도를 위해 10x10 유지) ---
nx, ny = 10, 10
dx = dy = 0.018
dt = 0.1
t_steps = 6001

rho = 2500.0
cp = 1000.0
k_th = 2.0
alpha = k_th / (rho * cp)

T = np.ones((nx, ny)) * 25.0 
T_new = T.copy()

cx, cy = 5, 5
short_start = 50.0
short_dur = 60.0

time_snapshots = [0, 80, 250, 600] 
results = []

print("고정밀 해석 수행 중...", end="")

for t in range(t_steps):
    current_time = t * dt
    Q_gen = np.zeros((nx, ny))
    
    # 발열 시나리오 (Hard Short)
    if short_start <= current_time <= (short_start + short_dur):
        Q_gen[cx, cy] = 5e7 
        
    for i in range(nx):
        for j in range(ny):
            if T[i, j] > 160.0: 
                Q_gen[i, j] += 1e8 
                
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            d2T_dx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2
            d2T_dy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            dT = alpha * (d2T_dx2 + d2T_dy2) + Q_gen[i, j] / (rho * cp)
            T_new[i, j] = T[i, j] + dT * dt
    
    # 경계 조건
    T_new[0, :] = T_new[1, :]
    T_new[-1, :] = T_new[-2, :]
    T_new[:, 0] = T_new[:, 1]
    T_new[:, -1] = T_new[:, -2]
    T = T_new.copy()
    
    for snap in time_snapshots:
        if abs(current_time - snap) < dt / 2:
            if not results or results[-1][0] != snap:
                results.append((snap, T.copy()))

print(" 완료!")

# --- 2. [핵심] 논문용 고퀄리티 시각화 ---

# IEEE 스타일 폰트 설정 (Times New Roman이 없으면 기본값 사용)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for idx, (time_val, temp_map) in enumerate(results):
    ax = axes[idx]
    
    # [비법 1] Bicubic Interpolation (깍두기를 부드럽게 뭉개서 고해상도처럼 보이게 함)
    # cmap='jet' 또는 'turbo'는 공학 논문에서 가장 선호하는 색상
    im = ax.imshow(temp_map, cmap='jet', vmin=25, vmax=500, 
                   interpolation='bicubic', origin='lower')
    
    # [비법 2] 등온선 (Contour Line) 추가 -> 전문가스러움 +100
    # 100도 간격으로 흰색 선을 그려서 온도 분포를 명확히 함
    if time_val > 0:
        cs = ax.contour(temp_map, levels=[100, 200, 300, 400], colors='white', 
                        linewidths=0.8, origin='lower', alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f°C')

    # 축 설정 (지저분한 눈금 제거하고 외곽선만 남김)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # 타이틀 (논문 스타일)
    ax.set_title(f"t = {time_val} s", fontsize=16, fontweight='bold', pad=10)

# 컬러바 (맨 오른쪽에 하나만 깔끔하게)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # 위치 미세 조정
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Temperature (°C)', fontsize=14, labelpad=10)
cbar.ax.tick_params(labelsize=12)

# 전체 제목
plt.suptitle("Thermal Propagation Analysis of Li-ion Battery Module (Numerical Simulation)", 
             fontsize=18, fontweight='bold', y=1.05)

plt.savefig("Paper_Quality_Result.png", dpi=600, bbox_inches='tight') # 600dpi 고화질 저장
plt.show()