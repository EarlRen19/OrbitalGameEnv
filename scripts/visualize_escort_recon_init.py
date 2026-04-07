"""
护卫侦照场景初始构型可视化
三星场景：红色高价值星(Red HV) + 红色护卫星(Red Escort) + 蓝色侦照星(Blue Recon)

输入：六根数 a[km], e[-], i[rad], raan[rad], peri[rad], M[rad]
太阳角定义与 cpp solar_illumination_angle 一致：
    目标(红HV)→太阳 与 目标(红HV)→追击者(蓝) 的夹角

默认初始时间：BJT 2027-09-01 20:00 = UTC 2027-09-01 12:00:00
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import argparse
from datetime import datetime, timezone, timedelta

MU = 398600.4418   # km^3/s^2  (与 cpp constants.h 一致)
AU = 149597870.691  # km        (与 cpp constants.h 一致)
RE = 6371.0         # km

# BJT = UTC+8，默认时间
DEFAULT_BJT = datetime(2027, 9, 1, 20, 0, 0, tzinfo=timezone(timedelta(hours=8)))
DEFAULT_UTC = DEFAULT_BJT.astimezone(timezone.utc)

# 默认六根数（a 单位 km，角度单位 rad）
DEFAULT_RED_HV  = [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.419833]
DEFAULT_RED_ESC = [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.421133]
DEFAULT_BLUE    = [42169.502913, 0.0, 0.002287, 1.592829, 0.0, 0.424435]


# ─── 轨道力学（与 cpp utils.cpp 算法完全对应） ────────────────────────────────

def ma2ta(ma: float, ecc: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """平近点角 → 真近点角（对应 cpp ma2ta）"""
    E = ma if ecc < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (ma - E + ecc * np.sin(E)) / (1.0 - ecc * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    ta = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    return ta % (2 * np.pi)


def coe2rv(a: float, e: float, incl: float, raan: float, w: float, M: float):
    """六根数(含平近点角M) → ECI 位置向量 km（对应 cpp coe2rv，TA 由 ma2ta 转换）"""
    ta = ma2ta(M, e)
    h = np.sqrt(a * MU * (1.0 - e * e))
    r_mag = (h * h / MU) / (1.0 + e * np.cos(ta))
    rp = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])

    def Rz(ang): return np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang),  np.cos(ang), 0],
                                   [0,            0,           1]])
    def Rx(ang): return np.array([[1, 0,           0          ],
                                   [0, np.cos(ang), -np.sin(ang)],
                                   [0, np.sin(ang),  np.cos(ang)]])

    Q = Rz(raan) @ Rx(incl) @ Rz(w)
    return Q @ rp


def orbit_points(a: float, e: float, incl: float, raan: float, w: float, n: int = 360):
    """生成完整轨道点序列（用于绘图）"""
    tas = np.linspace(0, 2 * np.pi, n)
    h = np.sqrt(a * MU * (1.0 - e * e))

    def Rz(ang): return np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang),  np.cos(ang), 0],
                                   [0,            0,           1]])
    def Rx(ang): return np.array([[1, 0,           0          ],
                                   [0, np.cos(ang), -np.sin(ang)],
                                   [0, np.sin(ang),  np.cos(ang)]])
    Q = Rz(raan) @ Rx(incl) @ Rz(w)

    pts = []
    for ta in tas:
        r_mag = (h * h / MU) / (1.0 + e * np.cos(ta))
        rp = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])
        pts.append(Q @ rp)
    return np.array(pts)


def julian_day(utc_dt: datetime) -> float:
    """UTC datetime → 儒略日（对应 cpp JulianDay）"""
    epoch_seconds = utc_dt.timestamp()
    return epoch_seconds / 86400.0 + 2440587.5


def solar_position_j2000(jd: float) -> np.ndarray:
    """
    太阳在 J2000 ECI 的位置向量 (km)
    与 cpp solar_position(jd, ...) 完全一致（Meeus 低精度算法）
    """
    T = (jd - 2451545.0) / 36525.0

    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    L0 = L0 % 360.0
    if L0 < 0:
        L0 += 360.0

    M_deg = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    M_deg = M_deg % 360.0
    if M_deg < 0:
        M_deg += 360.0
    M_rad = np.deg2rad(M_deg)

    e_orb = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T

    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * np.sin(M_rad)
         + (0.019993 - 0.000101 * T) * np.sin(2.0 * M_rad)
         + 0.000289 * np.sin(3.0 * M_rad))

    sun_lon = L0 + C
    v_rad = np.deg2rad(M_deg + C)

    R_km = AU * 1.000001018 * (1.0 - e_orb * e_orb) / (1.0 + e_orb * np.cos(v_rad))

    epsilon = 23.439291 - 0.0130042 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T
    eps_rad = np.deg2rad(epsilon)
    lam_rad = np.deg2rad(sun_lon)

    pos = np.array([
        R_km * np.cos(lam_rad),
        R_km * np.cos(eps_rad) * np.sin(lam_rad),
        R_km * np.sin(eps_rad) * np.sin(lam_rad),
    ])
    return pos


def solar_illumination_angle(pos_sun: np.ndarray,
                              pos_evader: np.ndarray,
                              pos_chaser: np.ndarray) -> float:
    """
    光照角（rad）：目标→太阳 与 目标→追击者 的夹角
    与 cpp solar_illumination_angle 完全一致
    """
    to_sun    = pos_sun    - pos_evader
    to_chaser = pos_chaser - pos_evader
    to_sun    = to_sun    / np.linalg.norm(to_sun)
    to_chaser = to_chaser / np.linalg.norm(to_chaser)
    return np.arccos(np.clip(np.dot(to_sun, to_chaser), -1.0, 1.0))


# ─── 可视化主函数 ─────────────────────────────────────────────────────────────

def plot_escort_recon_scenario(
    red_hv:  list,
    red_esc: list,
    blue:    list,
    utc_dt:  datetime = DEFAULT_UTC,
    output:  str = "escort_recon_init.png",
):
    # 位置向量
    r_hv  = coe2rv(*red_hv)
    r_esc = coe2rv(*red_esc)
    r_bl  = coe2rv(*blue)

    # 距离
    d_bl_hv  = np.linalg.norm(r_bl  - r_hv)
    d_bl_esc = np.linalg.norm(r_bl  - r_esc)
    d_esc_hv = np.linalg.norm(r_esc - r_hv)

    # 太阳位置
    jd = julian_day(utc_dt)
    pos_sun = solar_position_j2000(jd)

    # 光照角（与 cpp 定义一致：目标→太阳 vs 目标→追击者）
    # 蓝星侦照红HV：evader=红HV, chaser=蓝
    angle_bl_hv_rad  = solar_illumination_angle(pos_sun, r_hv,  r_bl)
    # 红护卫侦照蓝星：evader=蓝, chaser=红护卫
    angle_esc_bl_rad = solar_illumination_angle(pos_sun, r_bl,  r_esc)
    angle_bl_hv_deg  = np.rad2deg(angle_bl_hv_rad)
    angle_esc_bl_deg = np.rad2deg(angle_esc_bl_rad)

    # 侦照成功条件：距离≤20km 且 光照角≤60°
    recon_bl_hv  = d_bl_hv  <= 20.0 and angle_bl_hv_deg  <= 60.0
    recon_esc_bl = d_bl_esc <= 20.0 and angle_esc_bl_deg <= 60.0

    # 轨道点
    orb_hv  = orbit_points(*red_hv[:5])
    orb_esc = orbit_points(*red_esc[:5])
    orb_bl  = orbit_points(*blue[:5])

    # 太阳方向单位向量（从原点出发，用于全局图）
    sun_dir = pos_sun / np.linalg.norm(pos_sun)
    # 红HV → 太阳 单位向量（用于相对位置图，与 solar_illumination_angle 定义一致）
    hv_to_sun = pos_sun - r_hv
    hv_to_sun_dir = hv_to_sun / np.linalg.norm(hv_to_sun)

    bjt_str = utc_dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M BJT")
    utc_str = utc_dt.strftime("%Y-%m-%d %H:%M UTC")

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Escort-Recon Scenario — Initial Configuration\n{bjt_str}  ({utc_str})",
                 fontsize=15, fontweight='bold')

    # ── 子图1：3D 全局视图 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax1.plot_surface(RE*np.cos(u)*np.sin(v), RE*np.sin(u)*np.sin(v), RE*np.cos(v),
                     color='deepskyblue', alpha=0.12, linewidth=0)

    ax1.plot(orb_hv[:,0],  orb_hv[:,1],  orb_hv[:,2],  'r-',  lw=1.0, alpha=0.45, label='Red HV orbit')
    ax1.plot(orb_esc[:,0], orb_esc[:,1], orb_esc[:,2], color='darkorange', ls='--', lw=1.0, alpha=0.45, label='Red Escort orbit')
    ax1.plot(orb_bl[:,0],  orb_bl[:,1],  orb_bl[:,2],  'b-',  lw=1.0, alpha=0.45, label='Blue Recon orbit')

    ax1.scatter(*r_hv,  c='red',        s=140, marker='*', zorder=5, label='Red HV ★')
    ax1.scatter(*r_esc, c='darkorange', s=110, marker='^', zorder=5, label='Red Escort ▲')
    ax1.scatter(*r_bl,  c='blue',       s=110, marker='s', zorder=5, label='Blue Recon ■')

    for p1, p2, col, lbl in [
        (r_bl,  r_hv,  'purple', f'Blue↔HV   {d_bl_hv:.3f} km'),
        (r_bl,  r_esc, 'green',  f'Blue↔Esc  {d_bl_esc:.3f} km'),
        (r_esc, r_hv,  'gray',   f'Esc↔HV    {d_esc_hv:.3f} km'),
    ]:
        ax1.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                 color=col, lw=1.5, ls=':', alpha=0.85, label=lbl)

    scale = red_hv[0] * 0.35
    ax1.quiver(0, 0, 0, sun_dir[0]*scale, sun_dir[1]*scale, sun_dir[2]*scale,
               color='gold', linewidth=2.5, arrow_length_ratio=0.12, label='Sun dir (from Earth)')
    # 红HV → 太阳方向箭头（从红HV出发，长度=轨道半径*0.25）
    hv_sun_scale = red_hv[0] * 0.25
    ax1.quiver(r_hv[0], r_hv[1], r_hv[2],
               hv_to_sun_dir[0]*hv_sun_scale, hv_to_sun_dir[1]*hv_sun_scale, hv_to_sun_dir[2]*hv_sun_scale,
               color='yellow', linewidth=2.0, arrow_length_ratio=0.15, label='Red HV → Sun')

    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Global View (J2000 ECI)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.85)

    # ── 子图2：XY 平面投影 ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    earth2 = plt.Circle((0, 0), RE, color='deepskyblue', alpha=0.18)
    ax2.add_patch(earth2)
    ax2.plot(orb_hv[:,0],  orb_hv[:,1],  'r-',  lw=1, alpha=0.4)
    ax2.plot(orb_esc[:,0], orb_esc[:,1], color='darkorange', ls='--', lw=1, alpha=0.4)
    ax2.plot(orb_bl[:,0],  orb_bl[:,1],  'b-',  lw=1, alpha=0.4)
    ax2.scatter(r_hv[0],  r_hv[1],  c='red',        s=100, marker='*', zorder=5, label='Red HV')
    ax2.scatter(r_esc[0], r_esc[1], c='darkorange', s=80,  marker='^', zorder=5, label='Red Escort')
    ax2.scatter(r_bl[0],  r_bl[1],  c='blue',       s=80,  marker='s', zorder=5, label='Blue Recon')
    for p1, p2, col in [(r_bl, r_hv, 'purple'), (r_bl, r_esc, 'green'), (r_esc, r_hv, 'gray')]:
        ax2.plot([p1[0],p2[0]], [p1[1],p2[1]], color=col, lw=1.2, ls=':', alpha=0.8)
    # 太阳方向箭头（从原点，2D）
    scale2d = red_hv[0] * 0.25
    ax2.annotate('', xy=(sun_dir[0]*scale2d, sun_dir[1]*scale2d), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='gold', lw=2))
    ax2.text(sun_dir[0]*scale2d*1.05, sun_dir[1]*scale2d*1.05, 'Sun', color='goldenrod', fontsize=8)
    # 红HV → 太阳方向箭头（从红HV出发）
    hv_sun_scale2d = red_hv[0] * 0.18
    ax2.annotate('', xy=(r_hv[0] + hv_to_sun_dir[0]*hv_sun_scale2d,
                          r_hv[1] + hv_to_sun_dir[1]*hv_sun_scale2d),
                 xytext=(r_hv[0], r_hv[1]),
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=2.0))
    ax2.text(r_hv[0] + hv_to_sun_dir[0]*hv_sun_scale2d*1.08,
             r_hv[1] + hv_to_sun_dir[1]*hv_sun_scale2d*1.08,
             'HV→Sun', color='goldenrod', fontsize=7, ha='center')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Projection (J2000)', fontsize=11)
    ax2.legend(fontsize=7)

    # ── 子图3：以红色高价值星为中心的相对位置 ─────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    rel_esc = r_esc - r_hv
    rel_bl  = r_bl  - r_hv

    ax3.scatter(0, 0, c='red', s=160, marker='*', zorder=5, label='Red HV (center)')
    ax3.scatter(rel_esc[0], rel_esc[1], c='darkorange', s=100, marker='^', zorder=5,
                label=f'Red Escort  {d_esc_hv:.1f} km')
    ax3.scatter(rel_bl[0],  rel_bl[1],  c='blue',       s=100, marker='s', zorder=5,
                label=f'Blue Recon  {d_bl_hv:.1f} km')

    recon_circle = plt.Circle((0, 0), 20, color='green', alpha=0.13,
                               ls='--', lw=1.5, fill=True, label='Recon zone 20 km')
    escort_ref   = plt.Circle((0, 0), 80, color='darkorange', alpha=0.07,
                               ls=':', lw=1.5, fill=True, label='Escort ref 80 km')
    ax3.add_patch(recon_circle)
    ax3.add_patch(escort_ref)

    ax3.plot([0, rel_esc[0]], [0, rel_esc[1]], color='gray',   lw=1.2, ls=':')
    ax3.plot([0, rel_bl[0]],  [0, rel_bl[1]],  color='purple', lw=1.2, ls=':')
    ax3.plot([rel_esc[0], rel_bl[0]], [rel_esc[1], rel_bl[1]], color='green', lw=1.2, ls=':')

    # 红HV → 太阳方向箭头（从原点出发，长度=lim3*0.35）
    lim3 = max(d_bl_hv, d_esc_hv, 200) * 1.25
    sun_arrow_len3 = lim3 * 0.35
    ax3.annotate('', xy=(hv_to_sun_dir[0]*sun_arrow_len3, hv_to_sun_dir[1]*sun_arrow_len3),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='gold', lw=2.5))
    ax3.text(hv_to_sun_dir[0]*sun_arrow_len3*1.08, hv_to_sun_dir[1]*sun_arrow_len3*1.08,
             'HV→Sun', color='goldenrod', fontsize=8, ha='center', fontweight='bold')
    # 标注光照角弧线（蓝星方向 vs 太阳方向）
    if np.linalg.norm(rel_bl[:2]) > 1e-3:
        arc_r = lim3 * 0.15
        bl_angle  = np.arctan2(rel_bl[1],  rel_bl[0])
        sun_angle = np.arctan2(hv_to_sun_dir[1], hv_to_sun_dir[0])
        angles = np.linspace(sun_angle, bl_angle, 40)
        ax3.plot(arc_r*np.cos(angles), arc_r*np.sin(angles), color='cyan', lw=1.5, alpha=0.8)
        mid_angle = (sun_angle + bl_angle) / 2
        ax3.text(arc_r*1.25*np.cos(mid_angle), arc_r*1.25*np.sin(mid_angle),
                 f'{angle_bl_hv_deg:.1f}°', color='cyan', fontsize=8, ha='center')
    ax3.set_xlim(-lim3, lim3); ax3.set_ylim(-lim3, lim3)
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('ΔX (km)'); ax3.set_ylabel('ΔY (km)')
    ax3.set_title('Relative to Red HV (XY)', fontsize=11)
    ax3.legend(fontsize=7, loc='upper right')

    # ── 子图5：以蓝星为中心的相对位置 ────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    rel_hv_bl  = r_hv  - r_bl
    rel_esc_bl = r_esc - r_bl

    ax5.scatter(0, 0, c='blue', s=120, marker='s', zorder=5, label='Blue Recon (center)')
    ax5.scatter(rel_hv_bl[0],  rel_hv_bl[1],  c='red',        s=140, marker='*', zorder=5,
                label=f'Red HV  {d_bl_hv:.1f} km')
    ax5.scatter(rel_esc_bl[0], rel_esc_bl[1], c='darkorange', s=100, marker='^', zorder=5,
                label=f'Red Escort  {d_bl_esc:.1f} km')

    recon_hv = plt.Circle((rel_hv_bl[0], rel_hv_bl[1]), 20,
                           color='green', alpha=0.15, ls='--', lw=1.5, fill=True,
                           label='Recon zone (HV) 20 km')
    init_ref = plt.Circle((0, 0), 200, color='blue', alpha=0.05,
                           ls=':', lw=1.5, fill=True, label='Init range 200 km')
    ax5.add_patch(recon_hv)
    ax5.add_patch(init_ref)

    ax5.plot([0, rel_hv_bl[0]],  [0, rel_hv_bl[1]],  color='purple', lw=1.2, ls=':')
    ax5.plot([0, rel_esc_bl[0]], [0, rel_esc_bl[1]], color='green',  lw=1.2, ls=':')

    # 红HV → 太阳方向箭头（从红HV相对位置出发，在蓝星坐标系中）
    lim5 = max(d_bl_hv, d_bl_esc, 200) * 1.25
    sun_arrow_len5 = lim5 * 0.30
    ax5.annotate('', xy=(rel_hv_bl[0] + hv_to_sun_dir[0]*sun_arrow_len5,
                          rel_hv_bl[1] + hv_to_sun_dir[1]*sun_arrow_len5),
                 xytext=(rel_hv_bl[0], rel_hv_bl[1]),
                 arrowprops=dict(arrowstyle='->', color='gold', lw=2.5))
    ax5.text(rel_hv_bl[0] + hv_to_sun_dir[0]*sun_arrow_len5*1.1,
             rel_hv_bl[1] + hv_to_sun_dir[1]*sun_arrow_len5*1.1,
             'HV→Sun', color='goldenrod', fontsize=7, ha='center', fontweight='bold')
    ax5.set_xlim(-lim5, lim5); ax5.set_ylim(-lim5, lim5)
    ax5.set_aspect('equal'); ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('ΔX (km)'); ax5.set_ylabel('ΔY (km)')
    ax5.set_title('Relative to Blue Recon (XY)', fontsize=11)
    ax5.legend(fontsize=7, loc='upper right')

    # ── 子图6：信息面板 ────────────────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    def fmt_oe(label, oe):
        a, e, i, raan, peri, M = oe
        # 保持与输入等效精度：a 6位小数，e 6位小数，角度 rad→deg 保留6位小数
        return (f"{label}\n"
                f"  a    = {a:.6f} km\n"
                f"  e    = {e:.6f}\n"
                f"  i    = {i:.6f} rad  ({np.rad2deg(i):.6f}°)\n"
                f"  RAAN = {raan:.6f} rad  ({np.rad2deg(raan):.6f}°)\n"
                f"  ω    = {peri:.6f} rad  ({np.rad2deg(peri):.6f}°)\n"
                f"  M    = {M:.6f} rad  ({np.rad2deg(M):.6f}°)")

    def check(ok): return "✓ MET" if ok else "✗ NOT MET"

    info = "\n".join([
        fmt_oe("Red HV  ★", red_hv),
        "",
        fmt_oe("Red Escort ▲", red_esc),
        "",
        fmt_oe("Blue Recon ■", blue),
        "",
        "─── Distances ──────────────────────",
        f"  Blue ↔ Red HV   : {d_bl_hv:.6f} km",
        f"  Blue ↔ Escort   : {d_bl_esc:.6f} km",
        f"  Escort ↔ Red HV : {d_esc_hv:.6f} km",
        "",
        "─── Solar Illumination Angle ───────",
        "  (target→sun vs target→chaser)",
        f"  Blue→HV  : {angle_bl_hv_deg:.6f}°",
        f"  Esc→Blue : {angle_esc_bl_deg:.6f}°",
        "",
        "─── Recon Condition (d≤20 & θ≤60°) ─",
        f"  Blue recon HV  : {check(recon_bl_hv)}",
        f"  Esc  recon Blue: {check(recon_esc_bl)}",
    ])

    ax6.text(0.04, 0.98, info,
             transform=ax6.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax6.set_title('Scenario Info', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"图像已保存: {output}")
    plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_oe(s: str) -> list:
    vals = [float(x) for x in s.split(',')]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError("需要6个值: a,e,i,raan,peri,M")
    return vals


def parse_bjt(s: str) -> datetime:
    """解析 'YYYY-MM-DD HH:MM' 格式的 BJT 时间，返回 UTC datetime"""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    bjt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
    return bjt.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(
        description="护卫侦照场景初始构型可视化（三星：红HV + 红护卫 + 蓝侦照）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""默认值对应 BJT 2027-09-01 20:00 的 GEO 附近三星构型。
示例（使用默认值）:
  python visualize_escort_recon.py

示例（自定义）:
  python visualize_escort_recon.py \\
    --red-hv  42169.5,0,0.002287,1.5929,0,0.4198 \\
    --red-esc 42169.5,0,0.002287,1.5929,0,0.4211 \\
    --blue    42169.5,0,0.002287,1.5928,0,0.4244 \\
    --bjt "2027-09-01 20:00"
""")
    parser.add_argument('--red-hv',  type=parse_oe, default=DEFAULT_RED_HV,
                        metavar='a,e,i,raan,peri,M', help='红色高价值星六根数')
    parser.add_argument('--red-esc', type=parse_oe, default=DEFAULT_RED_ESC,
                        metavar='a,e,i,raan,peri,M', help='红色护卫星六根数')
    parser.add_argument('--blue',    type=parse_oe, default=DEFAULT_BLUE,
                        metavar='a,e,i,raan,peri,M', help='蓝色侦照星六根数')
    parser.add_argument('--bjt',     type=parse_bjt, default=DEFAULT_UTC,
                        metavar='YYYY-MM-DD HH:MM',
                        help='初始时间（BJT，默认 2027-09-01 20:00）')
    parser.add_argument('--output',  type=str, default='escort_recon_init.png',
                        help='输出图片路径')
    args = parser.parse_args()

    plot_escort_recon_scenario(
        red_hv=args.red_hv,
        red_esc=args.red_esc,
        blue=args.blue,
        utc_dt=args.bjt,
        output=args.output,
    )


if __name__ == '__main__':
    main()
