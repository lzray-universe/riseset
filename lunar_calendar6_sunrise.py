# -*- coding: utf-8 -*-
"""
精确农历与节气计算（DE440/DE441 + IAU 2006/2000A/2000B, 光行时+光行差）

要点：
- 使用 JPL DE440/DE441 历表（spiceypy/CSPICE）。
- 太阳/月球：含光行时迭代与周年光行差（相对论公式）；转入瞬时黄道。
- 岁差：IAU 2006；章动：优先 ERFA 2000A，次选 2000B；缺库时用简化近似。
- 求根：牛顿-拉弗森，导数单位统一为“弧度/日”。
- 时间尺度：TDB≈TT（误差<2ms），用闰秒表与 ΔT 外推将 TDB↔UTC/UTC+8。

用法:
    python lunar_calendar_fixed.py <path/to/de441.bsp> [year | start-end | y1,y2,...]
"""
from __future__ import annotations

import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, ClassVar

from astropy.time import Time, TimeDelta

from joblib import Parallel, delayed, cpu_count

try:
    import spiceypy as spice  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("需要安装 spiceypy：pip install spiceypy") from exc

# ------------------------------ ERFA (pyerfa) ------------------------------
try:
    import erfa  # type: ignore
    _HAS_ERFA = True
except Exception:
    erfa = None  # type: ignore
    _HAS_ERFA = False

# ----------------------------- JPL ephem -------------------------------
# ------------------------------ 配置：岁差模型 ------------------------------
# AUTO：距 J2000 超过阈值（世纪）时，优先使用 Vondrák 长时间岁差模型（ERFA: ltpequ）。
# 仅当安装了 pyerfa/erfa 时启用；否则保持 IAU 2006。
PRECESSION_MODEL = "AUTO"   # "AUTO" | "IAU2006" | "VONDRAK"
LONG_TERM_THRESHOLD_CENTURIES = 100.0  # |T|>=100 世纪（±1 万年）采用 Vondrák


AU_KM = 149_597_870.7
SEC_PER_DAY = 86400.0
C_AU_PER_DAY = 173.144632674  # 光速 (AU/day)

UTC8_OFFSET = TimeDelta(8 * 3600, format='sec')


@dataclass
class DE441Reader:
    """基于 spiceypy 读取 DE440/DE441 历表；封装常用组合段。"""

    filepath: str
    _loaded_paths: ClassVar[set[str]] = set()

    def __post_init__(self) -> None:
        self.filepath = os.path.abspath(self.filepath)
        self._ensure_kernel_loaded()
        # 常用编号
        self.SSB = 0       # 太阳系质心
        self.SUN = 10      # 太阳
        self.EMB = 3       # 地月质心
        self.EARTH = 399   # 地球
        self.MOON = 301    # 月球

        self._id_to_name = {
            self.SSB: "SOLAR SYSTEM BARYCENTER",
            self.SUN: "SUN",
            self.EMB: "EARTH BARYCENTER",
            self.EARTH: "EARTH",
            self.MOON: "MOON",
        }

    def _ensure_kernel_loaded(self) -> None:
        need_load = False
        try:
            count = spice.ktotal("SPK")
            if count == 0:
                need_load = True
        except Exception:
            need_load = True

        if self.filepath not in self._loaded_paths:
            need_load = True

        if need_load:
            spice.furnsh(self.filepath)
        self._loaded_paths.add(self.filepath)

    def _to_name(self, code: int) -> str:
        try:
            return self._id_to_name[code]
        except KeyError as exc:
            raise ValueError(f"未知的目标/观测者编号: {code}") from exc

    @staticmethod
    def _et_from_jd_tdb(jd_tdb: float) -> float:
        return (jd_tdb - 2451545.0) * SEC_PER_DAY

    def get_state(self, target: int, observer: int, jd_tdb: float) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_kernel_loaded()
        et = self._et_from_jd_tdb(jd_tdb)
        target_name = self._to_name(target)
        observer_name = self._to_name(observer)
        state, _ = spice.spkezr(target_name, et, "J2000", "NONE", observer_name)
        pos = np.array(state[:3]) / AU_KM
        vel = np.array(state[3:]) * (SEC_PER_DAY / AU_KM)
        return pos, vel

    # ---- 位置（单位：AU）----
    def get_position(self, target: int, observer: int, jd_tdb: float) -> np.ndarray:
        pos, _ = self.get_state(target, observer, jd_tdb)
        return pos

    # ---- 速度（单位：AU/day）----
    def get_velocity(self, target: int, observer: int, jd_tdb: float) -> np.ndarray:
        _, vel = self.get_state(target, observer, jd_tdb)
        return vel


@dataclass(frozen=True)
class RootTask:
    task_id: str
    kind: str
    target: float
    jd_initial: float
    eps_days: float = 1e-8
    max_iter: int = 20


# ----------------------------- 时间尺度/闰秒 -------------------------------
class TimeScale:
    """时间尺度换算：TDB≈TT；利用闰秒表与 ΔT 外推进行 TDB↔UTC。"""

    @staticmethod
    def tdb_to_tt(jd_tdb: float) -> float:
        return jd_tdb  # TDB≈TT (误差 < 2 ms)

    @staticmethod
    def tt_to_tdb(jd_tt: float) -> float:
        return jd_tt

    @staticmethod
    def tt_to_tai(jd_tt: float) -> float:
        # TT = TAI + 32.184s
        return jd_tt - 32.184 / SEC_PER_DAY

    @staticmethod
    def _leap_seconds_utc(jd_utc: float) -> int:
        """自 1972-01-01 起累计闰秒。
        注意：闰秒是人类决策，未来不可预测。本程序仅使用已发布表；超出表范围不外推。
        若需超长期天文计算，请优先在 TT/TDB 时间标下工作，或使用 ΔT 近似而非“预测闰秒”。
        """
        # (UTC JD, leaps) —— 列表按时间升序
        table = [
            (2441317.5, 10),  # 1972-01-01
            (2441499.5, 11),  # 1972-07-01
            (2441683.5, 12),  # 1973-01-01
            (2442048.5, 13),  # 1974-01-01
            (2442413.5, 14),  # 1975-01-01
            (2442778.5, 15),  # 1976-01-01
            (2443144.5, 16),  # 1977-01-01
            (2443509.5, 17),  # 1978-01-01
            (2443874.5, 18),  # 1979-01-01
            (2444239.5, 19),  # 1980-01-01
            (2444786.5, 20),  # 1981-07-01
            (2445151.5, 21),  # 1982-07-01
            (2445516.5, 22),  # 1983-07-01
            (2446247.5, 23),  # 1985-07-01
            (2447161.5, 24),  # 1988-01-01
            (2447892.5, 25),  # 1990-01-01
            (2448257.5, 26),  # 1991-01-01
            (2448804.5, 27),  # 1992-07-01
            (2449169.5, 28),  # 1993-07-01
            (2449534.5, 29),  # 1994-07-01
            (2450083.5, 30),  # 1996-01-01
            (2450630.5, 31),  # 1997-07-01
            (2451179.5, 32),  # 1999-01-01
            (2453736.5, 33),  # 2006-01-01
            (2454832.5, 34),  # 2009-01-01
            (2456109.5, 35),  # 2012-07-01
            (2457204.5, 36),  # 2015-07-01
            (2457754.5, 37),  # 2017-01-01
        ]
        leaps = 0
        for jd, l in table:
            if jd_utc >= jd:
                leaps = l
            else:
                break
        return leaps

    @staticmethod
    def tdb_to_utc(jd_tdb: float) -> float:
        """TDB -> UTC/UT1

        - 1972年以前：输出 UT1（UT1 = TT - ΔT）。
        - 1972–2025年：按闰秒表精确换算 UTC（TT -> TAI -> UTC）。
        - 2026年及以后：未来改走 ΔT 分支，用 TT-UTC ≈ ΔT 近似得到 UTC。

        TDB≈TT（误差 < 2 ms）。ΔT 采用 Stephenson+Morrison 外推公式。
        """
        jd_tt = TimeScale.tdb_to_tt(jd_tdb)

        # 以 TT 近似求年份
        year = 2000.0 + (jd_tt - 2451544.5) / 365.2425

        # ΔT 外推（秒）
        def _delta_t_seconds_from_year(y: float) -> float:
            import math
            t = (y - 1825.0) / 100.0
            return -150.568 + 31.4115 * (t * t) + 284.8436 * math.cos(2.0*math.pi*(t + 0.75)/14.0)

        # (b) 1972 年前：返回 UT1（而非 UTC）
        if year < 1972.0:
            delta_t = _delta_t_seconds_from_year(year)
            jd_ut1 = jd_tt - delta_t / SEC_PER_DAY
            return jd_ut1

        # (a) 2026 年及以后：未来改走 ΔT 分支，近似 UTC
        if year >= 2026.0:
            delta_t = _delta_t_seconds_from_year(year)
            jd_utc_approx = jd_tt - delta_t / SEC_PER_DAY
            return jd_utc_approx

        # 1972–2025：按闰秒表
        jd_tai = jd_tt - 32.184 / SEC_PER_DAY
        jd_utc = jd_tai
        for _ in range(2):
            leaps = TimeScale._leap_seconds_utc(jd_utc)
            jd_utc = jd_tai - leaps / SEC_PER_DAY
        return jd_utc

    @staticmethod
    def utc_to_tdb(jd_utc: float) -> float:
        """UTC -> TDB：UTC→TT 需 ΔT≈TT–UT1 作为近似（外推式）。"""
        year = 2000.0 + (jd_utc - 2451544.5) / 365.2425
        t = (year - 1825.0) / 100.0
        delta_t = -150.568 + 31.4115 * t * t + 284.8436 * math.cos(2.0*math.pi*(t+0.75)/14.0)
        # 以 TT-UTC≈ΔT 近似（误差 < 0.9s），再 TT≈TDB
        jd_tdb = jd_utc + delta_t / SEC_PER_DAY
        return jd_tdb


# -------------------------- 坐标变换/岁差章动 --------------------------
class CoordinateTransform:
    @staticmethod
    def R1(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    @staticmethod
    def R3(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    @staticmethod
    def frame_bias_matrix() -> np.ndarray:
        # ICRS -> mean J2000.0 frame bias (IERS 2003)
        return np.array([
            [0.9999999999999942, -7.078279744199198e-8,  8.056148940257979e-8],
            [7.078279477857338e-8,  0.9999999999999969,  3.306041454222136e-8],
            [-8.056149173973727e-8, -3.306040883980552e-8, 0.9999999999999962],
        ])



class PrecessionNutation:
    """IAU 2006 precession + 2000A(优先)/2000B(备选) nutation.
       远离 J2000（|T|>=LONG_TERM_THRESHOLD_CENTURIES）且安装了 ERFA 时，
       使用 Vondrák 长时间岁差模型（erfa.ltp, ±200 kyr 有效）。
    """

    @staticmethod
    def precession_matrix(jd_tdb: float) -> np.ndarray:
        # ERFA 优先：根据设置选择 IAU2006 或 Vondrák 长时间岁差
        if _HAS_ERFA:
            # ERFA 例程按 TT/TT 朱利安历元（Julian epoch）给出
            epj = 2000.0 + (jd_tdb - 2451545.0) / 365.25
            if PRECESSION_MODEL == "VONDRAK" or (PRECESSION_MODEL == "AUTO" and abs(epj - 2000.0) >= LONG_TERM_THRESHOLD_CENTURIES):
                # 长时间岁差（仅“岁差”，不含 frame bias/章动）：优先使用 erfa.ltp（Vondrák 模型）
                try:
                    return np.array(erfa.ltp(epj))
                except Exception:
                    # 若缺少 ltp，则退回 IAU 2006 P03
                    pass
            # 默认：IAU 2006 P03
            d1 = float(int(jd_tdb))
            d2 = float(jd_tdb - d1)
            return np.array(erfa.pmat06(d1, d2))

        T = (jd_tdb - 2451545.0) / 36525.0
        # IAU 2006 P03 angles (arcsec)
        psi_A = (5038.481507*T - 1.0790069*T**2 - 0.00114045*T**3 +
                 0.000132851*T**4 - 0.0000000951*T**5)
        omega_A = (84381.406 - 0.025754*T + 0.0512623*T**2 - 0.00772503*T**3
                   - 0.000000467*T**4 + 0.0000000337*T**5)
        chi_A = (10.556403*T - 2.3814292*T**2 - 0.00121197*T**3 +
                 0.000170663*T**4 - 0.0000000560*T**5)
        eps0 = 84381.406

        as2rad = math.pi / 648000.0
        psi_A *= as2rad; omega_A *= as2rad; chi_A *= as2rad; eps0 *= as2rad

        R1 = CoordinateTransform.R1
        R3 = CoordinateTransform.R3
        return R3(chi_A) @ R1(-omega_A) @ R3(-psi_A) @ R1(eps0)

    @staticmethod
    def mean_obliquity(jd_tdb: float) -> float:
        if _HAS_ERFA:
            d1 = float(int(jd_tdb)); d2 = float(jd_tdb - d1)
            return erfa.obl06(d1, d2)
        T = (jd_tdb - 2451545.0) / 36525.0
        as2rad = math.pi / 648000.0
        return (84381.406 - 46.836769*T - 0.0001831*T**2 + 0.00200340*T**3
                - 0.000000576*T**4 - 0.0000000434*T**5) * as2rad

    @staticmethod
    def nutation_angles(jd_tdb: float) -> Tuple[float, float]:
        """(Δψ, Δε) in radians. Prefer 2000A; fallback 2000B; else short approx."""
        if _HAS_ERFA:
            d1 = float(int(jd_tdb)); d2 = float(jd_tdb - d1)
            # 优先 2000A
            try:
                dpsi, deps = erfa.nut00a(d1, d2)
                return dpsi, deps
            except Exception:
                dpsi, deps = erfa.nut00b(d1, d2)
                return dpsi, deps

        # very short truncation (seconds-level)
        T = (jd_tdb - 2451545.0) / 36525.0
        # fundamental arguments (deg)
        l  = (134.96340251 + 1717915923.2178*T + 31.8792*T**2 + 0.051635*T**3 - 0.00024470*T**4)
        lp = (357.52910918 + 129596581.0481*T  - 0.5532*T**2 + 0.000136*T**3 - 0.00001149*T**4)
        F  = (93.27209062  + 1739527262.8478*T - 12.7512*T**2 - 0.001037*T**3 + 0.00000417*T**4)
        D  = (297.85019547 + 1602961601.2090*T - 6.3706*T**2 + 0.006593*T**3 - 0.00003169*T**4)
        Om = (125.04455501 - 6962890.5431*T    + 7.4722*T**2 + 0.007702*T**3 - 0.00005939*T**4)
        # to rad
        d2r = math.pi/180.0
        l*=d2r; lp*=d2r; F*=d2r; D*=d2r; Om*=d2r
        as2rad = math.pi/648000.0
        dpsi = (-17.20642418*math.sin(Om)
                + 0.003386*math.cos(Om)
                - 1.31709122*math.sin(2*F - 2*D + 2*Om)
                - 0.0013696*math.cos(2*F - 2*D + 2*Om)) * as2rad
        deps = ( 0.0015377*math.sin(Om)
                + 9.2052331*math.cos(Om)
                - 0.0004587*math.sin(2*F - 2*D + 2*Om)
                + 0.5730336*math.cos(2*F - 2*D + 2*Om)) * as2rad
        return dpsi, deps

    @staticmethod
    def nutation_matrix(jd_tdb: float) -> np.ndarray:
        if _HAS_ERFA:
            d1 = float(int(jd_tdb))
            d2 = float(jd_tdb - d1)
            return np.array(erfa.num06a(d1, d2))

        dpsi, deps = PrecessionNutation.nutation_angles(jd_tdb)
        epsA = PrecessionNutation.mean_obliquity(jd_tdb)
        R1 = CoordinateTransform.R1
        R3 = CoordinateTransform.R3
        eps = epsA + deps
        # N = R1(-ε) R3(-Δψ) R1(ε_A)
        return R1(-eps) @ R3(-dpsi) @ R1(epsA)


# -------------------------- 光行时 + 光行差（年度） --------------------------
class AberrationCorrection:
    """行星光行差：迭代光行时 + 相对论周年光行差，返回地心“视”向量（AU）。"""

    @staticmethod
    def _light_time_days(vec_au: np.ndarray) -> float:
        # r/c, 单位：日
        r = float(np.linalg.norm(vec_au))
        return r / C_AU_PER_DAY

    @staticmethod
    def geocentric_apparent_vector(ephem: DE441Reader, target: int, jd_tdb: float,
                                   max_iter: int = 3) -> np.ndarray:
        # 地球（SSB）在观测时刻 t
        xE = ephem.get_position(ephem.EARTH, ephem.SSB, jd_tdb)
        vE = ephem.get_velocity(ephem.EARTH, ephem.SSB, jd_tdb)  # AU/day

        # 迭代目标的“迟延时刻”
        tr = jd_tdb
        xt = ephem.get_position(target, ephem.SSB, tr)
        for _ in range(max_iter):
            r_geo = xt - xE
            lt = AberrationCorrection._light_time_days(r_geo)
            tr_new = jd_tdb - lt
            if abs(tr_new - tr) < 1e-12:
                break
            tr = tr_new
            xt = ephem.get_position(target, ephem.SSB, tr)

        # 几何向量（含光行时）
        r_geo = xt - xE
        r = float(np.linalg.norm(r_geo))
        n = r_geo / r

        # 相对论周年光行差（β=v/c, 单位 AU/day）
        beta = vE / C_AU_PER_DAY
        beta2 = float(beta @ beta)
        gamma_inv = math.sqrt(max(0.0, 1.0 - beta2))
        nb = float(n @ beta)
        n_app = (gamma_inv*n + beta + (nb*beta)/(1.0 + gamma_inv)) / (1.0 + nb)

        # 数值上 n_app 可能因舍入误差偏离单位向量，需归一化后再恢复距离
        n_app_norm = float(np.linalg.norm(n_app))
        if n_app_norm == 0.0:
            return n * r

        return (n_app / n_app_norm) * r


# ------------------------ 地心视黄经 λ 与导数 λ̇ ------------------------
class ApparentEclipticLongitude:
    def __init__(self, ephemeris: DE441Reader):
        self.eph = ephemeris
        self._frame_bias = CoordinateTransform.frame_bias_matrix()
        self._prec_cache_jd: float | None = None
        self._prec_cache: np.ndarray | None = None
        self._r1n_cache_jd: float | None = None
        self._r1n_cache: np.ndarray | None = None
        self._rot_cache_jd: float | None = None
        self._rot_cache: np.ndarray | None = None

    @staticmethod
    def _epsA(jd_tdb: float) -> float:
        return PrecessionNutation.mean_obliquity(jd_tdb)

    def _R1_eps_N(self, jd_tdb: float) -> np.ndarray:
        # R1(ε) @ N
        if self._r1n_cache is None or self._r1n_cache_jd != jd_tdb:
            epsA = PrecessionNutation.mean_obliquity(jd_tdb)
            dpsi, deps = PrecessionNutation.nutation_angles(jd_tdb)
            eps = epsA + deps
            R1 = CoordinateTransform.R1
            N = PrecessionNutation.nutation_matrix(jd_tdb)
            self._r1n_cache = R1(eps) @ N
            self._r1n_cache_jd = jd_tdb
        return self._r1n_cache

    def _precession_matrix(self, jd_tdb: float) -> np.ndarray:
        if self._prec_cache is None or self._prec_cache_jd != jd_tdb:
            self._prec_cache = PrecessionNutation.precession_matrix(jd_tdb)
            self._prec_cache_jd = jd_tdb
        return self._prec_cache

    def _rotation_matrix(self, jd_tdb: float) -> np.ndarray:
        if self._rot_cache is None or self._rot_cache_jd != jd_tdb:
            P = self._precession_matrix(jd_tdb)
            R1N = self._R1_eps_N(jd_tdb)
            self._rot_cache = R1N @ P @ self._frame_bias
            self._rot_cache_jd = jd_tdb
        return self._rot_cache

    def compute_sun(self, jd_tdb: float) -> Tuple[float, float]:
        """太阳地心视黄经及其时间导数（单位：弧度、弧度/日）。"""
        # 视向量（AU, ICRS）
        X = AberrationCorrection.geocentric_apparent_vector(self.eph, self.eph.SUN, jd_tdb)
        # 旋转到瞬时黄道
        rotation = self._rotation_matrix(jd_tdb)
        Xec = rotation @ X
        lam = math.atan2(Xec[1], Xec[0])
        if lam < 0:
            lam += 2*math.pi

        # 速度差（AU/day, ICRS）。注意：此处忽略光行差对导数的二阶影响
        v_earth = self.eph.get_velocity(self.eph.EARTH, self.eph.SSB, jd_tdb)
        v_sun = self.eph.get_velocity(self.eph.SUN, self.eph.SSB, jd_tdb)
        V = v_sun - v_earth
        Xec_dot = rotation @ V
        lam_dot = (Xec[0]*Xec_dot[1] - Xec[1]*Xec_dot[0]) / (Xec[0]**2 + Xec[1]**2)
        return lam, lam_dot  # rad, rad/day

    def compute_moon(self, jd_tdb: float) -> Tuple[float, float]:
        """月球地心视黄经及其时间导数（单位：弧度、弧度/日）。"""
        # 视向量（AU, ICRS）：同样计及光行时+周年光行差
        X = AberrationCorrection.geocentric_apparent_vector(self.eph, self.eph.MOON, jd_tdb)
        rotation = self._rotation_matrix(jd_tdb)
        Xec = rotation @ X
        lam = math.atan2(Xec[1], Xec[0])
        if lam < 0:
            lam += 2*math.pi

        # 速度（AU/day）用地月相对速度近似
        v = self.eph.get_velocity(self.eph.MOON, self.eph.EARTH, jd_tdb)
        Xec_dot = rotation @ v
        lam_dot = (Xec[0]*Xec_dot[1] - Xec[1]*Xec_dot[0]) / (Xec[0]**2 + Xec[1]**2)
        return lam, lam_dot


# ------------------------ 二十四节气 & 朔上望下 ------------------------
class SolarTermsAndLunarPhases:
    SOLAR_TERMS = {
        'Z2': ('春分', 0.0),
        'J3': ('清明',  np.pi/12),      # 15°
        'Z3': ('谷雨',  np.pi/6),       # 30°
        'J4': ('立夏',  np.pi/4),       # 45°
        'Z4': ('小满',  np.pi/3),       # 60°
        'J5': ('芒种',  5*np.pi/12),    # 75°
        'Z5': ('夏至',  np.pi/2),       # 90°
        'J6': ('小暑',  7*np.pi/12),    # 105°
        'Z6': ('大暑',  2*np.pi/3),     # 120°
        'J7': ('立秋',  3*np.pi/4),     # 135°
        'Z7': ('处暑',  5*np.pi/6),     # 150°
        'J8': ('白露',  11*np.pi/12),   # 165°
        'Z8': ('秋分',  np.pi),         # 180°
        'J9': ('寒露',  -11*np.pi/12),  # 195°
        'Z9': ('霜降',  -5*np.pi/6),    # 210°
        'J10':('立冬',  -3*np.pi/4),    # 225°
        'Z10':('小雪',  -2*np.pi/3),    # 240°
        'J11':('大雪',  -7*np.pi/12),   # 255°
        'Z11':('冬至',  -np.pi/2),      # 270°
        'J12':('小寒',  -5*np.pi/12),   # 285°
        'Z12':('大寒',  -np.pi/3),      # 300°
        'J1': ('立春',  -np.pi/4),      # 315°
        'Z1': ('雨水',  -np.pi/6),      # 330°
        'J2': ('惊蛰',  -np.pi/12),     # 345°
    }

    SOLAR_TERM_INIT_MONTH = {
        'Z11': 12, 'J12': 1, 'Z12': 1,
        'J1': 2, 'Z1': 2, 'J2': 3, 'Z2': 3,
        'J3': 4, 'Z3': 4, 'J4': 5, 'Z4': 5,
        'J5': 6, 'Z5': 6, 'J6': 7, 'Z6': 7,
        'J7': 8, 'Z7': 8, 'J8': 9, 'Z8': 9,
        'J9': 10, 'Z9': 10, 'J10': 11, 'Z10': 11, 'J11': 12,
    }

    LUNAR_PHASES = {
        'new_moon': ('朔', 0.0),
        'first_quarter': ('上弦',  np.pi/2),
        'full_moon': ('望',  np.pi),
        'last_quarter': ('下弦', -np.pi/2),
    }

    LUNAR_PHASE_OFFSETS = {
        'new_moon': 0.0,
        'first_quarter': 7.0,
        'full_moon': 15.0,
        'last_quarter': 22.0,
    }

    def __init__(self, eph: DE441Reader):
        self.eph = eph
        self.app = ApparentEclipticLongitude(eph)

    @staticmethod
    def _norm(angle: float) -> float:
        return angle - 2*np.pi*np.floor((angle + np.pi) / (2*np.pi))

    @classmethod
    def _solar_initial_guess_tdb(cls, year: int, code: str) -> float:
        m = cls.SOLAR_TERM_INIT_MONTH.get(code, 1)
        d = 22 if code == 'Z11' else 15
        jd0 = float(cls._utc_time(year, m, d).utc.jd)
        return TimeScale.utc_to_tdb(jd0)

    def f_solar_term(self, jd_tdb: float, target_lambda: float, *, lam: float | None = None) -> float:
        if lam is None:
            lam, _ = self.app.compute_sun(jd_tdb)
        return self._norm(lam - target_lambda)

    def f_lunar_phase(self, jd_tdb: float, phase_angle: float,
                      *, lam_s: float | None = None, lam_m: float | None = None) -> float:
        if lam_s is None or lam_m is None:
            lam_s, _ = self.app.compute_sun(jd_tdb)
            lam_m, _ = self.app.compute_moon(jd_tdb)
        return self._norm(lam_m - lam_s - phase_angle)

    def _value_and_derivative(self, kind: str, jd_tdb: float, target: float) -> Tuple[float, float]:
        if kind == 'solar':
            lam, lam_dot = self.app.compute_sun(jd_tdb)
            f = self.f_solar_term(jd_tdb, target, lam=lam)
            fdot = lam_dot
        else:
            lam_s, lam_dot_s = self.app.compute_sun(jd_tdb)
            lam_m, lam_dot_m = self.app.compute_moon(jd_tdb)
            f = self.f_lunar_phase(jd_tdb, target, lam_s=lam_s, lam_m=lam_m)
            fdot = lam_dot_m - lam_dot_s
        return f, fdot

    def _solve_tasks_parallel(self, tasks: List[RootTask]) -> Dict[str, float]:
        if not tasks:
            return {}

        n_jobs = max(1, min(cpu_count(), len(tasks)))

        if n_jobs == 1:
            return {
                task.task_id: self.newton(
                    task.kind, task.jd_initial, task.target,
                    eps_days=task.eps_days, max_iter=task.max_iter,
                )
                for task in tasks
            }

        results = Parallel(
            n_jobs=n_jobs,
            initializer=_worker_initialize,
            initargs=(self.eph.filepath,),
        )(
            delayed(_solve_task)(task, self.eph.filepath) for task in tasks
        )
        return dict(results)

    def newton(self, kind: str, jd_initial: float, target: float,
               eps_days: float = 1e-8, max_iter: int = 20) -> float:
        jd = jd_initial
        f, fdot = self._value_and_derivative(kind, jd, target)
        if abs(f) < 1e-12:
            return jd

        for _ in range(max_iter):
            if abs(fdot) < 1e-12:
                break

            delta = f / fdot              # days (因 f:rad, fdot:rad/day)
            delta = max(-3.0, min(3.0, delta))
            jd_new = jd - delta
            f_new, fdot_new = self._value_and_derivative(kind, jd_new, target)

            backtracks = 0
            while abs(f_new) > abs(f) and abs(delta) > eps_days and backtracks < 20:
                delta *= 0.5
                jd_new = jd - delta
                f_new, fdot_new = self._value_and_derivative(kind, jd_new, target)
                backtracks += 1

            if abs(f_new) > abs(f) and abs(delta) > eps_days:
                break

            if abs(delta) < eps_days or abs(f_new) < 1e-12:
                return jd_new

            jd, f, fdot = jd_new, f_new, fdot_new

        def _f_only(jd_val: float) -> float:
            val, _ = self._value_and_derivative(kind, jd_val, target)
            return val

        scan_step = 0.5
        scan_limit = 3.0
        f_center = f
        jd_center = jd
        if abs(f_center) < 1e-12:
            return jd_center

        intervals = []
        for direction in (-1, 1):
            prev_jd = jd_center
            prev_f = f_center
            steps = int(scan_limit / scan_step)
            for i in range(1, steps + 1):
                cand_jd = jd_center + direction * i * scan_step
                cand_f = _f_only(cand_jd)
                if abs(cand_f) < 1e-12:
                    return cand_jd
                if prev_f * cand_f <= 0:
                    left = min(prev_jd, cand_jd)
                    right = max(prev_jd, cand_jd)
                    f_left = prev_f if left == prev_jd else cand_f
                    f_right = cand_f if right == cand_jd else prev_f
                    intervals.append((left, right, f_left, f_right))
                    break
                prev_jd = cand_jd
                prev_f = cand_f

        for left, right, f_left, f_right in intervals:
            if f_left == 0.0:
                return left
            if f_right == 0.0:
                return right
            for _ in range(40):
                mid = 0.5 * (left + right)
                f_mid = _f_only(mid)
                if abs(f_mid) < 1e-12 or (right - left) * 0.5 < eps_days:
                    return mid
                if f_left * f_mid <= 0:
                    right = mid
                    f_right = f_mid
                else:
                    left = mid
                    f_left = f_mid

        raise RuntimeError("牛顿-拉弗森法未收敛")

    # ---- public helpers ----
    @staticmethod
    def make_local_time(year: int, month: int, day: int,
                        hour: int = 0, minute: int = 0, second: float = 0.0) -> Time:
        base = Time(
            {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second,
            },
            format='ymdhms',
            scale='utc',
        )
        return base + UTC8_OFFSET

    @staticmethod
    def _utc_time(year: int, month: int, day: int,
                  hour: int = 0, minute: int = 0, second: float = 0.0) -> Time:
        return Time(
            {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second,
            },
            format='ymdhms',
            scale='utc',
        )

    @staticmethod
    def _local_time_to_utc_jd(local_time: Time) -> float:
        return float((local_time - UTC8_OFFSET).utc.jd)

    @staticmethod
    def _utc_jd_to_local_time(jd_utc: float) -> Time:
        return Time(jd_utc, format='jd', scale='utc') + UTC8_OFFSET

    @staticmethod
    def _format_time(time: Time) -> str:
        return time.to_value('iso', subfmt='date_hms')

    # ---- concrete solvers ----
    def find_solar_term(self, code: str, year: int) -> Time:
        name, lam_target = self.SOLAR_TERMS[code]
        jd_tdb0 = self._solar_initial_guess_tdb(year, code)
        jd_tdb = self.newton('solar', jd_tdb0, lam_target)
        jd_utc = TimeScale.tdb_to_utc(jd_tdb)
        return self._utc_jd_to_local_time(jd_utc)

    def find_lunar_phase_near(self, phase_key: str, jd_near_tdb: float) -> Time:
        name, ang = self.LUNAR_PHASES[phase_key]
        jd_tdb = self.newton('lunar', jd_near_tdb, ang)
        jd_utc = TimeScale.tdb_to_utc(jd_tdb)
        return self._utc_jd_to_local_time(jd_utc)

    def compute_year(self, year: int) -> Dict:
        print(f"正在计算 {year} 年 ...")
        out = {'year': year, 'solar_terms': {}, 'lunar_phases': []}

        # 计算上一年冬至，作为朔望月初始锚点
        prev_task_id = f"solar_prev:{year - 1}:Z11"
        prev_task = RootTask(
            prev_task_id,
            'solar',
            self.SOLAR_TERMS['Z11'][1],
            self._solar_initial_guess_tdb(year - 1, 'Z11'),
        )
        prev_result = self._solve_tasks_parallel([prev_task])
        ws_prev_jd_tdb = prev_result[prev_task_id]
        ws_prev_local = self._utc_jd_to_local_time(TimeScale.tdb_to_utc(ws_prev_jd_tdb))

        start_local = self.make_local_time(year, 1, 1)
        end_local = self.make_local_time(year + 1, 1, 1)

        tasks: List[RootTask] = []
        task_meta: Dict[str, Dict[str, object]] = {}

        # 节气任务
        for code, (name, lam_target) in self.SOLAR_TERMS.items():
            task_id = f"solar:{code}"
            jd_tdb0 = self._solar_initial_guess_tdb(year, code)
            tasks.append(RootTask(task_id, 'solar', lam_target, jd_tdb0))
            task_meta[task_id] = {'type': 'solar', 'code': code, 'name': name}

        # 朔望月任务：上一年冬至前 ~45 天为起点，提前准备 18 组朔望月根
        jd_anchor = self._local_time_to_utc_jd(ws_prev_local) - 45.0
        jd_anchor_tdb = TimeScale.utc_to_tdb(jd_anchor)
        for idx in range(18):
            base_jd = jd_anchor_tdb + idx * 29.530588
            for phase_key, (_, phase_angle) in self.LUNAR_PHASES.items():
                offset = self.LUNAR_PHASE_OFFSETS[phase_key]
                task_id = f"lunar:{phase_key}:{idx}"
                tasks.append(RootTask(task_id, 'lunar', phase_angle, base_jd + offset))
                task_meta[task_id] = {
                    'type': 'lunar',
                    'phase': phase_key,
                    'index': idx,
                }

        results = self._solve_tasks_parallel(tasks)

        # 处理节气结果
        for code, (name, _) in self.SOLAR_TERMS.items():
            task_id = f"solar:{code}"
            jd_tdb = results.get(task_id)
            if jd_tdb is None:
                continue
            jd_utc = TimeScale.tdb_to_utc(jd_tdb)
            dt = self._utc_jd_to_local_time(jd_utc)
            out['solar_terms'][code] = {'name': name, 'datetime': dt}
            print(f"  {name}: {self._format_time(dt)}")

        # 处理朔望月结果
        lunar_results: Dict[Tuple[str, int], float] = {}
        for task_id, meta in task_meta.items():
            if meta.get('type') != 'lunar':
                continue
            value = results.get(task_id)
            if value is None:
                continue
            lunar_results[(meta['phase'], meta['index'])] = value

        added = 0
        for idx in sorted({index for (_, index) in lunar_results.keys()}):
            key = ('new_moon', idx)
            if key not in lunar_results:
                continue
            jd_new_tdb = lunar_results[key]
            jd_new_utc = TimeScale.tdb_to_utc(jd_new_tdb)
            dt_new = self._utc_jd_to_local_time(jd_new_utc)
            if dt_new < start_local:
                continue
            if dt_new >= end_local and added > 0:
                break

            try:
                dt_fq = self._utc_jd_to_local_time(
                    TimeScale.tdb_to_utc(lunar_results[('first_quarter', idx)])
                )
                dt_full = self._utc_jd_to_local_time(
                    TimeScale.tdb_to_utc(lunar_results[('full_moon', idx)])
                )
                dt_lq = self._utc_jd_to_local_time(
                    TimeScale.tdb_to_utc(lunar_results[('last_quarter', idx)])
                )
            except KeyError as exc:
                print(f"  朔望月索引 {idx} 结果不完整: {exc}")
                continue

            out['lunar_phases'].append({
                'new_moon': dt_new,
                'first_quarter': dt_fq,
                'full_moon': dt_full,
                'last_quarter': dt_lq,
            })
            added += 1

        # 打印摘要
        print("\n二十四节气（UTC+8）")
        print("-"*64)
        order = ['J12','Z12','J1','Z1','J2','Z2','J3','Z3','J4','Z4','J5','Z5',
                 'J6','Z6','J7','Z7','J8','Z8','J9','Z9','J10','Z10','J11','Z11']
        for code in order:
            if code in out['solar_terms']:
                dt = out['solar_terms'][code]['datetime']
                print(f"{self.SOLAR_TERMS[code][0]:<6} {self._format_time(dt)}")

        print("\n朔望月相（UTC+8）")
        print("-"*64)
        for i, ph in enumerate(out['lunar_phases'], 1):
            print(
                "第{i:02d}月  朔:{nm}  上弦:{fq}  望:{full}  下弦:{lq}".format(
                    i=i,
                    nm=self._format_time(ph['new_moon']),
                    fq=self._format_time(ph['first_quarter']),
                    full=self._format_time(ph['full_moon']),
                    lq=self._format_time(ph['last_quarter']),
                )
            )

        return out


# ------------------------------- CLI --------------------------------

_worker_solver: "SolarTermsAndLunarPhases" | None = None
_worker_solver_path: str | None = None


def _worker_initialize(ephemeris_path: str) -> None:
    _get_worker_solver(ephemeris_path)


def _get_worker_solver(ephemeris_path: str) -> "SolarTermsAndLunarPhases":
    global _worker_solver, _worker_solver_path
    if _worker_solver is None or _worker_solver_path != ephemeris_path:
        eph = DE441Reader(ephemeris_path)
        _worker_solver = SolarTermsAndLunarPhases(eph)
        _worker_solver_path = ephemeris_path
    return _worker_solver


def _solve_task(task: RootTask, ephemeris_path: str) -> Tuple[str, float]:
    solver = _get_worker_solver(ephemeris_path)
    try:
        result = solver.newton(
            task.kind,
            task.jd_initial,
            task.target,
            eps_days=task.eps_days,
            max_iter=task.max_iter,
        )
    except Exception as exc:  # pragma: no cover - parallel error propagation
        raise RuntimeError(f"求解任务 {task.task_id} 失败") from exc
    return task.task_id, result


def parse_year_arguments(arg: str) -> List[int]:
    """解析年份参数，支持单年、范围以及逗号分隔列表。"""

    years: List[int] = []
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    if not parts:
        raise ValueError("年份参数为空")

    for part in parts:
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"范围 {part} 结束年份早于开始年份")
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))

    # 去重同时保持输入顺序
    seen = set()
    ordered_years: List[int] = []
    for year in years:
        if year not in seen:
            ordered_years.append(year)
            seen.add(year)

    return ordered_years


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python lunar_calendar_fixed.py <de441.bsp路径> [年份]")
        sys.exit(1)

    path = sys.argv[1]
    year_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if year_arg is None:
        years = [2025]
    else:
        try:
            years = parse_year_arguments(year_arg)
        except ValueError as exc:
            print(f"年份参数无效: {exc}")
            sys.exit(1)

    eph = DE441Reader(path)
    solver = SolarTermsAndLunarPhases(eph)

    for idx, year in enumerate(years):
        if idx:
            print("\n" + "=" * 72 + "\n")
        solver.compute_year(year)



# =============================
# Sunrise/Sunset (topocentric)
# =============================
# High-precision sunrise/sunset using:
# - DE440/DE441 (spiceypy) for geocentric *apparent* solar vector (light-time + annual aberration)
# - ERFA for precise Earth orientation (c2t chain), polar motion, UT1, TT, diurnal aberration
# - Full topocentric geometry (observer ECEF -> GCRS), refraction (Saemundsson), solar semidiameter,
#   and geometric horizon dip by elevation.
#
# API:
#   calc = SunriseSunsetCalculator(DE441Reader('/path/to/de441.bsp'))
#   result = calc.sunrise_sunset(
#       year=2025, month=10, day=14,
#       lon_deg=139.6917, lat_deg=35.6895, elev_m=40.0,
#       pressure_hpa=1013.25, temperature_c=10.0,
#       dut1_sec=0.0, xp_arcsec=0.0, yp_arcsec=0.0,
#   )
#   print(result)  # {'sunrise_utc': '2025-10-13T20:XX:XXZ', 'sunset_utc': '2025-10-14T08:XX:XXZ', 'status': 'ok'}
#
# CLI (added to main):  python lunar_calendar6_sunrise.py <de441.bsp> --sun lat lon elev [--date YYYY-MM-DD]

AU_M = AU_KM * 1000.0
R_EARTH_M = 6378137.0  # WGS84 equatorial radius (approx for dip)
R_SUN_M = 695700000.0  # nominal photospheric radius (m)
OMEGA_EARTH = 7.2921150e-5  # rad/s
AS2R = math.pi / (180.0 * 3600.0)

def _datetime_to_jd_utc(dt):
    u1, u2 = erfa.dtf2d('UTC', dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
    return float(u1 + u2)

def _jd_to_iso_utc(jd):
    y,m,d, fd = erfa.jd2cal(jd, 0.0)
    # fd is fraction of day
    sec = fd * 86400.0
    hh = int(sec // 3600); sec -= hh*3600
    mm = int(sec // 60); ss = sec - mm*60
    return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:06.3f}Z"

def _refraction_saemundsson(alt_rad, pressure_hpa, temperature_c):
    # Saemundsson (1986), as in Meeus: h is APPARENT altitude (deg).
    # We'll iterate with h_app. Clamp near horizon to avoid blow-up.
    alt_deg = max(-2.0, min(90.0, math.degrees(alt_rad)))
    R_arcmin = (1.02 / math.tan(math.radians(alt_deg + 10.3/(alt_deg + 5.11)))) * (pressure_hpa/1010.0) * (283.0/(273.0 + temperature_c))
    return math.radians(R_arcmin / 60.0)

def _horizon_dip_rad(elev_m):
    if elev_m <= 0.0:
        return 0.0
    # Geometric dip of the visible sea-level horizon below the true horizontal.
    # Exact: cos(dip) = Re/(Re + h). For small h, dip ≈ sqrt(2h/Re).
    ratio = R_EARTH_M / (R_EARTH_M + elev_m)
    ratio = max(0.0, min(1.0, ratio))
    return math.acos(ratio)

class SunriseSunsetCalculator:
    def __init__(self, ephemeris: DE441Reader):
        self.eph = ephemeris

    def _rc2t_matrix(self, utc_dt, dut1_sec=0.0, xp_arcsec=0.0, yp_arcsec=0.0):
        # UTC -> UT1, TT
        u1, u2 = erfa.dtf2d('UTC', utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second + utc_dt.microsecond*1e-6)
        ut11, ut12 = erfa.utcut1(u1, u2, dut1_sec)
        tai1, tai2 = erfa.utctai(u1, u2)
        tt1, tt2 = erfa.taitt(tai1, tai2)

        # Celestial-to-intermediate
        rc2i = erfa.c2i06a(tt1, tt2)
        # Earth rotation angle
        era = erfa.era00(ut11, ut12)
        # Polar motion
        sp = erfa.sp00(tt1, tt2)  # TIO locator s'
        rpom = erfa.pom00(xp_arcsec*AS2R, yp_arcsec*AS2R, sp)
        # Celestial-to-terrestrial (GCRS -> ITRS)
        rc2t = erfa.c2tcio(rc2i, era, rpom)
        return rc2t, (ut11, ut12, tt1, tt2)

    def _sun_geocentric_apparent_gcrs(self, jd_utc):
        jd_tdb = TimeScale.utc_to_tdb(jd_utc)
        # Apparent (light-time + annual aberration)
        return AberrationCorrection.geocentric_apparent_vector(self.eph, self.eph.SUN, jd_tdb)

    def _observer_itrs(self, lon_rad, lat_rad, elev_m):
        # ITRS Cartesian (meters) from geodetic (WGS84)
        if hasattr(erfa, "gd2gc"):
            # Modern pyerfa exposes gd2gc with an integer ellipsoid selector.
            xyz = erfa.gd2gc(erfa.WGS84, lon_rad, lat_rad, elev_m)
            x, y, z = xyz
        else:  # pragma: no cover - legacy API fallback
            a = 6378137.0
            f = 1.0 / 298.257223563
            x, y, z = erfa.gd2gce(a, f, lon_rad, lat_rad, elev_m)
        return np.array([x, y, z], dtype=float)

    def _apply_diurnal_aberration(self, r_topo_gcrs, v_obs_gcrs_mps):
        # Convert to AU/day then use same relativistic aberration formula as annual
        v_au_per_day = v_obs_gcrs_mps * (SEC_PER_DAY / AU_M)
        beta = v_au_per_day / C_AU_PER_DAY  # vector
        n = r_topo_gcrs / np.linalg.norm(r_topo_gcrs)
        beta2 = float(beta @ beta)
        if beta2 < 1e-20:
            return r_topo_gcrs
        gamma_inv = math.sqrt(max(0.0, 1.0 - beta2))
        nb = float(n @ beta)
        n_app = (gamma_inv*n + beta + (nb*beta)/(1.0 + gamma_inv)) / (1.0 + nb)
        n_app /= np.linalg.norm(n_app)
        return n_app * np.linalg.norm(r_topo_gcrs)

    def _sun_altitude_components(self, utc_dt, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec):
        """Return apparent center altitude, semidiameter, and horizon dip."""
        lon = math.radians(lon_deg); lat = math.radians(lat_deg)

        # ERFA matrices for this moment
        rc2t, _ = self._rc2t_matrix(utc_dt, dut1_sec, xp_arcsec, yp_arcsec)

        # Observer in ITRS
        r_obs_itrs = self._observer_itrs(lon, lat, elev_m)

        # Transform observer to GCRS
        r_obs_gcrs = rc2t.T @ r_obs_itrs  # transpose: ITRS -> GCRS
        r_obs_gcrs_au = r_obs_gcrs / AU_M

        # Sun geocentric apparent (GCRS, AU)
        jd_utc = _datetime_to_jd_utc(utc_dt)
        r_sun_gcrs_au = self._sun_geocentric_apparent_gcrs(jd_utc)

        # Topocentric geometric LOS (GCRS)
        r_topo_gcrs = r_sun_gcrs_au - r_obs_gcrs_au

        # Diurnal velocity of observer in ITRS and rotate to GCRS
        omega = np.array([0.0, 0.0, OMEGA_EARTH])  # rad/s
        v_obs_itrs = np.cross(omega, r_obs_itrs)   # m/s
        v_obs_gcrs = rc2t.T @ v_obs_itrs           # m/s

        # Apply diurnal aberration
        r_topo_gcrs = self._apply_diurnal_aberration(r_topo_gcrs, v_obs_gcrs)

        # Transform topocentric LOS to ITRS then to local ENU to get altitude
        r_topo_itrs = rc2t @ (r_topo_gcrs * AU_M)  # now meters; only direction matters

        # ECEF -> ENU frame at site
        sinl, cosl = math.sin(lat), math.cos(lat)
        sinL, cosL = math.sin(lon), math.cos(lon)
        e_hat = np.array([-sinL,  cosL, 0.0])
        n_hat = np.array([-sinl*cosL, -sinl*sinL, cosl])
        u_hat = np.array([ cosl*cosL,  cosl*sinL, sinl])

        r_unit = r_topo_itrs / np.linalg.norm(r_topo_itrs)
        s_u = float(u_hat @ r_unit)
        # True geometric altitude of center
        h_true = math.asin(max(-1.0, min(1.0, s_u)))

        # Apparent altitude with refraction (iterative on apparent altitude)
        h_app = h_true
        for _ in range(2):
            R = _refraction_saemundsson(h_app, pressure_hpa, temperature_c)
            h_app = h_true + R

        # Semidiameter (from topocentric distance)
        dist_m = np.linalg.norm(r_topo_gcrs) * AU_M
        sd = math.asin(min(1.0, R_SUN_M / dist_m))

        # Horizon dip by elevation
        dip = _horizon_dip_rad(elev_m)

        return h_app, sd, dip

    def _altitude_app_center(self, utc_dt, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec):
        h_app, sd, dip = self._sun_altitude_components(utc_dt, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec)
        # For limb touching visible horizon: h_app(center) + sd + dip = 0
        return h_app + sd + dip

    def sunrise_sunset(self, year, month, day, lon_deg, lat_deg, elev_m,
                       pressure_hpa=1013.25, temperature_c=10.0,
                       dut1_sec=0.0, xp_arcsec=0.0, yp_arcsec=0.0,
                       tz_offset_hours: float | None = None):
        from datetime import datetime, timezone, timedelta

        if tz_offset_hours is None:
            tz_offset_hours = round(lon_deg / 15.0)
        tz_offset_hours = max(-12.0, min(14.0, float(tz_offset_hours)))
        tz_offset_seconds = int(round(tz_offset_hours * 3600.0))
        tz = timezone(timedelta(seconds=tz_offset_seconds))

        local_midnight = datetime(year, month, day, 0, 0, 0, tzinfo=tz)
        day0 = local_midnight.astimezone(timezone.utc)
        # Define altitude conditions for sunrise/sunset and twilights
        twilight_angles = {
            'civil': -6.0,
            'nautical': -12.0,
            'astronomical': -18.0,
        }

        def _limb_condition(h_app, sd, dip):
            return h_app + sd + dip

        event_specs = {
            'sun': {
                'func': _limb_condition,
                'labels': ('sunrise', 'sunset'),
            }
        }
        for name, angle in twilight_angles.items():
            rad = math.radians(angle)

            def _make_func(target_rad):
                return lambda h_app, sd, dip, target=target_rad: h_app - target

            event_specs[name] = {
                'func': _make_func(rad),
                'labels': (f'{name}_dawn', f'{name}_dusk'),
            }

        # sample every 30 minutes to robustly bracket events (handles high latitudes better)
        samples = []
        for k in range(49):  # 0..24h by 0.5h
            t = day0 + timedelta(minutes=30*k)
            h_app, sd, dip = self._sun_altitude_components(t, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec)
            values = {key: spec['func'](h_app, sd, dip) for key, spec in event_specs.items()}
            samples.append((t, values))

        # Find sign-change intervals for each event
        event_brackets = {key: [] for key in event_specs}
        for (t1, vals1), (t2, vals2) in zip(samples[:-1], samples[1:]):
            for key in event_specs:
                f1 = vals1[key]
                f2 = vals2[key]
                if f1 == 0.0:
                    event_brackets[key].append((t1, t1))
                elif f1 * f2 < 0.0:
                    event_brackets[key].append((t1, t2))

        # Refine by bisection to ~0.1 s
        def refine(t1, t2, func):
            from datetime import timedelta
            if t1 == t2:
                return t1
            f1 = func(*self._sun_altitude_components(t1, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec))
            f2 = func(*self._sun_altitude_components(t2, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec))
            for _ in range(80):
                mid = t1 + (t2 - t1)/2
                fm = func(*self._sun_altitude_components(mid, lon_deg, lat_deg, elev_m, pressure_hpa, temperature_c, dut1_sec, xp_arcsec, yp_arcsec))
                if abs(fm) < 1e-12 or (t2 - t1).total_seconds() < 0.1:
                    return mid
                if f1 * fm <= 0.0:
                    t2 = mid
                    f2 = fm
                else:
                    t1 = mid
                    f1 = fm
            return t1 + (t2 - t1)/2

        event_times = {}
        for key, spec in event_specs.items():
            func = spec['func']
            brackets = event_brackets[key]
            refined = [refine(a, b, func) for (a, b) in brackets[:2]]
            event_times[key] = sorted(refined)

        sun_events = event_times.get('sun', [])
        if not sun_events:
            sun_values = [vals['sun'] for _, vals in samples]
            above = all(f > 0 for f in sun_values)
            below = all(f < 0 for f in sun_values)
            status = 'polar_day' if above else ('polar_night' if below else 'no_event')
            return {
                'sunrise_utc': None,
                'sunset_utc': None,
                'sunrise_local': None,
                'sunset_local': None,
                'civil_dawn_utc': None,
                'civil_dawn_local': None,
                'civil_dusk_utc': None,
                'civil_dusk_local': None,
                'nautical_dawn_utc': None,
                'nautical_dawn_local': None,
                'nautical_dusk_utc': None,
                'nautical_dusk_local': None,
                'astronomical_dawn_utc': None,
                'astronomical_dawn_local': None,
                'astronomical_dusk_utc': None,
                'astronomical_dusk_local': None,
                'status': status,
                'tz_offset_hours': tz_offset_hours,
            }

        # Sort chronologically
        out = {
            'sunrise_utc': None,
            'sunset_utc': None,
            'sunrise_local': None,
            'sunset_local': None,
            'civil_dawn_utc': None,
            'civil_dawn_local': None,
            'civil_dusk_utc': None,
            'civil_dusk_local': None,
            'nautical_dawn_utc': None,
            'nautical_dawn_local': None,
            'nautical_dusk_utc': None,
            'nautical_dusk_local': None,
            'astronomical_dawn_utc': None,
            'astronomical_dawn_local': None,
            'astronomical_dusk_utc': None,
            'astronomical_dusk_local': None,
            'status': 'ok',
            'tz_offset_hours': tz_offset_hours,
        }

        def _assign(label, dt_value):
            if dt_value is None:
                out[f'{label}_utc'] = None
                out[f'{label}_local'] = None
            else:
                out[f'{label}_utc'] = _jd_to_iso_utc(_datetime_to_jd_utc(dt_value))
                out[f'{label}_local'] = dt_value.astimezone(tz).isoformat()

        sun_events.sort()
        if len(sun_events) >= 1:
            _assign('sunrise', sun_events[0])
        if len(sun_events) >= 2:
            _assign('sunset', sun_events[1])

        for key, spec in event_specs.items():
            if key == 'sun':
                continue
            labels = spec['labels']
            times = event_times.get(key, [])
            times.sort()
            if len(labels) >= 1:
                _assign(labels[0], times[0] if len(times) >= 1 else None)
            if len(labels) >= 2:
                _assign(labels[1], times[1] if len(times) >= 2 else None)
        return out


# --------------- CLI extension ---------------
def _cli_sunrise_sunset(args):
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sun', action='store_true', help='Compute sunrise/sunset mode')
    parser.add_argument('--date', type=str, default=None, help='YYYY-MM-DD (UTC date)')
    parser.add_argument('--lat', type=float, required=False)
    parser.add_argument('--lon', type=float, required=False)
    parser.add_argument('--elev', type=float, default=0.0)
    parser.add_argument('--pressure', type=float, default=1013.25)
    parser.add_argument('--temp', type=float, default=10.0)
    parser.add_argument('--dut1', type=float, default=0.0)
    parser.add_argument('--xp', type=float, default=0.0, help='polar motion X (arcsec)')
    parser.add_argument('--yp', type=float, default=0.0, help='polar motion Y (arcsec)')
    parser.add_argument('--tz', type=float, default=None, help='time zone offset from UTC in hours (default: round(lon/15))')
    ns, unknown = parser.parse_known_args(args)

    if not ns.sun:
        return None  # let original main() run

    if ns.lat is None or ns.lon is None:
        raise SystemExit('Missing --lat/--lon.')

    # Parse date
    if ns.date is None:
        dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        y,m,d = dt.year, dt.month, dt.day
    else:
        y, m, d = map(int, ns.date.split('-'))

    # Use the first CLI arg as ephemeris path
    # The launcher passes [<thisfile> <de441.bsp>] before these flags
    import sys
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python lunar_calendar6_sunrise.py <de441.bsp> --sun --date YYYY-MM-DD --lat ... --lon ... [--elev ...]')
    eph_path = sys.argv[1]

    eph = DE441Reader(eph_path)
    calc = SunriseSunsetCalculator(eph)
    res = calc.sunrise_sunset(y, m, d, ns.lon, ns.lat, ns.elev, ns.pressure, ns.temp, ns.dut1, ns.xp, ns.yp, ns.tz)
    print(res)
    return 'DONE'

# Hook into original CLI if invoked directly
if __name__ == "__main__":
    import sys
    # If user passed --sun, run the sunrise/sunset mode; otherwise fall back to original main()
    if any(arg == "--sun" for arg in sys.argv[2:]):  # skip <script> <bsp>
        _cli_sunrise_sunset(sys.argv[2:])
    else:
        main()
