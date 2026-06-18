import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


# ============================================================
# Перечисление типов
# ============================================================

class ElementType(Enum):
    HORN_PYRAMIDAL = "Пирамидальный рупор"
    HORN_H_SECTOR = "H-секториальный рупор"
    HORN_E_SECTOR = "E-секториальный рупор"
    HORN_CONICAL = "Конический рупор"
    PATCH = "Печатный излучатель"
    DIPOLE = "Вибратор"
    ISOTROPIC = "Изотропный"


# ============================================================
# Параметры излучателей
# ============================================================

@dataclass
class ElementParams:
    frequency: float = 8.33

    @property
    def wavelength(self) -> float:
        return 300 / self.frequency


@dataclass
class HornParams(ElementParams):
    L: float = 3.0
    wall_thickness: float = 0.02


@dataclass
class PyramidalHornParams(HornParams):
    a: float = 2.5
    b: float = 1.96
    a_wg: float = 0.77
    b_wg: float = 0.33


@dataclass
class PatchParams(ElementParams):
    substrate_eps: float = 4.4
    substrate_h: float = 1.6



class AntennaElement(ABC):
    def __init__(self, params: ElementParams):
        self.params = params
        self._pattern_cache = {}
        self._validation_result = None

    @abstractmethod
    def validate(self) -> dict:
        pass

    @abstractmethod
    def pattern(self, theta: np.ndarray, phi: float = 0.0,
                mode: str = 'engineer') -> np.ndarray:
        pass

    @abstractmethod
    def get_physical_size(self) -> tuple:
        pass

    @abstractmethod
    def get_gain_estimate(self) -> float:
        pass

    def clear_cache(self):
        self._pattern_cache.clear()
        self._validation_result = None

    def pattern_db(self, theta, phi=0.0, mode='engineer'):
        pat = self.pattern(theta, phi, mode)
        return 20 * np.log10(np.maximum(pat, 1e-10))

    def get_geometry_svg(self, width=400, height=300) -> str:
        return f'''<svg width="{width}" height="{height}">
            <text x="20" y="{height//2}" font-size="14" fill="#999">
                Эскиз не реализован
            </text></svg>'''


# ============================================================
# Физическое ядро: апертурный метод с отражением
# ============================================================

# ============================================================
# Физическое ядро: три режима расчёта
# ============================================================

def _aperture_field(a, b, L, Nx=200, Ny=160):
    """
    Касательное электрическое поле в раскрыве рупора.
    Режимы 2 и 3 (апертурный и полный).
    """
    k = 2 * np.pi

    x = np.linspace(-a/2, a/2, Nx)
    y = np.linspace(-b/2, b/2, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Амплитуда: основная мода TE10
    E_amp = np.cos(np.pi * X / a)

    # Геометрическая фаза (сферический фронт)
    # Радиусы кривизны фазового фронта
    R_H = L * a / (a - 0.77) if a > 0.78 else L  # a_wg=0.77λ
    R_E = L * b / (b - 0.33) if b > 0.34 else L  # b_wg=0.33λ

    # Эффективный радиус (среднегеометрическое)
    R_eff = np.sqrt(R_H * R_E)

    distance = np.sqrt(R_eff**2 + X**2 + Y**2)
    phase_incident = k * distance

    # Импедансное отражение
    Z0 = 1.0
    r = np.sqrt(X**2 + Y**2)
    theta_local = np.arctan2(r, L)
    cos_theta = np.cos(theta_local)
    Z_freespace = Z0 / (cos_theta + 1e-15)

    lambda_norm = 1.0
    Z_wg = Z0 / np.sqrt(max(1 - (lambda_norm/(2*a))**2, 0.1))
    theta_edge = np.arctan2(a/2, L)
    Z_edge = Z0 / np.cos(theta_edge)

    r_norm = r / (a/2 + 1e-15)
    r_norm = np.clip(r_norm, 0, 1)
    Z_horn = Z_wg + (Z_edge - Z_wg) * r_norm**0.5

    Gamma = (Z_freespace - Z_horn) / (Z_freespace + Z_horn + 1e-15)
    Gamma = np.clip(Gamma, -1, 1)

    E_incident = E_amp * np.exp(-1j * phase_incident)
    E_ap = E_incident * (1 - np.abs(Gamma))

    return X, Y, E_ap, dx, dy

def _edge_waves_H(theta, a, b, L, E_ap, dx, dy):
    """
    PTD-дифракция на кромках в H-плоскости.
    Две кромки при x = ±a/2.
    
    Коэффициент связи с апертурным полем определяется
    из условия непрерывности тока на кромке (Майкснер, Уфимцев).
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    edge_field = np.zeros(len(theta), dtype=np.complex128)

    if not np.any(visible):
        return edge_field

    # Поле на кромках: среднее по y
    edge_left = np.mean(E_ap[:, 0])
    edge_right = np.mean(E_ap[:, -1])

    # Амплитуда краевой волны (PTD)
    # Пропорциональна скачку поля на кромке
    # Для мягкой границы (E параллельно кромке):
    # E_edge = E_aperture(edge) * D_soft(θ) * sqrt(λ/r)
    E0 = (np.abs(edge_left) + np.abs(edge_right)) / 2

    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)

    # Угол раствора рупора
    half_angle = np.arctan(a / (2 * L))

    # Коэффициент дифракции для клина 90° (мягкая граница)
    # D_soft = -exp(-jπ/4) / (2√(2πk)) · [1/cos((θ-θ')/2) - 1/cos((θ+θ')/2)]

    D0 = -np.exp(-1j * np.pi / 4) / (2 * np.sqrt(2 * np.pi * k))

    # Угол падения (от грани клина)
    theta_prime = np.pi/2 + half_angle

    # Углы наблюдения для левой и правой кромок
    theta1 = np.pi/2 - half_angle - theta_v
    theta2 = np.pi/2 - half_angle + theta_v

    # Защита от сингулярности
    eps = 1e-12
    cos_diff1 = np.cos((theta1 - theta_prime) / 2)
    cos_sum1 = np.cos((theta1 + theta_prime) / 2)
    cos_diff2 = np.cos((theta2 - theta_prime) / 2)
    cos_sum2 = np.cos((theta2 + theta_prime) / 2)

    D1 = 1/(cos_diff1 + eps) - 1/(cos_sum1 + eps)
    D2 = 1/(cos_diff2 + eps) - 1/(cos_sum2 + eps)

    # Фаза: разность хода от двух кромок
    phase1 = k * (a/2) * sin_t
    phase2 = -k * (a/2) * sin_t

    # Краевое поле
    edge_visible = E0 * D0 * (
        D1 * np.exp(1j * phase1) + D2 * np.exp(1j * phase2)
    )

    # Множитель наклона для краевых волн
    obliquity_edge = (1 + cos_t) / 2
    edge_field[visible] = edge_visible * obliquity_edge

    return edge_field

def _edge_waves_E(theta, a, b, L, E_ap, dx, dy):
    """
    PTD-дифракция на кромках в E-плоскости.
    Две кромки при y = ±b/2.
    Жёсткая граница (E перпендикулярно кромке).
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    edge_field = np.zeros(len(theta), dtype=np.complex128)

    if not np.any(visible):
        return edge_field

    edge_bottom = np.mean(E_ap[0, :])
    edge_top = np.mean(E_ap[-1, :])
    E0 = (np.abs(edge_bottom) + np.abs(edge_top)) / 2

    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)

    half_angle = np.arctan(b / (2 * L))

    D0 = -np.exp(-1j * np.pi / 4) / (2 * np.sqrt(2 * np.pi * k))

    theta_prime = np.pi/2 + half_angle

    theta1 = np.pi/2 - half_angle - theta_v
    theta2 = np.pi/2 - half_angle + theta_v

    eps = 1e-12
    cos_diff1 = np.cos((theta1 - theta_prime) / 2)
    cos_sum1 = np.cos((theta1 + theta_prime) / 2)
    cos_diff2 = np.cos((theta2 - theta_prime) / 2)
    cos_sum2 = np.cos((theta2 + theta_prime) / 2)

    # Жёсткая граница: знак +
    D1 = 1/(cos_diff1 + eps) + 1/(cos_sum1 + eps)
    D2 = 1/(cos_diff2 + eps) + 1/(cos_sum2 + eps)

    phase1 = k * (b/2) * sin_t
    phase2 = -k * (b/2) * sin_t

    edge_visible = E0 * D0 * (
        D1 * np.exp(1j * phase1) + D2 * np.exp(1j * phase2)
    )

    obliquity_edge = (1 + cos_t) / 2
    edge_field[visible] = edge_visible * obliquity_edge

    return edge_field

def _horn_pattern_engineer(theta, a, b, L, phi):
    """
    Режим 1: Инженерный (Mathcad).
    Формула Баланиса для синфазного раскрыва, без учёта L.
    Быстро, точность ~3°.
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)

    if not np.any(visible):
        return result

    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)

    if np.abs(phi) < 1e-10 or np.abs(phi - np.pi) < 1e-10:
        # H-плоскость
        u = (k * a / 2) * sin_t
        denominator = 1 - (2 * u / np.pi)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.abs((1 + cos_t) / 2 * np.cos(u) / denominator)
            singular = np.abs(denominator) < 1e-8
            if np.any(singular):
                pattern[singular] = (np.pi/4) * (1 + cos_t[singular]) / 2
    else:
        # E-плоскость
        v = (k * b / 2) * sin_t
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(
                np.abs(v) < 1e-10,
                (1 + cos_t) / 2,
                np.abs((1 + cos_t) / 2 * np.sin(v) / v)
            )

    pattern = np.nan_to_num(pattern, nan=0.0)
    max_val = np.max(pattern)
    if max_val > 1e-10:
        pattern /= max_val
    result[visible] = pattern
    return result

def _horn_pattern_aperture(theta, a, b, L, phi):
    """
    Режим 2: Апертурный.
    Интеграл по раскрыву с отражением, без дифракции.
    Средняя скорость, точность ~1.5°.
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)

    if not np.any(visible):
        return result

    theta_v = theta[visible]
    X, Y, E_ap, dx, dy = _aperture_field(a, b, L)

    u = np.sin(theta_v) * np.cos(phi)
    v = np.sin(theta_v) * np.sin(phi)

    pattern = np.zeros(len(theta_v), dtype=np.complex128)
    for idx in range(len(theta_v)):
        phase_factor = np.exp(1j * k * (X * u[idx] + Y * v[idx]))
        pattern[idx] = np.sum(E_ap * phase_factor) * dx * dy

    obliquity = (1 + np.cos(theta_v)) / 2
    pattern_mag = np.abs(pattern) * obliquity

    max_val = np.max(pattern_mag)
    if max_val > 1e-10:
        pattern_mag /= max_val

    pattern_mag = np.nan_to_num(pattern_mag, nan=0.0)
    result[visible] = pattern_mag
    return result

def _horn_pattern_waveguide(theta, a_wg, b_wg, phi):
    """
    ДН открытого конца волновода (без рупора).
    Мода H10.
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)
    
    if not np.any(visible):
        return result
    
    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)
    
    if np.abs(phi) < 1e-10 or np.abs(phi - np.pi) < 1e-10:
        # H-плоскость
        u = (k * a_wg / 2) * sin_t
        denominator = 1 - (2 * u / np.pi)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.abs((1 + cos_t) / 2 * np.cos(u) / denominator)
            singular = np.abs(denominator) < 1e-8
            if np.any(singular):
                pattern[singular] = (np.pi/4) * (1 + cos_t[singular]) / 2
    else:
        # E-плоскость
        v = (k * b_wg / 2) * sin_t
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(
                np.abs(v) < 1e-10,
                (1 + cos_t) / 2,
                np.abs((1 + cos_t) / 2 * np.sin(v) / v)
            )
    
    pattern = np.nan_to_num(pattern, nan=0.0)
    max_val = np.max(pattern)
    if max_val > 1e-10:
        pattern /= max_val
    result[visible] = pattern
    return result

def _horn_pattern_full(theta, a, b, L, phi):
    """
    Режим 3: Полный.
    H-плоскость: апертурный интеграл + PTD-дифракция на кромках.
    E-плоскость: апертурный интеграл (PTD нестабилен для жёсткой границы).
    """
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)

    if not np.any(visible):
        return result

    theta_v = theta[visible]

    # Апертурное поле
    X, Y, E_ap, dx, dy = _aperture_field(a, b, L)

    u = np.sin(theta_v) * np.cos(phi)
    v = np.sin(theta_v) * np.sin(phi)

    pattern_ap = np.zeros(len(theta_v), dtype=np.complex128)
    for idx in range(len(theta_v)):
        phase_factor = np.exp(1j * k * (X * u[idx] + Y * v[idx]))
        pattern_ap[idx] = np.sum(E_ap * phase_factor) * dx * dy

    # Краевые волны — только для H-плоскости
    if np.abs(phi) < 1e-10 or np.abs(phi - np.pi) < 1e-10:
        pattern_edge = _edge_waves_H(theta_v, a, b, L, E_ap, dx, dy)
        pattern_total = pattern_ap + pattern_edge
    else:
        # E-плоскость и промежуточные углы — чистая апертура
        pattern_total = pattern_ap

    obliquity = (1 + np.cos(theta_v)) / 2
    pattern_mag = np.abs(pattern_total) * obliquity

    max_val = np.max(pattern_mag)
    if max_val > 1e-10:
        pattern_mag /= max_val

    pattern_mag = np.nan_to_num(pattern_mag, nan=0.0)
    result[visible] = pattern_mag
    return result

def _horn_pattern_small_H(theta, a, a_wg):
    """H-плоскость малого рупора. a, a_wg в lambda."""
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)
    
    if not np.any(visible):
        return result
    
    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)
    
    a_eff = (a_wg + a) / 2
    u = (k * a_eff / 2) * sin_t
    denominator = 1 - (2 * u / np.pi)**2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.abs((1 + cos_t) / 2 * np.cos(u) / denominator)
        singular = np.abs(denominator) < 1e-8
        if np.any(singular):
            pattern[singular] = (np.pi / 4) * (1 + cos_t[singular]) / 2
    
    pattern = np.nan_to_num(pattern, nan=0.0)
    max_val = np.max(pattern)
    if max_val > 1e-10:
        pattern /= max_val
    result[visible] = pattern
    return result


def _horn_pattern_small_E(theta, b, b_wg):
    """E-плоскость малого рупора. b, b_wg в длинах волн."""
    k = 2 * np.pi
    theta = np.asarray(theta, dtype=np.float64)
    visible = np.abs(theta) <= np.pi/2
    result = np.zeros_like(theta)
    
    if not np.any(visible):
        return result
    
    theta_v = theta[visible]
    sin_t = np.sin(theta_v)
    cos_t = np.cos(theta_v)
    
    b_eff = (b_wg + b) / 2
    v = (k * b_eff / 2) * sin_t
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(
            np.abs(v) < 1e-10,
            (1 + cos_t) / 2,
            np.abs((1 + cos_t) / 2 * np.sin(v) / v)
        )
    
    pattern = np.nan_to_num(pattern, nan=0.0)
    max_val = np.max(pattern)
    if max_val > 1e-10:
        pattern /= max_val
    result[visible] = pattern
    return result

@dataclass
class ConicalHornParams(HornParams):
    """Конический рупор"""
    D: float = 2.5      # диаметр раскрыва, λ
    d_wg: float = 0.77  # диаметр волновода, λ


class ConicalHorn(AntennaElement):
    """Конический рупор. Модель с эффективным размером раскрыва."""
    
    def __init__(self, params: ConicalHornParams):
        super().__init__(params)
        self.p = params
        
        if self.p.L > 0 and self.p.D > self.p.d_wg:
            R = self.p.L * self.p.D / (self.p.D - self.p.d_wg)
            self._phase_error = np.pi * self.p.D**2 / (8 * R)
        elif self.p.L > 0:
            self._phase_error = np.pi * self.p.D**2 / (8 * self.p.L)
        else:
            self._phase_error = float('inf')
    
    def validate(self) -> dict:
        if self._validation_result is not None:
            return self._validation_result
        
        errors = []
        warnings = []
        
        if self.p.D < 0.5:
            errors.append(f"Диаметр D={self.p.D:.2f}λ < 0.5λ")
        if self.p.L < 0.5:
            errors.append(f"Длина L={self.p.L:.2f}λ < 0.5λ")
        
        MAX_PHASE = 3 * np.pi / 4
        
        if self._phase_error > MAX_PHASE:
            errors.append(
                f"Фазовая ошибка {np.degrees(self._phase_error):.0f}° > 135°"
            )
        
        phys_size = self.p.D + 2 * self.p.wall_thickness
        
        result = {
            'valid': len(errors) == 0,
            'optimal': len(errors) == 0 and len(warnings) == 0,
            'warnings': warnings,
            'errors': errors,
            'phase_error_deg': np.degrees(self._phase_error),
            'physical_size': (phys_size, phys_size),
            'gain_estimate': self.get_gain_estimate()
        }
        
        self._validation_result = result
        return result
    
    def pattern(self, theta, phi=0.0, mode='engineer'):
        cache_key = (len(theta), mode)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()
        
        pat = self._pattern_calc(theta)
        self._pattern_cache[cache_key] = pat.copy()
        return pat
    
    def _pattern_calc(self, theta):
        """ДН конического рупора."""
        k = 2 * np.pi
        theta = np.asarray(theta, dtype=np.float64)
        visible = np.abs(theta) <= np.pi/2
        result = np.zeros_like(theta)
        
        if not np.any(visible):
            return result
        
        theta_v = theta[visible]
        sin_t = np.sin(theta_v)
        cos_t = np.cos(theta_v)
        
        # Без эффективного размера — используем геометрический диаметр
        D = self.p.D
        
        u = (k * D / 2) * sin_t
        
        try:
            from scipy.special import j1
        except ImportError:
            j1 = lambda x: np.where(np.abs(x) > 0.1,
                                    np.sqrt(2/(np.pi*np.abs(x))) * np.sin(x - np.pi/4),
                                    0.5 * x * (1 - x**2/8))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(
                np.abs(u) < 1e-8,
                1.0,
                np.abs(2 * j1(u) / u)
            )
            pattern = pattern * (1 + cos_t) / 2
        
        pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_val = np.max(pattern)
        if max_val > 1e-10:
            pattern /= max_val
        
        result[visible] = pattern
        return result
    
    def get_physical_size(self) -> tuple:
        s = self.p.D + 2 * self.p.wall_thickness
        return (s, s)
    
    def get_gain_estimate(self) -> float:
        S = self._phase_error / np.pi
        D_eff = self.p.D / np.sqrt(1 + S**2)
        A_eff = np.pi * (D_eff/2)**2 * 0.84
        G_linear = 4 * np.pi * A_eff
        return 10 * np.log10(max(G_linear, 1.0))
    
    def get_geometry_svg(self, width=400, height=300) -> str:
        D_px = int(self.p.D * 40)
        L_px = int(self.p.L * 25)
        L_px = min(L_px, height - 150)
        D_px = min(D_px, width - 100)
        
        cx = width // 2
        y_top = 60
        y_bot = y_top + L_px
        
        return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .horn {{ fill: #dce8f5; stroke: #2a6090; stroke-width: 2; }}
            .wg   {{ fill: #e0e0e0; stroke: #666; stroke-width: 1.5; }}
            .dim  {{ font-size: 11px; fill: #444; font-family: Times New roman; }}
            .lbl  {{ font-size: 12px; fill: #222; font-weight: bold; font-family: Times New roman; }}
            .dash {{ stroke: #999; stroke-width: 1; stroke-dasharray: 4,4; fill: none; }}
        </style>
        <text x="{cx-60}" y="22" class="lbl">Конический рупор</text>
        <polygon points="{cx-D_px//2},{y_top} {cx+D_px//2},{y_top} {cx+D_px//4},{y_bot} {cx-D_px//4},{y_bot}" class="horn"/>
        <rect x="{cx-D_px//6}" y="{y_bot}" width="{D_px//3}" height="25" class="wg"/>
        <line x1="{cx-D_px//2}" y1="{y_top-15}" x2="{cx+D_px//2}" y2="{y_top-15}" class="dash"/>
        <text x="{cx-25}" y="{y_top-20}" class="dim">D = {self.p.D:.2f}λ</text>
        <text x="{width//2}" y="{height-10}" text-anchor="middle" font-size="10" fill="#888">
            Конический рупор · {self.p.frequency:.2f} ГГц
        </text>
        </svg>'''


def special_j1(x):
    """J₁(x)/x с обработкой особенности в нуле."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = np.abs(x) > 1e-10
    result[mask] = scipy_j1(x[mask]) / x[mask]
    result[~mask] = 0.5
    return result


def scipy_j1(x):
    """J₁(x) — функция Бесселя первого рода первого порядка."""
    try:
        from scipy.special import j1
        return j1(x)
    except ImportError:
        # Аппроксимация при отсутствии scipy
        return np.sin(x - np.pi/4) / np.sqrt(np.pi * np.abs(x) / 2)

class PyramidalHorn(AntennaElement):
    """Пирамидальный рупор. Полная электродинамическая модель."""

    def __init__(self, params: PyramidalHornParams):
        super().__init__(params)
        self.p = params

        if self.p.L > 0:
            # Радиусы кривизны фазового фронта (вершины треугольника рупора)
            R_H = (self.p.L * self.p.a / (self.p.a - self.p.a_wg) 
                if self.p.a > self.p.a_wg + 0.01 else self.p.L)
            R_E = (self.p.L * self.p.b / (self.p.b - self.p.b_wg) 
                if self.p.b > self.p.b_wg + 0.01 else self.p.L)
            
            self._phase_error_H = np.pi * self.p.a**2 / (4 * R_H)
            self._phase_error_E = np.pi * self.p.b**2 / (4 * R_E)
        else:
            self._phase_error_H = float('inf')
            self._phase_error_E = float('inf')

    def validate(self) -> dict:
        if self._validation_result is not None:
            return self._validation_result
        
        errors = []
        warnings = []
        
        if self.p.a < 0.2:
            errors.append(f"Раскрыв a={self.p.a:.2f}λ < 0.2λ")
        if self.p.b < 0.2:
            errors.append(f"Раскрыв b={self.p.b:.2f}λ < 0.2λ")
        if self.p.L < 0.2:
            errors.append(f"Длина L={self.p.L:.2f}λ < 0.2λ")
        
        # Проверка фазовых ошибок (без "оптимальности")
        MAX_PHASE = 3 * np.pi / 4  # 135°
        
        if self._phase_error_H > MAX_PHASE:
            errors.append(
                f"Фазовая ошибка H-плоскости {np.degrees(self._phase_error_H):.0f}° > 135°"
            )
        if self._phase_error_E > MAX_PHASE:
            errors.append(
                f"Фазовая ошибка E-плоскости {np.degrees(self._phase_error_E):.0f}° > 135°"
            )
        
        phys_w = self.p.a + 2 * self.p.wall_thickness
        phys_h = self.p.b + 2 * self.p.wall_thickness
        
        result = {
            'valid': len(errors) == 0,
            'optimal': len(errors) == 0 and len(warnings) == 0,
            'warnings': warnings,
            'errors': errors,
            'phase_error_H_deg': np.degrees(self._phase_error_H),
            'phase_error_E_deg': np.degrees(self._phase_error_E),
            'physical_size': (phys_w, phys_h),
            'gain_estimate': self.get_gain_estimate()
        }
        
        self._validation_result = result
        return result

    def pattern(self, theta, phi=0.0, mode='engineer'):
        cache_key = (len(theta), round(float(phi), 4), mode)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()

        pat = self._pattern_calc(theta, phi)
        self._pattern_cache[cache_key] = pat.copy()
        return pat

    def _horn_pattern_small_E(theta, b, b_wg, L):
        """
        E-плоскость для электрически малого рупора (b < 1.0λ).
        Использует модель открытого конца волновода с фазовой поправкой.
        """
        k = 2 * np.pi
        theta = np.asarray(theta, dtype=np.float64)
        visible = np.abs(theta) <= np.pi/2
        result = np.zeros_like(theta)
        
        if not np.any(visible):
            return result
        
        theta_v = theta[visible]
        sin_t = np.sin(theta_v)
        cos_t = np.cos(theta_v)
        
        # Эффективный размер: среднее между волноводом и раскрывом
        b_eff = np.sqrt(b_wg * b) if b > b_wg else b
        
        v = (k * b_eff / 2) * sin_t
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(
                np.abs(v) < 1e-10,
                (1 + cos_t) / 2,
                np.abs((1 + cos_t) / 2 * np.sin(v) / v)
            )
        
        pattern = np.nan_to_num(pattern, nan=0.0)
        max_val = np.max(pattern)
        if max_val > 1e-10:
            pattern /= max_val
        result[visible] = pattern
        return result


    def _horn_pattern_small_H(theta, a, a_wg, L):
        """
        H-плоскость для электрически малого рупора (a < 1.0λ).
        """
        k = 2 * np.pi
        theta = np.asarray(theta, dtype=np.float64)
        visible = np.abs(theta) <= np.pi/2
        result = np.zeros_like(theta)
        
        if not np.any(visible):
            return result
        
        theta_v = theta[visible]
        sin_t = np.sin(theta_v)
        cos_t = np.cos(theta_v)
        
        a_eff = np.sqrt(a_wg * a) if a > a_wg else a
        
        u = (k * a_eff / 2) * sin_t
        denominator = 1 - (2 * u / np.pi)**2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.abs((1 + cos_t) / 2 * np.cos(u) / denominator)
            singular = np.abs(denominator) < 1e-8
            if np.any(singular):
                pattern[singular] = (np.pi/4) * (1 + cos_t[singular]) / 2
        
        pattern = np.nan_to_num(pattern, nan=0.0)
        max_val = np.max(pattern)
        if max_val > 1e-10:
            pattern /= max_val
        result[visible] = pattern
        return result

    def _pattern_calc(self, theta, phi, mode='full'):
        phi_norm = phi % (2 * np.pi)
        
        if phi_norm < 1e-10 or abs(phi_norm - np.pi) < 1e-10:
            phi_calc = 0.0
        elif abs(phi_norm - np.pi/2) < 1e-10:
            phi_calc = np.pi/2
        else:
            phi_calc = phi_norm
        
        # Для малых рупоров — простые аналитические формулы
        if self.p.a < 0.8 or self.p.b < 0.8:
            if np.abs(phi_calc) < 1e-10 or abs(phi_calc - np.pi) < 1e-10:
                return _horn_pattern_small_H(theta, self.p.a, self.p.a_wg)
            elif abs(phi_calc - np.pi/2) < 1e-10:
                return _horn_pattern_small_E(theta, self.p.b, self.p.b_wg)
        
        # Для нормальных рупоров — апертурный метод
        if mode == 'engineer':
            return _horn_pattern_engineer(theta, self.p.a, self.p.b, self.p.L, phi_calc)
        elif mode == 'aperture':
            return _horn_pattern_aperture(theta, self.p.a, self.p.b, self.p.L, phi_calc)
        else:
            return _horn_pattern_full(theta, self.p.a, self.p.b, self.p.L, phi_calc)

    def pattern(self, theta, phi=0.0, mode='full'):
        """
        ДН рупора.
        mode: 'engineer' (быстро), 'aperture' (средне), 'full' (точно)
        """
        cache_key = (len(theta), round(float(phi), 4), mode)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()

        phi_norm = phi % (2 * np.pi)
        
        if phi_norm < 1e-10 or abs(phi_norm - np.pi) < 1e-10:
            pat = self._pattern_calc(theta, phi, mode)
        elif abs(phi_norm - np.pi/2) < 1e-10:
            pat = self._pattern_calc(theta, phi, mode)
        else:
            pat_H = self._pattern_calc(theta, 0.0, mode)
            pat_E = self._pattern_calc(theta, np.pi/2, mode)
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            pat = np.sqrt(pat_H**2 * cphi**2 + pat_E**2 * sphi**2)

        self._pattern_cache[cache_key] = pat.copy()
        return pat

    def get_physical_size(self) -> tuple:
        return (
            self.p.a + 2 * self.p.wall_thickness,
            self.p.b + 2 * self.p.wall_thickness
        )

    def get_gain_estimate(self) -> float:
        A_eff = self.p.a * self.p.b * 0.81
        G_linear = 4 * np.pi * A_eff
        return 10 * np.log10(max(G_linear, 1.0))

    def get_geometry_svg(self, width=400, height=300) -> str:
        a_px = int(self.p.a * 40)
        b_px = int(self.p.b * 40)
        L_px = int(self.p.L * 25)
        L_px = min(L_px, height - 150)
        a_px = min(a_px, width - 100)
        b_px = min(b_px, width - 100)
        cx = width // 2
        y_h = 50
        y_e = y_h + L_px + 60

        return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow" markerWidth="6" markerHeight="4" refX="3" refY="2" orient="auto">
                <polygon points="0,0 6,2 0,4" fill="#555"/>
            </marker>
        </defs>
        <style>
            .horn {{ fill: #dce8f5; stroke: #2a6090; stroke-width: 2; }}
            .wg   {{ fill: #e0e0e0; stroke: #666; stroke-width: 1.5; }}
            .dim  {{ font-size: 11px; fill: #444; font-family: Times New roman; }}
            .lbl  {{ font-size: 12px; fill: #222; font-weight: bold; font-family: Times New roman; }}
            .dash {{ stroke: #999; stroke-width: 1; stroke-dasharray: 4,4; fill: none; }}
        </style>
        <text x="{cx-60}" y="22" class="lbl">H-плоскость</text>
        <polygon points="{cx-a_px//2},{y_h} {cx+a_px//2},{y_h} {cx+a_px//3},{y_h+L_px} {cx-a_px//3},{y_h+L_px}" class="horn"/>
        <rect x="{cx-a_px//4}" y="{y_h+L_px}" width="{a_px//2}" height="25" class="wg"/>
        <line x1="{cx-a_px//2}" y1="{y_h-15}" x2="{cx+a_px//2}" y2="{y_h-15}" class="dash" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="{cx-25}" y="{y_h-20}" class="dim">a = {self.p.a:.2f}λ</text>
        <line x1="{cx+a_px//2+15}" y1="{y_h}" x2="{cx+a_px//2+15}" y2="{y_h+L_px}" class="dash" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="{cx+a_px//2+18}" y="{y_h+L_px//2+5}" class="dim">L = {self.p.L:.1f}λ</text>
        <text x="{cx-60}" y="{y_e-10}" class="lbl">E-плоскость</text>
        <polygon points="{cx-b_px//2},{y_e} {cx+b_px//2},{y_e} {cx+b_px//3},{y_e+L_px} {cx-b_px//3},{y_e+L_px}" class="horn"/>
        <rect x="{cx-b_px//4}" y="{y_e+L_px}" width="{b_px//2}" height="25" class="wg"/>
        <line x1="{cx-b_px//2}" y1="{y_e-15}" x2="{cx+b_px//2}" y2="{y_e-15}" class="dash" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
        <text x="{cx-25}" y="{y_e-20}" class="dim">b = {self.p.b:.2f}λ</text>
        <text x="{width//2}" y="{height-10}" text-anchor="middle" font-size="10" fill="#888">
            Пирамидальный рупор · {self.p.frequency:.2f} ГГц
        </text>
        </svg>'''


# ============================================================
# Заглушки
# ============================================================

class PatchAntenna(AntennaElement):
    def __init__(self, params: PatchParams = None):
        if params is None:
            params = PatchParams()
        super().__init__(params)

    def validate(self) -> dict:
        return {
            'valid': True, 'optimal': True,
            'warnings': ['Заглушка: точность ограничена'],
            'errors': [],
            'physical_size': (0.5, 0.5),
            'gain_estimate': self.get_gain_estimate()
        }

    def pattern(self, theta, phi=0.0, mode='engineer'):
        cache_key = len(theta)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()
        theta = np.asarray(theta)
        pat = np.cos(theta)**1.5
        pat[np.abs(theta) > np.pi/2] = 0
        pat /= np.max(pat) if np.max(pat) > 0 else 1
        self._pattern_cache[cache_key] = pat.copy()
        return pat

    def get_physical_size(self) -> tuple:
        return (0.5, 0.5)

    def get_gain_estimate(self) -> float:
        return 7.0


class DipoleAntenna(AntennaElement):
    def __init__(self, params: ElementParams = None):
        if params is None:
            params = ElementParams()
        super().__init__(params)

    def validate(self) -> dict:
        return {
            'valid': True, 'optimal': True,
            'warnings': ['Заглушка: аналитическая ДН'],
            'errors': [],
            'physical_size': (0.5, 0.05),
            'gain_estimate': self.get_gain_estimate()
        }

    def pattern(self, theta, phi=0.0, mode='engineer'):
        cache_key = len(theta)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()
        theta = np.asarray(theta)
        with np.errstate(divide='ignore', invalid='ignore'):
            pat = np.abs(np.cos(np.pi/2 * np.cos(theta)) / np.sin(theta))
            pat[np.isnan(pat)] = 1.0
        pat[np.abs(theta) > np.pi/2] = 0
        max_val = np.max(pat)
        if max_val > 0:
            pat /= max_val
        self._pattern_cache[cache_key] = pat.copy()
        return pat

    def get_physical_size(self) -> tuple:
        return (0.5, 0.05)

    def get_gain_estimate(self) -> float:
        return 2.15


class IsotropicElement(AntennaElement):
    def __init__(self, params: ElementParams = None):
        if params is None:
            params = ElementParams()
        super().__init__(params)

    def validate(self) -> dict:
        return {
            'valid': True, 'optimal': True,
            'warnings': [], 'errors': [],
            'physical_size': (0, 0),
            'gain_estimate': 0.0
        }

    def pattern(self, theta, phi=0.0, mode='engineer'):
        theta = np.asarray(theta)
        pat = np.ones_like(theta)
        pat[np.abs(theta) > np.pi/2] = 0
        return pat

    def get_physical_size(self) -> tuple:
        return (0.0, 0.0)

    def get_gain_estimate(self) -> float:
        return 0.0


# ============================================================
# Фабрика
# ============================================================

def create_element(element_type: ElementType, **kwargs) -> AntennaElement:
    freq = kwargs.get('frequency', 8.33)

    if element_type == ElementType.HORN_PYRAMIDAL:
        params = PyramidalHornParams(
            frequency=freq,
            a=kwargs.get('a', 2.5),
            b=kwargs.get('b', 1.96),
            L=kwargs.get('L', 3.0),
            a_wg=kwargs.get('a_wg', 0.77),
            b_wg=kwargs.get('b_wg', 0.33)
        )
        return PyramidalHorn(params)

    elif element_type == ElementType.PATCH:
        return PatchAntenna(PatchParams(frequency=freq))

    elif element_type == ElementType.DIPOLE:
        return DipoleAntenna(ElementParams(frequency=freq))

    elif element_type == ElementType.ISOTROPIC:
        return IsotropicElement(ElementParams(frequency=freq))
    elif element_type == ElementType.HORN_CONICAL:
        params = ConicalHornParams(
            frequency=freq,
            D=kwargs.get('D', 2.5),
            d_wg=kwargs.get('d_wg', 0.77),
            L=kwargs.get('L', 3.0)
        )
        return ConicalHorn(params)

    else:
        raise ValueError(f"Неизвестный тип: {element_type}")