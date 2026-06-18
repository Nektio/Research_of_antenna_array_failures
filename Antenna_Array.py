# antenna_array_simulator.py
import sys
import os
import traceback
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from functools import lru_cache
import hashlib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.integrate import trapezoid
    SCIPY_AVAILABLE = True
    from antenna_elements import (AntennaElement, ElementType, ElementParams,
                                PyramidalHornParams, PyramidalHorn,
                                PatchParams, PatchAntenna,
                                DipoleAntenna, IsotropicElement,
                                create_element)
except ImportError:
    SCIPY_AVAILABLE = False

# Создаем лог-файл с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"antenna_debug_{timestamp}.log"

def write_log(msg, level="INFO"):
    """Записать сообщение в лог с уровнем"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted_msg = f"[{ts}] [{level}] {msg}"
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + "\n")
    except:
        pass
    print(formatted_msg)

# Очищаем старый лог
if os.path.exists(log_file):
    os.remove(log_file)

write_log("=" * 60, "INFO")
write_log("НАЧАЛО РАБОТЫ ПРОГРАММЫ", "INFO")
write_log(f"Python: {sys.version}", "INFO")
write_log(f"Рабочая директория: {os.getcwd()}", "INFO")
write_log("=" * 60, "INFO")

try:
    write_log("Импортируем необходимые библиотеки...", "INFO")
    
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QLabel, QPushButton, 
                             QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
                             QTabWidget, QGridLayout, QTextEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QDialog, QMessageBox, QProgressBar, QSplitter,
                             QSplashScreen, QScrollArea, QFrame, QFileDialog)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QBrush, QFont
    
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    
    write_log("Все импорты успешны", "INFO")
    
except Exception as e:
    write_log(f"ОШИБКА ПРИ ИМПОРТЕ: {str(e)}", "ERROR")
    write_log(traceback.format_exc(), "ERROR")
    input("Нажмите Enter для выхода...")
    sys.exit(1)


class SplashScreen(QSplashScreen):
    """Заставка при запуске программы"""
    
    def __init__(self):
        from PyQt5.QtGui import QPixmap, QPainter, QPen
        
        pixmap = QPixmap(700, 400)
        pixmap.fill(QColor(245, 245, 245))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(QColor(0, 100, 180), 2))
        painter.setFont(QFont('Times New roman', 20, QFont.Bold))
        painter.drawText(0, 70, 700, 40, Qt.AlignCenter, "Исследование влияния отказов на характеристики ФАР")
        
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setFont(QFont('Times New roman', 14))
        painter.drawText(0, 120, 700, 30, Qt.AlignCenter, "Лабораторная работа")
        
        painter.setFont(QFont('Times New roman', 11))
        painter.drawText(0, 160, 700, 30, Qt.AlignCenter, "Компенсация отказов")
        
        # Рисуем схематичную решетку
        painter.setPen(QPen(QColor(0, 120, 200), 1))
        painter.setBrush(QBrush(QColor(200, 225, 250)))
        
        start_x, start_y = 200, 230
        spacing = 35
        for i in range(4):
            for j in range(8):
                x = start_x + j * spacing
                y = start_y + i * spacing
                painter.drawRect(x, y, 28, 28)
        
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setFont(QFont('Times New roman', 10))
        painter.drawText(0, 370, 700, 20, Qt.AlignCenter, "Кликните для продолжения")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    
    def mousePressEvent(self, event):
        self.close()


class AntennaArray:
    """Модель антенной решетки с кэшированием"""
    
    def __init__(self, Nx=5, Ny=5, dx=0.594, dy=0.594):
        self.cached_pattern_ideal_H = None
        self.cached_pattern_ideal_E = None
        write_log(f"Создание решетки {Nx}x{Ny}, dx={dx}, dy={dy}", "DEBUG")
        self.Nx = max(1, min(Nx, 50))
        self.Ny = max(1, min(Ny, 50))
        self.dx = max(0.1, min(dx, 2.0))
        self.dy = max(0.1, min(dy, 2.0))

        default_params = PyramidalHornParams(
            frequency=8.33,
            a=0.5,
            b=0.5,
            L=0.5,
            a_wg=0.77,
            b_wg=0.33
        )

        # Единый излучатель (объект AntennaElement)
        self.element = PyramidalHorn(default_params)
        self.element_type = 'horn'
        self.horn_a = 0.5
        self.horn_b = 0.5
        self.delta = 0.5
        self.horn_L = 0.5
        
        self.element_mask = np.ones((self.Ny, self.Nx), dtype=bool)
        self._base_amplitudes = np.ones((self.Ny, self.Nx))
        self.amplitudes = np.ones((self.Ny, self.Nx))
        self.phases = np.zeros((self.Ny, self.Nx))
        
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._update_counter = 0
        
        self.last_pattern = None
        self.last_theta = None
        self.last_phi = 0
        
        # Хранилище состояний для сравнения до/после
        self.state_history = {
            'ideal': None,
            'failed': None,
            'compensated': None
        }
        
        self._precalculate_coordinates()
        write_log("Решетка создана успешно", "DEBUG")
    
    def cache_ideal_pattern(self):
        """Сохранить текущую ДН как идеальную (до отказов)"""
        theta_H, pattern_H = self.calculate_pattern(phi_slice=0)
        theta_E, pattern_E = self.calculate_pattern(phi_slice=np.pi/2)
        self.cached_pattern_ideal_H = (theta_H.copy(), pattern_H.copy())
        self.cached_pattern_ideal_E = (theta_E.copy(), pattern_E.copy())

    def set_element(self, element: AntennaElement):
        """Установить излучатель"""
        self.element = element
        if isinstance(element, PyramidalHorn):
            self.element_type = 'horn'
            self.horn_a = element.p.a
            self.horn_b = element.p.b
            self.horn_L = element.p.L
        elif isinstance(element, PatchAntenna):
            self.element_type = 'patch'
        elif isinstance(element, DipoleAntenna):
            self.element_type = 'dipole'
        elif isinstance(element, IsotropicElement):
            self.element_type = 'isotropic'
        self._cache.clear()
    
    def save_state(self, state_name):
        """Сохранить текущее состояние решётки для сравнения"""
        self.state_history[state_name] = {
            'amplitudes': self.amplitudes.copy(),
            'phases': self.phases.copy(),
            'mask': self.element_mask.copy(),
            'element_type': self.element_type,
            'dx': self.dx,
            'dy': self.dy
        }
    
    def restore_state(self, state_name):
        """Восстановить сохранённое состояние"""
        state = self.state_history.get(state_name)
        if state is None:
            return False
        self.amplitudes = state['amplitudes'].copy()
        self.phases = state['phases'].copy()
        self.element_mask = state['mask'].copy()
        self.dx = state['dx']
        self.dy = state['dy']
        self._cache.clear()
        return True
    
    def _precalculate_coordinates(self):
        j_indices = np.arange(self.Nx)
        i_indices = np.arange(self.Ny)
        self._J, self._I = np.meshgrid(j_indices, i_indices)
        
        # Нормированные координаты для амплитудного распределения
        if self.Nx > 1:
            self._X_norm = np.linspace(-1, 1, self.Nx)
        else:
            self._X_norm = np.array([0.0])

        if self.Ny > 1:
            self._Y_norm = np.linspace(-1, 1, self.Ny)
        else:
            self._Y_norm = np.array([0.0])
        
        self._X_norm_grid, self._Y_norm_grid = np.meshgrid(self._X_norm, self._Y_norm)
        
        center_x = (self.Nx - 1) / 2 * self.dx
        center_y = (self.Ny - 1) / 2 * self.dy
        
        self._X_pos = self._J * self.dx - center_x
        self._Y_pos = self._I * self.dy - center_y
    
    def get_min_element_spacing(self):
        if self.element_type == 'horn':
            return round(max(self.horn_a, self.horn_b) * 1.05, 3)
        elif self.element_type == 'dipole':
            return 0.25
        elif self.element_type == 'patch':
            return 0.3
        else:
            return 0.1
    
    def validate_spacing(self, dx, dy):
        min_spacing = self.get_min_element_spacing()
        warnings_list = []
        if dx < min_spacing:
            warnings_list.append(f"dx={dx:.3f} меньше рекомендуемого {min_spacing:.3f}")
        if dy < min_spacing:
            warnings_list.append(f"dy={dy:.3f} меньше рекомендуемого {min_spacing:.3f}")
        return warnings_list
    
    def _get_cache_key(self):
        state = {
            'Nx': self.Nx, 'Ny': self.Ny,
            'dx': self.dx, 'dy': self.dy,
            'element_type': self.element_type,
            'amplitudes_hash': hashlib.md5(self.amplitudes.tobytes()).hexdigest(),
            'phases_hash': hashlib.md5(self.phases.tobytes()).hexdigest(),
            'mask_hash': hashlib.md5(self.element_mask.tobytes()).hexdigest()
        }
        return hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
    
    def set_element_type(self, element_type):
        valid_types = ['horn', 'dipole', 'patch', 'isotropic']
        if element_type not in valid_types:
            write_log(f"Неверный тип излучателя: {element_type}", "WARNING")
            return
        self.element_type = element_type
        self._cache.clear()
        write_log(f"Тип излучателя изменен на: {element_type}", "DEBUG")
    
    def get_element_pattern(self, theta, phi=0, mode='engineer'):
        """ДН элемента через объект AntennaElement"""
        try:
            if hasattr(self.element, 'pattern'):
                return self.element.pattern(theta, phi, mode)
        except Exception:
            pass
        
        # Fallback: старые модели (если модуль не загружен)
        return self._get_element_pattern_legacy(theta, phi)


    def _get_element_pattern_legacy(self, theta, phi=0):
        """Старая модель ДН (на случай отсутствия antenna_elements)"""
        try:
            if self.element_type == 'horn':
                if np.abs(phi) < 1e-10 or np.abs(phi - np.pi) < 1e-10:
                    return self._horn_pattern_H_legacy(theta)
                elif np.abs(phi - np.pi/2) < 1e-10 or np.abs(phi - 3*np.pi/2) < 1e-10:
                    return self._horn_pattern_E_legacy(theta)
                else:
                    pattern_H = self._horn_pattern_H_legacy(theta)
                    pattern_E = self._horn_pattern_E_legacy(theta)
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)
                    return np.sqrt(pattern_H**2 * cos_phi**2 + pattern_E**2 * sin_phi**2)
            elif self.element_type == 'dipole':
                theta_arr = np.asarray(theta)
                result = np.zeros_like(theta_arr)
                valid_mask = np.abs(theta_arr) <= np.pi/2
                if np.any(valid_mask):
                    theta_valid = theta_arr[valid_mask]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pattern = np.where(
                            np.abs(np.sin(theta_valid)) < 1e-10,
                            1.0,
                            np.abs(np.cos(np.pi/2 * np.cos(theta_valid)) / np.sin(theta_valid))
                        )
                        pattern = np.nan_to_num(pattern, nan=0.0)
                    max_val = np.max(pattern)
                    if max_val > 0:
                        pattern /= max_val
                    result[valid_mask] = pattern
                return result
            elif self.element_type == 'patch':
                theta_arr = np.asarray(theta)
                result = np.cos(theta_arr) ** 1.5
                result = np.where(np.abs(theta_arr) <= np.pi/2, result, 0)
                return result
            else:
                theta_arr = np.asarray(theta)
                result = np.ones_like(theta_arr)
                result[np.abs(theta_arr) > np.pi/2] = 0
                return result
        except Exception:
            theta_arr = np.asarray(theta)
            result = np.ones_like(theta_arr)
            result[np.abs(theta_arr) > np.pi/2] = 0
            return result


    def _horn_pattern_H_legacy(self, theta):
        """Заглушка H-плоскости (старая модель)"""
        try:
            a = self.horn_a
            k = 2 * np.pi
            theta = np.asarray(theta)
            valid_mask = np.abs(theta) <= np.pi/2
            result = np.zeros_like(theta)
            if np.any(valid_mask):
                theta_valid = theta[valid_mask]
                sin_theta = np.sin(theta_valid)
                u = (k * a / 2) * sin_theta
                denominator = 1 - (2 * a * sin_theta / np.pi)**2
                safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
                with np.errstate(divide='ignore', invalid='ignore'):
                    pattern = np.abs((1 + np.cos(theta_valid)) / 2 * np.cos(u) / safe_denom)
                    singular = np.abs(denominator) < 1e-10
                    if np.any(singular):
                        pattern[singular] = (np.pi/4) * (1 + np.cos(theta_valid[singular])) / 2
                    pattern = np.nan_to_num(pattern, nan=0.0)
                max_val = np.max(pattern)
                if max_val > 0:
                    pattern /= max_val
                result[valid_mask] = pattern
            return result
        except Exception:
            return np.ones_like(theta)


    def _horn_pattern_E_legacy(self, theta):
        """Заглушка E-плоскости (старая модель)"""
        try:
            b = self.horn_b
            k = 2 * np.pi
            theta = np.asarray(theta)
            valid_mask = np.abs(theta) <= np.pi/2
            result = np.zeros_like(theta)
            if np.any(valid_mask):
                theta_valid = theta[valid_mask]
                u = (k * b / 2) * np.sin(theta_valid)
                with np.errstate(divide='ignore', invalid='ignore'):
                    pattern = np.where(
                        np.abs(u) < 1e-10,
                        (1 + np.cos(theta_valid)) / 2,
                        np.abs((1 + np.cos(theta_valid)) / 2 * np.sin(u) / u)
                    )
                    pattern = np.nan_to_num(pattern, nan=0.0)
                max_val = np.max(pattern)
                if max_val > 0:
                    pattern /= max_val
                result[valid_mask] = pattern
            return result
        except Exception:
            return np.ones_like(theta)


    def get_min_element_spacing(self):
        """Минимальный шаг решётки для текущего излучателя"""
        if hasattr(self.element, 'get_physical_size'):
            phys_w, phys_h = self.element.get_physical_size()
            return round(max(phys_w, phys_h) * 1.1, 3)
        
        if self.element_type == 'horn':
            return round(max(self.horn_a, self.horn_b) * 1.1, 3)
        elif self.element_type == 'dipole':
            return 0.25
        elif self.element_type == 'patch':
            return 0.3
        else:
            return 0.1
    
    def update_amplitude_distribution(self, dist_type, **params):
        write_log(f"Обновление распределения: {dist_type}", "DEBUG")
        try:
            valid_types = ["uniform", "cosine", "cosine_pedestal", 
                          "triangular", "gaussian", "quadratic"]
            
            if dist_type not in valid_types:
                write_log(f"Неверный тип распределения: {dist_type}", "WARNING")
                return
            
            # Для линейной решётки одна из координат = 0
            if self.Nx == 1:
                X = np.zeros((self.Ny, self.Nx))
            else:
                X = self._X_norm_grid

            if self.Ny == 1:
                Y = np.zeros((self.Ny, self.Nx))
            else:
                Y = self._Y_norm_grid

            # Если обе координаты существуют — используем сетку
            if self.Nx > 1 and self.Ny > 1:
                X = self._X_norm_grid
                Y = self._Y_norm_grid
            elif self.Nx == 1 and self.Ny > 1:
                X = np.zeros((self.Ny, self.Nx))
                Y = self._Y_norm_grid
            elif self.Nx > 1 and self.Ny == 1:
                X = self._X_norm_grid
                Y = np.zeros((self.Ny, self.Nx))
            else:
                X = np.zeros((1, 1))
                Y = np.zeros((1, 1))
            
            if dist_type == "uniform":
                amp = np.ones_like(X)
            elif dist_type == "cosine":
                amp = np.cos(X * np.pi/2) * np.cos(Y * np.pi/2)
            elif dist_type == "cosine_pedestal":
                pedestal = np.clip(params.get('pedestal', 0.2), 0.0, 1.0)
                amp = np.cos(X * np.pi/2) * np.cos(Y * np.pi/2)
                amp_min, amp_max = np.min(amp), np.max(amp)
                if amp_max > amp_min:
                    amp_norm = (amp - amp_min) / (amp_max - amp_min)
                    amp = pedestal + (1 - pedestal) * amp_norm
            elif dist_type == "triangular":
                amp = (1 - np.abs(X)) * (1 - np.abs(Y))
                amp = np.maximum(amp, 0)
            elif dist_type == "gaussian":
                sigma = np.clip(params.get('sigma', 0.5), 0.1, 2.0)
                amp = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            elif dist_type == "quadratic":
                delta = np.clip(params.get('delta', 0.5), 0.0, 1.0)
                amp_x = 1 - (1 - delta) * X**2
                amp_y = 1 - (1 - delta) * Y**2
                amp = amp_x * amp_y
                amp = np.maximum(amp, delta)
            
            max_amp = np.max(amp)
            if max_amp > 0:
                amp = amp / max_amp
            
            self._base_amplitudes = amp
            self._update_current_amplitudes()
            self._cache.clear()
            
        except Exception as e:
            write_log(f"Ошибка в update_amplitude_distribution: {e}", "ERROR")
    
    def _update_current_amplitudes(self):
        self.amplitudes = self._base_amplitudes.copy()
        self.amplitudes[~self.element_mask] = 0
        self._update_counter += 1
    
    def set_element_state(self, i, j, state):
        if 0 <= i < self.Ny and 0 <= j < self.Nx:
            old_state = self.element_mask[i, j]
            self.element_mask[i, j] = state
            
            if state:
                self.amplitudes[i, j] = self._base_amplitudes[i, j]
            else:
                self.amplitudes[i, j] = 0
            
            if old_state != state:
                self._cache.clear()
                self._update_counter += 1
    
    def set_scan_angle(self, theta_x_deg, theta_y_deg=0):
        """Независимое сканирование по X (H) и Y (E)"""
        try:
            theta_x = np.radians(np.clip(theta_x_deg, -90, 90))
            theta_y = np.radians(np.clip(theta_y_deg, -90, 90))
            
            k = 2 * np.pi
            
            self.phases = -k * (self._X_pos * np.sin(theta_x) + 
                            self._Y_pos * np.sin(theta_y))
            self._cache.clear()
            
        except Exception as e:
            write_log(f"Ошибка в set_scan_angle: {e}", "ERROR")
    
    def _array_factor(self, theta, phi=0):
        try:
            k = 2 * np.pi
            theta = np.asarray(theta).reshape(-1, 1, 1)
            
            sin_theta = np.sin(theta)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            spatial_phase = k * (self._X_pos.reshape(1, self.Ny, self.Nx) * sin_theta * cos_phi +
                               self._Y_pos.reshape(1, self.Ny, self.Nx) * sin_theta * sin_phi)
            
            amps = self.amplitudes.reshape(1, self.Ny, self.Nx)
            phases = self.phases.reshape(1, self.Ny, self.Nx)
            mask = self.element_mask.reshape(1, self.Ny, self.Nx)
            
            contributions = amps * mask * np.exp(1j * (spatial_phase + phases))
            af = np.sum(contributions, axis=(1, 2))
            
            af_magnitude = np.abs(af.flatten())
            
            if self.dx > 0.5 or self.dy > 0.5:
                active_count = self.get_active_count()
                if active_count > 0:
                    af_magnitude = af_magnitude / active_count
                    max_val = np.max(af_magnitude)
                    if max_val > 0:
                        af_magnitude = af_magnitude / max_val
            
            return af_magnitude
            
        except Exception as e:
            write_log(f"Ошибка в _array_factor: {e}", "ERROR")
            theta = np.asarray(theta)
            return np.ones_like(theta)
    
    def calculate_pattern(self, phi_slice=0):
        try:
            cache_key = (phi_slice, self._get_cache_key())
            
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            
            self._cache_misses += 1
            
            theta = np.linspace(-np.pi/2, np.pi/2, 361)
            elem_pattern = self.get_element_pattern(theta, phi_slice)
            array_factor = self._array_factor(theta, phi_slice)
            
            max_af = np.max(array_factor)
            if max_af > 0:
                array_factor /= max_af
            
            total_pattern = elem_pattern * array_factor
            
            max_val = np.max(total_pattern)
            if max_val > 0:
                total_pattern /= max_val
            
            self.last_theta = theta
            self.last_pattern = total_pattern
            self.last_phi = phi_slice
            
            self._cache[cache_key] = (theta.copy(), total_pattern.copy())
            
            return theta, total_pattern
            
        except Exception as e:
            write_log(f"Ошибка в calculate_pattern: {e}", "ERROR")
            theta = np.linspace(-np.pi/2, np.pi/2, 181)
            pattern = np.cos(theta) ** 2
            return theta, pattern
    
    def calculate_directivity(self):
        try:
            if self.last_pattern is None or self.last_theta is None:
                return 0
            
            theta = self.last_theta
            pattern = self.last_pattern
            
            visible_mask = np.abs(theta) <= np.pi/2
            theta_vis = theta[visible_mask]
            pattern_vis = pattern[visible_mask]
            
            if len(theta_vis) < 2:
                return 0
            
            power_pattern = pattern_vis**2 * np.sin(np.abs(theta_vis))
            
            integral = 0
            if SCIPY_AVAILABLE and len(theta_vis) >= 3:
                integral = trapezoid(power_pattern, theta_vis)
            elif len(theta_vis) >= 3:
                integral = np.sum((power_pattern[:-1] + power_pattern[1:]) * np.diff(theta_vis)) / 2
            else:
                integral = np.sum(power_pattern) * (theta_vis[1] - theta_vis[0]) if len(theta_vis) > 1 else 0
            
            if integral > 1e-10:
                max_power = np.max(pattern_vis**2)
                D = 4 * np.pi * max_power / integral
            else:
                D = 1
            
            active_count = self.get_active_count()
            S_eff = self.Nx * self.dx * self.Ny * self.dy * (active_count / (self.Nx * self.Ny))
            D_max_theoretical = 4 * np.pi * S_eff
            D = min(D, D_max_theoretical)
            
            if self.dx > 0.5 or self.dy > 0.5:
                grating_loss_factor = min(1.0, 0.5 / max(self.dx, self.dy))
                D *= grating_loss_factor
            
            D_dBi = 10 * np.log10(max(D, 1e-10))
            return D_dBi
            
        except Exception as e:
            write_log(f"Ошибка в calculate_directivity: {e}", "ERROR")
            return 0

    def calculate_aperture_efficiency(self):
        try:
            if self.last_pattern is None:
                self.calculate_pattern()
            
            D_real_dBi = self.calculate_directivity()
            D_real = 10 ** (D_real_dBi / 10)
            
            S_ideal = self.Nx * self.dx * self.Ny * self.dy
            D_ideal = 4 * np.pi * S_ideal
            
            active_ratio = self.get_active_count() / (self.Nx * self.Ny)
            D_ideal *= active_ratio
            
            efficiency = (D_real / D_ideal) * 100 if D_ideal > 0 else 0
            return min(efficiency, 100)
            
        except Exception as e:
            write_log(f"Ошибка в calculate_aperture_efficiency: {e}", "ERROR")
            return 0
    
    def calculate_sll(self):
        try:
            if self.last_pattern is None:
                return -100
            
            pattern = self.last_pattern
            theta = self.last_theta
            
            pattern_db = 20 * np.log10(np.maximum(pattern, 1e-10))
            gradient = np.gradient(pattern_db)
            
            peaks = []
            for i in range(1, len(gradient)-1):
                if gradient[i-1] > 0 and gradient[i] <= 0:
                    peaks.append(i)
            
            if not peaks:
                return -100
            
            main_peak_idx = peaks[np.argmax([pattern_db[p] for p in peaks])]
            sidelobe_peaks = [p for p in peaks if p != main_peak_idx]
            
            if sidelobe_peaks:
                max_sidelobe = max([pattern_db[p] for p in sidelobe_peaks])
                return max_sidelobe
            else:
                return -100
                
        except Exception as e:
            write_log(f"Ошибка в calculate_sll: {e}", "ERROR")
            return -100
    
    def get_beamwidth_3db(self):
        try:
            if self.last_pattern is None or self.last_theta is None:
                return 0
            
            pattern_db = 20 * np.log10(np.maximum(self.last_pattern, 1e-10))
            theta_deg = np.degrees(self.last_theta)
            
            max_idx = np.argmax(pattern_db)
            max_level = pattern_db[max_idx]
            target_level = max_level - 3
            
            left_idx = None
            for i in range(max_idx, 0, -1):
                if pattern_db[i] <= target_level:
                    left_idx = i
                    break
            
            right_idx = None
            for i in range(max_idx, len(pattern_db)):
                if pattern_db[i] <= target_level:
                    right_idx = i
                    break
            
            if left_idx is None or right_idx is None:
                return 0
            
            if left_idx > 0 and pattern_db[left_idx-1] > pattern_db[left_idx]:
                t1, p1 = theta_deg[left_idx-1], pattern_db[left_idx-1]
                t2, p2 = theta_deg[left_idx], pattern_db[left_idx]
                if p1 != p2:
                    left_angle = t1 + (target_level - p1) * (t2 - t1) / (p2 - p1)
                else:
                    left_angle = theta_deg[left_idx]
            else:
                left_angle = theta_deg[left_idx]
            
            if right_idx < len(pattern_db)-1 and pattern_db[right_idx] > pattern_db[right_idx+1]:
                t1, p1 = theta_deg[right_idx], pattern_db[right_idx]
                t2, p2 = theta_deg[right_idx+1], pattern_db[right_idx+1]
                if p1 != p2:
                    right_angle = t1 + (target_level - p1) * (t2 - t1) / (p2 - p1)
                else:
                    right_angle = theta_deg[right_idx]
            else:
                right_angle = theta_deg[right_idx]
            
            return abs(right_angle - left_angle)
            
        except Exception as e:
            write_log(f"Ошибка в get_beamwidth_3db: {e}", "ERROR")
            return 0
    
    def get_active_count(self):
        return np.sum(self.element_mask)
    
    def check_grating_lobes(self, scan_angle_deg=0):
        theta_scan = np.radians(scan_angle_deg)
        warnings_list = []
        
        if self.dx > 0.5:
            try:
                sin_arg = 1/self.dx - np.sin(theta_scan)
                if -1 <= sin_arg <= 1:
                    angle_x = np.degrees(np.arcsin(sin_arg))
                    warnings_list.append(f"dx={self.dx:.2f} > 0.5! Grating lobe в H-плоскости на {angle_x:.1f} град")
                else:
                    warnings_list.append(f"dx={self.dx:.2f} > 0.5! Grating lobes возможны за пределами видимой области")
            except:
                warnings_list.append(f"dx={self.dx:.2f} > 0.5! Возможны grating lobes")
        
        if self.dy > 0.5:
            try:
                sin_arg = 1/self.dy - np.sin(theta_scan)
                if -1 <= sin_arg <= 1:
                    angle_y = np.degrees(np.arcsin(sin_arg))
                    warnings_list.append(f"dy={self.dy:.2f} > 0.5! Grating lobe в E-плоскости на {angle_y:.1f} град")
                else:
                    warnings_list.append(f"dy={self.dy:.2f} > 0.5! Grating lobes возможны за пределами видимой области")
            except:
                warnings_list.append(f"dy={self.dy:.2f} > 0.5! Возможны grating lobes")
        
        if not warnings_list:
            warnings_list.append("Grating lobes отсутствуют")
        
        return warnings_list
    
    def disable_random_elements(self, count):
        try:
            active_indices = [(i, j) for i in range(self.Ny) for j in range(self.Nx) 
                             if self.element_mask[i, j]]
            
            if not active_indices:
                return
            
            count = max(1, min(count, len(active_indices)))
            selected = np.random.choice(len(active_indices), count, replace=False)
            
            for idx in selected:
                i, j = active_indices[idx]
                self.set_element_state(i, j, False)
            
            write_log(f"Отключено {count} случайных элементов", "DEBUG")
            
        except Exception as e:
            write_log(f"Ошибка в disable_random_elements: {e}", "ERROR")
    
    def reset_all_elements(self):
        self.element_mask = np.ones((self.Ny, self.Nx), dtype=bool)
        self._update_current_amplitudes()
        self._cache.clear()
    
    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }


class CollapsiblePanel(QWidget):
    """Сворачиваемая панель"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_expanded = True
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.toggle_btn = QPushButton("v " + title)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                padding: 8px;
                text-align: left;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d3d9df;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_panel)
        self.main_layout.addWidget(self.toggle_btn)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_widget.setLayout(self.content_layout)
        self.content_widget.setVisible(True)
        
        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)
    
    def toggle_panel(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        if self.is_expanded:
            self.toggle_btn.setText("v " + self.title)
        else:
            self.toggle_btn.setText("> " + self.title)
    
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class ElementTableDialog(QDialog):
    """Таблица элементов"""
    
    def __init__(self, array, parent=None):
        super().__init__(parent)
        self.array = array
        self.parent_window = parent
        self.setWindowTitle(f"Таблица элементов {array.Nx}x{array.Ny}")
        self.setGeometry(300, 200, 600, 500)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        info_label = QLabel("Кликните по ячейке для переключения состояния элемента")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        self.table = QTableWidget()
        self.table.setRowCount(self.array.Ny)
        self.table.setColumnCount(self.array.Nx)
        
        self.update_table()
        
        for j in range(self.array.Nx):
            header_item = QTableWidgetItem(f"X{j}")
            header_item.setFont(QFont('Times New roman', 9, QFont.Bold))
            self.table.setHorizontalHeaderItem(j, header_item)
        
        for i in range(self.array.Ny):
            header_item = QTableWidgetItem(f"Y{i}")
            header_item.setFont(QFont('Times New roman', 9, QFont.Bold))
            self.table.setVerticalHeaderItem(i, header_item)
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellClicked.connect(self.on_cell_clicked)
        
        stats_widget = QWidget()
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Рабочих элементов: 0/0 (0%)")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stats_label)
        stats_widget.setLayout(stats_layout)
        layout.addWidget(stats_widget)
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        
        enable_all_btn = QPushButton("Включить все")
        enable_all_btn.clicked.connect(self.enable_all)
        btn_layout.addWidget(enable_all_btn)
        
        disable_all_btn = QPushButton("Выключить все")
        disable_all_btn.clicked.connect(self.disable_all)
        btn_layout.addWidget(disable_all_btn)
        
        random_btn = QPushButton("Случайно")
        random_btn.clicked.connect(self.random_toggle)
        btn_layout.addWidget(random_btn)
        
        layout.addLayout(btn_layout)
        
        action_layout = QHBoxLayout()
        
        save_btn = QPushButton("Сохранить конфигурацию")
        save_btn.clicked.connect(self.save_configuration)
        action_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Загрузить конфигурацию")
        load_btn.clicked.connect(self.load_configuration)
        action_layout.addWidget(load_btn)
        
        layout.addLayout(action_layout)
        
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def update_table(self):
        for i in range(self.array.Ny):
            for j in range(self.array.Nx):
                item = QTableWidgetItem()
                if self.array.element_mask[i, j]:
                    amp = self.array.amplitudes[i, j]
                    phase = self.array.phases[i, j]
                    item.setText(f"{amp:.2f}")
                    if amp > 0.7:
                        item.setBackground(QBrush(QColor(200, 255, 200)))
                    elif amp > 0.3:
                        item.setBackground(QBrush(QColor(255, 255, 200)))
                    else:
                        item.setBackground(QBrush(QColor(255, 200, 200)))
                    item.setToolTip(f"Амплитуда: {amp:.3f}\nФаза: {phase:.2f} рад")
                else:
                    item.setText("ВЫКЛ")
                    item.setBackground(QBrush(QColor(220, 220, 220)))
                    item.setToolTip("Элемент отключен")
                
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont('Times New roman', 8))
                self.table.setItem(i, j, item)
        
        self.update_stats()
    
    def update_stats(self):
        if hasattr(self, 'stats_label'):
            active = self.array.get_active_count()
            total = self.array.Nx * self.array.Ny
            self.stats_label.setText(f"Рабочих элементов: {active}/{total} ({active/total*100:.1f}%)")
    
    def on_cell_clicked(self, row, col):
        current_state = self.array.element_mask[row, col]
        self.array.set_element_state(row, col, not current_state)
        self.update_table()
    
    def enable_all(self):
        self.array.reset_all_elements()
        self.update_table()
    
    def disable_all(self):
        for i in range(self.array.Ny):
            for j in range(self.array.Nx):
                self.array.set_element_state(i, j, False)
        self.update_table()
    
    def random_toggle(self):
        count = self.array.Nx * self.array.Ny // 4
        active_indices = [(i, j) for i in range(self.array.Ny) 
                         for j in range(self.array.Nx) 
                         if self.array.element_mask[i, j]]
        
        inactive_indices = [(i, j) for i in range(self.array.Ny) 
                           for j in range(self.array.Nx) 
                           if not self.array.element_mask[i, j]]
        
        to_enable = np.random.choice(len(inactive_indices), 
                                    min(count//2, len(inactive_indices)), 
                                    replace=False) if inactive_indices else []
        
        to_disable = np.random.choice(len(active_indices), 
                                     min(count//2, len(active_indices)), 
                                     replace=False) if active_indices else []
        
        for idx in to_enable:
            i, j = inactive_indices[idx]
            self.array.set_element_state(i, j, True)
        
        for idx in to_disable:
            i, j = active_indices[idx]
            self.array.set_element_state(i, j, False)
        
        self.update_table()
    
    def save_configuration(self):
        config = {
            'Nx': self.array.Nx,
            'Ny': self.array.Ny,
            'mask': self.array.element_mask.tolist(),
            'amplitudes': self.array.amplitudes.tolist(),
            'phases': self.array.phases.tolist()
        }
        
        filename = f"antenna_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Сохранено", f"Конфигурация сохранена в {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить: {e}")
    
    def load_configuration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить конфигурацию", "", "JSON files (*.json)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            if config['Nx'] != self.array.Nx or config['Ny'] != self.array.Ny:
                QMessageBox.warning(self, "Ошибка", 
                                  "Конфигурация не совместима с текущим размером решетки")
                return
            
            self.array.element_mask = np.array(config['mask'], dtype=bool)
            self.array.amplitudes = np.array(config['amplitudes'])
            self.array.phases = np.array(config['phases'])
            
            self.update_table()
            QMessageBox.information(self, "Загружено", "Конфигурация успешно загружена")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить: {e}")


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        write_log("\nИнициализация главного окна", "INFO")
        super().__init__()
        try:
            self.array = None
            self._update_pending = False
            
            self.plot_timer = QTimer()
            self.plot_timer.setSingleShot(True)
            self.plot_timer.timeout.connect(self.update_plot_delayed)
            
            self.init_ui()
            self.create_array(8, 8, 0.5, 0.5)
            
            write_log("Главное окно инициализировано успешно", "INFO")
            
        except Exception as e:
            write_log(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", "ERROR")
            write_log(traceback.format_exc(), "ERROR")
            raise
    
    def init_ui(self):
        write_log("Создание интерфейса", "DEBUG")
        
        self.setWindowTitle("Исследование влияние отказов на харакетристики ФАР")
        self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель управления
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setMaximumWidth(450)
        
        # 1. Параметры решетки
        main_group = QGroupBox("Параметры решетки")
        main_layout_grid = QGridLayout()
        
        main_layout_grid.addWidget(QLabel("Количесто элементов по горизантали:"), 0, 0)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 100)
        self.nx_spin.setValue(5)
        self.nx_spin.valueChanged.connect(self.on_size_changed)
        self.nx_spin.setToolTip("Количество элементов по горизонтали")
        main_layout_grid.addWidget(self.nx_spin, 0, 1)
        
        main_layout_grid.addWidget(QLabel("Количество элементов по вертикали:"), 1, 0)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 100)
        self.ny_spin.setValue(5)
        self.ny_spin.valueChanged.connect(self.on_size_changed)
        self.ny_spin.setToolTip("Количество элементов по вертикали")
        main_layout_grid.addWidget(self.ny_spin, 1, 1)
        
        main_layout_grid.addWidget(QLabel("Расстояние между элементами по горизонтали в длинах волн:"), 2, 0)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 2.0)
        self.dx_spin.setSingleStep(0.05)
        self.dx_spin.setValue(0.5)
        self.dx_spin.setDecimals(3)
        self.dx_spin.setSuffix(" λ")
        self.dx_spin.valueChanged.connect(self.on_param_changed)
        self.dx_spin.setToolTip("Расстояние между элементами по горизонтали в длинах волн")
        main_layout_grid.addWidget(self.dx_spin, 2, 1)
        
        main_layout_grid.addWidget(QLabel("Расстояние между элементами по вертикали в длинах волн:"), 3, 0)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 2.0)
        self.dy_spin.setSingleStep(0.05)
        self.dy_spin.setValue(0.5)
        self.dy_spin.setDecimals(3)
        self.dy_spin.setSuffix(" λ")
        self.dy_spin.valueChanged.connect(self.on_param_changed)
        self.dy_spin.setToolTip("Расстояние между элементами по вертикали в длинах волн")
        main_layout_grid.addWidget(self.dy_spin, 3, 1)
        
        main_group.setLayout(main_layout_grid)
        left_layout.addWidget(main_group)

                # Кнопка конструктора излучателя
        self.design_element_btn = QPushButton("Конструктор излучателя")
        self.design_element_btn.clicked.connect(self.show_element_designer)
        self.design_element_btn.setStyleSheet("""
            QPushButton {
                background-color: #337ab7;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border: 1px solid #2e6da4;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #286090; }
        """)
        self.design_element_btn.setMinimumHeight(35)
        left_layout.addWidget(self.design_element_btn)
        
        # 2. Амплитудное распределение
        amp_group = QGroupBox("Амплитудное распределение")
        amp_layout = QVBoxLayout()
        
        self.amp_combo = QComboBox()
        self.amp_combo.addItems([
            "Равномерное",
            "Косинус на пьедестале",
            "Треугольное",
            "Гауссово",
        ])
        self.amp_combo.currentTextChanged.connect(self.on_amplitude_changed)
        amp_layout.addWidget(self.amp_combo)
        
        self.param_widget = QWidget()
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Пьедестал:"))
        self.pedestal_spin = QDoubleSpinBox()
        self.pedestal_spin.setRange(0.0, 0.9)
        self.pedestal_spin.setSingleStep(0.1)
        self.pedestal_spin.setValue(0.2)
        self.pedestal_spin.valueChanged.connect(self.on_amplitude_changed)
        param_layout.addWidget(self.pedestal_spin)
        self.param_widget.setLayout(param_layout)
        self.param_widget.setVisible(False)
        amp_layout.addWidget(self.param_widget)
        
        amp_group.setLayout(amp_layout)
        left_layout.addWidget(amp_group)
        
        # Фазовое сканирование
        scan_group = QGroupBox("Фазовое сканирование")
        scan_layout = QGridLayout()

        # --- H-плоскость ---
        scan_layout.addWidget(QLabel("Угол H:"), 0, 0)

        self.theta_scan_input = QSpinBox()
        self.theta_scan_input.setRange(-65, 65)
        self.theta_scan_input.setValue(0)
        self.theta_scan_input.setSuffix(" град")
        self.theta_scan_input.setFixedWidth(90)
        self.theta_scan_input.valueChanged.connect(self.on_scan_input_changed)
        scan_layout.addWidget(self.theta_scan_input, 0, 1)

        self.theta_scan_slider = QSlider(Qt.Horizontal)
        self.theta_scan_slider.setRange(-65, 65)
        self.theta_scan_slider.setValue(0)
        self.theta_scan_slider.valueChanged.connect(self.on_scan_slider_changed)
        self.theta_scan_slider.setTickPosition(QSlider.TicksBelow)
        self.theta_scan_slider.setTickInterval(10)
        scan_layout.addWidget(self.theta_scan_slider, 0, 2)

        self.theta_label = QLabel("0 град")
        self.theta_label.setMinimumWidth(60)
        self.theta_label.setAlignment(Qt.AlignCenter)
        self.theta_label.setFont(QFont('Times New Roman', 11, QFont.Bold))
        scan_layout.addWidget(self.theta_label, 0, 3)

        # --- E-плоскость ---
        scan_layout.addWidget(QLabel("Угол E:"), 1, 0)

        self.phi_scan_input = QSpinBox()
        self.phi_scan_input.setRange(-65, 65)
        self.phi_scan_input.setValue(0)
        self.phi_scan_input.setSuffix(" град")
        self.phi_scan_input.setFixedWidth(90)
        self.phi_scan_input.valueChanged.connect(self.on_scan_input_changed)
        scan_layout.addWidget(self.phi_scan_input, 1, 1)

        self.phi_scan_slider = QSlider(Qt.Horizontal)
        self.phi_scan_slider.setRange(-65, 65)
        self.phi_scan_slider.setValue(0)
        self.phi_scan_slider.valueChanged.connect(self.on_scan_slider_changed)
        self.phi_scan_slider.setTickPosition(QSlider.TicksBelow)
        self.phi_scan_slider.setTickInterval(10)
        scan_layout.addWidget(self.phi_scan_slider, 1, 2)

        self.phi_label = QLabel("0 град")
        self.phi_label.setMinimumWidth(60)
        self.phi_label.setAlignment(Qt.AlignCenter)
        self.phi_label.setFont(QFont('Times New Roman', 11, QFont.Bold))
        scan_layout.addWidget(self.phi_label, 1, 3)

        scan_group.setLayout(scan_layout)
        left_layout.addWidget(scan_group)
        
        # 4. Управление элементами
        self.control_panel = CollapsiblePanel("Управление элементами")
        
        self.table_btn = QPushButton("Открыть таблицу элементов")
        self.table_btn.clicked.connect(self.show_element_table)
        self.control_panel.addWidget(self.table_btn)
        
        random_layout = QHBoxLayout()
        random_layout.addWidget(QLabel("Случайно выключить:"))
        self.random_count_spin = QSpinBox()
        self.random_count_spin.setRange(1, 400)
        self.random_count_spin.setValue(5)
        random_layout.addWidget(self.random_count_spin)
        
        self.random_btn = QPushButton("Применить")
        self.random_btn.clicked.connect(self.on_random_disable)
        random_layout.addWidget(self.random_btn)
        
        self.control_panel.addLayout(random_layout)
        
        self.reset_btn = QPushButton("Сбросить все элементы")
        self.reset_btn.clicked.connect(self.on_reset)
        self.control_panel.addWidget(self.reset_btn)
        
        # Кнопки компенсации
        self.compensate_btn = QPushButton("Компенсация отказов")
        self.compensate_btn.clicked.connect(self.on_compensate)
        self.compensate_btn.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border: 1px solid #c9302c;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
        """)
        self.compensate_btn.setMinimumHeight(40)
        self.control_panel.addWidget(self.compensate_btn)
        
        self.rollback_btn = QPushButton("Откат компенсации")
        self.rollback_btn.clicked.connect(self.on_rollback)
        self.rollback_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: black;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border: 1px solid #eea236;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #eea236;
            }
        """)
        self.control_panel.addWidget(self.rollback_btn)
        
        left_layout.addWidget(self.control_panel)
        
        # 5. Маркеры на графиках
        self.markers_panel = CollapsiblePanel("Маркеры на графиках")
        
        h_label = QLabel("Горизонтальные линии (дБ):")
        h_label.setFont(QFont('Times New roman', 9, QFont.Bold))
        self.markers_panel.addWidget(h_label)
        
        h1_layout = QHBoxLayout()
        self.h_line1_check = QPushButton("Линия 1")
        self.h_line1_check.setCheckable(True)
        self.h_line1_check.setChecked(True)
        self.h_line1_check.toggled.connect(self.on_marker_changed)
        self.h_line1_check.setMaximumWidth(80)
        h1_layout.addWidget(self.h_line1_check)
        
        self.h_line1_spin = QDoubleSpinBox()
        self.h_line1_spin.setRange(-100, 10)
        self.h_line1_spin.setValue(-3)
        self.h_line1_spin.setSuffix(" дБ")
        self.h_line1_spin.valueChanged.connect(self.on_marker_changed)
        h1_layout.addWidget(self.h_line1_spin)
        self.markers_panel.addLayout(h1_layout)
        
        h2_layout = QHBoxLayout()
        self.h_line2_check = QPushButton("Линия 2")
        self.h_line2_check.setCheckable(True)
        self.h_line2_check.setChecked(False)
        self.h_line2_check.toggled.connect(self.on_marker_changed)
        self.h_line2_check.setMaximumWidth(80)
        h2_layout.addWidget(self.h_line2_check)
        
        self.h_line2_spin = QDoubleSpinBox()
        self.h_line2_spin.setRange(-100, 10)
        self.h_line2_spin.setValue(-10)
        self.h_line2_spin.setSuffix(" дБ")
        self.h_line2_spin.valueChanged.connect(self.on_marker_changed)
        h2_layout.addWidget(self.h_line2_spin)
        self.markers_panel.addLayout(h2_layout)
        
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        self.markers_panel.addWidget(sep1)
        
        v_label = QLabel("Вертикальные линии (град):")
        v_label.setFont(QFont('Times New roman', 9, QFont.Bold))
        self.markers_panel.addWidget(v_label)
        
        v1_layout = QHBoxLayout()
        self.v_line1_check = QPushButton("Линия 1")
        self.v_line1_check.setCheckable(True)
        self.v_line1_check.setChecked(False)
        self.v_line1_check.toggled.connect(self.on_marker_changed)
        self.v_line1_check.setMaximumWidth(80)
        v1_layout.addWidget(self.v_line1_check)
        
        self.v_line1_spin = QDoubleSpinBox()
        self.v_line1_spin.setRange(-90, 90)
        self.v_line1_spin.setValue(0)
        self.v_line1_spin.setSuffix(" град")
        self.v_line1_spin.valueChanged.connect(self.on_marker_changed)
        v1_layout.addWidget(self.v_line1_spin)
        self.markers_panel.addLayout(v1_layout)
        
        v2_layout = QHBoxLayout()
        self.v_line2_check = QPushButton("Линия 2")
        self.v_line2_check.setCheckable(True)
        self.v_line2_check.setChecked(False)
        self.v_line2_check.toggled.connect(self.on_marker_changed)
        self.v_line2_check.setMaximumWidth(80)
        v2_layout.addWidget(self.v_line2_check)
        
        self.v_line2_spin = QDoubleSpinBox()
        self.v_line2_spin.setRange(-90, 90)
        self.v_line2_spin.setValue(30)
        self.v_line2_spin.setSuffix(" град")
        self.v_line2_spin.valueChanged.connect(self.on_marker_changed)
        v2_layout.addWidget(self.v_line2_spin)
        self.markers_panel.addLayout(v2_layout)
        
        left_layout.addWidget(self.markers_panel)
        
        # 6. Характеристики ДН
        self.params_panel = CollapsiblePanel("Характеристики ДН")
        
        self.params_text = QTextEdit()
        self.params_text.setMaximumHeight(300)
        self.params_text.setReadOnly(True)
        self.params_text.setFont(QFont('Times New roman', 9))
        self.params_panel.addWidget(self.params_text)
        
        left_layout.addWidget(self.params_panel)
        
        # Справочные материалы
        self.help_panel = CollapsiblePanel("Справочные материалы")
        self.help_panel.is_expanded = False
        self.help_panel.content_widget.setVisible(False)
        self.help_panel.toggle_btn.setText("> Справочные материалы")
        
        self.help_btn = QPushButton("Открыть теорию по антенным решеткам")
        self.help_btn.clicked.connect(self.show_help_material)
        self.help_panel.addWidget(self.help_btn)
        
        left_layout.addWidget(self.help_panel)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        
        # Правая панель с графиками
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(12, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(800, 600)
        
        right_layout.addWidget(self.canvas)
        right_panel.setLayout(right_layout)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 1150])
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        
        self.statusBar().showMessage("Готов к работе")
        
        self.on_marker_changed()
        write_log("Интерфейс создан успешно", "DEBUG")
    
    def on_marker_changed(self):
        if self.h_line1_check.isChecked():
            self.h_line1_check.setStyleSheet("background-color: #5bc0de; color: white;")
        else:
            self.h_line1_check.setStyleSheet("background-color: #f0f0f0; color: gray;")
        
        if self.h_line2_check.isChecked():
            self.h_line2_check.setStyleSheet("background-color: #5cb85c; color: white;")
        else:
            self.h_line2_check.setStyleSheet("background-color: #f0f0f0; color: gray;")
        
        if self.v_line1_check.isChecked():
            self.v_line1_check.setStyleSheet("background-color: #d9534f; color: white;")
        else:
            self.v_line1_check.setStyleSheet("background-color: #f0f0f0; color: gray;")
        
        if self.v_line2_check.isChecked():
            self.v_line2_check.setStyleSheet("background-color: #f0ad4e; color: black;")
        else:
            self.v_line2_check.setStyleSheet("background-color: #f0f0f0; color: gray;")
        
        self.update_plot()
    
    def clear_all_markers(self):
        self.h_line1_check.setChecked(False)
        self.h_line2_check.setChecked(False)
        self.v_line1_check.setChecked(False)
        self.v_line2_check.setChecked(False)
        self.on_marker_changed()
    
    def show_help_material(self):
        help_text = """ТЕОРИЯ АНТЕННЫХ РЕШЁТОК

    1. ОСНОВНЫЕ ПОНЯТИЯ
    Антенная решётка (АР) — совокупность излучающих элементов,
    расположенных в пространстве упорядоченно (линейно, плоскостно,
    конформно). Каждый элемент возбуждается с заданной амплитудой
    и фазой, формируя суммарную диаграмму направленности (ДН).

    2. МНОЖИТЕЛЬ РЕШЁТКИ (ARRAY FACTOR)
    ДН решётки = ДН элемента × Множитель решётки

    F(θ,φ) = F₁(θ,φ) × AF(θ,φ)

    Для плоской прямоугольной решётки Nx×Ny с шагом dx, dy:

    AF(θ,φ) = ΣₙΣₘ Aₙₘ · exp[j·k·(xₙ·sinθ·cosφ + yₘ·sinθ·sinφ) + j·φₙₘ]

    где:
    k = 2π/λ — волновое число
    xₙ = (n - (Nx-1)/2)·dx — координата элемента по X
    yₘ = (m - (Ny-1)/2)·dy — координата элемента по Y
    Aₙₘ — амплитуда элемента
    φₙₘ — фазовый сдвиг элемента

    3. УСЛОВИЕ ОТСУТСТВИЯ ДИФРАКЦИОННЫХ ЛЕПЕСТКОВ
    Для прямоугольной решётки с шагом d:

    d/λ < 1 / (1 + |sin(θ_scan)|)

    При θ_scan = 0°:   d < 1.00λ
    При θ_scan = 30°:  d < 0.67λ
    При θ_scan = 60°:  d < 0.54λ

    4. КОЭФФИЦИЕНТ НАПРАВЛЕННОГО ДЕЙСТВИЯ (КНД)

    D_max = 4π × S_эфф / λ²

    S_эфф = Nx·Ny·dx·dy — эффективная площадь раскрыва

    КНД в дБи:  D_dBi = 10·log₁₀(D_max)

    5. АМПЛИТУДНЫЕ РАСПРЕДЕЛЕНИЯ

    Равномерное: Aₙₘ = 1
    УБЛ = -13.2 дБ, максимальный КНД

    Косинус на пьедестале: Aₙₘ = p + (1-p)·cos(...)
    p ∈ [0,1] — уровень пьедестала
    УБЛ регулируется от -13 до -23 дБ

    Треугольное: Aₙₘ = (1-|xₙ|)·(1-|yₘ|)
    УБЛ ≈ -26 дБ, широкий луч

    Гауссово: Aₙₘ = exp(-(xₙ²+yₘ²)/(2·σ²))
    σ ∈ [0.1, 2.0] — ширина распределения
    Плавное спадание, низкий УБЛ при малых σ

    6. ФАЗОВОЕ СКАНИРОВАНИЕ

    Фазовый сдвиг для отклонения луча на угол θ_scan:

    По оси X:  Δφₙ = -k · xₙ · sin(θ_x)
    По оси Y:  Δφₘ = -k · yₘ · sin(θ_y)

    Максимальный угол сканирования ограничен шагом решётки
    и появлением дифракционных лепестков.

    7. ВЛИЯНИЕ ОТКАЗОВ ЭЛЕМЕНТОВ

    • Снижение КНД пропорционально доле отказавших элементов:
    D_eff ≈ D_ideal × (N_раб / N_общ)

    • Рост УБЛ из-за нарушения амплитудного распределения:
    ΔУБЛ ≈ 2-5 дБ при 10-20% отказов

    • Искажение формы главного лепестка:
    - Смещение максимума (ошибка пеленга)
    - Расширение луча
    - Асимметрия ДН

    • Наибольшее влияние — отказы центральных элементов

    8. КОМПЕНСАЦИЯ ОТКАЗОВ

    Принцип: энергия отказавшего элемента перераспределяется
    на работающие соседние элементы.

    Алгоритм:
    1. Определение отказавших элементов
    2. Поиск соседей в радиусе 2 шагов решётки
    3. Распределение мощности с весом w = 1/r²
        (r — расстояние до соседа)
    4. Ограничение: мощность элемента ≤ 2 × P_ном

    Режимы компенсации:
    • Линейный — все соседи в пределах допуска
    • Насыщение — часть соседей на максимуме мощности

    Эффективность компенсации:
    • Полная (100%) — для одиночных отказов
    • Частичная (~70%) — для групповых отказов
    • Невозможна — при отказе >50% элементов

    9. ИЗЛУЧАТЕЛИ

    Пирамидальный рупор:
    • Апертурный метод расчёта ДН
    • Учёт фазовой ошибки: Δφ = π·a²/(4·λ·R)
    • Оптимальный: Δφ = 90° (компромисс КУ/УБЛ)
    • КУ = 4π·a·b·0.81/λ²

    Конический рупор:
    • Осесимметричная ДН
    • Распределение поля: J₁(r)/r (мода TE₁₁)
    • ДН: 2·J₁(u)/u, u = π·D·sinθ/λ
    • УБЛ ≈ -17.6 дБ (равномерное), -24 дБ (TE₁₁)

    10. ХАРАКТЕРИСТИКИ ДН

    Ширина луча по уровню -3 дБ:
    θ₃ = arcsin(λ/(N·d))  — для равномерной линейной решётки

    Уровень боковых лепестков (УБЛ):
    Определяется амплитудным распределением

    Коэффициент использования площади (КИП):
    η = D_real / D_ideal ≤ 1
    Учитывает потери на неравномерность амплитуд"""

        dialog = QDialog(self)
        dialog.setWindowTitle("Справочный материал")
        dialog.setGeometry(150, 100, 800, 700)
        
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(help_text)
        text_edit.setFont(QFont('Times New Roman', 12))
        text_edit.setStyleSheet("line-height: 1.6;")
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Закрыть")
        close_btn.setFont(QFont('Times New Roman', 12))
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_element_designer(self):
        """Открыть конструктор излучателя"""
        try:
            from horn_design_dialog import ElementDesignDialog
            dialog = ElementDesignDialog(self, default_freq=8.33)
            if dialog.exec_() == QDialog.Accepted:
                element = dialog.get_element()
                if element:
                    self.array.set_element(element)
                    self.statusBar().showMessage(
                        f"Излучатель: {element.__class__.__name__} | "
                        f"КУ ≈ {element.get_gain_estimate():.1f} дБи"
                    )
                    self.update_plot()
        except ImportError:
            QMessageBox.warning(self, "Ошибка", 
                            "Модуль конструктора (horn_design_dialog.py) не найден")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть конструктор:\n{str(e)}")
    
    def create_array(self, Nx, Ny, dx, dy):
        write_log(f"Создание решетки: {Nx}x{Ny}, dx={dx}, dy={dy}", "INFO")
        try:
            # Сохраняем текущий излучатель
            old_element = self.array.element if self.array else None
            
            self.array = AntennaArray(Nx, Ny, dx, dy)
            
            # Восстанавливаем излучатель
            if old_element is not None:
                self.array.set_element(old_element)
            
            self.on_amplitude_changed()
            self.on_scan_changed()
            self.update_params()
            self.update_plot()
            self.statusBar().showMessage(f"Решетка {Nx}x{Ny} создана")
        except Exception as e:
            write_log(f"ОШИБКА в create_array: {str(e)}", "ERROR")
            self.statusBar().showMessage(f"Ошибка: {str(e)}")
        
    def on_size_changed(self):
        self.create_array(
            self.nx_spin.value(),
            self.ny_spin.value(),
            self.dx_spin.value(),
            self.dy_spin.value()
        )
    
    def on_param_changed(self):
        if not self.array:
            return
        
        new_dx = self.dx_spin.value()
        new_dy = self.dy_spin.value()
        
        # Минимальный шаг = размер излучателя + 5%
        min_dx = self.array.get_min_element_spacing()
        min_dy = min_dx  # для квадратного рупора одинаково
        
        # Обновляем минимумы спинов
        self.dx_spin.blockSignals(True)
        self.dy_spin.blockSignals(True)
        self.dx_spin.setMinimum(min_dx)
        self.dy_spin.setMinimum(min_dy)
        self.dx_spin.blockSignals(False)
        self.dy_spin.blockSignals(False)
        
        # Если текущие значения меньше минимума — подтягиваем
        if new_dx < min_dx:
            self.dx_spin.setValue(min_dx)
            new_dx = min_dx
        if new_dy < min_dy:
            self.dy_spin.setValue(min_dy)
            new_dy = min_dy
        
        # Применяем
        self.array.dx = new_dx
        self.array.dy = new_dy
        self.array._precalculate_coordinates()
        self.array._cache.clear()
        
        self.statusBar().showMessage(
            f"Мин. шаг: {min_dx:.3f}λ | Текущий: dx={new_dx:.3f}λ, dy={new_dy:.3f}λ", 0)
        
        self._update_pending = False
        self.update_plot_delayed()
    
    def show_element_designer(self):
        """Открыть конструктор излучателя"""
        try:
            from horn_design_dialog import ElementDesignDialog
            
            # Передаём текущие размеры, если рупор уже настроен
            current_params = {}
            if hasattr(self.array.element, 'p'):
                p = self.array.element.p
                current_params = {
                    'a': p.a,
                    'b': p.b,
                    'L': p.L,
                    'frequency': p.frequency
                }
            
            dialog = ElementDesignDialog(self, default_freq=8.33, **current_params)
            if dialog.exec_() == QDialog.Accepted:
                element = dialog.get_element()
                if element:
                    self.array.set_element(element)
                    
                    phys_w, phys_h = element.get_physical_size()
                    min_dx = phys_w * 1.05
                    min_dy = phys_h * 1.05
                    
                    self.dx_spin.setMinimum(min_dx)
                    self.dy_spin.setMinimum(min_dy)
                    self.dx_spin.setValue(min_dx)
                    self.dy_spin.setValue(min_dy)
                    
                    self.array.dx = min_dx
                    self.array.dy = min_dy
                    self.array._precalculate_coordinates()
                    self.array._cache.clear()
                    
                    self.statusBar().showMessage(
                        f"Излучатель: {element.__class__.__name__} | "
                        f"Шаг: dx={min_dx:.3f}, dy={min_dy:.3f} | "
                        f"КУ = {element.get_gain_estimate():.1f} дБи"
                    )
                    self.update_plot()
        except ImportError:
            QMessageBox.warning(self, "Ошибка", 
                            "Модуль конструктора (horn_design_dialog.py) не найден")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть конструктор:\n{str(e)}")
    
    def on_amplitude_changed(self):
        if not self.array:
            return
        
        dist_name = self.amp_combo.currentText()
        self.param_widget.setVisible("пьедестале" in dist_name.lower())
        
        dist_map = {
            "Равномерное": ("uniform", {}),
            "Косинусное": ("cosine", {}),
            "Косинус на пьедестале": ("cosine_pedestal", {'pedestal': self.pedestal_spin.value()}),
            "Треугольное": ("triangular", {}),
            "Гауссово": ("gaussian", {'sigma': 0.5}),
            "Квадратичное": ("quadratic", {'delta': 0.5})
        }
        
        if dist_name in dist_map:
            dist_type, params = dist_map[dist_name]
            self.array.update_amplitude_distribution(dist_type, **params)
        
        self.update_plot()
    
    def on_scan_slider_changed(self):
        """Слайдер двигает поле ввода"""
        theta_x = self.theta_scan_slider.value()
        theta_y = self.phi_scan_slider.value()
        
        self.theta_scan_input.blockSignals(True)
        self.phi_scan_input.blockSignals(True)
        self.theta_scan_input.setValue(theta_x)
        self.phi_scan_input.setValue(theta_y)
        self.theta_scan_input.blockSignals(False)
        self.phi_scan_input.blockSignals(False)
        
        self._apply_scan()

    def on_scan_input_changed(self):
        """Поле ввода двигает слайдер"""
        theta_x = self.theta_scan_input.value()
        theta_y = self.phi_scan_input.value()
        
        self.theta_scan_slider.blockSignals(True)
        self.phi_scan_slider.blockSignals(True)
        self.theta_scan_slider.setValue(theta_x)
        self.phi_scan_slider.setValue(theta_y)
        self.theta_scan_slider.blockSignals(False)
        self.phi_scan_slider.blockSignals(False)
        
        self._apply_scan()

    def _apply_scan(self):
        """Применить сканирование"""
        if not self.array:
            return
        
        theta_x = self.theta_scan_slider.value()
        theta_y = self.phi_scan_slider.value()
        self.theta_label.setText(f"{theta_x} град")
        self.phi_label.setText(f"{theta_y} град")
        self.array.set_scan_angle(theta_x, theta_y)
        self.update_plot()
    
    def on_random_disable(self):
        if self.array:
            # Сохраняем идеальную ДН перед первым отказом
            if self.array.cached_pattern_ideal_H is None:
                self.array.cache_ideal_pattern()
            
            count = self.random_count_spin.value()
            self.array.disable_random_elements(count)
            self.update_params()
            self.update_plot()
            self.statusBar().showMessage(f"Случайно отключено {count} элементов")
    
    def on_reset(self):
        if self.array:
            self.array.reset_all_elements()
            self.on_amplitude_changed()
            self.update_params()
            self.update_plot()
            self.array.cached_pattern_ideal_H = None
            self.array.cached_pattern_ideal_E = None
            self.statusBar().showMessage("Все элементы сброшены")
    
    def on_compensate(self):
        """Компенсация отказов с физическим ограничением мощности"""
        if not self.array:
            return
        
        failed_mask = ~self.array.element_mask
        failed_count = np.sum(failed_mask)
        
        if failed_count == 0:
            QMessageBox.information(self, "Компенсация", "Нет отказавших элементов!")
            return
        
        write_log(f"Запуск компенсации. Отказавших: {failed_count}", "INFO")
        
        # Сохраняем состояние ДО компенсации
        self.array.save_state('failed')
        
        # Максимальное увеличение мощности на элементе (50%)
        MAX_POWER_INCREASE = 1.0
        
        base_amps = self.array._base_amplitudes.copy()
        working_amps = self.array.amplitudes.copy()
        active_mask = self.array.element_mask.copy()
        
        failed_indices = np.argwhere(failed_mask)
        total_compensations = 0
        saturation_count = 0
        
        for i_fail, j_fail in failed_indices:
            failed_power = base_amps[i_fail, j_fail]**2
            
            # Ищем соседей с весами по расстоянию
            neighbors = []
            for di in [-2, -1, 0, 1, 2]:
                for dj in [-2, -1, 0, 1, 2]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i_fail + di, j_fail + dj
                    if (0 <= ni < self.array.Ny and 
                        0 <= nj < self.array.Nx and 
                        active_mask[ni, nj]):
                        dist = np.sqrt(di**2 + dj**2)
                        if dist <= 2.5:  # в пределах 2 шагов
                            neighbors.append((ni, nj, 1.0 / (dist**2)))
            
            if not neighbors:
                continue
            
            total_weight = sum(w for _, _, w in neighbors)
            
            # Распределяем 70% мощности отказавшего
            power_to_distribute = failed_power * 1
            
            for ni, nj, weight in neighbors:
                current_amp = working_amps[ni, nj]
                current_power = current_amp**2
                max_allowed_power = (base_amps[ni, nj]**2) * (1 + MAX_POWER_INCREASE)
                
                add_power = power_to_distribute * (weight / total_weight)
                new_power = current_power + add_power
                
                if new_power > max_allowed_power:
                    working_amps[ni, nj] = np.sqrt(max_allowed_power)
                    saturation_count += 1
                else:
                    working_amps[ni, nj] = np.sqrt(new_power)
                
                total_compensations += 1
        
        self.array.amplitudes = working_amps
        self.array._cache.clear()
        
        # Сохраняем состояние ПОСЛЕ компенсации
        self.array.save_state('compensated')
        
        mode_text = "линейный" if saturation_count == 0 else f"насыщение ({saturation_count} эл.)"
        write_log(f"Компенсация: {total_compensations} коррекций, режим: {mode_text}", "INFO")
        
        self._update_pending = False
        self.update_plot_delayed()
        self.update_params()
        
        self.statusBar().showMessage(
            f"Компенсация выполнена! Режим: {mode_text}", 5000)
    
    def on_rollback(self):
        if hasattr(self.array, '_original_amplitudes'):
            reply = QMessageBox.question(self, "Откат", 
                                        "Вернуть исходные амплитуды?",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.array.amplitudes = self.array._original_amplitudes.copy()
                self.array.phases = self.array._original_phases.copy()
                self.array.element_mask = self.array._original_mask.copy()
                self.array._cache.clear()
                
                self._update_pending = False
                self.update_plot_delayed()
                self.update_params()
                
                self.statusBar().showMessage("Компенсация отменена", 3000)
                write_log("Компенсация отменена", "INFO")
        else:
            QMessageBox.information(self, "Откат", "Нечего откатывать")
    
    def show_element_table(self):
        if self.array:
            dialog = ElementTableDialog(self.array, self)
            if dialog.exec_():
                self.update_plot()
                self.statusBar().showMessage("Таблица элементов закрыта")
    
    def update_params(self):
        if not self.array:
            return
        
        try:
            theta_H, pattern_H = self.array.calculate_pattern(phi_slice=0)
            self.array.last_pattern = pattern_H
            self.array.last_theta = theta_H
            D_dBi_H = self.array.calculate_directivity()
            sll_db_H = self.array.calculate_sll()
            beamwidth_H = self.array.get_beamwidth_3db()
            
            self.array.calculate_pattern(phi_slice=np.pi/2)
            D_dBi_E = self.array.calculate_directivity()
            sll_db_E = self.array.calculate_sll()
            beamwidth_E = self.array.get_beamwidth_3db()
            
            active = self.array.get_active_count()
            total = self.array.Nx * self.array.Ny
            cache_stats = self.array.get_cache_stats()
            
            grating_warnings = self.array.check_grating_lobes(self.theta_scan_slider.value())
            grating_text = "\n".join(grating_warnings)
            
            efficiency = self.array.calculate_aperture_efficiency()
            
            comp_info = ""
            if hasattr(self.array, '_original_amplitudes'):
                failed = np.sum(~self.array.element_mask)
                if failed > 0:
                    diff = self.array.amplitudes - self.array._original_amplitudes
                    active_mask = self.array.element_mask
                    if np.any(active_mask):
                        boosted = np.sum(diff[active_mask] > 0.001)
                        max_add = np.max(diff[active_mask]) if boosted > 0 else 0
                        comp_info = f"""
КОМПЕНСАЦИЯ:
  Отказало:     {failed}
  Усилено:      {boosted}
  Макс. добавка: {max_add:.3f}"""
            
            params_text = f"""ХАРАКТЕРИСТИКИ ДН:

H-ПЛОСКОСТЬ (phi=0 град):
  КНД:        {D_dBi_H:6.2f} дБи
  УБЛ:        {sll_db_H:6.1f} дБ
  Ширина:     {beamwidth_H:6.1f} град

E-ПЛОСКОСТЬ (phi=90 град):
  КНД:        {D_dBi_E:6.2f} дБи
  УБЛ:        {sll_db_E:6.1f} дБ
  Ширина:     {beamwidth_E:6.1f} град

ПАРАМЕТРЫ РЕШЕТКИ:
  Размер:     {self.array.Nx}x{self.array.Ny}
  Рабочих элементов:   {active}/{total} ({active/total*100:.1f}%)
  d/lambda:   dx={self.array.dx:.2f}, dy={self.array.dy:.2f}
  КИП:        {efficiency:.1f}%
  Излучатель: {self.array.element.__class__.__name__ if hasattr(self.array.element, '__class__') else 'Не настроен'}
  Распредел.: {self.amp_combo.currentText()}
  Сканирован.: {self.theta_scan_slider.value()} град

КЭШИРОВАНИЕ:
  Попаданий:  {cache_stats['hits']}
  Промахов:   {cache_stats['misses']}
  Эффект.:    {cache_stats['hit_rate']:.1f}%"""
            
            self.params_text.setText(params_text)
            
        except Exception as e:
            write_log(f"Ошибка в update_params: {e}", "ERROR")
    
    def update_plot(self):
        if not self._update_pending:
            self._update_pending = True
            self.plot_timer.start(50)
    

    def update_plot_delayed(self):
        self._update_pending = False
        
        if not self.array:
            return
        
        try:
            self.figure.clf()
            
            # Текущая ДН
            theta_H, pattern_H = self.array.calculate_pattern(phi_slice=0)
            theta_deg_H = np.degrees(theta_H)
            pattern_db_H = 20 * np.log10(np.maximum(pattern_H, 1e-10))
            
            theta_E, pattern_E = self.array.calculate_pattern(phi_slice=np.pi/2)
            theta_deg_E = np.degrees(theta_E)
            pattern_db_E = 20 * np.log10(np.maximum(pattern_E, 1e-10))
            
            # Идеальная ДН (пунктир) — из кэша
            has_ideal = self.array.cached_pattern_ideal_H is not None
            if has_ideal:
                theta_H_i, pattern_H_i = self.array.cached_pattern_ideal_H
                theta_E_i, pattern_E_i = self.array.cached_pattern_ideal_E
                pattern_db_H_i = 20 * np.log10(np.maximum(pattern_H_i, 1e-10))
                pattern_db_E_i = 20 * np.log10(np.maximum(pattern_E_i, 1e-10))
            
            # График 1: H-плоскость
            ax1 = self.figure.add_subplot(221)
            ax1.plot(theta_deg_H, pattern_db_H, 'b-', linewidth=2, label='Текущая')
            if has_ideal:
                ax1.plot(theta_deg_H, pattern_db_H_i, 'r--', linewidth=1.5, alpha=0.7, label='Исходная')
            ax1.fill_between(theta_deg_H, -50, pattern_db_H, alpha=0.1, color='blue')
            ax1.set_xlabel('Угол, град', fontsize=10)
            ax1.set_ylabel('Уровень, дБ', fontsize=10)
            ax1.set_title('H-плоскость (phi=0 град)', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_xlim(-90, 90)
            ax1.set_ylim(-50, 5)
            
            self.array.last_pattern = pattern_H
            self.array.last_theta = theta_H
            sll_db_H = self.array.calculate_sll()
            beamwidth_H = self.array.get_beamwidth_3db()
            
            if sll_db_H > -100:
                ax1.axhline(y=sll_db_H, color='g', linestyle=':', alpha=0.7, 
                           label=f'УБЛ={sll_db_H:.1f} дБ')
            
            if beamwidth_H > 0:
                max_idx = np.argmax(pattern_db_H)
                theta_max = theta_deg_H[max_idx]
                ax1.axvspan(theta_max - beamwidth_H/2, theta_max + beamwidth_H/2, 
                           alpha=0.2, color='yellow')
                ax1.text(theta_max, -45, f'{beamwidth_H:.1f} град', 
                        ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            

            # График 2: E-плоскость
            ax2 = self.figure.add_subplot(222)
            ax2.plot(theta_deg_E, pattern_db_E, 'g-', linewidth=2, label='Текущая')
            if has_ideal:
                ax2.plot(theta_deg_E, pattern_db_E_i, 'r--', linewidth=1.5, alpha=0.7, label='Исходная')
            ax2.fill_between(theta_deg_E, -50, pattern_db_E, alpha=0.1, color='green')
            ax2.set_xlabel('Угол, град', fontsize=10)
            ax2.set_ylabel('Уровень, дБ', fontsize=10)
            ax2.set_title('E-плоскость (phi=90 град)', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(-90, 90)
            ax2.set_ylim(-50, 5)
            
            self.array.last_pattern = pattern_E
            self.array.last_theta = theta_E
            sll_db_E = self.array.calculate_sll()
            beamwidth_E = self.array.get_beamwidth_3db()
            
            if sll_db_E > -100:
                ax2.axhline(y=sll_db_E, color='g', linestyle=':', alpha=0.7, 
                           label=f'УБЛ={sll_db_E:.1f} дБ')
            
            if beamwidth_E > 0:
                max_idx = np.argmax(pattern_db_E)
                theta_max = theta_deg_E[max_idx]
                ax2.axvspan(theta_max - beamwidth_E/2, theta_max + beamwidth_E/2, 
                           alpha=0.2, color='yellow')
                ax2.text(theta_max, -45, f'{beamwidth_E:.1f} град', 
                        ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Маркеры
            line_styles = {"Сплошная": '-', "Штриховая": '--', "Пунктирная": ':', "Штрих-пунктир": '-.'}
            current_style = '--'
            
            if self.h_line1_check.isChecked():
                ax1.axhline(y=self.h_line1_spin.value(), color='blue', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
                ax2.axhline(y=self.h_line1_spin.value(), color='blue', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
            
            if self.h_line2_check.isChecked():
                ax1.axhline(y=self.h_line2_spin.value(), color='green', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
                ax2.axhline(y=self.h_line2_spin.value(), color='green', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
            
            if self.v_line1_check.isChecked():
                ax1.axvline(x=self.v_line1_spin.value(), color='red', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
                ax2.axvline(x=self.v_line1_spin.value(), color='red', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
            
            if self.v_line2_check.isChecked():
                ax1.axvline(x=self.v_line2_spin.value(), color='orange', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
                ax2.axvline(x=self.v_line2_spin.value(), color='orange', 
                           linestyle=current_style, alpha=0.7, linewidth=1.5)
            
            # Grating lobes
            if self.array.dx > 0.5:
                scan_theta = np.radians(self.theta_scan_slider.value())
                sin_arg = 1/self.array.dx - np.sin(scan_theta)
                if -1 <= sin_arg <= 1:
                    grating_angle = np.degrees(np.arcsin(sin_arg))
                    if -90 <= grating_angle <= 90:
                        ax1.axvline(x=grating_angle, color='red', linestyle='--', alpha=0.5)
            
            if self.array.dy > 0.5:
                sin_arg_e = 1/self.array.dy - np.sin(np.radians(self.theta_scan_slider.value()))
                if -1 <= sin_arg_e <= 1:
                    grating_angle_e = np.degrees(np.arcsin(sin_arg_e))
                    if -90 <= grating_angle_e <= 90:
                        ax2.axvline(x=grating_angle_e, color='red', linestyle='--', alpha=0.5)
            
            ax1.legend(fontsize=8, loc='upper right')
            ax2.legend(fontsize=8, loc='upper right')
            
            # График 3: Амплитудное распределение с числами
            ax3 = self.figure.add_subplot(223)
            display_data = self.array.amplitudes.copy()
            display_data[~self.array.element_mask] = np.nan
            
            im = ax3.imshow(display_data, cmap='RdYlGn', 
                           vmin=0, vmax=1, aspect='auto',
                           interpolation='nearest')
            
            # Добавляем числа на ячейки
            for i in range(self.array.Ny):
                for j in range(self.array.Nx):
                    if self.array.element_mask[i, j]:
                        val = self.array.amplitudes[i, j]
                        color = 'white' if val < 0.5 else 'black'
                        ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color=color, fontweight='bold', fontsize=7)
                    else:
                        ax3.text(j, i, 'OFF', ha='center', va='center',
                                color='black', fontweight='bold', fontsize=6)
            
            ax3.set_xlabel('Индекс по X', fontsize=10)
            ax3.set_ylabel('Индекс по Y', fontsize=10)
            ax3.set_title(f'Амплитудное распределение\nРабочих элементов: {self.array.get_active_count()}/{self.array.Nx*self.array.Ny}', 
                         fontsize=10, fontweight='bold')
            
            cbar1 = self.figure.colorbar(im, ax=ax3, label='Амплитуда')
            cbar1.ax.tick_params(labelsize=8)
            
            # График 4: Фазовое распределение с числами
            ax4 = self.figure.add_subplot(224)
            phase_data = np.where(self.array.element_mask, self.array.phases, np.nan)
            
            im2 = ax4.imshow(phase_data, cmap='hsv', aspect='auto',
                            interpolation='nearest', vmin=-np.pi, vmax=np.pi)
            
            # Добавляем числа фаз
            for i in range(self.array.Ny):
                for j in range(self.array.Nx):
                    if self.array.element_mask[i, j]:
                        val = self.array.phases[i, j]
                        ax4.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color='black', fontweight='bold', fontsize=6)
            
            ax4.set_xlabel('Индекс по X', fontsize=10)
            ax4.set_ylabel('Индекс по Y', fontsize=10)
            ax4.set_title(f'Фазовое распределение (рад)\nСканирование: {self.theta_scan_slider.value()} град', 
                         fontsize=10, fontweight='bold')
            
            cbar2 = self.figure.colorbar(im2, ax=ax4, label='Фаза (рад)')
            cbar2.ax.tick_params(labelsize=8)
            
            self.figure.tight_layout(pad=3.0)
            self.canvas.draw()
            
            self.update_params()
            self.statusBar().showMessage("Графики обновлены", 2000)
            
        except Exception as e:
            write_log(f"Ошибка в update_plot_delayed: {e}", "ERROR")
            self.statusBar().showMessage(f"Ошибка обновления: {str(e)}")


def main():
    write_log("\n" + "="*60, "INFO")
    write_log("ЗАПУСК ПРИЛОЖЕНИЯ", "INFO")
    write_log("="*60, "INFO")
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("Antenna Array Simulator")
        
        splash = SplashScreen()
        splash.show()
        
        window = MainWindow()
        
        while splash.isVisible():
            app.processEvents()
        
        window.show()
        write_log("Главное окно отображено", "INFO")
        
        exit_code = app.exec_()
        write_log(f"Приложение завершено с кодом: {exit_code}", "INFO")
        
        return exit_code
        
    except Exception as e:
        write_log(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", "ERROR")
        write_log(traceback.format_exc(), "ERROR")
        
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        
        QMessageBox.critical(None, "Критическая ошибка", 
                           f"Произошла критическая ошибка:\n{str(e)}\n\n"
                           f"Подробности в лог-файле: {log_file}")
        return 1


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Исследование влияния отазов на характеристики ФАР")
    print("="*70)
    print(f"Логи сохраняются в: {log_file}")
    print("="*70 + "\n")
    
    exit_code = main()
    
    write_log(f"\n" + "="*60, "INFO")
    write_log(f"ПРОГРАММА ЗАВЕРШИЛАСЬ С КОДОМ: {exit_code}", "INFO")
    write_log("="*60, "INFO")
    
    sys.exit(exit_code)