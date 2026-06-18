"""
Окно конструктора излучателя.
Позволяет выбрать тип и настроить параметры.
Блокирует основную программу, пока излучатель не валиден.
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QGroupBox, QLabel, QPushButton, QTabWidget,
                             QDoubleSpinBox, QTextEdit, QWidget, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from antenna_elements import (AntennaElement, ElementType, ElementParams,
                              PyramidalHornParams, PyramidalHorn,
                              PatchParams, PatchAntenna,
                              DipoleAntenna, IsotropicElement,
                              create_element)


class ElementDesignDialog(QDialog):
    """Диалог проектирования излучателя"""
    
    def __init__(self, parent=None, default_freq=8.33, **kwargs):
        super().__init__(parent)
        self.setWindowTitle("Конструктор излучателя")
        self.setGeometry(200, 100, 750, 680)
        self.setMinimumSize(700, 600)
        
        self.element = None
        self.current_element = None
        
        # Сохраняем переданные параметры
        self.initial_params = kwargs
        
        self.init_ui()
        
        # Устанавливаем переданные значения
        if 'a' in kwargs:
            self.pyr_a.setValue(kwargs['a'])
        if 'b' in kwargs:
            self.pyr_b.setValue(kwargs['b'])
        if 'L' in kwargs:
            self.pyr_L.setValue(kwargs['L'])
        if 'frequency' in kwargs:
            self.freq_spin.setValue(kwargs['frequency'])
        
        # Загружаем пирамидальный рупор
        self.on_pyramidal_changed()
    
    # ============================================================
    # Интерфейс
    # ============================================================
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # --- Частота (общая) ---
        freq_group = QGroupBox("Рабочая частота")
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Частота:"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1.0, 100.0)
        self.freq_spin.setValue(8.33)
        self.freq_spin.setSuffix(" ГГц")
        self.freq_spin.setDecimals(2)
        self.freq_spin.valueChanged.connect(self.on_frequency_changed)
        freq_layout.addWidget(self.freq_spin)
        
        self.wavelength_label = QLabel("λ = 36.0 мм")
        freq_layout.addWidget(self.wavelength_label)
        freq_layout.addStretch()
        freq_group.setLayout(freq_layout)
        main_layout.addWidget(freq_group)
        
        # --- Вкладки типов ---
        self.tab_widget = QTabWidget()
        
        self.pyramidal_tab = self._build_pyramidal_tab()
        self.tab_widget.addTab(self.pyramidal_tab, "Пирамидальный рупор")
        
        self.conical_tab = self._build_conical_tab()
        self.tab_widget.addTab(self.conical_tab, "Конический рупор")
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tab_widget)
        
        bottom = QHBoxLayout()

        status_widget = QWidget()
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_label = QLabel("Настройте параметры излучателя")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont('Times New roman', 12, QFont.Bold))
        status_layout.addWidget(self.status_label)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(180)
        self.details_text.setFont(QFont('Times New roman', 9))
        status_layout.addWidget(self.details_text)
        
        status_widget.setLayout(status_layout)
        bottom.addWidget(status_widget, 1)
        
        # График ДН элемента
        self.figure = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        bottom.addWidget(self.canvas, 1)
        
        main_layout.addLayout(bottom)
        
        # --- Кнопки ---
        btn_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Применить излучатель и продолжить")
        self.apply_btn.setMinimumHeight(45)
        self.apply_btn.setFont(QFont('Times New roman', 12, QFont.Bold))
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.on_apply)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #5cb85c; color: white;
                border: 2px solid #4cae4c; border-radius: 6px;
            }
            QPushButton:hover { background-color: #449d44; }
            QPushButton:disabled { background-color: #ccc; border-color: #aaa; }
        """)
        btn_layout.addWidget(self.apply_btn)
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
    
    # ============================================================
    # Вкладка: Пирамидальный рупор
    # ============================================================
    
    def _build_pyramidal_tab(self):
        widget = QWidget()
        layout = QGridLayout()
        layout.setVerticalSpacing(12)
        
        # Размер a
        layout.addWidget(QLabel("Размер раскрыва в плоскости H:"), 0, 0)
        self.pyr_a = QDoubleSpinBox()
        self.pyr_a.setRange(0.2, 15.0)
        self.pyr_a.setValue(0.5)
        self.pyr_a.setSuffix(" λ")
        self.pyr_a.setDecimals(3)
        self.pyr_a.setSingleStep(0.1)
        self.pyr_a.setToolTip("Размер раскрыва в плоскости H (магнитной)")
        self.pyr_a.valueChanged.connect(self.on_pyramidal_changed)
        layout.addWidget(self.pyr_a, 0, 1)
        
        # Размер b
        layout.addWidget(QLabel("Размер раскрыва в плоскости E:"), 1, 0)
        self.pyr_b = QDoubleSpinBox()
        self.pyr_b.setRange(0.2, 15.0)
        self.pyr_b.setValue(0.5)
        self.pyr_b.setSuffix(" λ")
        self.pyr_b.setDecimals(3)
        self.pyr_b.setSingleStep(0.1)
        self.pyr_b.setToolTip("Размер раскрыва в плоскости E")
        self.pyr_b.valueChanged.connect(self.on_pyramidal_changed)
        layout.addWidget(self.pyr_b, 1, 1)
        
        # Длина L
        layout.addWidget(QLabel("Длина рупора h:"), 2, 0)
        self.pyr_L = QDoubleSpinBox()
        self.pyr_L.setRange(0.2, 30.0)
        self.pyr_L.setValue(0.5)
        self.pyr_L.setSuffix(" λ")
        self.pyr_L.setDecimals(2)
        self.pyr_L.setSingleStep(0.5)
        self.pyr_L.setToolTip("Длина рупора от горловины до раскрыва")
        self.pyr_L.valueChanged.connect(self.on_pyramidal_changed)
        layout.addWidget(self.pyr_L, 2, 1)
        
        # Информация о фазовых ошибках
        self.pyr_phase_label = QLabel()
        self.pyr_phase_label.setFont(QFont('Times New roman', 10))
        self.pyr_phase_label.setStyleSheet("background: #f8f8f8; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.pyr_phase_label, 4, 0, 1, 2)
        
        widget.setLayout(layout)
        return widget
    
    # ============================================================
    # Вкладки-заглушки
    # ============================================================
    
    def _build_stub_tab(self, title, description):
        widget = QWidget()
        layout = QVBoxLayout()
        
        text = QLabel(f"<b>{title}</b><br><br>{description}")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        text.setFont(QFont('Times New roman', 11))
        layout.addWidget(text)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def _build_patch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        text = QLabel(
            "<b>Печатный излучатель (заглушка)</b><br><br>"
            "Используется упрощённая модель ДН: cos¹·⁵(θ).<br>"
            "Полноценный расчёт с учётом подложки будет добавлен позже."
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        text.setFont(QFont('Times New roman', 11))
        layout.addWidget(text)
        
        apply_stub = QPushButton("Использовать печатный излучатель (заглушка)")
        apply_stub.clicked.connect(lambda: self._use_stub('patch'))
        layout.addWidget(apply_stub)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget

    def _build_conical_tab(self):
        """Вкладка конического рупора"""
        widget = QWidget()
        layout = QGridLayout()
        layout.setVerticalSpacing(12)
        
        layout.addWidget(QLabel("Диаметр раскрыва:"), 0, 0)
        self.con_D = QDoubleSpinBox()
        self.con_D.setRange(0.2, 15.0)
        self.con_D.setValue(0.5)
        self.con_D.setSuffix(" λ")
        self.con_D.setDecimals(3)
        self.con_D.setSingleStep(0.1)
        self.con_D.setToolTip("Диаметр раскрыва рупора")
        self.con_D.valueChanged.connect(self.on_conical_changed)
        layout.addWidget(self.con_D, 0, 1)
        
        layout.addWidget(QLabel("Диаметр волновода:"), 1, 0)
        self.con_dwg = QDoubleSpinBox()
        self.con_dwg.setRange(0.1, 5.0)
        self.con_dwg.setValue(0.77)
        self.con_dwg.setSuffix(" λ")
        self.con_dwg.setDecimals(3)
        self.con_dwg.setSingleStep(0.05)
        self.con_dwg.setToolTip("Диаметр питающего волновода")
        self.con_dwg.valueChanged.connect(self.on_conical_changed)
        layout.addWidget(self.con_dwg, 1, 1)
        
        layout.addWidget(QLabel("Длина рупора:"), 2, 0)
        self.con_L = QDoubleSpinBox()
        self.con_L.setRange(0.2, 30.0)
        self.con_L.setValue(1.0)
        self.con_L.setSuffix(" λ")
        self.con_L.setDecimals(2)
        self.con_L.setSingleStep(0.5)
        self.con_L.setToolTip("Длина рупора от горловины до раскрыва")
        self.con_L.valueChanged.connect(self.on_conical_changed)
        layout.addWidget(self.con_L, 2, 1)
        
        self.con_phase_label = QLabel()
        self.con_phase_label.setFont(QFont('Times New roman', 10))
        self.con_phase_label.setStyleSheet("background: #f8f8f8; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.con_phase_label, 3, 0, 1, 2)
        
        widget.setLayout(layout)
        return widget

    def _build_dipole_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        text = QLabel(
            "<b>Полуволновый вибратор (заглушка)</b><br><br>"
            "Аналитическая ДН: |cos(π/2·cosθ)/sinθ|.<br>"
            "Расширенные настройки — в будущих версиях."
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        text.setFont(QFont('Times New roman', 11))
        layout.addWidget(text)
        
        apply_stub = QPushButton("Использовать вибратор (заглушка)")
        apply_stub.clicked.connect(lambda: self._use_stub('dipole'))
        layout.addWidget(apply_stub)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def _build_isotropic_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        text = QLabel(
            "<b>Изотропный излучатель</b><br><br>"
            "Идеальный ненаправленный элемент. F(θ) = 1.<br>"
            "Для отладки и сравнения с теорией."
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        text.setFont(QFont('Times New roman', 11))
        layout.addWidget(text)
        
        apply_stub = QPushButton("Использовать изотропный")
        apply_stub.clicked.connect(lambda: self._use_stub('isotropic'))
        layout.addWidget(apply_stub)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    # ============================================================
    # Логика
    # ============================================================
    
    def on_frequency_changed(self):
        freq = self.freq_spin.value()
        wl_mm = 300 / freq
        self.wavelength_label.setText(f"λ = {wl_mm:.1f} мм")
        
        # Пересчитываем текущий элемент с новой частотой
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 0:  # пирамидальный
            self.on_pyramidal_changed()
    
    def on_tab_changed(self, index):
        if index == 0:  # пирамидальный
            self.on_pyramidal_changed()
        elif index == 3:  # конический
            self.on_conical_changed()
    
    def on_pyramidal_changed(self):
        """Создать/обновить пирамидальный рупор"""
        params = PyramidalHornParams(
            frequency=self.freq_spin.value(),
            a=self.pyr_a.value(),
            b=self.pyr_b.value(),
            L=self.pyr_L.value()
        )
        self.current_element = PyramidalHorn(params)
        self._update_status()
        self._update_plot()

    def on_conical_changed(self):
        """Создать/обновить конический рупор"""
        from antenna_elements import ConicalHornParams, ConicalHorn
        
        params = ConicalHornParams(
            frequency=self.freq_spin.value(),
            D=self.con_D.value(),
            d_wg=self.con_dwg.value(),
            L=self.con_L.value()
        )
        self.current_element = ConicalHorn(params)
        self._update_status()
        self._update_plot()    
    def on_optimize_pyramidal(self):
        L = self.pyr_L.value()
    
        # Условие оптимальности: Δφ = π/2
        # Δφ = π·a²/(4L) = π/2  →  a² = 2L  →  a = √(2L)
        a_opt = np.sqrt(2 * L)
    
        # Для E-плоскости обычно b ≈ 0.8·a
        b_opt = a_opt * 0.8
    
        # Ограничиваем разумными значениями
        a_opt = min(max(a_opt, 0.5), 10.0)
        b_opt = min(max(b_opt, 0.5), 10.0)
    
        self.pyr_a.blockSignals(True)
        self.pyr_b.blockSignals(True)
        self.pyr_a.setValue(round(a_opt, 3))
        self.pyr_b.setValue(round(b_opt, 3))
        self.pyr_a.blockSignals(False)
        self.pyr_b.blockSignals(False)
    
        self.on_pyramidal_changed()
    
    def _use_stub(self, stub_type):
        """Применить заглушку"""
        freq = self.freq_spin.value()
        
        if stub_type == 'patch':
            self.current_element = PatchAntenna(PatchParams(frequency=freq))
        elif stub_type == 'dipole':
            self.current_element = DipoleAntenna(ElementParams(frequency=freq))
        elif stub_type == 'isotropic':
            self.current_element = IsotropicElement(ElementParams(frequency=freq))
        
        self._update_status()
        self._update_plot()
    
    def _update_status(self):
        if self.current_element is None:
            return
        
        v = self.current_element.validate()
        
        # Статусная строка
        if v['valid']:
            self.status_label.setText("Размеры излучателя выбраны правильно")
            self.status_label.setStyleSheet("color: #2d6a2d; font-weight: bold;")
            self.apply_btn.setEnabled(True)
        else:
            self.status_label.setText("Излучатель недопустим - исправьте ошибки")
            self.status_label.setStyleSheet("color: #a00; font-weight: bold;")
            self.apply_btn.setEnabled(False)
        
        # Детали
        details = "ХАРАКТЕРИСТИКИ ИЗЛУЧАТЕЛЯ\n"
        details += "-" * 40 + "\n"
        details += f"  Тип:           {self.current_element.__class__.__name__}\n"
        details += f"  Частота:       {self.current_element.params.frequency:.2f} ГГц\n"
        details += f"  Длина волны:   {self.current_element.params.wavelength:.1f} мм\n\n"
        
        if hasattr(self.current_element, '_phase_error_H'):
            details += "ФАЗОВЫЕ ОШИБКИ НА КРАЮ РАСКРЫВА:\n"
            details += f"  H-плоскость:   {v['phase_error_H_deg']:.1f} град"
            if v['phase_error_H_deg'] <= 135:
                details += " (<= 135, OK)\n"
            else:
                details += " (> 135, ошибка)\n"
            
            details += f"  E-плоскость:   {v['phase_error_E_deg']:.1f} град"
            if v['phase_error_E_deg'] <= 135:
                details += " (<= 135, OK)\n"
            else:
                details += " (> 135, ошибка)\n"
            details += "\n"
        elif hasattr(self.current_element, '_phase_error') and not hasattr(self.current_element, '_phase_error_H'):
            ph = v.get('phase_error_deg', 0)
            details += "ФАЗОВАЯ ОШИБКА НА КРАЮ РАСКРЫВА:\n"
            details += f"  Delta_phi = {ph:.1f}° "
            if ph <= 135:
                details += "(<= 135, OK)\n"
            else:
                details += "(> 135, ошибка)\n"
            details += "\n"        
        details += f"  КУ (оценка):   {self.current_element.get_gain_estimate():.1f} дБи\n"
        details += f"  Физ. размер:   {v['physical_size'][0]:.2f} x {v['physical_size'][1]:.2f} lambda\n\n"
   
        if v['warnings']:
            details += "ПРЕДУПРЕЖДЕНИЯ:\n"
            for w in v['warnings']:
                details += f"  - {w}\n"
            details += "\n"
        
        if v['errors']:
            details += "ОШИБКИ:\n"
            for e in v['errors']:
                details += f"  - {e}\n"
        
        self.details_text.setText(details)
        
        if hasattr(self, 'pyr_phase_label') and hasattr(self.current_element, '_phase_error_H'):
            ph_H = v['phase_error_H_deg']
            ph_E = v['phase_error_E_deg']
            
            def status_text(val):
                return "OK" if val <= 135 else "ОШИБКА"
            
            self.pyr_phase_label.setText(
                f"Фазовая ошибка:  H: {ph_H:.1f} град [{status_text(ph_H)}]    "
                f"E: {ph_E:.1f} град [{status_text(ph_E)}]\n"
                f"Предельно допустимо: 135 град"
            )
        if hasattr(self, 'con_phase_label') and hasattr(self.current_element, '_phase_error') and not hasattr(self.current_element, '_phase_error_H'):
            ph = v.get('phase_error_deg', 0)
            status = "OK" if ph <= 135 else "ОШИБКА"
            self.con_phase_label.setText(
                f"Фазовая ошибка: {ph:.1f}° [{status}]\n"
                f"Предельно допустимо: 135°"
            )
    
    def _update_plot(self):
        if self.current_element is None:
            return
        
        self.figure.clf()
        theta = np.linspace(-np.pi/2, np.pi/2, 721)  # больше точек для точности
        theta_deg = np.degrees(theta)
        
        # ========================================
        # H-плоскость
        # ========================================
        ax1 = self.figure.add_subplot(121)
        pat_H = self.current_element.pattern(theta, phi=0)
        pat_H_db = 20 * np.log10(np.maximum(pat_H, 1e-10))
        
        ax1.plot(theta_deg, pat_H_db, 'b-', linewidth=2, label='ДН')
        ax1.fill_between(theta_deg, -50, pat_H_db, alpha=0.06, color='blue')
        
        # --- Анализ H-плоскости ---
        bw_H, left_H, right_H = self._calc_beamwidth(theta_deg, pat_H_db)
        sll_H, sll_angle_H = self._calc_sll(theta_deg, pat_H_db)
        
        # Ширина луча
        if bw_H is not None:
            ax1.axvspan(left_H, right_H, alpha=0.15, color='cyan')
            ax1.axvline(x=left_H, color='blue', linestyle='--', alpha=0.6, linewidth=1)
            ax1.axvline(x=right_H, color='blue', linestyle='--', alpha=0.6, linewidth=1)
            ax1.annotate(f'{bw_H:.1f}°', 
                        xy=((left_H+right_H)/2, -3),
                        fontsize=9, ha='center', color='blue',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7))
        
        # Уровень -3 дБ
        ax1.axhline(y=-3, color='blue', linestyle=':', alpha=0.4, linewidth=1)
        
        # УБЛ
        if sll_H is not None:
            ax1.axhline(y=sll_H, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
            ax1.annotate(f'УБЛ={sll_H:.1f} дБ', 
                        xy=(sll_angle_H, sll_H),
                        fontsize=8, color='red', ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        if bw_H is not None and sll_H is not None:
            title_H = f'H-плоскость (φ=0°)\nШирина: {bw_H:.1f}° | УБЛ: {sll_H:.1f} дБ'
        elif bw_H is not None:
            title_H = f'H-плоскость (φ=0°)\nШирина: {bw_H:.1f}°'
        else:
            title_H = 'H-плоскость (φ=0°)'
        ax1.set_title(title_H, fontsize=10, fontweight='bold')
        ax1.set_xlabel('θ, град')
        ax1.set_ylabel('Уровень, дБ')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(-90, 90)
        ax1.set_ylim(-50, 3)
        ax1.legend(fontsize=8, loc='upper right')
        
        # ========================================
        # E-плоскость
        # ========================================
        ax2 = self.figure.add_subplot(122)
        pat_E = self.current_element.pattern(theta, phi=np.pi/2)
        pat_E_db = 20 * np.log10(np.maximum(pat_E, 1e-10))
        
        ax2.plot(theta_deg, pat_E_db, 'g-', linewidth=2, label='ДН')
        ax2.fill_between(theta_deg, -50, pat_E_db, alpha=0.06, color='green')
        
        # --- Анализ E-плоскости ---
        bw_E, left_E, right_E = self._calc_beamwidth(theta_deg, pat_E_db)
        sll_E, sll_angle_E = self._calc_sll(theta_deg, pat_E_db)
        
        if bw_E is not None:
            ax2.axvspan(left_E, right_E, alpha=0.15, color='yellow')
            ax2.axvline(x=left_E, color='green', linestyle='--', alpha=0.6, linewidth=1)
            ax2.axvline(x=right_E, color='green', linestyle='--', alpha=0.6, linewidth=1)
            ax2.annotate(f'{bw_E:.1f}°',
                        xy=((left_E+right_E)/2, -3),
                        fontsize=9, ha='center', color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.axhline(y=-3, color='green', linestyle=':', alpha=0.4, linewidth=1)
        
        if sll_E is not None:
            ax2.axhline(y=sll_E, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
            ax2.annotate(f'УБЛ={sll_E:.1f} дБ',
                        xy=(sll_angle_E, sll_E),
                        fontsize=8, color='red', ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        if bw_E is not None and sll_E is not None:
            title_E = f'E-плоскость (φ=90°)\nШирина: {bw_E:.1f}° | УБЛ: {sll_E:.1f} дБ'
        elif bw_E is not None:
            title_E = f'E-плоскость (φ=90°)\nШирина: {bw_E:.1f}°'
        else:
            title_E = 'E-плоскость (φ=90°)'
        ax2.set_title(title_E, fontsize=10, fontweight='bold')
        ax2.set_xlabel('θ, град')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-90, 90)
        ax2.set_ylim(-50, 3)
        ax2.legend(fontsize=8, loc='upper right')
        
        self.figure.tight_layout(pad=2.5)
        self.canvas.draw()


    def _calc_beamwidth(self, theta_deg, pattern_db):
        """Расчёт ширины луча по уровню -3 дБ"""
        # Ищем максимум
        max_idx = np.argmax(pattern_db)
        max_level = pattern_db[max_idx]
        target = max_level - 3
        
        # Левый край
        left_idx = None
        for i in range(max_idx, -1, -1):
            if pattern_db[i] <= target:
                left_idx = i
                break
        
        # Правый край
        right_idx = None
        for i in range(max_idx, len(pattern_db)):
            if pattern_db[i] <= target:
                right_idx = i
                break
        
        if left_idx is None or right_idx is None:
            return None, None, None
        
        # Линейная интерполяция для точности
        if left_idx > 0:
            t1, t2 = theta_deg[left_idx-1], theta_deg[left_idx]
            p1, p2 = pattern_db[left_idx-1], pattern_db[left_idx]
            if abs(p1 - p2) > 1e-10:
                left_angle = t1 + (target - p1) * (t2 - t1) / (p2 - p1)
            else:
                left_angle = theta_deg[left_idx]
        else:
            left_angle = theta_deg[left_idx]
        
        if right_idx < len(pattern_db) - 1:
            t1, t2 = theta_deg[right_idx], theta_deg[right_idx+1]
            p1, p2 = pattern_db[right_idx], pattern_db[right_idx+1]
            if abs(p1 - p2) > 1e-10:
                right_angle = t1 + (target - p1) * (t2 - t1) / (p2 - p1)
            else:
                right_angle = theta_deg[right_idx]
        else:
            right_angle = theta_deg[right_idx]
        
        bw = abs(right_angle - left_angle)
        return bw, left_angle, right_angle


    def _calc_sll(self, theta_deg, pattern_db):
        """Расчёт уровня первого бокового лепестка"""
        max_idx = np.argmax(pattern_db)
        
        left_null = None
        for i in range(max_idx-1, 1, -1):
            if pattern_db[i] > pattern_db[i-1] and pattern_db[i] > pattern_db[i+1]:
                continue  # это максимум, пропускаем
            if pattern_db[i] <= -25 and pattern_db[i-1] > -25:
                left_null = i
                break
        
        # Правый ноль
        right_null = None
        for i in range(max_idx+1, len(pattern_db)-2):
            if pattern_db[i] > pattern_db[i-1] and pattern_db[i] > pattern_db[i+1]:
                continue
            if pattern_db[i] <= -25 and pattern_db[i+1] < -25:
                right_null = i
                break
        
        # Ищем боковые лепестки за нулями
        sidelobes = []
        
        # Слева
        if left_null is not None:
            start = max(1, left_null - 30)
            for i in range(left_null-1, start, -1):
                if i > 1 and i < len(pattern_db)-1:
                    if pattern_db[i] > pattern_db[i-1] and pattern_db[i] >= pattern_db[i+1]:
                        sidelobes.append((theta_deg[i], pattern_db[i]))
                        break
        
        # Справа
        if right_null is not None:
            end = min(len(pattern_db)-2, right_null + 30)
            for i in range(right_null+1, end):
                if i > 1 and i < len(pattern_db)-1:
                    if pattern_db[i] > pattern_db[i-1] and pattern_db[i] >= pattern_db[i+1]:
                        sidelobes.append((theta_deg[i], pattern_db[i]))
                        break
        
        if not sidelobes:
            # Запасной метод: просто ищем максимум за пределами главного луча
            main_lobe_start = max(0, max_idx - 15)
            main_lobe_end = min(len(pattern_db), max_idx + 15)
            
            for i in range(15, len(pattern_db)-15):
                if main_lobe_start <= i <= main_lobe_end:
                    continue
                if pattern_db[i] > pattern_db[i-1] and pattern_db[i] >= pattern_db[i+1]:
                    sidelobes.append((theta_deg[i], pattern_db[i]))
            
            if sidelobes:
                # Берём самый сильный
                strongest = max(sidelobes, key=lambda x: x[1])
                sidelobes = [strongest]
        
        if sidelobes:
            strongest = max(sidelobes, key=lambda x: x[1])
            return strongest[1], strongest[0]
        
        return None, None

    
    def on_apply(self):
        """Проверить и принять излучатель"""
        if self.current_element is None:
            QMessageBox.warning(self, "Ошибка", "Излучатель не построен")
            return
        
        v = self.current_element.validate()
        
        if not v['valid']:
            msg = "Излучатель не прошёл валидацию:\n\n"
            for e in v['errors']:
                msg += f"  ❌ {e}\n"
            QMessageBox.warning(self, "Ошибка валидации", msg)
            return
        
        if not v['optimal'] and v['warnings']:
            msg = "Излучатель не оптимален:\n\n"
            for w in v['warnings']:
                msg += f"  ⚠ {w}\n"
            msg += "\nВсё равно использовать?"
            reply = QMessageBox.question(self, "Предупреждение", msg,
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        self.element = self.current_element
        self.accept()
    
    def get_element(self) -> AntennaElement:
        """Получить созданный излучатель"""
        return self.element