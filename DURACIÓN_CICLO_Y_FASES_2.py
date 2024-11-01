import os
import sys
from datetime import datetime
from PyQt6.QtCore import Qt, QPoint
from PyQt6.uic import loadUi
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter, butter
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from PyQt6 import QtCore
from scipy.signal import lfilter
import csv
import math
from scipy.stats import rayleigh


class MainWindow(QMainWindow):
    def __init__(self, archivo, channel_colors, stim_colors, values):
        """Inicializa la ventana principal de la aplicación."""
        super().__init__()

        # Cargar la interfaz gráfica desde el archivo .ui
        loadUi('DURACION_CICLO_Y_FASES_GUI_con_HP_e_Hyst_desc.ui', self)

        # Configurar el widget central y el diseño principal de la ventana
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)
        central_widget.setLayout(layout)

        # Crear widgets de trazado para diferentes gráficos y añadirlos a los layouts correspondientes
        self.plot_widget_1 = pg.PlotWidget()
        self.Graph_layout_1.addWidget(self.plot_widget_1)

        self.plot_widget_2 = pg.PlotWidget()
        self.Graph_layout_2.addWidget(self.plot_widget_2)

        self.plot_widget_envolvente_1 = pg.PlotWidget()
        self.Graph_layout_3.addWidget(self.plot_widget_envolvente_1)

        self.plot_widget_envolvente_2 = pg.PlotWidget()
        self.Graph_layout_4.addWidget(self.plot_widget_envolvente_2)

        # Configurar menús desplegables para selección de canales
        options = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                   'CH 10', 'CH 11', 'CH 12']
        # Añadir las opciones al menú desplegable de canales para la señal 1
        self.column_combobox_1.addItems(options)
        # Conectar la señal de cambio de selección a la función correspondiente
        self.column_combobox_1.currentIndexChanged.connect(self.channel_1)

        # Crear un diccionario que asocia cada opción del menú desplegable con un color
        self.channel_color_map = dict(zip(options, channel_colors))

        # Configurar el menú desplegable de canales para la señal 2
        self.column_combobox_2.addItems(options)
        self.column_combobox_2.currentIndexChanged.connect(self.channel_2)

        # Conectar los cambios en los umbrales a métodos que actualizan los valores correspondientes
        self.threshold_1_sb.valueChanged.connect(self.update_threshold_1)
        self.threshold_2_sb.valueChanged.connect(self.update_threshold_2)

        # Conectar los cambios en los controles de histéresis a sus métodos de actualización
        self.SB_Hysteresis_asc_1.valueChanged.connect(self.actualizar_hyst_asc_1)
        self.SB_Hysteresis_asc_2.valueChanged.connect(self.actualizar_hyst_asc_2)
        self.SB_Hysteresis_desc_1.valueChanged.connect(self.actualizar_hyst_desc_1)
        self.SB_Hysteresis_desc_2.valueChanged.connect(self.actualizar_hyst_desc_2)

        # Conectar los cambios en los controles de frecuencia de corte de filtros a sus métodos de actualización
        self.cutoff_LP_1.valueChanged.connect(self.actualizar_LP_fc_1)
        self.cutoff_LP_2.valueChanged.connect(self.actualizar_LP_fc_2)
        self.cutoff_HP_1.valueChanged.connect(self.actualizar_HP_fc_1)
        self.cutoff_HP_2.valueChanged.connect(self.actualizar_HP_fc_2)

        # Conectar los cambios en los tiempos de retención a sus métodos de actualización
        self.hold_off_spinbox_1.valueChanged.connect(self.actualizar_hold_off_1)
        self.hold_off_spinbox_2.valueChanged.connect(self.actualizar_hold_off_2)

        # Conectar los botones de filtro y detección de cruces a sus métodos correspondientes
        self.btn_hp_1.clicked.connect(self.update_filter_1)
        self.btn_hp_2.clicked.connect(self.update_filter_2)
        self.btn_cruces_1.clicked.connect(self.detectar_cruces_1)
        self.btn_cruces_2.clicked.connect(self.detectar_cruces_2)

        # Conectar el cambio en el tiempo de ventana a su método de actualización
        self.btn_window_time.valueChanged.connect(self.update_window_time)

        # Conectar los botones para avanzar y retroceder gráficos a sus métodos correspondientes
        self.btn_avanzar.clicked.connect(self.avanzar_graficas)
        self.btn_retroceder.clicked.connect(self.retroceder_graficas)

        # Conectar el botón para iniciar el análisis de ciclos a su método correspondiente
        self.bt_go.clicked.connect(self.cycle_duration_analysis)

        # Conectar los cambios en los rangos de amplitud a sus métodos de actualización
        self.y_max_1.valueChanged.connect(self.update_y_range_1)
        self.y_min_1.valueChanged.connect(self.update_y_range_1)
        self.y_max_2.valueChanged.connect(self.update_y_range_2)
        self.y_min_2.valueChanged.connect(self.update_y_range_2)
        self.y_max_norm_1.valueChanged.connect(self.update_y_range_norm_1)
        self.y_min_norm_1.valueChanged.connect(self.update_y_range_norm_1)
        self.y_max_norm_2.valueChanged.connect(self.update_y_range_norm_2)
        self.y_min_norm_2.valueChanged.connect(self.update_y_range_norm_2)

        # Conectar el botón de guardar a su método correspondiente
        self.save_function.clicked.connect(self.save_active)

        # Conectar los cambios en los tiempos inicial y final a sus métodos de actualización
        self.t_inic_a.valueChanged.connect(self.update_t_inic_a)
        self.t_fin_a.valueChanged.connect(self.update_t_fin_a)

        # Configurar las opciones de estímulo
        stims = ['STIM 1', 'STIM 2', 'STIM 3']
        self.stim_colors = stim_colors
        self.stims_options.addItems(stims)
        self.stims_options.currentIndexChanged.connect(self.update_stim)
        self.stim_color_map = dict(zip(stims, self.stim_colors))

        # Conectar el cambio en el nombre del archivo a su método de actualización
        self.name_file.textChanged.connect(self.name_file_update)

        # Conectar eventos de clic del mouse en los widgets de trazado a sus métodos correspondientes
        self.plot_widget_envolvente_1.scene().sigMouseClicked.connect(self.on_plot_clicked_3)
        self.plot_widget_envolvente_2.scene().sigMouseClicked.connect(self.on_plot_clicked_4)

        # Inicializar parámetros de archivo y datos
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)
        self.data = pd.read_csv(self.ruta_archivo)  # Leer datos del archivo CSV
        self.data_copy = pd.read_csv(self.ruta_archivo)  # Copia de los datos para uso futuro

        # Configurar requisitos de filtro estándar
        self.fs = 3300  # Frecuencia de muestreo
        self.cutoff_1_LP = 100  # Frecuencia de corte del filtro pasa bajas 1
        self.cutoff_2_LP = 100  # Frecuencia de corte del filtro pasa bajas 2
        self.cutoff_1_HP = 0  # Frecuencia de corte del filtro pasa altas 1
        self.cutoff_2_HP = 0  # Frecuencia de corte del filtro pasa altas 2
        self.delay_1 = float(values[3])  # Retraso 1
        self.delay_2 = float(values[3])  # Retraso 2
        self.hold_off_1 = (int(values[2])) / 1000  # Tiempo de retención 1 en segundos
        self.hold_off_spinbox_1.setValue(int(values[2]))
        self.hold_off_2 = (int(values[2])) / 1000  # Tiempo de retención 2 en segundos
        self.hold_off_spinbox_2.setValue(int(values[2]))
        self.threshold_1 = float(values[1])  # Umbral 1
        self.threshold_1_sb.setValue(self.threshold_1)
        self.threshold_2 = float(values[1])  # Umbral 2
        self.threshold_2_sb.setValue(self.threshold_2)
        self.hyst_asc_1 = float(values[4])  # Histéresis ascendente 1
        self.SB_Hysteresis_asc_1.setValue(self.hyst_asc_1)
        self.hyst_asc_2 = float(values[4])  # Histéresis ascendente 2
        self.SB_Hysteresis_asc_2.setValue(self.hyst_asc_2)
        self.hyst_desc_1 = float(values[5])  # Histéresis descendente 1
        self.SB_Hysteresis_desc_1.setValue(self.hyst_desc_1)
        self.hyst_desc_2 = float(values[5])  # Histéresis descendente 2
        self.SB_Hysteresis_desc_2.setValue(self.hyst_desc_2)

        # Inicializar datos de tiempo y estímulos
        self.tiempo = self.data['TIME'].to_numpy()
        self.stim_plot_1 = self.data['STIM 1'].to_numpy()
        self.stim_plot_2 = self.data['STIM 2'].to_numpy()
        self.stim_plot_3 = self.data['STIM 3'].to_numpy()
        self.new_stim = 'STIM 1'  # Estímulo seleccionado por defecto
        self.stim = self.data[self.new_stim].to_numpy()
        self.tag_graph = np.where(self.stim == 1, self.stim, 0)  # Datos para gráfico de etiqueta
        self.stim_graph = np.where(self.stim == 2, self.stim, 0)  # Datos para gráfico de estímulo

        # Configuración de parámetros de gráficos
        self.width_line = 2  # Ancho de línea para gráficos
        self.width_crossing = 1  # Ancho de línea para cruces

        # Inicializar parámetros de señales y colores
        self.s_1 = 'CH 1'  # Columna seleccionada para señal 1
        self.s_2 = 'CH 1'  # Columna seleccionada para señal 2
        self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
        self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
        self.signals = self.data[[f'{self.s_1}', f'{self.s_2}', f'{self.s_1}', f'{self.s_2}']].to_numpy()
        self.original_1 = self.signals[:, 2]
        self.original_2 = self.signals[:, 3]

        # Configuración de colores para señales y estímulos
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2]
        }

        self.stim_map = {
            self.new_stim: self.stim_color_map[self.new_stim],
        }
        self.original_color = '#5B5B5B'  # Color para señal original
        self.filtered_color = '#00FFFF'  # Color para señal filtrada
        self.stim_color = 'y'  # Color para estímulo
        self.up_color = 'm'  # Color para detección de subida
        self.down_color = '#FFFFFF'  # Color para detección de bajada
        self.threshold_color = 'g'  # Color para umbral

        # Configuración inicial de parámetros de visualización y estados
        self.start_a = 0
        self.end_a = 20
        self.window_time = self.end_a - self.start_a
        self.tiempo_inicial = self.start_a
        self.tiempo_final = self.end_a
        self.datos_por_s = round(1 / (self.tiempo[1] - self.tiempo[0]))
        self.start_a_num = round(self.start_a * self.datos_por_s)
        self.end_a_num = round(self.end_a * self.datos_por_s)
        self.tiempo_segment = self.tiempo[self.start_a_num: self.end_a_num]
        self.min_amplitude_1 = 0
        self.max_amplitude_1 = 2
        self.min_amplitude_2 = 0
        self.max_amplitude_2 = 2
        self.min_amplitude_norm_1 = 0
        self.max_amplitude_norm_1 = 1
        self.min_amplitude_norm_2 = 0
        self.max_amplitude_norm_2 = 1
        self.avanza = False
        self.retrocede = False
        self.window = False
        self.crossing_detected = False
        self.crossing_hysteresis_detected = False
        self.crossing_descendente = False
        self.first_up = False
        self.first_down = False
        self.cruces_1 = False
        self.cruces_2 = False
        self.filter_1_HP = False
        self.filter_2_HP = False
        self.filter_1_LP = False
        self.filter_2_LP = False
        self.save_function_flag = False
        self.new_file_name = "CYCLE_AND_PHASE"

        # Actualizar gráficos iniciales
        self.update_plots_1()
        self.update_plots_2()

    def save_active(self):
        """Alterna el estado del indicador de función de guardado.

        Cambia el valor de `self.save_function_flag` entre True y False.
        Si `self.save_function_flag` es False, se establece en True,
        indicando que la función de guardado está activa.
        Si es True, se establece en False, indicando que la función de guardado está desactivada."""
        if not self.save_function_flag:
            # Si la función de guardado no está activa, activarla
            self.save_function_flag = True
        else:
            # Si la función de guardado está activa, desactivarla
            self.save_function_flag = False

    def name_file_update(self, new_file_name):
        """Actualiza el nombre del archivo de salida.

        :param new_file_name: El nuevo nombre del archivo que se va a usar para guardar los datos."""
        self.new_file_name = new_file_name

    def update_t_inic_a(self, value):
        """Actualiza el tiempo inicial para el análisis de datos.

        :param value: Nuevo tiempo inicial en segundos."""
        self.start_a = value  # Establece el nuevo tiempo inicial en segundos
        self.start_a_num = round(self.start_a * self.datos_por_s)  # Convierte el tiempo inicial a índice de datos
        self.window_time = self.end_a - self.start_a  # Calcula la duración de la ventana
        self.btn_window_time.setValue(int(self.window_time))  # Actualiza el valor en el botón de tiempo de ventana
        self.avanza = False  # Reinicia la bandera de avance
        self.retrocede = False  # Reinicia la bandera de retroceso
        self.window = False  # Reinicia la bandera de ventana activa

        self.mover_ventana()  # Llama al método para actualizar la ventana de visualización

    def update_t_fin_a(self, value):
        """Actualiza el tiempo final para el análisis de datos.

        :param value: Nuevo tiempo final en segundos."""
        self.end_a = value  # Establece el nuevo tiempo final en segundos
        self.end_a_num = round(self.end_a * self.datos_por_s)  # Convierte el tiempo final a índice de datos
        self.window_time = self.end_a - self.start_a  # Calcula la duración de la ventana
        self.btn_window_time.setValue(int(self.window_time))  # Actualiza el valor en el botón de tiempo de ventana
        self.avanza = False  # Reinicia la bandera de avance
        self.retrocede = False  # Reinicia la bandera de retroceso
        self.window = False  # Reinicia la bandera de ventana activa

        self.mover_ventana()  # Llama al método para actualizar la ventana de visualización

    def update_stim(self):
        """Actualiza la selección de estímulo y ajusta las visualizaciones correspondientes.

        Obtiene el estímulo seleccionado del menú desplegable, actualiza los datos de estímulo
        y actualiza los mapas de color y gráficos relacionados."""
        # Obtener el estímulo seleccionado del menú desplegable
        self.new_stim = self.stims_options.currentText()

        # Extraer los datos del nuevo estímulo y convertirlos en un array de NumPy
        self.stim = self.data[self.new_stim].to_numpy()

        # Actualizar el mapa de colores del estímulo
        self.stim_map = {
            self.new_stim: self.stim_color_map[self.new_stim],
        }

        # Preparar datos para los gráficos:
        # Datos para el gráfico de etiquetas: valores donde el estímulo es 1
        self.tag_graph = np.where(self.stim == 1, self.stim, 0)
        # Datos para el gráfico de estímulos: valores donde el estímulo es 2
        self.stim_graph = np.where(self.stim == 2, self.stim, 0)

        # Actualizar las visualizaciones de los gráficos
        self.update_plots_1()
        self.update_plots_2()

    def update_y_range_1(self):
        """Actualiza el rango del eje Y para el primer gráfico de datos.

        Obtiene los valores mínimo y máximo del eje Y del primer gráfico de datos
        desde los widgets correspondientes y ajusta el rango del eje Y del gráfico."""
        # Obtener los valores mínimo y máximo del eje Y desde los widgets de entrada
        self.min_amplitude_1 = self.y_min_1.value()
        self.max_amplitude_1 = self.y_max_1.value()

        # Establecer el rango del eje Y en el primer gráfico de datos
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)

    def update_y_range_2(self):
        """Actualiza el rango del eje Y para el segundo gráfico de datos.

        Obtiene los valores mínimo y máximo del eje Y del segundo gráfico de datos
        desde los widgets correspondientes y ajusta el rango del eje Y del gráfico."""
        # Obtener los valores mínimo y máximo del eje Y desde los widgets de entrada
        self.min_amplitude_2 = self.y_min_2.value()
        self.max_amplitude_2 = self.y_max_2.value()

        # Establecer el rango del eje Y en el segundo gráfico de datos
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)

    def update_y_range_norm_1(self):
        """Actualiza el rango del eje Y para el primer gráfico de envolvente normalizada.

        Obtiene los valores mínimo y máximo del eje Y del primer gráfico de envolvente
        normalizada desde los widgets correspondientes y ajusta el rango del eje Y del gráfico."""
        # Obtener los valores mínimo y máximo del eje Y desde los widgets de entrada
        self.min_amplitude_norm_1 = self.y_min_norm_1.value()
        self.max_amplitude_norm_1 = self.y_max_norm_1.value()

        # Establecer el rango del eje Y en el primer gráfico de envolvente normalizada
        self.plot_widget_envolvente_1.setYRange(self.min_amplitude_norm_1, self.max_amplitude_norm_1)

    def update_y_range_norm_2(self):
        """Actualiza el rango del eje Y para el segundo gráfico de envolvente normalizada.

        Obtiene los valores mínimo y máximo del eje Y del segundo gráfico de envolvente
        normalizada desde los widgets correspondientes y ajusta el rango del eje Y del gráfico."""
        # Obtener los valores mínimo y máximo del eje Y desde los widgets de entrada
        self.min_amplitude_norm_2 = self.y_min_norm_2.value()
        self.max_amplitude_norm_2 = self.y_max_norm_2.value()

        # Establecer el rango del eje Y en el segundo gráfico de envolvente normalizada
        self.plot_widget_envolvente_2.setYRange(self.min_amplitude_norm_2, self.max_amplitude_norm_2)

    def keyPressEvent(self, event):
        """ Maneja el evento de presión de una tecla.

        Cambia el estado de la variable `self.key_pressed` según la tecla presionada.
        - Si se presiona la tecla 'U', `self.key_pressed` se establece en 'U'.
        - Si se presiona la tecla 'D', `self.key_pressed` se establece en 'D'."""
        if event.key() == Qt.Key.Key_U:
            self.key_pressed = 'U'
        elif event.key() == Qt.Key.Key_D:
            self.key_pressed = 'D'

    def keyReleaseEvent(self, event):
        """Maneja el evento de liberación de una tecla.

        Establece `self.key_pressed` en None cuando se libera cualquier tecla."""
        self.key_pressed = None

    def on_plot_clicked_3(self, event):
        """Maneja el evento de clic en el primer gráfico de envolvente normalizada.

        Dependiendo del botón del ratón y las teclas presionadas, se realizan las siguientes acciones:
        - Si se hace clic con el botón izquierdo y se mantiene presionada la tecla Control:
            - Se ajusta una ventana alrededor del punto clickeado.
            - Se elimina el cruce más cercano en esa ventana (si existe) y se actualizan las líneas de cruce.
        - Si solo se hace clic con el botón izquierdo:
            - Se marca un cruce como ascendente ('U') o descendente ('D') en la posición más cercana al clic, según la tecla presionada.
            - Se actualizan las líneas de cruce y se recalculan los índices de cruce ascendentes y descendentes."""
        modifiers = QApplication.keyboardModifiers()

        if event.button() == Qt.MouseButton.LeftButton and (modifiers & Qt.KeyboardModifier.ControlModifier):
            # Clic izquierdo con la tecla Control presionada
            pos = event.scenePos()
            pos_mapped = self.plot_widget_envolvente_1.plotItem.vb.mapSceneToView(pos)
            index_clicked = np.searchsorted(self.tiempo, pos_mapped.x())

            if 0 <= index_clicked < len(self.tiempo):
                ventana_indices = 334  # Tamaño de ventana en datos, aproximadamente 0.1 segundos
                inicio_ventana = max(0, index_clicked - ventana_indices)
                fin_ventana = min(len(self.tiempo), index_clicked + ventana_indices + 1)

                # Encuentra los índices de cruce ascendentes y descendentes dentro de la ventana
                plot_1_indices = np.where((self.cross_up_1 == 1) | (self.cross_down_1 == 1))[0]
                cross_indices_UP_1 = np.where(self.cross_up_1 == 1)[0]
                cross_indices_DOWN_1 = np.where(self.cross_down_1 == 1)[0]

                indices_ventana_1 = plot_1_indices[
                    (plot_1_indices >= inicio_ventana) & (plot_1_indices < fin_ventana)]

                if len(indices_ventana_1) > 0:
                    # Encuentra el índice más cercano al clic
                    closest_index = min(indices_ventana_1, key=lambda x: abs(x - index_clicked))

                    # Elimina el cruce más cercano (ascendente o descendente)
                    if closest_index in cross_indices_UP_1:
                        self.cross_up_1[closest_index] = 0
                        self.cross_1_line_up.setData(self.tiempo, self.cross_up_1)
                    elif closest_index in cross_indices_DOWN_1:
                        self.cross_down_1[closest_index] = 0
                        self.cross_1_line_down.setData(self.tiempo, self.cross_down_1)

        elif event.button() == Qt.MouseButton.LeftButton:
            # Clic izquierdo sin la tecla Control presionada
            pos = event.scenePos()
            pos_mapped = self.plot_widget_envolvente_1.plotItem.vb.mapSceneToView(pos)
            closest_index = np.searchsorted(self.tiempo, pos_mapped.x())

            if self.key_pressed == 'U':
                # Marcar un cruce ascendente en la posición más cercana al clic
                self.cross_up_1[closest_index] = 1
                self.cross_1_line_up.setData(self.tiempo, self.cross_up_1)
            elif self.key_pressed == 'D':
                # Marcar un cruce descendente en la posición más cercana al clic
                self.cross_down_1[closest_index] = 1
                self.cross_1_line_down.setData(self.tiempo, self.cross_down_1)

        # Actualizar las listas de cruces ascendentes y descendentes
        self.UP_1 = np.where(self.cross_up_1 == 1)[0]
        self.UP_1 = self.tiempo[self.UP_1]
        self.DOWN_1 = np.where(self.cross_down_1 == 1)[0]
        self.DOWN_1 = self.tiempo[self.DOWN_1]

        # Restablecer la tecla presionada
        self.key_pressed = None

    def on_plot_clicked_4(self, event):
        """Maneja el evento de clic en el segundo gráfico de envolvente normalizada.

        Dependiendo del botón del ratón y las teclas presionadas, realiza las siguientes acciones:
        - Si se hace clic con el botón izquierdo y se mantiene presionada la tecla Control:
            - Ajusta una ventana alrededor del punto clickeado.
            - Elimina el cruce más cercano en esa ventana (si existe) y actualiza las líneas de cruce.
        - Si solo se hace clic con el botón izquierdo:
            - Marca un cruce como ascendente ('U') o descendente ('D') en la posición más cercana al clic, según la tecla presionada.
            - Actualiza las líneas de cruce y las listas de cruces ascendentes y descendentes."""
        modifiers = QApplication.keyboardModifiers()

        if event.button() == Qt.MouseButton.LeftButton and (modifiers & Qt.KeyboardModifier.ControlModifier):
            # Clic izquierdo con la tecla Control presionada
            pos = event.scenePos()
            pos_mapped = self.plot_widget_envolvente_2.plotItem.vb.mapSceneToView(pos)
            index_clicked = np.searchsorted(self.tiempo, pos_mapped.x())

            # Definir el tamaño de la ventana en datos, aproximadamente 0.1 segundos
            ventana_indices_2 = 334
            inicio_ventana_2 = max(0, index_clicked - ventana_indices_2)
            fin_ventana_2 = min(len(self.tiempo), index_clicked + ventana_indices_2 + 1)

            # Encuentra los índices de cruce ascendentes y descendentes dentro de la ventana
            plot_2_indices = np.where((self.cross_up_2 == 1) | (self.cross_down_2 == 1))[0]
            cross_indices_UP_2 = np.where(self.cross_up_2 == 1)[0]
            cross_indices_DOWN_2 = np.where(self.cross_down_2 == 1)[0]
            indices_ventana_2 = plot_2_indices[
                (plot_2_indices >= inicio_ventana_2) & (plot_2_indices < fin_ventana_2)]

            if 0 <= index_clicked < len(self.tiempo):
                if len(indices_ventana_2) > 0:
                    # Encuentra el índice más cercano al clic dentro de la ventana
                    closest_index_2 = min(indices_ventana_2, key=lambda y: abs(y - index_clicked))

                    # Elimina el cruce más cercano (ascendente o descendente)
                    if closest_index_2 in cross_indices_UP_2:
                        self.cross_up_2[closest_index_2] = 0
                        self.cross_2_line_up.setData(self.tiempo, self.cross_up_2)
                        self.UP_2 = np.where(self.cross_up_2 == 1)[0]
                        self.UP_2 = self.tiempo[self.UP_2]

                    elif closest_index_2 in cross_indices_DOWN_2:
                        self.cross_down_2[closest_index_2] = 0
                        self.cross_2_line_down.setData(self.tiempo, self.cross_down_2)
                        self.DOWN_2 = np.where(self.cross_down_2 == 1)[0]
                        self.DOWN_2 = self.tiempo[self.DOWN_2]

        elif event.button() == Qt.MouseButton.LeftButton:
            # Clic izquierdo sin la tecla Control presionada
            pos = event.scenePos()
            pos_mapped = self.plot_widget_envolvente_2.plotItem.vb.mapSceneToView(pos)
            closest_index_2 = np.searchsorted(self.tiempo, pos_mapped.x())

            if self.key_pressed == 'U':
                # Marcar un cruce ascendente en la posición más cercana al clic
                self.cross_up_2[closest_index_2] = 1
                self.cross_2_line_up.setData(self.tiempo, self.cross_up_2)
                self.UP_2 = np.where(self.cross_up_2 == 1)[0]
                self.UP_2 = self.tiempo[self.UP_2]

            elif self.key_pressed == 'D':
                # Marcar un cruce descendente en la posición más cercana al clic
                self.cross_down_2[closest_index_2] = 1
                self.cross_2_line_down.setData(self.tiempo, self.cross_down_2)
                self.DOWN_2 = np.where(self.cross_down_2 == 1)[0]
                self.DOWN_2 = self.tiempo[self.DOWN_2]

        # Restablecer la tecla presionada
        self.key_pressed = None

    def update_threshold_1(self, value):
        """Actualiza el umbral para el primer canal y actualiza el gráfico correspondiente.

        :param value: El nuevo valor del umbral para el primer canal."""
        self.threshold_1 = value  # Establece el nuevo umbral para el primer canal.
        self.update_plots_1()  # Actualiza el gráfico del primer canal para reflejar el nuevo umbral.

    def update_threshold_2(self, value):
        """Actualiza el umbral para el segundo canal y actualiza el gráfico correspondiente.

        :param value: El nuevo valor del umbral para el segundo canal."""
        self.threshold_2 = value  # Establece el nuevo umbral para el segundo canal.
        self.update_plots_2()  # Actualiza el gráfico del segundo canal para reflejar el nuevo umbral.

    def update_window_time(self, value):
        """Actualiza el tiempo de ventana para el análisis de datos, detiene el avance y retroceso de las gráficas,
        y activa el modo de ventana.

        :param value: El nuevo tiempo de ventana en segundos."""
        self.window_time = value  # Establece el nuevo tiempo de ventana.
        self.avanza = False  # Detiene el avance de las gráficas.
        self.retrocede = False  # Detiene el retroceso de las gráficas.
        self.window = True  # Activa el modo de ventana.
        self.mover_ventana()  # Ajusta la ventana de visualización en el gráfico.

    def avanzar_graficas(self):
        """Activa el avance de las gráficas, detiene el retroceso y desactiva el modo de ventana.

        Este método se usa para mover las gráficas hacia adelante en el tiempo."""
        self.avanza = True  # Activa el avance de las gráficas.
        self.retrocede = False  # Detiene el retroceso de las gráficas.
        self.window = False  # Desactiva el modo de ventana.
        self.mover_ventana()  # Ajusta la ventana de visualización en el gráfico.

    def retroceder_graficas(self):
        """Activa el retroceso de las gráficas, detiene el avance y desactiva el modo de ventana.

        Este método se usa para mover las gráficas hacia atrás en el tiempo."""
        self.retrocede = True  # Activa el retroceso de las gráficas.
        self.avanza = False  # Detiene el avance de las gráficas.
        self.window = False  # Desactiva el modo de ventana.
        self.mover_ventana()  # Ajusta la ventana de visualización en el gráfico.

    def actualizar_LP_fc_1(self, value):
        """Actualiza la frecuencia de corte del filtro pasa bajo (LPF) para el primer canal
        y activa el filtro.

        :param value: La nueva frecuencia de corte del filtro pasa bajo en Hz."""
        self.cutoff_1_LP = value  # Actualiza la frecuencia de corte del filtro pasa bajo para el primer canal.
        self.filter_1_LP = True  # Activa el filtro pasa bajo para el primer canal.
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro.

    def actualizar_LP_fc_2(self, value):
        """Actualiza la frecuencia de corte del filtro pasa bajo (LPF) para el segundo canal
        y activa el filtro.

        :param value: La nueva frecuencia de corte del filtro pasa bajo en Hz."""
        self.cutoff_2_LP = value  # Actualiza la frecuencia de corte del filtro pasa bajo para el segundo canal.
        self.filter_2_LP = True  # Activa el filtro pasa bajo para el segundo canal.
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro.

    def actualizar_HP_fc_1(self, value):
        """Actualiza la frecuencia de corte del filtro pasa alto (HPF) para el primer canal.

        :param value: La nueva frecuencia de corte del filtro pasa alto en Hz."""
        self.cutoff_1_HP = value  # Actualiza la frecuencia de corte del filtro pasa alto para el primer canal.

    def actualizar_HP_fc_2(self, value):
        """Actualiza la frecuencia de corte del filtro pasa alto (HPF) para el segundo canal.

        :param value: La nueva frecuencia de corte del filtro pasa alto en Hz."""
        self.cutoff_2_HP = value  # Actualiza la frecuencia de corte del filtro pasa alto para el segundo canal.

    def update_filter_1(self):
        """Activa el filtro pasa alto (HPF) para el primer canal y actualiza el gráfico.

        Este método marca el filtro pasa alto del primer canal como activo y actualiza el gráfico
        para reflejar los cambios en el filtro."""
        self.filter_1_HP = True  # Activa el filtro pasa alto para el primer canal.
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro.

    def update_filter_2(self):
        """Activa el filtro pasa alto (HPF) para el segundo canal y actualiza el gráfico.

        Este método marca el filtro pasa alto del segundo canal como activo y actualiza el gráfico
        para reflejar los cambios en el filtro."""
        self.filter_2_HP = True  # Activa el filtro pasa alto para el segundo canal.
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro.

    def actualizar_hold_off_1(self, value):
        """Actualiza el valor del hold off para el primer canal.

        :param value: El nuevo valor del hold off en milisegundos, que se convertirá a segundos."""
        self.hold_off_1 = value / 1000  # Convierte el valor de milisegundos a segundos.

    def actualizar_hold_off_2(self, value):
        """Actualiza el valor del hold off para el segundo canal.

        :param value: El nuevo valor del hold off en milisegundos, que se convertirá a segundos."""
        self.hold_off_2 = value / 1000  # Convierte el valor de milisegundos a segundos.

    def actualizar_hyst_asc_1(self, value):
        """Actualiza el valor de histéresis ascendente para el primer canal.

        :param value: El nuevo valor de histéresis ascendente."""
        self.hyst_asc_1 = value  # Establece el nuevo valor de histéresis ascendente.

    def actualizar_hyst_asc_2(self, value):
        """Actualiza el valor de histéresis ascendente para el segundo canal.

        :param value: El nuevo valor de histéresis ascendente."""
        self.hyst_asc_2 = value  # Establece el nuevo valor de histéresis ascendente.

    def actualizar_hyst_desc_1(self, value):
        """Actualiza el valor de histéresis descendente para el primer canal.

        :param value: El nuevo valor de histéresis descendente."""
        self.hyst_desc_1 = value  # Establece el nuevo valor de histéresis descendente.

    def actualizar_hyst_desc_2(self, value):
        """Actualiza el valor de histéresis descendente para el segundo canal.

        :param value: El nuevo valor de histéresis descendente."""
        self.hyst_desc_2 = value  # Establece el nuevo valor de histéresis descendente.

    def prueba_rayleigh(self, r, n, alpha):
        """Prueba de Rayleigh para uniformidad de datos circulares.

        Parámetros:
        r: valor R calculado a partir de las direcciones angulares.
        n: número de observaciones.
        alpha: nivel de significancia (por defecto 0.05).

        Devuelve:
        estadistico: valor del estadístico de Rayleigh.
        valor_critico: valor crítico para el nivel de significancia especificado.
        significativo: booleano indicando si el acoplamiento es significativo."""
        # Calcula el estadístico de Rayleigh
        estadistico = 2 * n * (r ** 2)

        # Obtiene el valor crítico para el nivel de significancia especificado usando la
        # función de percentil inversa (ppf) de la distribución de Rayleigh
        valor_critico = rayleigh.ppf(1 - alpha)

        # Determina si el estadístico es significativo comparándolo con el valor crítico
        significativo = estadistico > valor_critico

        return estadistico, valor_critico, significativo

    def cycle_duration_analysis(self):

        if not self.cross_up_1.any() or not self.cross_up_2.any():
            print("Asegúrate de haber detectado los cruces ascendentes en ambas señales.")
            return

        if self.save_function_flag:
            self.save_and_insert_data(f"{self.new_file_name}_CYCLE_DURATION_DATA")

        self.new_stim = self.stims_options.currentText()
        self.stim = self.data[self.new_stim].to_numpy()
        start_index = int(self.start_a * self.datos_por_s)
        end_index = int(self.end_a * self.datos_por_s)

        self.stim_a = self.stim.copy()
        self.stim_a[:start_index] = 0
        self.stim_a[end_index:] = 0
        self.tiempo_a = self.tiempo.copy()
        self.tiempo_a[:start_index] = 0
        self.tiempo_a[end_index:] = 0
        self.cross_up_1_a = self.cross_up_1.copy()
        self.cross_up_1_a[:start_index] = 0
        self.cross_up_1_a[end_index:] = 0
        self.cross_down_1_a = self.cross_down_1.copy()
        self.cross_down_1_a[:start_index] = 0
        self.cross_down_1_a[end_index:] = 0
        self.cross_up_2_a = self.cross_up_2.copy()
        self.cross_up_2_a[:start_index] = 0
        self.cross_up_2_a[end_index:] = 0
        self.cross_down_2_a = self.cross_down_2.copy()
        self.cross_down_2_a[:start_index] = 0
        self.cross_down_2_a[end_index:] = 0

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        P_1 = []
        TAG_1 = []
        STIM_1 = []
        index_tag_1 = []
        index_stim_1 = []

        indices_cruces_1 = np.where(self.cross_up_1_a == 1)[0]
        self.UP_1_a = self.tiempo_a[indices_cruces_1]
        indices_cruces_2 = np.where(self.cross_up_2_a == 1)[0]
        self.UP_2_a = self.tiempo_a[indices_cruces_2]

        if self.UP_1_a[0] < self.UP_2_a[0]:
            a = 0
            b = 1
        elif self.UP_1_a[0] > self.UP_2_a[0]:
            a = 1
            b = 0
        else:
            a = 0
            b = 0

        for i in range(len(self.UP_1_a) - 1):
            P_i = self.UP_1_a[i + 1] - self.UP_1_a[i]
            P_1.append(P_i)
            stim_found = False  # Variable booleana para verificar si se encontró un estímulo

            for j in range(indices_cruces_1[i], indices_cruces_1[i + 1]):
                if self.stim_a[j] == 2 and not stim_found:
                    STIM_1.append(P_i)
                    index_stim_1.append(a + 1)
                    stim_found = True
            # print("VALOR DE j", j)

            if not stim_found:  # Solo agrega a TAG_1 si no se encontró un estímulo
                TAG_1.append(P_i)
                index_tag_1.append(a + 1)
            a += 1

        if index_tag_1 or index_stim_1:
            # Si al menos una de las listas no está vacía, combinar las listas
            index_combined_c1 = index_tag_1 + index_stim_1
            # Combinar los valores de TAG y STIM
            combined_values_c1 = TAG_1 + STIM_1
            # Ordenar los índices combinados y los valores combinados
            index_combined_sorted_c1, combined_values_sorted_c1 = zip(
                *sorted(zip(index_combined_c1, combined_values_c1)))
        else:
            # Si ambas listas están vacías, asignar listas vacías como valores predeterminados
            index_combined_sorted_c1, combined_values_sorted_c1 = [], []

        P_2 = []
        TAG_2 = []
        STIM_2 = []
        index_stim_2 = []
        index_tag_2 = []

        for i in range(len(indices_cruces_2) - 1):
            P_i = self.UP_2_a[i + 1] - self.UP_2_a[i]
            P_2.append(P_i)

            stim_2_found = False  # Variable booleana para verificar si se encontró un estímulo

            for k in range(indices_cruces_2[i], indices_cruces_2[i + 1]):
                if self.stim_a[k] == 2 and not stim_2_found:
                    STIM_2.append(P_i)
                    index_stim_2.append(b + 1)
                    stim_2_found = True

            if not stim_2_found:  # Solo agrega a TAG_1 si no se encontró un estímulo
                TAG_2.append(P_i)
                index_tag_2.append(b + 1)

            b += 1

        if index_tag_2 or index_stim_2:
            # Si al menos una de las listas no está vacía, combinar las listas
            index_combined_c2 = index_tag_2 + index_stim_2
            # Combinar los valores de TAG y STIM
            combined_vales_c2 = TAG_2 + STIM_2
            # Ordenar los índices combinados y los valores combinados
            index_combined_sorted_c2, combined_values_sorted_c2 = zip(
                *sorted(zip(index_combined_c2, combined_vales_c2)))
        else:
            # Si ambas listas están vacías, asignar listas vacías como valores predeterminados
            index_combined_sorted_c2, combined_values_sorted_c2 = [], []

            # Trazar la línea que pasa por los puntos de TAG y STIM combinados
        axs[0].plot(index_combined_sorted_c1, combined_values_sorted_c1, linestyle='-', color=self.color_map[self.s_1])
        axs[0].plot(index_tag_1, TAG_1, marker='o', color=self.color_map[self.s_1], linestyle='none',
                    label=f'CONTROL {self.s_1}; S={len(TAG_1)}')
        axs[0].plot(index_stim_1, STIM_1, marker='*', color=self.stim_map[self.new_stim], linestyle='none',
                    label=f'{self.new_stim}; S={len(STIM_1)}')

        # Trazar la línea que pasa por los puntos de TAG2 y STIM2 combinados
        axs[0].plot(index_combined_sorted_c2, combined_values_sorted_c2, linestyle='-', color=self.color_map[self.s_2])
        axs[0].plot(index_tag_2, TAG_2, marker='o', color=self.color_map[self.s_2], linestyle='none',
                    label=f'CONTROL {self.s_2}; S={len(TAG_2)}')
        axs[0].plot(index_stim_2, STIM_2, marker='v', color=self.stim_map[self.new_stim], linestyle='none',
                    label=f'{self.new_stim}; S={len(STIM_2)}')


        # _____________
        Phase_1 = []
        TAG_p1 = []
        index_tag_p1 = []
        STIM_p1 = []
        index_stim_p1 = []

        indices_cruces_d = np.where(self.cross_down_1_a == 1)[0]

        if indices_cruces_d[0] < indices_cruces_1[0]:
            indices_cruces_d = np.delete(indices_cruces_d, 0)
        else:
            pass

        self.DOWN_1_a = self.tiempo_a[indices_cruces_d]

        min_phase_1 = min(len(indices_cruces_1), len(indices_cruces_d))

        if self.UP_1_a[0] < self.UP_2_a[0]:
            c = 0
            d = 1
        elif self.UP_1_a[0] > self.UP_2_a[0]:
            c = 1
            d = 0
        else:
            c = 0
            d = 0

        for i in range(min_phase_1):
            P_i = self.DOWN_1_a[i] - self.UP_1_a[i]
            Phase_1.append(P_i)
            stim_phase_found = False  # Variable booleana para verificar si se encontró un estímulo

            for o in range((indices_cruces_1[i]), indices_cruces_d[i]):
                if self.stim_a[o] == 2 and not stim_phase_found:
                    STIM_p1.append(P_i)
                    index_stim_p1.append(c + 1)
                    stim_phase_found = True

            if not stim_phase_found:  # Solo agrega a TAG_1 si no se encontró un estímulo
                TAG_p1.append(P_i)
                index_tag_p1.append(c + 1)

            c += 1

        if index_tag_p1 or index_stim_p1:
            # Si al menos una de las listas no está vacía, combinar las listas
            index_combined_p1 = index_tag_p1 + index_stim_p1
            combined_values_p1 = TAG_p1 + STIM_p1
            # Ordenar los índices combinados y los valores combinados
            index_combined_sorted_p1, combined_values_sorted_p1 = zip(
                *sorted(zip(index_combined_p1, combined_values_p1)))
        else:
            # Si ambas listas están vacías, asignar listas vacías como valores predeterminados
            index_combined_sorted_p1, combined_values_sorted_p1 = [], []

        Phase_2 = []
        TAG_p2 = []
        index_tag_p2 = []
        STIM_p2 = []
        index_stim_p2 = []

        indices_cruces_d_2 = np.where(self.cross_down_2_a == 1)[0]

        if indices_cruces_d_2[0] < indices_cruces_2[0]:
            indices_cruces_d_2 = np.delete(indices_cruces_d_2, 0)
        else:
            pass

        self.DOWN_2_a = self.tiempo_a[indices_cruces_d_2]

        min_phase_2 = min(len(indices_cruces_2), len(indices_cruces_d_2))
        # print("INDICES STIM P1", index_stim_p1)
        for i in range(min_phase_2):
            # Calcular la duración del período P_i
            P_i_2 = self.DOWN_2_a[i] - self.UP_2_a[i]
            Phase_2.append(P_i_2)
            stim_phase_found = False  # Variable booleana para verificar si se encontró un estímulo

            for m in range(len(index_stim_p1)):
                # print("INDEX CONDITION", index_stim_p1[m], "d+1", d+1, "m", m)
                if d + 1 == index_stim_p1[m]:
                    # print("ESTIM P1", i, d+1)
                    STIM_p2.append(P_i_2)
                    index_stim_p2.append(d + 1)
                    stim_phase_found = True

            if not stim_phase_found:  # Solo agrega a TAG_1 si no se encontró un estímulo
                TAG_p2.append(P_i_2)
                index_tag_p2.append(d + 1)
            d += 1

        if index_tag_p2 or index_stim_p2:
            # Si al menos una de las listas no está vacía, combinar las listas
            index_combined_p2 = index_tag_p2 + index_stim_p2
            combined_values_p2 = TAG_p2 + STIM_p2
            # Ordenar los índices combinados y los valores combinados
            index_combined_sorted_p2, combined_values_sorted_p2 = zip(
                *sorted(zip(index_combined_p2, combined_values_p2)))
        else:
            # Si ambas listas están vacías, asignar listas vacías como valores predeterminados
            index_combined_sorted_p2, combined_values_sorted_p2 = [], []

        # Trazar la línea que pasa por los puntos de TAG y STIM combinados
        axs[1].plot(index_combined_sorted_p2, combined_values_sorted_p2, linestyle='-', color=self.color_map[self.s_2])
        axs[1].plot(index_tag_p2, TAG_p2, marker='s', color=self.color_map[self.s_2], linestyle='none',
                    label=f'CONTROL {self.s_2}; S={len(TAG_p2)}')
        axs[1].plot(index_stim_p2, STIM_p2, marker='v', color=self.stim_map[self.new_stim], linestyle='none',
                    label=f'{self.new_stim}; S={len(STIM_p2)}')

        # Trazar la línea que pasa por los puntos de TAG y STIM combinados
        axs[1].plot(index_combined_sorted_p1, combined_values_sorted_p1, linestyle='-', color=self.color_map[self.s_1])
        axs[1].plot(index_tag_p1, TAG_p1, marker='s', color=self.color_map[self.s_1], linestyle='none',
                    label=f'CONTROL {self.s_1}; S={len(TAG_p1)}')
        axs[1].plot(index_stim_p1, STIM_p1, marker='*', color=self.stim_map[self.new_stim], linestyle='none',
                    label=f'{self.new_stim}; S={len(STIM_p1)}')

        fig.patch.set_facecolor('none')
        for ax in axs.flatten():
            ax.set_facecolor('none')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(frameon=False)

        axs[0].set_xlabel('Events')  # Etiqueta del eje x
        axs[0].set_ylabel('Duration [s]')  # Etiqueta del eje y
        axs[0].set_title('CYCLE DURATION', color='gray')  # Título del gráfico
        axs[1].set_xlabel('Events')  # Etiqueta del eje x
        axs[1].set_ylabel('Duration [s]')  # Etiqueta del eje y
        axs[1].set_title('ACTIVE PHASE DURATION', color='gray')  # Título del gráfico

        axs[0].set_ylim((min((*P_1, *P_2))) - 0.5, (max((*P_1, *P_2))) + 0.5)
        axs[1].set_ylim((min((*Phase_1, *Phase_2))) - 0.5, (max((*Phase_1, *Phase_2))) + 0.5)

        # Guardar la figura de las duraciones
        self.save_fig(f"{self.new_file_name}_DURATION_PHASES_ANALYSIS")

        fig.patch.set_facecolor('black')

        # Configurar el color de los ejes y las etiquetas en gris
        for ax in axs.flatten():
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('gray')  # Color gris para el eje x
            ax.spines['left'].set_color('gray')  # Color gris para el eje y
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(frameon=False, labelcolor='gray')

        # _______POLAR

        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        # Eliminar el box que rodea al gráfico
        ax.spines['polar'].set_visible(False)
        plt.ylim(0, 1.1)
        alpha = 0.05  # Valor de referencia para comparar el coeficiente de Rayleigh

        min_len = min(len(self.UP_1_a), len(self.UP_2_a), len(P_1))
        L_tag = []
        L_stim = []
        P_tag = []
        P_stim = []


        for i in range(min_len):
            L_i = round(self.UP_2_a[i] - self.UP_1_a[i], 4)
            P_i = P_1[i]
            stim_found_p = False

            if i < min_len - 1:
                for j in range(indices_cruces_1[i], indices_cruces_1[i + 1]):
                    if self.stim_a[j] == 2 and not stim_found_p:
                        # print(f"{j} VALOR DE STIM", self.stim_a[j], "valor de i", i)
                        L_stim.append(L_i)
                        P_stim.append(P_i)

                        stim_found_p = True

            if not stim_found_p:
                L_tag.append(L_i)
                P_tag.append(P_i)

        tan_tag = []
        X_tag = []
        Y_tag = []
        phis_tag = []
        min_len_L_tag = min(len(L_tag), len(P_tag))

        for i in range(min_len_L_tag):
            angle_tag = (L_tag[i] / P_tag[i]) * 2 * np.pi
            x = math.cos(angle_tag)
            y = math.sin(angle_tag)
            phis_tag.append(angle_tag)
            tan_2_tag = math.atan2(y, x)  # atan2 da el ángulo correcto en radianes
            tan_tag.append(tan_2_tag)
            X_tag.append(x)
            Y_tag.append(y)

        n_tag = len(tan_tag)
        ax.plot(tan_tag, np.ones(n_tag), marker='o', linestyle='none', color=self.filtered_color,
                label=f'CONTROL; S={n_tag}')

        X_prom_tag = np.mean([np.cos(phi) for phi in phis_tag])
        Y_prom_tag = np.mean([np.sin(phi) for phi in phis_tag])
        phi_vector_tag = math.atan2(Y_prom_tag, X_prom_tag)  # Esto sigue en radianes
        r_tag = math.sqrt((X_prom_tag ** 2) + (Y_prom_tag ** 2))

        # Vector de fase promedio para TAG
        plt.quiver(0, 0, phi_vector_tag, r_tag, angles='xy', scale_units='xy', scale=1, color='#FF1493', width=0.005,
                   headwidth=4, label='CONTROL')

        tan_stim = []
        X_stim = []
        Y_stim = []
        phis_stim = []
        min_len_L_stim = min(len(L_stim), len(P_stim))

        for i in range(min_len_L_stim):
            angle_stim = (L_stim[i] / P_stim[i]) * 2 * np.pi
            x = math.cos(angle_stim)
            y = math.sin(angle_stim)
            phis_stim.append(angle_stim)
            tan_2_stim = math.atan2(y, x)
            tan_stim.append(tan_2_stim)
            X_stim.append(x)
            Y_stim.append(y)

        n_stim = len(tan_stim)
        # print("VECTOR DE ANGULOS STIM", tan_stim, "RAD", tan_stim_radians)
        ax.plot(tan_stim, np.ones(n_stim), marker='*', linestyle='none', color=self.stim_map[self.new_stim],
                label=f'{self.new_stim}; S={n_stim}')

        X_prom_stim = np.mean([np.cos(phi) for phi in phis_stim])
        Y_prom_stim = np.mean([np.sin(phi) for phi in phis_stim])
        phi_vector_stim = math.atan2(Y_prom_stim, X_prom_stim)
        r_stim = math.sqrt((X_prom_stim ** 2) + (Y_prom_stim ** 2))

        # Vector de fase promedio para STIM
        plt.quiver(0, 0, phi_vector_stim, r_stim, angles='xy', scale_units='xy', scale=1, color='yellow',
                   width=0.005, headwidth=4, label=f'{self.new_stim}')

        estadistico_rayleigh_tag, valor_critico_tag, significativo_tag = self.prueba_rayleigh(r_tag, n_tag, alpha)

        estadistico_rayleigh_stim, valor_critico_stim, significativo_stim = self.prueba_rayleigh(r_stim, n_stim, alpha)

        ax.set_title(f'PHASE ANALYSIS BETWEEN {self.s_1}-{self.s_2}; S = {n_tag + n_stim}',
                     color='gray')  # Título del gráfico

        # Añadir el valor crítico de significancia al gráfico como una leyenda en la parte inferior
        text_color = 'gray'
        if significativo_tag:
            ax.text(0.5, -0.05,
                    f'Coupling of control values IS significant. Rayleigh statistical value: {estadistico_rayleigh_tag}. Critical value: {valor_critico_tag}. Alpha: {alpha}',
                    transform=ax.transAxes, fontsize=12, color=text_color,
                    ha='center', va='center')
        else:
            ax.text(0.5, -0.05,
                    f'Coupling of control values is NOT significant. Rayleigh statistical value: {estadistico_rayleigh_tag}. Critical value: {valor_critico_tag}. Alpha: {alpha}',
                    transform=ax.transAxes, fontsize=12, color=text_color, ha='center', va='center')

        if significativo_stim:
            ax.text(0.5, -0.1,
                    f'Coupling of stimulus values IS significant. Rayleigh statistical value: {estadistico_rayleigh_stim}. Critical value: {valor_critico_stim}. Alpha: {alpha}',
                    transform=ax.transAxes, fontsize=12, color=text_color,
                    ha='center', va='center')
        else:
            ax.text(0.5, -0.1,
                    f'Coupling of stimulus values is NOT significant. Rayleigh statistical value: {estadistico_rayleigh_stim}. Critical value: {valor_critico_stim}. Alpha: {alpha}',
                    transform=ax.transAxes, fontsize=12, color=text_color, ha='center', va='center')

        # Guardar la figura como SVG con diferentes colores de fondo
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.grid(color='black', linestyle='-')  # Cambiar el color del grid a gris
        # ax.grid('gray')
        ax.legend(frameon=False, labelcolor='black')

        # Guardar figura análisis de fases
        self.save_fig(f"{self.new_file_name}_PHASES_ANALYSIS")

        ax.set_facecolor('black')  # Establecer el color de fondo de los ejes como transparente
        fig.patch.set_facecolor('black')
        ax.grid(color='gray', linestyle='-')  # Cambiar el color del grid a gris
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        ax.legend(frameon=False, labelcolor='gray')

        # Generar la gráfica lineal------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))  # Crear figura y ejes
        # Eliminar el cuadro que rodea al gráfico
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Establecer límites en el eje x
        ax.set_xlim(0, min_len)
        ax.set_xlabel('Number of events')
        ax.set_ylabel('Time [s]')

        ax.plot(np.arange(len(P_1)), P_1, marker='o', linestyle='-', color=self.color_map[self.s_1], markersize=8,
                label=f'{self.s_1}')
        ax.plot(np.arange(len(P_2)), P_2, marker='o', linestyle='-', color=self.color_map[self.s_2], markersize=8,
                label=f'{self.s_2}')

        ax.set_title(f'PHASES ANALYSIS; n={n_tag + n_stim}', color='gray')  # Título del gráfico

        # Guardar la figura como SVG con diferentes colores de fondo
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.legend(frameon=False, labelcolor='black')

        # Guardar la figura del analisis lineal
        self.save_fig(f"{self.new_file_name}_PHASES_LINEAL_ANALYSIS")

        # Cambiar el color de los ejes a gris
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Cambiar el color de los ticks de los ejes a gris
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')

        # Cambiar el color del fondo de los ejes a negro
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.legend(frameon=False, labelcolor='gray')

        # Cambiar el color del fondo de los ejes a negro
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.legend(frameon=False, labelcolor='gray')

        # Generar la gráfica del registro------------------------------------------
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))

        axs[0].plot(self.tiempo[self.start_a_num: self.end_a_num], self.s_1_norm[self.start_a_num: self.end_a_num],
                    label=f'{self.s_1}/LP={self.cutoff_1_LP}Hz/HP={self.cutoff_1_HP}/HO={self.hold_off_1}ms', color=self.color_map[self.s_1])
        axs[0].plot(self.tiempo[self.start_a_num: self.end_a_num], self.cross_up_1[self.start_a_num: self.end_a_num],
                    color=self.up_color, label=f'Hyst.Asc.{self.hyst_asc_1}')
        axs[0].plot(self.tiempo[self.start_a_num: self.end_a_num], self.cross_down_1[self.start_a_num: self.end_a_num],
                    label=f'Hyst.Desc.{self.hyst_desc_1}', color='gray', linestyle='--')
        axs[0].plot(self.tiempo[self.start_a_num: self.end_a_num], self.stim_graph[self.start_a_num: self.end_a_num],
                    label=f'{self.new_stim}', color=self.stim_map[self.new_stim], linestyle='--')
        axs[0].plot(self.tiempo[self.start_a_num: self.end_a_num], self.tag_graph[self.start_a_num: self.end_a_num],
                    label='Tag', color=self.stim_map[self.new_stim])
        axs[0].axhline(y=self.threshold_norm_1, label='Threshold', color=self.threshold_color)

        axs[1].plot(self.tiempo[self.start_a_num: self.end_a_num], self.s_2_norm[self.start_a_num: self.end_a_num],
                    label=f'{self.s_2}/LP={self.cutoff_2_LP}Hz/HP={self.cutoff_2_HP}/HO={self.hold_off_2}ms', color=self.color_map[self.s_2])
        axs[1].plot(self.tiempo[self.start_a_num: self.end_a_num], self.cross_up_2[self.start_a_num: self.end_a_num],
                    color=self.up_color, label=f'Hyst.Asc.{self.hyst_asc_2}')
        axs[1].plot(self.tiempo[self.start_a_num: self.end_a_num], self.cross_down_2[self.start_a_num: self.end_a_num],
                    label=f'Hyst.Desc.{self.hyst_desc_2}', color='gray', linestyle='--')
        axs[1].plot(self.tiempo[self.start_a_num: self.end_a_num], self.stim_graph[self.start_a_num: self.end_a_num],
                    label=f'{self.new_stim}', color=self.stim_map[self.new_stim], linestyle='--')
        axs[1].plot(self.tiempo[self.start_a_num: self.end_a_num], self.tag_graph[self.start_a_num: self.end_a_num],
                    label='Tag', color=self.stim_map[self.new_stim])
        axs[1].axhline(y=self.threshold_norm_2, label='Threshold', color=self.threshold_color)

        fig.patch.set_facecolor('none')

        for ax in axs.flatten():
            ax.set_facecolor('none')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(frameon=False, labelcolor='gray')

        axs[0].set_facecolor('none')  # Establecer el color de fondo de los ejes como transparente
        axs[1].set_facecolor('none')  # Establecer el color de fondo de los ejes como transparente
        axs[0].set_xlabel('Time [s]', color='gray')  # Etiqueta del eje x
        axs[0].set_ylabel('Amplitude [V]', color='gray')  # Etiqueta del eje y
        axs[0].set_title(f'SIGNAL {self.s_1}', color='gray')  # Título del gráfico
        axs[1].set_xlabel('Time [s]', color='gray')  # Etiqueta del eje x
        axs[1].set_ylabel('Amplitude [V]', color='gray')  # Etiqueta del eje y
        axs[1].set_title(f'SIGNAL {self.s_2}', color='gray')  # Título del gráfico

        axs[0].set_ylim(self.min_amplitude_norm_1, self.max_amplitude_norm_1)
        axs[1].set_ylim(self.min_amplitude_norm_2, self.max_amplitude_norm_2)

        # Guardar figura del registro
        self.save_fig(f"{self.new_file_name}_SIGNALS_PHASES_ANALYSIS")

        fig.patch.set_facecolor('black')
        # Configurar el color de los ejes y las etiquetas en gris
        for ax in axs.flatten():
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('gray')  # Color gris para el eje x
            ax.spines['left'].set_color('gray')  # Color gris para el eje y
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(frameon=False, labelcolor='gray')

        plt.show()  # Mostrar la figura

    def save_and_insert_data(self, name):
        """Guarda los datos de la sesión en un archivo CSV y añade columnas adicionales para el análisis de ciclos
        activos e inactivos, tiempos de ciclo y frecuencias para dos señales."""

        # Crear un nombre de archivo único con timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nombre_archivo_data = f"{name}_{timestamp}.csv"
        self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para uso futuro

        # Ruta completa del archivo CSV
        file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)

        # Preparar los datos para guardar, incluyendo señales y cruces
        data = self.data_copy[['TIME', 'CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                               'CH 10', 'CH 11', 'CH 12', 'TAG OUT']].to_numpy()
        up_1 = self.cross_up_1
        down_1 = self.cross_down_1
        up_2 = self.cross_up_2
        down_2 = self.cross_down_2
        stim_1 = self.data['STIM 1']
        stim_2 = self.data['STIM 2']
        stim_3 = self.data['STIM 3']

        # Definir las etiquetas de las columnas en el archivo CSV
        column_labels = ['TIME', 'CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                         'CH 10', 'CH 11', 'CH 12', 'TAG OUT', f'UP {self.s_1}', f'DOWN {self.s_1}',
                         f'UP {self.s_2}', f'DOWN {self.s_2}', 'STIM 1', 'STIM 2', 'STIM 3']

        # Combinar los datos en una matriz para guardar
        save_array = np.column_stack((data, up_1, down_1, up_2, down_2, stim_1, stim_2, stim_3))

        # Guardar los datos en el archivo CSV
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_labels)  # Escribir las etiquetas de columna
            for row in save_array:
                writer.writerow(row)

        # Leer el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(file_path)
        num_columns = len(df.columns)  # Obtener el número actual de columnas en el DataFrame

        # CALCULAR CICLO ACTIVO
        duracion_ciclo = []
        for index, row in df.iterrows():
            if row[f'DOWN {self.s_1}'] == 1:
                index_down = index
                index_up = df.loc[:index_down, f'UP {self.s_1}'][
                           ::-1].idxmax()  # Encuentra el índice más cercano en UP con valor 1
                duracion = row['TIME'] - df.at[index_up, 'TIME']  # Calcula la duración del ciclo
                duracion_ciclo.append(duracion)
            else:
                duracion_ciclo.append(None)  # Inserta None si DOWN no es 1
        df.insert(num_columns, f'ACTIVE CYCLE {self.s_1}', duracion_ciclo)

        # CALCULAR CICLO INACTIVO
        inactive_cycle = []
        for index, row in df.iterrows():
            if row[f'UP {self.s_1}'] == 1:
                index_up = index
                index_down = df.loc[:index_up, f'DOWN {self.s_1}'][
                             ::-1].idxmax()  # Encuentra el índice más cercano en DOWN con valor 1
                delta = df.at[index_up, 'TIME'] - df.at[index_down, 'TIME']  # Calcula la duración del ciclo inactivo
                inactive_cycle.append(delta)
            else:
                inactive_cycle.append(None)  # Inserta None si UP no es 1
        df.insert(num_columns + 1, f'INACTIVE CYCLE {self.s_1}', inactive_cycle)

        # CALCULAR TIEMPO ENTRE CICLOS
        delta_cycle = [None] * len(df)
        up_indices = df[df[f'UP {self.s_1}'] == 1].index
        for i in range(len(up_indices) - 1):
            time_diff = df.at[up_indices[i + 1], 'TIME'] - df.at[up_indices[i], 'TIME']
            delta_cycle[up_indices[i]] = time_diff
        df.insert(num_columns + 2, f'CYCLE TIME {self.s_1}', delta_cycle)

        # CALCULAR FRECUENCIA
        frequency_1 = 1 / df[f'CYCLE TIME {self.s_1}'].mean() if not df[
            f'CYCLE TIME {self.s_1}'].isnull().all() else None
        df.insert(num_columns + 3, f'FREQUENCY {self.s_1}', None)  # Insertar columna de frecuencia vacía
        df.at[0, f'FREQUENCY {self.s_1}'] = frequency_1  # Asignar la frecuencia calculada a la primera fila

        # CICLO ACTIVO PARA LA SEGUNDA SEÑAL
        duracion_ciclo_2 = []
        for index, row in df.iterrows():
            if row[f'DOWN {self.s_2}'] == 1:
                index_down = index
                index_up = df.loc[:index_down, f'UP {self.s_2}'][
                           ::-1].idxmax()  # Encuentra el índice más cercano en UP con valor 1
                duracion = row['TIME'] - df.at[index_up, 'TIME']  # Calcula la duración del ciclo
                duracion_ciclo_2.append(duracion)
            else:
                duracion_ciclo_2.append(None)  # Inserta None si DOWN no es 1
        df.insert(num_columns + 4, f'ACTIVE CYCLE {self.s_2}', duracion_ciclo_2)

        # CICLO INACTIVO PARA LA SEGUNDA SEÑAL
        inactive_cycle_2 = []
        for index, row in df.iterrows():
            if row[f'UP {self.s_2}'] == 1:
                index_up = index
                index_down = df.loc[:index_up, f'DOWN {self.s_2}'][
                             ::-1].idxmax()  # Encuentra el índice más cercano en DOWN con valor 1
                delta = df.at[index_up, 'TIME'] - df.at[index_down, 'TIME']  # Calcula la duración del ciclo inactivo
                inactive_cycle_2.append(delta)
            else:
                inactive_cycle_2.append(None)  # Inserta None si UP no es 1
        df.insert(num_columns + 5, f'INACTIVE CYCLE {self.s_2}', inactive_cycle_2)

        # CALCULAR TIEMPO ENTRE CICLOS PARA LA SEGUNDA SEÑAL
        delta_cycle_2 = [None] * len(df)
        up_indices_2 = df[df[f'UP {self.s_2}'] == 1].index
        for i in range(len(up_indices_2) - 1):
            time_diff = df.at[up_indices_2[i + 1], 'TIME'] - df.at[up_indices_2[i], 'TIME']
            delta_cycle_2[up_indices_2[i]] = time_diff
        df.insert(num_columns + 6, f'CYCLE TIME {self.s_2}', delta_cycle_2)

        # CALCULAR FRECUENCIA PARA LA SEGUNDA SEÑAL
        frequency_2 = 1 / df[f'CYCLE TIME {self.s_2}'].mean() if not df[
            f'CYCLE TIME {self.s_2}'].isnull().all() else None
        df.insert(num_columns + 7, f'FREQUENCY {self.s_2}', None)  # Insertar columna de frecuencia vacía
        df.at[0, f'FREQUENCY {self.s_2}'] = frequency_2  # Asignar la frecuencia calculada a la primera fila

        # AGREGAR DATOS DE CONFIGURACIÓN PARA LA PRIMERA SEÑAL
        df.insert(num_columns + 8, f'DATA {self.s_1}', None)  # Insertar columna para datos de configuración
        df.at[0, f'DATA {self.s_1}'] = "Threshold"
        df.at[1, f'DATA {self.s_1}'] = "Hold off"
        df.at[2, f'DATA {self.s_1}'] = "cut off LP"
        df.at[3, f'DATA {self.s_1}'] = "Hysteresis asc"
        df.at[4, f'DATA {self.s_1}'] = "Hysteresis desc"
        df.at[5, f'DATA {self.s_1}'] = self.new_stim

        df.insert(num_columns + 9, f'VALUES {self.s_1}', None)  # Insertar columna para valores de configuración
        df.at[0, f'VALUES {self.s_1}'] = self.threshold_1
        df.at[1, f'VALUES {self.s_1}'] = self.hold_off_1
        df.at[2, f'VALUES {self.s_1}'] = self.cutoff_1_LP
        df.at[3, f'VALUES {self.s_1}'] = self.hyst_asc_1
        df.at[4, f'VALUES {self.s_1}'] = self.hyst_desc_1

        # AGREGAR DATOS DE CONFIGURACIÓN PARA LA SEGUNDA SEÑAL
        df.insert(num_columns + 10, f'DATA {self.s_2}', None)  # Insertar columna para datos de configuración
        df.at[0, f'DATA {self.s_2}'] = "Threshold"
        df.at[1, f'DATA {self.s_2}'] = "Hold off"
        df.at[2, f'DATA {self.s_2}'] = "cut off LP"
        df.at[3, f'DATA {self.s_2}'] = "Hysteresis asc"
        df.at[4, f'DATA {self.s_2}'] = "Hysteresis desc"
        df.at[5, f'DATA {self.s_2}'] = self.new_stim

        df.insert(num_columns + 11, f'VALUES {self.s_2}', None)  # Insertar columna para valores de configuración
        df.at[0, f'VALUES {self.s_2}'] = self.threshold_2
        df.at[1, f'VALUES {self.s_2}'] = self.hold_off_2
        df.at[2, f'VALUES {self.s_2}'] = self.cutoff_2_LP
        df.at[3, f'VALUES {self.s_2}'] = self.hyst_asc_2
        df.at[4, f'VALUES {self.s_2}'] = self.hyst_desc_2

        # Guardar el DataFrame modificado de vuelta en el archivo CSV
        df.to_csv(file_path, index=False)

        print("DATOS GUARDADOS", file_path)  # Confirmación de que los datos se han guardado correctamente

    def save_fig(self, name):
        """Guarda la figura actual en un archivo SVG con un nombre que incluye una marca de tiempo.

        Esta función realiza las siguientes tareas:
        1. Obtiene la marca de tiempo actual en el formato `YYYYMMDDHHMMSS`.
        2. Crea un nombre de archivo único para la imagen, que incluye el nombre proporcionado (`name`) y la marca de tiempo.
        3. Construye la ruta completa del archivo usando la carpeta de datos especificada y el nombre del archivo.
        4. Guarda la figura actual como un archivo SVG en la ruta especificada.
        5. Imprime un mensaje de confirmación con el nombre del archivo guardado.

        Parámetros:
        - name (str): El nombre base para el archivo de imagen. Se le añade una marca de tiempo para crear un nombre único.

        No devuelve valores."""
        # Obtener la marca de tiempo actual en el formato YYYYMMDDHHMMSS
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Crear el nombre del archivo añadiendo la marca de tiempo al nombre base
        nombre_archivo_actualizado = f"{name}_{timestamp}.svg"

        # Construir la ruta completa para guardar el archivo
        ruta_imagen_actualizada = os.path.join(self.carpeta_datos, nombre_archivo_actualizado)

        # Guardar la figura actual como un archivo SVG
        plt.savefig(ruta_imagen_actualizada, format='svg')

        # Imprimir un mensaje de confirmación con el nombre del archivo guardado
        print(f"ARCHIVO GUARDADO {nombre_archivo_actualizado}")

    def channel_1(self):
        """Actualiza los datos y la visualización del canal 1 basado en la selección actual en el combobox de la columna 1.

        Esta función realiza las siguientes tareas:
        1. Obtiene el nuevo nombre de la columna seleccionado en el combobox de la columna 1.
        2. Actualiza el atributo `s_1` con el nuevo nombre de columna.
        3. Extrae los datos del canal 1 de `self.data` usando el nuevo nombre de columna y los convierte en un array de NumPy.
        4. Actualiza los datos en el atributo `self.signals` para el canal 1.
        5. Guarda una copia original de los datos del canal 1 en `self.original_1`.
        6. Actualiza el mapa de colores `self.color_map` para reflejar el color del nuevo canal seleccionado.
        7. Llama a la función `self.update_plots_1()` para actualizar las visualizaciones del canal 1.

        No toma parámetros ni devuelve valores."""
        # Obtener el nuevo nombre de la columna para el canal 1 desde el combobox
        new_s_1 = self.column_combobox_1.currentText()
        self.s_1 = new_s_1  # Actualizar el nombre del canal 1

        # Extraer y actualizar los datos del canal 1 en el array de señales
        self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
        self.signals[:, 0] = self.signal_1
        self.original_1 = self.data[f'{self.s_1}'].to_numpy()
        self.signals[:, 2] = self.original_1

        # Actualizar el mapa de colores para los canales
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2]
        }

        # Actualizar la visualización para el canal 1
        self.update_plots_1()

    def channel_2(self):
        """Actualiza los datos y la visualización del canal 2 basado en la selección actual en el combobox de la columna 2."""
        # Obtener el nuevo nombre de la columna para el canal 2 desde el combobox
        new_s_2 = self.column_combobox_2.currentText()
        self.s_2 = new_s_2  # Actualizar el nombre del canal 2

        # Extraer y actualizar los datos del canal 2 en el array de señales
        self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
        self.signals[:, 1] = self.signal_2
        self.original_2 = self.data[f'{self.s_2}'].to_numpy()
        self.signals[:, 3] = self.original_2

        # Actualizar el mapa de colores para los canales
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2]
        }

        # Actualizar la visualización para el canal 2
        self.update_plots_2()

    def filter_LP(self, data, cutoff):
        """Aplica un filtro pasabajo de Butterworth a los datos.

        Este método utiliza un filtro pasabajo de Butterworth para filtrar los datos de entrada.
        Los coeficientes del filtro se calculan utilizando un orden fijo de 5 y la frecuencia de corte proporcionada.

        Parámetros:
        - data (array-like): Datos que se desean filtrar.
        - cutoff (float): Frecuencia de corte del filtro pasabajo en Hz.

        Retorna:
        - array-like: Datos filtrados."""
        order = 5  # Orden del filtro
        # Calcula los coeficientes del filtro Butterworth
        b, a = butter(order, cutoff, btype='low', fs=self.fs)
        # Aplica el filtro a los datos
        z = lfilter(b, a, data)
        return z

    def filter_HP(self, signal, cutoff_HP):
        """Aplica un filtro pasaalto de Butterworth a la señal.

        Este método utiliza un filtro pasaalto de Butterworth para filtrar la señal de entrada.
        Los coeficientes del filtro se calculan utilizando un orden fijo de 2 y la frecuencia de corte proporcionada.

        Parámetros:
        - signal (array-like): Señal que se desea filtrar.
        - cutoff_HP (float): Frecuencia de corte del filtro pasaalto en Hz.

        Retorna:
        - array-like: Señal filtrada."""
        order = 2  # Orden del filtro
        # Calcula los coeficientes del filtro Butterworth
        b, a = butter(order, cutoff_HP, btype='high', fs=self.fs)
        # Aplica el filtro a la señal
        y = lfilter(b, a, signal)
        return y

    def normalizar(self, data, threshold):
        """Normaliza los datos a un rango [0, 1] y ajusta el umbral en consecuencia.

        Este método normaliza los datos de entrada a un rango [0, 1] y ajusta el umbral proporcionado
        para que se corresponda con el rango normalizado. Si todos los valores de los datos son iguales,
        la normalización se ajusta para evitar división por cero.

        Parámetros:
        - data (array-like): Datos que se desean normalizar.
        - threshold (float): Umbral que se desea ajustar a la escala normalizada.

        Retorna:
        - tuple: Una tupla con dos elementos:
            - s_normalizada (array-like): Datos normalizados.
            - threshold_norm (float): Umbral ajustado a la escala normalizada."""
        # Eliminar valores NaN de los datos de entrada, sustituyéndolos por 0.0
        data = np.nan_to_num(data, nan=0.0)

        # Calcular el rango máximo y mínimo de los datos
        max_range = np.max(data)
        min_range = np.min(data)

        # Manejar el caso donde max_range es igual a min_range
        if max_range == min_range:
            s_normalizada = np.zeros_like(data)
            threshold_norm = 0.0
        else:
            # Normalizar los datos y ajustar el umbral
            s_normalizada = (data - min_range) / (max_range - min_range)
            threshold_norm = (threshold - min_range) / (max_range - min_range)

        return s_normalizada, threshold_norm

    def actualizar_grafico(self):
        """Actualiza los gráficos aplicando filtros a las señales y luego actualiza las visualizaciones."""

        # Filtrado de paso bajo para la primera señal
        if self.filter_1_LP:
            # Aplicar el filtro de paso bajo
            filtered_signal = self.filter_LP(self.original_1, self.cutoff_1_LP)
            self.signals[:, 0] = filtered_signal
            # Desactivar el flag de filtro de paso bajo
            self.filter_1_LP = False
            # Actualizar el gráfico de la primera señal
            self.update_plots_1()

        # Filtrado de paso alto para la primera señal
        if self.filter_1_HP:
            # Aplicar el filtro de paso alto
            filtered_signal = self.filter_HP(self.signals[:, 0], self.cutoff_1_HP)
            self.signals[:, 0] = filtered_signal
            # Desactivar el flag de filtro de paso alto
            self.filter_1_HP = False
            # Actualizar el gráfico de la primera señal
            self.update_plots_1()

        # Filtrado de paso bajo para la segunda señal
        if self.filter_2_LP:
            # Aplicar el filtro de paso bajo
            filtered_signal = self.filter_LP(self.original_2, self.cutoff_2_LP)
            self.signals[:, 1] = filtered_signal
            # Desactivar el flag de filtro de paso bajo
            self.filter_2_LP = False
            # Actualizar el gráfico de la segunda señal
            self.update_plots_2()

        # Filtrado de paso alto para la segunda señal
        if self.filter_2_HP:
            # Aplicar el filtro de paso alto
            filtered_signal = self.filter_HP(self.signals[:, 1], self.cutoff_2_HP)
            self.signals[:, 1] = filtered_signal
            # Desactivar el flag de filtro de paso alto
            self.filter_2_HP = False
            # Actualizar el gráfico de la segunda señal
            self.update_plots_2()

    def mover_ventana(self):
        """Mueve la ventana de visualización de datos en función de los flags de avance, retroceso o ventana
        específica. Actualiza los valores de tiempo inicial y final, calcula el segmento de datos correspondiente
        y actualiza los gráficos."""

        # Avanzar la ventana
        if self.avanza:
            self.start_a = self.start_a + self.window_time
            self.end_a = self.start_a + self.window_time
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)

        # Retroceder la ventana
        if self.retrocede:
            self.end_a = self.start_a
            self.start_a = self.end_a - self.window_time
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)

        # Ajustar la ventana a una duración específica
        if self.window:
            self.start_a = self.start_a
            self.end_a = self.window_time + self.start_a
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)
            self.window = False

        # Calcular los índices de datos basados en el tiempo de la ventana
        self.start_a_num = round(self.start_a * self.datos_por_s)
        self.end_a_num = round(self.end_a * self.datos_por_s)

        # Obtener el segmento de datos correspondiente a la ventana actual
        self.tiempo_segment = self.tiempo[self.start_a_num: self.end_a_num]

        # Actualizar los gráficos con el nuevo segmento de datos
        self.update_plots_1()
        self.update_plots_2()

    def update_plots_1(self):
        """Actualiza los gráficos del primer conjunto de datos en el widget de trazado `plot_widget_1`.

        Esta función realiza las siguientes acciones:
        1. Limpia el widget de trazado para preparar el gráfico para nuevos datos.
        2. Traza la señal original y la señal filtrada para el primer conjunto de datos (`s_1`).
        3. Traza las etiquetas de los eventos (`tag_graph`) y las señales de estimulación (`stim_graph`).
        4. Añade una línea infinita que representa el umbral (`threshold_1`).
        5. Configura las etiquetas de los ejes, el título del gráfico y el rango de los ejes.
        6. Actualiza el gráfico de envolvente y cruces si están habilitados.
        7. Actualiza `self.data_copy` con la señal filtrada."""
        # Limpiar el widget de trazado
        self.plot_widget_1.clear()

        # Traza la señal original
        self.plot_widget_1.plot(self.tiempo_segment, self.original_1[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.original_color, width=self.width_line), name='Señal original')

        # Traza la señal filtrada
        self.plot_widget_1.plot(self.tiempo_segment, self.signals[:, 0][self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.color_map[self.s_1], width=self.width_line),
                                name='Señal filtrada')

        # Traza las etiquetas de eventos
        self.plot_widget_1.plot(self.tiempo_segment, self.tag_graph[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        # Traza las señales de estimulación
        self.plot_widget_1.plot(self.tiempo_segment, self.stim_graph[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line,
                                             style=QtCore.Qt.PenStyle.DashLine))

        # Añade una línea infinita para el umbral
        self.plot_widget_1.addItem(pg.InfiniteLine(self.threshold_1, angle=0, pen='g'))

        # Configura las etiquetas de los ejes y el título del gráfico
        self.plot_widget_1.setLabel('left', 'Amplitude [V]')
        self.plot_widget_1.setLabel('bottom', 'Time [s]')
        self.plot_widget_1.setTitle('Original and filtered signal')

        # Configura el rango de los ejes
        self.plot_widget_1.setXRange(self.start_a, self.end_a)
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)

        # Actualiza el gráfico de envolvente y cruces si están habilitados
        self.plot_widget_envolvente_1.setXRange(self.start_a, self.end_a)
        if self.cruces_1:
            self.envolvente_1.setData(self.tiempo_segment, self.s_1_norm[self.start_a_num: self.end_a_num])
            self.cross_1_line_up.setData(self.tiempo_segment, self.cross_up_1[self.start_a_num: self.end_a_num])
            self.cross_1_line_down.setData(self.tiempo_segment, self.cross_down_1[self.start_a_num: self.end_a_num])

        # Actualiza los datos copiados con la señal filtrada
        self.data_copy[f'{self.s_1}'] = self.signals[:, 0]

    def update_plots_2(self):
        """Actualiza los gráficos del segundo conjunto de datos en el widget de trazado `plot_widget_2`.

        Esta función realiza las siguientes acciones:
        1. Limpia el widget de trazado para preparar el gráfico para nuevos datos.
        2. Traza la señal original y la señal filtrada para el segundo conjunto de datos (`s_2`).
        3. Traza las etiquetas de los eventos (`tag_graph`) y las señales de estimulación (`stim_graph`).
        4. Añade una línea infinita que representa el umbral (`threshold_2`).
        5. Configura las etiquetas de los ejes, el título del gráfico y el rango de los ejes.
        6. Actualiza el gráfico de envolvente y cruces si están habilitados.
        7. Actualiza `self.data_copy` con la señal filtrada."""
        # Limpiar el widget de trazado
        self.plot_widget_2.clear()

        # Traza la señal original
        self.plot_widget_2.plot(self.tiempo_segment, self.original_2[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.original_color, width=self.width_line), name='Señal original')

        # Traza la señal filtrada
        self.plot_widget_2.plot(self.tiempo_segment, self.signals[:, 1][self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.color_map[self.s_2], width=self.width_line),
                                name='Señal filtrada')

        # Traza las etiquetas de eventos
        self.plot_widget_2.plot(self.tiempo_segment, self.tag_graph[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        # Traza las señales de estimulación
        self.plot_widget_2.plot(self.tiempo_segment, self.stim_graph[self.start_a_num: self.end_a_num],
                                pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line,
                                             style=QtCore.Qt.PenStyle.DashLine))

        # Añade una línea infinita para el umbral
        self.plot_widget_2.addItem(pg.InfiniteLine(self.threshold_2, angle=0, pen='g'))

        # Configura las etiquetas de los ejes y el título del gráfico
        self.plot_widget_2.setLabel('left', 'Amplitude [V]')
        self.plot_widget_2.setLabel('bottom', 'Time [s]')
        self.plot_widget_2.setTitle('Original and filtered signal')

        # Configura el rango de los ejes
        self.plot_widget_2.setXRange(self.start_a, self.end_a)
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)

        # Actualiza el gráfico de envolvente y cruces si están habilitados
        self.plot_widget_envolvente_2.setXRange(self.start_a, self.end_a)
        if self.cruces_2:
            self.envolvente_2.setData(self.tiempo_segment, self.s_2_norm[self.start_a_num: self.end_a_num])
            self.cross_2_line_up.setData(self.tiempo_segment, self.cross_up_2[self.start_a_num: self.end_a_num])
            self.cross_2_line_down.setData(self.tiempo_segment, self.cross_down_2[self.start_a_num: self.end_a_num])

        # Actualiza los datos copiados con la señal filtrada
        self.data_copy[f'{self.s_2}'] = self.signals[:, 1]

    def detectar_cruces(self, datos, threshold, histeresis_asc, histeresis_desc, hold_off):
        """Detecta los cruces de una señal con respecto a un umbral, considerando histéresis y tiempo de espera.

        Esta función realiza las siguientes acciones:
        1. Inicializa arrays para los cruces ascendentes y descendentes.
        2. Itera sobre los datos para detectar cruces ascendentes y descendentes.
        3. Utiliza histéresis para evitar detecciones erróneas y un tiempo de espera (hold-off) para estabilizar las detecciones.
        4. Registra los tiempos en que se detectan cruces ascendentes y descendentes.
        5. Devuelve listas de tiempos de los cruces ascendentes y descendentes, así como arrays binarios indicando los cruces en los datos.

        Parámetros:
        - `datos`: Array con los datos de la señal en la que se detectarán los cruces.
        - `threshold`: Valor del umbral para detectar los cruces.
        - `histeresis_asc`: Histéresis positiva para evitar detecciones erróneas al cruzar hacia arriba.
        - `histeresis_desc`: Histéresis negativa para evitar detecciones erróneas al cruzar hacia abajo.
        - `hold_off`: Tiempo en segundos que debe pasar antes de permitir otro cruce después de una detección."""
        # Inicialización de arrays y listas para almacenar los resultados
        self.cross_up = np.zeros(len(datos))  # Array para cruces ascendentes
        self.cross_down = np.zeros(len(datos))  # Array para cruces descendentes
        UP = []  # Lista para almacenar tiempos de cruces ascendentes
        DOWN = []  # Lista para almacenar tiempos de cruces descendentes
        up_detected = False  # Flag para detectar si se ha cruzado hacia arriba
        hold_off_timer = False  # Flag para manejar el tiempo de espera
        hold_start_time = 0  # Tiempo de inicio del hold-off

        # Iteración sobre los datos para detectar cruces
        for i in range(len(datos)):
            # Detectar cruce ascendente
            if not up_detected and not hold_off_timer:
                if datos[i] >= threshold + histeresis_asc:
                    up_detected = True
                    UP.append(self.tiempo[i])
                    self.cross_up[i] = 1
                    hold_start_time = self.tiempo[i]

            # Manejar el tiempo de espera después de un cruce ascendente
            if up_detected and not hold_off_timer:
                if self.tiempo[i] >= hold_start_time + hold_off:
                    hold_off_timer = True

            # Detectar cruce descendente
            if datos[i] <= threshold - histeresis_desc and up_detected and hold_off_timer:
                up_detected = False
                hold_off_timer = False
                DOWN.append(self.tiempo[i])
                self.cross_down[i] = 1

        return UP, DOWN, self.cross_up, self.cross_down

    def detectar_cruces_1(self):
        """Detecta los cruces en el primer conjunto de datos y actualiza la visualización correspondiente.

        Esta función realiza las siguientes acciones:
        1. Activa la detección de cruces para el primer canal de datos.
        2. Limpia el widget de trazado y configura los elementos visuales necesarios para mostrar los cruces y la señal normalizada.
        3. Llama a `detectar_cruces` para encontrar los cruces ascendentes y descendentes, y normaliza la señal.
        4. Actualiza los gráficos con los datos normalizados y los cruces detectados."""
        self.cruces_1 = True
        self.plot_widget_envolvente_1.clear()  # Limpiar el widget de trazado

        # Configurar la visualización para la envolvente y los cruces
        self.envolvente_1 = self.plot_widget_envolvente_1.plot(
            pen=pg.mkPen(color=self.color_map[self.s_1], width=self.width_line))
        self.cross_1_line_up = self.plot_widget_envolvente_1.plot(
            pen=pg.mkPen(color=self.up_color, width=self.width_crossing))
        self.cross_1_line_down = self.plot_widget_envolvente_1.plot(
            pen=pg.mkPen(color=self.down_color, width=self.width_crossing, style=QtCore.Qt.PenStyle.DashLine))

        # Configuración de etiquetas y título del gráfico
        self.plot_widget_envolvente_1.setLabel('left', 'Amplitude [V]')
        self.plot_widget_envolvente_1.setLabel('bottom', 'Time [s]')
        self.plot_widget_envolvente_1.setTitle('Normalized signal')

        # Detectar cruces y normalizar la señal
        self.UP_1, self.DOWN_1, self.cross_up_1, self.cross_down_1 = self.detectar_cruces(
            self.signals[:, 0],
            self.threshold_1,
            self.hyst_asc_1,
            self.hyst_desc_1,
            self.hold_off_1
        )
        self.s_1_norm, self.threshold_norm_1 = self.normalizar(self.signals[:, 0], self.threshold_1)

        # Actualizar los datos en los elementos gráficos
        self.envolvente_1.setData(self.tiempo, self.s_1_norm)
        self.cross_1_line_up.setData(self.tiempo, self.cross_up_1)
        self.cross_1_line_down.setData(self.tiempo, self.cross_down_1)
        self.plot_widget_envolvente_1.addItem(pg.InfiniteLine(self.threshold_norm_1, angle=0,
                                                              pen=pg.mkPen(color=self.threshold_color,
                                                                           width=self.width_line)))

        # Configurar el rango de los ejes del gráfico
        self.plot_widget_envolvente_1.setXRange(self.start_a, self.end_a)
        self.plot_widget_envolvente_1.setYRange(self.min_amplitude_norm_1, self.max_amplitude_norm_1)

        # Actualizar los gráficos del primer canal
        self.update_plots_1()

    def detectar_cruces_2(self):
        """ Detecta los cruces en el segundo conjunto de datos y actualiza la visualización correspondiente.

        Esta función realiza las siguientes acciones:
        1. Activa la detección de cruces para el segundo canal de datos.
        2. Limpia el widget de trazado y configura los elementos visuales necesarios para mostrar los cruces y la señal normalizada.
        3. Llama a `detectar_cruces` para encontrar los cruces ascendentes y descendentes, y normaliza la señal.
        4. Actualiza los gráficos con los datos normalizados y los cruces detectados."""
        self.cruces_2 = True
        self.plot_widget_envolvente_2.clear()  # Limpiar el widget de trazado

        # Configurar la visualización para la envolvente y los cruces
        self.envolvente_2 = self.plot_widget_envolvente_2.plot(
            pen=pg.mkPen(color=self.color_map[self.s_2], width=self.width_line))
        self.cross_2_line_up = self.plot_widget_envolvente_2.plot(
            pen=pg.mkPen(color=self.up_color, width=self.width_crossing))
        self.cross_2_line_down = self.plot_widget_envolvente_2.plot(
            pen=pg.mkPen(color=self.down_color, width=self.width_crossing, style=QtCore.Qt.PenStyle.DashLine))

        # Configuración de etiquetas y título del gráfico
        self.plot_widget_envolvente_2.setLabel('left', 'Amplitude [V]')
        self.plot_widget_envolvente_2.setLabel('bottom', 'Time [s]')
        self.plot_widget_envolvente_2.setTitle('Normalized signal')

        # Detectar cruces y normalizar la señal
        self.UP_2, self.DOWN_2, self.cross_up_2, self.cross_down_2 = self.detectar_cruces(
            self.signals[:, 1],
            self.threshold_2,
            self.hyst_asc_2,
            self.hyst_desc_2,
            self.hold_off_2
        )
        self.s_2_norm, self.threshold_norm_2 = self.normalizar(self.signals[:, 1], self.threshold_2)

        # Actualizar los datos en los elementos gráficos
        self.envolvente_2.setData(self.tiempo, self.s_2_norm)
        self.cross_2_line_up.setData(self.tiempo, self.cross_up_2)
        self.cross_2_line_down.setData(self.tiempo, self.cross_down_2)
        self.plot_widget_envolvente_2.addItem(pg.InfiniteLine(self.threshold_norm_2, angle=0,
                                                              pen=pg.mkPen(color=self.threshold_color,
                                                                           width=self.width_line)))

        # Configurar el rango de los ejes del gráfico
        self.plot_widget_envolvente_2.setXRange(self.start_a, self.end_a)
        self.plot_widget_envolvente_2.setYRange(self.min_amplitude_norm_2, self.max_amplitude_norm_2)

        # Imprimir mensaje de estado y actualizar los gráficos del segundo canal
        print("DETECCIÓN DE CRUCES 2")
        self.update_plots_2()


if __name__ == '__main__':
    """Punto de entrada principal del programa. Inicializa la aplicación PyQt y la ventana principal,
    luego ejecuta la aplicación en el bucle de eventos."""
    # Inicializa la aplicación Qt
    app = QApplication(sys.argv)

    # Obtiene el primer argumento de la línea de comandos como el nombre del archivo
    archivo = sys.argv[1]

    # Obtiene los colores de los canales desde los argumentos de la línea de comandos (índices 2 a 14)
    # Se espera una lista de colores en formato hexadecimal
    channel_colors = sys.argv[2:15]  #['#23BAC4', '#DAA520', '#E69DFB', '#F08080', '#7FFFD4', '#FF0000', '#00FF00', '#E7D40A', '#DA70D6', '#FF9900', '#F08080', '#FF689D']

    # Obtiene los colores de los estímulos desde los argumentos de la línea de comandos (índices 15 a 18)
    # Se espera una lista de colores en formato hexadecimal para los estímulos y el umbral
    stim_colors = sys.argv[15:19]  #[self.stim_color_1, self.stim_color_2, self.stim_color_3, self.threshold_color]

    # Obtiene los valores adicionales desde los argumentos de la línea de comandos (índices 19 en adelante)
    # Estos valores pueden incluir índices de canales, umbrales, tiempos de retención, tiempos de retardo, etc.
    values = sys.argv[19:]  #[self.channel_index, self.threshold, self.hold_off_time, self.delay_time, hysteresis_up, hysteresis_down]

    # Crea una instancia de la ventana principal, pasando los argumentos obtenidos
    mainWindow = MainWindow(archivo, channel_colors, stim_colors, values)

    # Muestra la ventana principal
    mainWindow.show()

    # Ejecuta el bucle de eventos de la aplicación Qt
    sys.exit(app.exec())
