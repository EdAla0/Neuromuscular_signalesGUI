import csv
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from scipy.signal import lfilter, butter
from scipy import signal


class MainWindow(QMainWindow):
    def __init__(self, archivo, channel_colors, stim_colors, values):
        """Inicializa una instancia de la clase, configurando la interfaz gráfica y los datos necesarios para el análisis de señales.

        Parámetros:
        archivo (str): Ruta del archivo CSV que contiene los datos de señales.
        channel_colors (list): Lista de colores para los canales de señal.
        stim_colors (list): Lista de colores para los estímulos.
        values (list): Lista de valores iniciales para configurar el análisis."""
        super().__init__()
        # Carga la interfaz gráfica desde el archivo .ui
        loadUi('Autocorrelation.ui', self)

        # Crea y configura el widget central y su layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)
        central_widget.setLayout(layout)

        # Inicializa el widget de trazado para la señal original y su envolvente suavizada 1
        self.plot_widget_1 = pg.PlotWidget()
        self.Graph_layout_1.addWidget(self.plot_widget_1)

        # Inicializa el widget de trazado para la señal original y su envolvente suavizada 2
        self.plot_widget_2 = pg.PlotWidget()
        self.Graph_layout_2.addWidget(self.plot_widget_2)

        # Inicializa el widget de trazado para la envolvente suavizada vs tiempo 1
        self.plot_widget_3 = pg.PlotWidget()
        self.Graph_layout_3.addWidget(self.plot_widget_3)

        # Inicializa el widget de trazado para la envolvente suavizada vs tiempo 2
        self.plot_widget_4 = pg.PlotWidget()
        self.Graph_layout_6.addWidget(self.plot_widget_4)

        # Configura las opciones para el menú desplegable de canales
        options = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                   'CH 10', 'CH 11', 'CH 12']

        # Crea un diccionario que mapea cada opción del menú con su color correspondiente
        self.channel_color_map = dict(zip(options, channel_colors))

        # Configura los comboboxes para seleccionar las columnas de datos
        comboboxes = [
            self.column_combobox_1, self.column_combobox_2, self.column_combobox_3, self.column_combobox_5
        ]
        for combobox in comboboxes:
            combobox.addItems(options)  # Añade las opciones al combobox
            combobox.currentIndexChanged.connect(
                self.channel)  # Conecta el cambio de selección con la función `channel`

        # Conecta los cambios en los QSpinBox con las funciones correspondientes para ajustar el rango del gráfico
        self.y_max_1.valueChanged.connect(self.update_y_range_1)
        self.y_min_1.valueChanged.connect(self.update_y_range_1)
        self.y_max_2.valueChanged.connect(self.update_y_range_2)
        self.y_min_2.valueChanged.connect(self.update_y_range_2)
        self.y_max_3.valueChanged.connect(self.update_y_range_3)
        self.y_min_3.valueChanged.connect(self.update_y_range_3)
        self.y_max_4.valueChanged.connect(self.update_y_range_4)
        self.y_min_4.valueChanged.connect(self.update_y_range_4)

        # Configura el QSpinBox para el tiempo de la ventana de visualización
        self.window_time.valueChanged.connect(self.update_window_time)  # Conecta al método de actualización

        # Conecta los QSpinBox de los filtros con sus funciones correspondientes para actualizar los filtros
        spinboxes = [
            (self.cutoff_s1_LP, self.actualizar_LP_fc_1), (self.cutoff_s2_LP, self.actualizar_LP_fc_2),
            (self.cutoff_s3_LP, self.actualizar_LP_fc_3), (self.cutoff_s4_LP, self.actualizar_LP_fc_4),
            (self.cutoff_s1_HP, self.actualizar_HP_fc_1), (self.cutoff_s2_HP, self.actualizar_HP_fc_2),
            (self.cutoff_s3_HP, self.actualizar_HP_fc_3), (self.cutoff_s4_HP, self.actualizar_HP_fc_4)
        ]
        for spinbox, function in spinboxes:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor con la función correspondiente

        # Conecta los botones de filtro con las funciones correspondientes para aplicar los filtros
        filter_buttons = [
            (self.btn_filter_1, self.update_filter_1), (self.btn_filter_2, self.update_filter_2),
            (self.btn_filter_3, self.update_filter_3), (self.btn_filter_4, self.update_filter_4)
        ]
        for button, function in filter_buttons:
            button.clicked.connect(function)  # Conecta el clic del botón con la función correspondiente

        # Configura el botón para guardar
        self.save_function.clicked.connect(self.save_active)


        self.stim_colors = stim_colors
        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        self.name_file.textChanged.connect(self.name_file_update)

        # Configura los botones para avanzar y retroceder en las gráficas
        self.btn_avanzar.clicked.connect(self.avanzar_graficas)
        self.btn_retroceder.clicked.connect(self.retroceder_graficas)

        # Configura el botón para realizar el análisis de Autocorrelation
        self.bt_go.clicked.connect(self.analisis_Autocorrelation)

        # Configura un diccionario para asociar los PlotWidgets con sus títulos
        self.plot_widgets = {
            self.plot_widget_1: "Signal 1",
            self.plot_widget_2: "Signal 2",
            self.plot_widget_3: "Signal 3",
            self.plot_widget_4: "Signal 4"
        }

        # Conecta el evento de clic del mouse en los PlotWidgets a la función correspondiente
        self.plot_widget_1.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_2.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_3.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_4.scene().sigMouseClicked.connect(self.on_plot_clicked)

        # Inicializa la ruta del archivo y carga los datos
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)  # Extrae la carpeta del archivo de datos
        self.data = pd.read_csv(self.ruta_archivo)  # Lee los datos del archivo CSV
        self.data_copy = pd.read_csv(self.ruta_archivo)  # Crea una copia de los datos
        self.time = self.data['TIME'].to_numpy()  # Extrae el tiempo de los datos
        
        self.num_dominant = 1

        # Inicializa variables de estado para el análisis
        self.crossing_detected = False
        self.crossing_hysteresis_detected = False
        self.crossing_descendente = False
        self.first_up = False
        self.first_down = False
        self.hysteresis = float(values[4])  # Configura el valor de histeresis
        self.s_1 = 'CH 1'  # Columna seleccionada para Signal 1
        self.s_2 = 'CH 1'  # Columna seleccionada para Signal 2
        self.s_3 = 'CH 1'  # Columna seleccionada para Signal 3
        self.s_4 = 'CH 1'  # Columna seleccionada para Signal 4
        self.signal_1 = self.data[f'{self.s_1}'].to_numpy()  # Extrae la señal 1
        self.signal_2 = self.data[f'{self.s_2}'].to_numpy()  # Extrae la señal 2
        self.signal_3 = self.data[f'{self.s_3}'].to_numpy()  # Extrae la señal 3
        self.signal_4 = self.data[f'{self.s_4}'].to_numpy()  # Extrae la señal 4

        # Configura las señales y sus envolventes
        self.signals = self.data[
            [f'{self.s_1}', f'{self.s_2}', f'{self.s_3}', f'{self.s_4}', f'{self.s_1}', f'{self.s_2}', f'{self.s_3}',
             f'{self.s_4}']].to_numpy()
        self.original_1 = self.signals[:, 4]
        self.original_2 = self.signals[:, 5]
        self.original_3 = self.signals[:, 6]
        self.original_4 = self.signals[:, 7]
        
        # Configura el checkbox para detrend.
        self.detrend_button.stateChanged.connect(self.detrend_active)
        # Configura el checkbox para normalización.
        self.normalize_button.stateChanged.connect(self.normalize_active)
        # Configura el QSpinBox para el sub muestreo.
        self.subsample_factor.valueChanged.connect(self.update_subsample_factor)  # Conecta al método de actualización
        
        # Configura el botón para aplicar el análisis unicamente en los datos mostrados en ventana.
        self.analysis_on_window_function.stateChanged.connect(self.analysis_on_window_active)
        
        # Configura los parámetros del análisis
        self.detrend = True #detrend (bool): Si True, elimina la tendencia lineal de las señales.
        self.normalize = True #normalize (bool): Si True, normaliza las señales.
        self.subsample_factor = 1 #subsample_factor (int): Se tomará solo cada enésimo valor de la señal.

        # Configura los parámetros de los filtros y las ventanas de visualización
        self.fs = 3300
        self.cutoff_1_LP = 1000
        self.cutoff_2_LP = 1000
        self.cutoff_3_LP = 1000
        self.cutoff_4_LP = 1000
        self.cutoff_1_HP = 0
        self.cutoff_2_HP = 0
        self.cutoff_3_HP = 0
        self.cutoff_4_HP = 0
        self.window_time = 10
        self.tiempo_inicial = 0  # Inicio de la ventana de visualización
        self.tiempo_final = self.window_time  # Fin de la ventana de visualización
        self.datos_por_s = round(1 / (self.time[1] - self.time[0]))  # Datos por segundo
        self.start_a_num = round(self.tiempo_inicial * self.datos_por_s)
        self.end_a_num = round(self.tiempo_final * self.datos_por_s)
        self.min_amplitude_1 = 0
        self.max_amplitude_1 = 1
        self.min_amplitude_2 = 0
        self.max_amplitude_2 = 1
        self.min_amplitude_3 = 0
        self.max_amplitude_3 = 1
        self.min_amplitude_4 = 0
        self.max_amplitude_4 = 1

        # Configura el mapeo de colores para los canales y estímulos
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2],
            self.s_3: self.channel_color_map[self.s_3],
            self.s_4: self.channel_color_map[self.s_4]
        }


        # Configura el ancho de las líneas y colores de las diferentes características en los gráficos
        self.width_line = 2
        self.width_crossing = 1
        self.original_color = '#00FFFF'
        self.filtered_color = '#00FFFF'
        self.stim_color = 'y'
        self.up_color = 'm'
        self.down_color = '#FFFFFF'
        self.threshold_color = 'g'

        # Inicializa flags y nombres de archivo
        self.avanza = False
        self.retrocede = False
        self.window = False
        self.filter_1_LP = False
        self.filter_2_LP = False
        self.filter_3_LP = False
        self.filter_4_LP = False
        self.filter_1_HP = False
        self.filter_2_HP = False
        self.filter_3_HP = False
        self.filter_4_HP = False
        self.save_function_flag = False
        self.analysis_on_window = False
        self.before = 1
        self.after = 1
        directory, file_name = os.path.split(self.ruta_archivo)
        self.new_file_name  = os.path.splitext(file_name)[0]
        # Actualiza el gráfico inicial
        self.actualizar_grafico()
        
    def detrend_active(self):
        """Activa o desactiva el detrend antes de realizar el análisis.

        La función cambia el estado de `detrend` entre `True` y `False`, y muestra un mensaje en la consola
        indicando si la funcionalidad está activa o desactivada."""
        # Verifica el estado actual
        if not self.detrend:
            # Si está desactivado, actívalo y muestra un mensaje en la consola
            self.detrend = True
            print("Detrend activado")
        else:
            # Si ya está activado, desactívalo y muestra un mensaje en la consola
            self.detrend = False
            print("Detrend desactivado")
            
    def normalize_active(self):
        """Activa o desactiva la normalización antes de realizar el análisis.

        La función cambia el estado de `normalize` entre `True` y `False`, y muestra un mensaje en la consola
        indicando si la funcionalidad está activa o desactivada."""
        # Verifica el estado actual
        if not self.normalize:
            # Si está desactivado, actívalo y muestra un mensaje en la consola
            self.normalize = True
            print("Normalización activada")
        else:
            # Si ya está activado, desactívalo y muestra un mensaje en la consola
            self.normalize = False
            print("Normalización desactivada")
    
    def update_subsample_factor(self, value):
        """Actualiza el enésimo valor subsample_factor a usar en el análisis.
        Parámetros:
            value (int): Nuevo valor de subsample_factor."""
        self.subsample_factor = value  # Actualiza subsample_factor

    def save_active(self):
        """Activa o desactiva la funcionalidad de guardado en función del estado actual del flag de guardado.

        La función cambia el estado del flag `save_function_flag` entre `True` y `False`, y muestra un mensaje en la consola
        indicando si la funcionalidad de guardado está activa o desactivada."""
        # Verifica el estado actual del flag de guardado
        if not self.save_function_flag:
            # Si el flag está desactivado, actívalo y muestra un mensaje en la consola
            self.save_function_flag = True
            print("ACTIVO SAVE")
        else:
            # Si el flag ya está activado, desactívalo y muestra un mensaje en la consola
            self.save_function_flag = False
            print("DESACTIVO SAVE")
            
    def analysis_on_window_active(self):
        """Activa o desactiva la funcionalidad de analisís en la ventana en función del estado actual del flag de guardado.

        La función cambia el estado del flag `analysis_on_window` entre `True` y `False`, y muestra un mensaje en la consola
        indicando si la funcionalidad de analisís en la ventana está activa o desactivada."""
        # Verifica el estado actual del flag de analisís en la ventana.
        if not self.analysis_on_window:
            # Si el flag está desactivado, actívalo y muestra un mensaje en la consola
            self.analysis_on_window = True
            print("ACTIVO ANALYSIS ON WINDOW")
        else:
            # Si el flag ya está activado, desactívalo y muestra un mensaje en la consola
            self.analysis_on_window = False
            print("DESACTIVO ANALYSIS ON WINDOW")
                    
    def name_file_update(self, new_file_name):
        """Actualiza el nombre del archivo de análisis."""
        self.new_file_name = new_file_name

    def update_y_range_1(self):
        """Actualiza el rango del eje Y del primer gráfico (`plot_widget_1`).

        Esta función obtiene los valores mínimos y máximos del rango Y desde los controles `y_min_1` y `y_max_1`
        y los aplica al gráfico `plot_widget_1`.

        El rango Y del gráfico se establece utilizando los valores obtenidos de los controles."""
        # Obtiene el valor mínimo del eje Y desde el control `y_min_1`
        self.min_amplitude_1 = self.y_min_1.value()
        # Obtiene el valor máximo del eje Y desde el control `y_max_1`
        self.max_amplitude_1 = self.y_max_1.value()
        # Establece el rango Y del gráfico `plot_widget_1` con los valores obtenidos
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)

    def update_y_range_2(self):
        """Actualiza el rango del eje Y del segundo gráfico (`plot_widget_2`).

        Esta función obtiene los valores mínimos y máximos del rango Y desde los controles `y_min_2` y `y_max_2`
        y los aplica al gráfico `plot_widget_2`.

        El rango Y del gráfico se establece utilizando los valores obtenidos de los controles."""
        # Obtiene el valor mínimo del eje Y desde el control `y_min_2`
        self.min_amplitude_2 = self.y_min_2.value()
        # Obtiene el valor máximo del eje Y desde el control `y_max_2`
        self.max_amplitude_2 = self.y_max_2.value()
        # Establece el rango Y del gráfico `plot_widget_2` con los valores obtenidos
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)

    def update_y_range_3(self):
        """Actualiza el rango del eje Y del tercer gráfico (`plot_widget_3`).

        Esta función obtiene los valores mínimos y máximos del rango Y desde los controles `y_min_3` y `y_max_3`
        y los aplica al gráfico `plot_widget_3`.

        El rango Y del gráfico se establece utilizando los valores obtenidos de los controles."""
        # Obtiene el valor mínimo del eje Y desde el control `y_min_3`
        self.min_amplitude_3 = self.y_min_3.value()
        # Obtiene el valor máximo del eje Y desde el control `y_max_3`
        self.max_amplitude_3 = self.y_max_3.value()
        # Establece el rango Y del gráfico `plot_widget_3` con los valores obtenidos
        self.plot_widget_3.setYRange(self.min_amplitude_3, self.max_amplitude_3)

    def update_y_range_4(self):
        """Actualiza el rango del eje Y del cuarto gráfico (`plot_widget_4`).

        Esta función obtiene los valores mínimos y máximos del rango Y desde los controles `y_min_4` y `y_max_4`
        y los aplica al gráfico `plot_widget_4`.

        El rango Y del gráfico se establece utilizando los valores obtenidos de los controles."""
        # Obtiene el valor mínimo del eje Y desde el control `y_min_4`
        self.min_amplitude_4 = self.y_min_4.value()
        # Obtiene el valor máximo del eje Y desde el control `y_max_4`
        self.max_amplitude_4 = self.y_max_4.value()
        # Establece el rango Y del gráfico `plot_widget_4` con los valores obtenidos
        self.plot_widget_4.setYRange(self.min_amplitude_4, self.max_amplitude_4)

    def on_plot_clicked(self, event):
        """Maneja el evento de clic en los gráficos de la interfaz de usuario.

        Si se hace clic con el botón izquierdo del ratón mientras se mantiene presionada la tecla Control,
        busca los índices de estimulación en una ventana alrededor del punto clickeado y los marca como 0.
        Si se hace clic con el botón izquierdo del ratón sin ninguna tecla modificadora, agrega un valor de 1
        en el vector de estimulación (self.stim) en el índice correspondiente al punto clickeado.

        Parámetros:
            event (QMouseEvent): Evento de clic del ratón."""
        # Obtiene los modificadores del teclado (como la tecla Control)
        modifiers = QApplication.keyboardModifiers()

        # Si se hace clic con el botón izquierdo del ratón y se mantiene presionada la tecla Control
        if event.button() == Qt.MouseButton.LeftButton and (modifiers & Qt.KeyboardModifier.ControlModifier):
            pos = event.scenePos()  # Obtiene la posición del clic en la escena
            for plot_widget, title in self.plot_widgets.items():
                # Mapea la posición del clic a las coordenadas del gráfico
                pos_mapped = plot_widget.plotItem.vb.mapSceneToView(pos)
                # Encuentra el índice más cercano en el vector de tiempo correspondiente a la posición del clic
                index_clicked = np.searchsorted(self.time, pos_mapped.x())
                if 0 <= index_clicked < len(self.time):
                    # Define la ventana de índices alrededor del índice clickeado
                    ventana_indices = 100
                    inicio_ventana = max(0, index_clicked - ventana_indices)
                    fin_ventana = min(len(self.time), index_clicked + ventana_indices + 1)

                    # Encuentra los índices donde self.stim es 1 o 2 dentro de la ventana
                    stim_indices = np.where((self.stim == 1) | (self.stim == 2))[0]
                    indices_ventana = stim_indices[(stim_indices >= inicio_ventana) & (stim_indices < fin_ventana)]

                    if len(indices_ventana) > 0:
                        # Si hay índices dentro de la ventana, marca el más cercano
                        closest_index = min(indices_ventana, key=lambda x: abs(x - index_clicked))
                        # Marca el índice como 0 en self.stim
                        self.stim[closest_index] = 0
                        self.update_plots()  # Actualiza los gráficos
                    else:
                        print("No se encontraron índices de estimulación (1 o 2) dentro de la ventana")
                    pass

        # Si se hace clic con el botón izquierdo del ratón sin ninguna tecla modificadora
        elif event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()  # Obtiene la posición del clic en la escena
            for plot_widget, title in self.plot_widgets.items():
                # Mapea la posición del clic a las coordenadas del gráfico
                pos_mapped = plot_widget.plotItem.vb.mapSceneToView(pos)
                # Encuentra el índice más cercano en el vector de tiempo correspondiente a la posición del clic
                index_clicked = np.searchsorted(self.time, pos_mapped.x())
                if 0 <= index_clicked < len(self.time):
                    # Agrega un valor de 1 en el vector de estimulación (self.stim) en el índice correspondiente
                    self.stim[index_clicked] = 1
                    self.update_plots()  # Actualiza los gráficos

    def update_window_time(self, value):
        """Actualiza el tiempo de la ventana de visualización y mueve la ventana de acuerdo al nuevo valor.

        Parámetros:
            value (int): Nuevo valor del tiempo de la ventana de visualización."""
        self.window_time = value  # Actualiza el tiempo de la ventana de visualización
        self.avanza = False  # Indica que no se está avanzando
        self.retrocede = False  # Indica que no se está retrocediendo
        self.window = True  # Indica que la acción es actualizar la ventana
        self.mover_ventana()  # Llama a la función para mover la ventana de visualización

    def avanzar_graficas(self):
        """Avanza la ventana de visualización en los gráficos."""
        self.avanza = True  # Indica que se está avanzando
        self.retrocede = False  # Indica que no se está retrocediendo
        self.window = False  # Indica que la acción no es actualizar la ventana
        self.mover_ventana()  # Llama a la función para mover la ventana de visualización

    def retroceder_graficas(self):
        """Retrocede la ventana de visualización en los gráficos."""
        self.retrocede = True  # Indica que se está retrocediendo
        self.avanza = False  # Indica que no se está avanzando
        self.window = False  # Indica que la acción no es actualizar la ventana
        self.mover_ventana()  # Llama a la función para mover la ventana de visualización


    def actualizar_LP_fc_1(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-bajos para la señal 1 y actualiza el gráfico.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-bajos de la señal 1."""
        self.cutoff_1_LP = value  # Actualiza la frecuencia de corte del filtro pasa-bajos de la señal 1
        self.filter_1_LP = True  # Indica que el filtro pasa-bajos para la señal 1 está activo
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_2(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-bajos para la señal 2 y actualiza el gráfico.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-bajos de la señal 2."""
        self.cutoff_2_LP = value  # Actualiza la frecuencia de corte del filtro pasa-bajos de la señal 2
        self.filter_2_LP = True  # Indica que el filtro pasa-bajos para la señal 2 está activo
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_3(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-bajos para la señal 3 y actualiza el gráfico.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-bajos de la señal 3."""
        self.cutoff_3_LP = value  # Actualiza la frecuencia de corte del filtro pasa-bajos de la señal 3
        self.filter_3_LP = True  # Indica que el filtro pasa-bajos para la señal 3 está activo
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_4(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-bajos para la señal 4 y actualiza el gráfico.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-bajos de la señal 4."""
        self.cutoff_4_LP = value  # Actualiza la frecuencia de corte del filtro pasa-bajos de la señal 4
        self.filter_4_LP = True  # Indica que el filtro pasa-bajos para la señal 4 está activo
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_HP_fc_1(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-altos para la señal 1.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-altos de la señal 1."""
        self.cutoff_1_HP = value  # Actualiza la frecuencia de corte del filtro pasa-altos de la señal 1

    def actualizar_HP_fc_2(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-altos para la señal 2.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-altos de la señal 2."""
        self.cutoff_2_HP = value  # Actualiza la frecuencia de corte del filtro pasa-altos de la señal 2

    def actualizar_HP_fc_3(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-altos para la señal 3.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-altos de la señal 3."""
        self.cutoff_3_HP = value  # Actualiza la frecuencia de corte del filtro pasa-altos de la señal 3

    def actualizar_HP_fc_4(self, value):
        """Actualiza la frecuencia de corte del filtro pasa-altos para la señal 4.

        Parámetros:
            value (float): Nuevo valor para la frecuencia de corte del filtro pasa-altos de la señal 4."""
        self.cutoff_4_HP = value  # Actualiza la frecuencia de corte del filtro pasa-altos de la señal 4

    def update_filter_1(self):
        """Activa el filtro pasa-altos para la señal 1 y actualiza el gráfico correspondiente."""
        self.filter_1_HP = True  # Activa el filtro pasa-altos para la señal 1
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico

    def update_filter_2(self):
        """Activa el filtro pasa-altos para la señal 2 y actualiza el gráfico correspondiente."""
        self.filter_2_HP = True  # Activa el filtro pasa-altos para la señal 2
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico

    def update_filter_3(self):
        """Activa el filtro pasa-altos para la señal 3 y actualiza el gráfico correspondiente."""
        self.filter_3_HP = True  # Activa el filtro pasa-altos para la señal 3
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico

    def update_filter_4(self):
        """Activa el filtro pasa-altos para la señal 4 y actualiza el gráfico correspondiente."""
        self.filter_4_HP = True  # Activa el filtro pasa-altos para la señal 4
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico

    def filter_HP(self, signal, cutoff_HP):
        """Aplica un filtro pasa-altos (high-pass) Butterworth a una señal.

        Parámetros:
            signal (numpy.ndarray): La señal a la que se le aplicará el filtro.
            cutoff_HP (float): Frecuencia de corte del filtro pasa-altos.

        Retorna:
            numpy.ndarray: La señal filtrada."""
        order = 5  # Orden del filtro
        # Calcula los coeficientes del filtro pasa-altos Butterworth
        b, a = butter(order, cutoff_HP, btype='high', fs=self.fs)
        # Aplica el filtro a la señal
        y = lfilter(b, a, signal)
        return y  # Retorna la señal filtrada

    def filter_LP(self, signal, cutoff_LP):
        """Aplica un filtro pasa-bajos (low-pass) Butterworth a una señal.

        Parámetros:
            signal (numpy.ndarray): La señal a la que se le aplicará el filtro.
            cutoff_LP (float): Frecuencia de corte del filtro pasa-bajos.

        Retorna:
            numpy.ndarray: La señal filtrada."""
        order = 5  # Orden del filtro
        # Calcula los coeficientes del filtro pasa-bajos Butterworth
        b, a = butter(order, cutoff_LP, btype='low', fs=self.fs)
        # Aplica el filtro a la señal
        z = lfilter(b, a, signal)
        return z  # Retorna la señal filtrada

    def channel(self):
        """Actualiza las señales de los canales seleccionados y las asigna a las variables correspondientes.
            Luego, actualiza el mapa de colores y las gráficas."""

        # Obtiene las nuevas selecciones de los comboboxes de canales
        new_s_1 = self.column_combobox_1.currentText()
        new_s_2 = self.column_combobox_2.currentText()
        new_s_3 = self.column_combobox_3.currentText()
        new_s_4 = self.column_combobox_5.currentText()

        # Actualiza las señales del canal 1 si ha cambiado la selección
        if new_s_1 != self.s_1:
            self.s_1 = new_s_1
            self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
            self.signals[:, 0] = self.signal_1
            self.s_5 = new_s_1
            self.original_1 = self.data[f'{self.s_5}'].to_numpy()
            self.signals[:, 4] = self.original_1

        # Actualiza las señales del canal 2 si ha cambiado la selección
        if new_s_2 != self.s_2:
            self.s_2 = new_s_2
            self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
            self.signals[:, 1] = self.signal_2
            self.s_6 = new_s_2
            self.original_2 = self.data[f'{self.s_6}'].to_numpy()
            self.signals[:, 5] = self.original_2

        # Actualiza las señales del canal 3 si ha cambiado la selección
        if new_s_3 != self.s_3:
            self.s_3 = new_s_3
            self.signal_3 = self.data[f'{self.s_3}'].to_numpy()
            self.signals[:, 2] = self.signal_3
            self.s_7 = new_s_3
            self.original_3 = self.data[f'{self.s_7}'].to_numpy()
            self.signals[:, 6] = self.original_3

        # Actualiza las señales del canal 4 si ha cambiado la selección
        if new_s_4 != self.s_4:
            self.s_4 = new_s_4
            self.signal_4 = self.data[f'{self.s_4}'].to_numpy()
            self.signals[:, 3] = self.signal_4
            self.s_8 = new_s_4
            self.original_4 = self.data[f'{self.s_8}'].to_numpy()
            self.signals[:, 7] = self.original_4

        # Actualiza el mapa de colores para los canales
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2],
            self.s_3: self.channel_color_map[self.s_3],
            self.s_4: self.channel_color_map[self.s_4]
        }

        # Llama a la función para actualizar las gráficas
        self.update_plots()

    def save_and_insert_data(self, name):
        """Guarda los datos de las señales en un archivo CSV y añade información adicional.

        Este método realiza las siguientes acciones:
        - Genera un nombre de archivo único basado en el nombre proporcionado y el timestamp actual.
        - Guarda las señales y la estimulación actuales en un archivo CSV.
        - Añade una columna adicional con información sobre las frecuencias de corte y la estimulación.

        Parámetros:
            name (str): Nombre base para el archivo.

        Variables:
            timestamp (str): Marca de tiempo actual en el formato YYYYMMDDHHMMSS.
            nombre_archivo_data (str): Nombre completo del archivo con el timestamp.
            file_path (str): Ruta completa del archivo donde se guardarán los datos.
            data_to_save (np.ndarray): Datos a guardar en el archivo CSV.
            column_labels (list): Lista de etiquetas de las columnas para el archivo CSV."""

        # Genera un timestamp y un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nombre_archivo_data = f"{name}_{timestamp}.csv"
        self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior

        # Ruta completa del archivo
        file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)

        # Actualiza las señales en el DataFrame copia
        self.data_copy[f'{self.s_1}'] = self.signal_1
        self.data_copy[f'{self.s_2}'] = self.signal_2
        self.data_copy[f'{self.s_3}'] = self.signal_3
        self.data_copy[f'{self.s_4}'] = self.signal_4
        
        # Convierte el DataFrame a un array de NumPy
        data_to_save = self.data_copy.to_numpy()

        # Crear una lista con las etiquetas de las columnas incluyendo las nuevas columnas de frecuencias de corte
        column_labels = ['TIME', 'CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                         'CH 10', 'CH 11', 'CH 12', 'TAG OUT', 'UP', 'DOWN', 'STIM 1', 'STIM 2', 'STIM 3',
                         'ACTIVE CYCLE',
                         'INACTIVE CYCLE', 'CYCLE TIME', 'FREQUENCY']

        # Escribe los datos en el archivo CSV
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_labels)  # Escribir las etiquetas de columna
            writer.writerows(data_to_save)  # Escribir las filas de datos

        # Lee el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(file_path)
        num_columns = len(df.columns)

        # Inserta una nueva columna 'DATA' llena de None
        df.insert(num_columns, f'DATA', None)

        # Asigna valores a la nueva columna 'DATA' con información de frecuencias de corte y estimulación
        df.at[0, 'DATA'] = f"cut off LP  {self.s_1}: {self.cutoff_1_LP}"
        df.at[1, 'DATA'] = f"cut off LP  {self.s_2}: {self.cutoff_2_LP}"
        df.at[2, 'DATA'] = f"cut off LP {self.s_3}: {self.cutoff_3_LP}"
        df.at[3, 'DATA'] = f"cut off LP {self.s_4}: {self.cutoff_4_LP}"


        # Guarda el DataFrame modificado de vuelta al mismo archivo CSV
        df.to_csv(file_path, index=False)

        print(f"Datos guardados en {file_path}")

    def analisis_Autocorrelation(self):
        """
        Realiza el análisis de autocorrelación de las señales filtradas, 
        genera gráficos de autocorrelación para cada señal, y guarda los resultados en un archivo CSV.

        La función calcula la autocorrelación para cada señal, genera gráficos con un formato personalizado
        (incluyendo colores y diseño del gráfico) y, opcionalmente, guarda los resultados de autocorrelación en un archivo CSV.

        Parameters:
        - None: Los parámetros necesarios se toman de los atributos del objeto (`self`), como `self.signals_filtradas`, 
        `self.subsample_factor`, `self.detrend`, `self.normalize`, `self.signals_labels`, entre otros.
        
        self.subsample_factor:

            Tipo: int
            Descripción: Factor de submuestreo que determina la frecuencia con la que se seleccionan las muestras de las señales originales. Si este valor es mayor que 1, la función tomará una muestra cada self.subsample_factor valores, reduciendo así la cantidad de datos analizados.
            Ejemplo: Si self.subsample_factor es 2, se seleccionará cada segunda muestra de las señales originales.
       
        self.detrend:

            Tipo: bool
            Descripción: Indicador que especifica si se debe eliminar la tendencia lineal de las señales antes de realizar el análisis de autocorrelación. Cuando está establecido en True, se aplica una eliminación de tendencia lineal para hacer que la señal oscile alrededor de cero y así facilitar el análisis de la autocorrelación.
            Ejemplo: Si self.detrend es True, se utilizará la función detrend de scipy.signal para eliminar la tendencia de la señal.
        
        self.normalize:

            Tipo: bool
            Descripción: Indicador que determina si se debe normalizar cada señal antes de realizar el análisis. Si está establecido en True, la señal será escalada para tener media 0 y desviación estándar 1, lo que facilita la comparación entre señales de diferente magnitud.
            Ejemplo: Si self.normalize es True, se normalizará la señal utilizando la media y desviación estándar de la misma.

        Returns:
        - None: Esta función no retorna ningún valor directamente, pero produce gráficos y guarda los resultados de autocorrelación 
        en un archivo CSV si se activa la opción `self.save_function_flag`.

        Output:
        - Gráficos de autocorrelación: Se muestran gráficos personalizados de la autocorrelación de cada señal.
        - Archivo CSV (opcional): Los resultados de la autocorrelación de las señales se guardan en un archivo CSV 
        si la opción `self.save_function_flag` está activada.
        """
    
        print("ANÁLISIS DE Autocorrelation EN CURSO...")
            

        #Configuración inicial
        detrend = self.detrend
        normalize = self.normalize
        subsample_factor = self.subsample_factor

        # Define las etiquetas de las señales
        
        self.signals_labels = [self.s_1, self.s_2, self.s_3, self.s_4]
        
        if self.analysis_on_window:
            # Combina las señales en una matriz
            self.signals_filtradas = np.column_stack((self.signal_1[self.start_a_num: self.end_a_num], 
                                                      self.signal_2[self.start_a_num: self.end_a_num], 
                                                      self.signal_3[self.start_a_num: self.end_a_num], 
                                                      self.signal_4[self.start_a_num: self.end_a_num]))
        else:
            # Combina las señales en una matriz
            self.signals_filtradas = np.column_stack((self.signal_1, self.signal_2, self.signal_3, self.signal_4))       
            

        # Combina las señales en una matriz
        self.signals_filtradas = np.column_stack((self.signal_1, self.signal_2, self.signal_3, self.signal_4))
        
        if subsample_factor > 1:
            self.signals_filtradas = self.signals_filtradas[::subsample_factor]
        # Reemplaza valores NaN en las señales combinadas con 0
        self.signals_filtradas = np.nan_to_num(self.signals_filtradas, nan=0.0)
        # Elimina la tendencia lineal si se especifica
        if detrend:
            self.signals_filtradas = signal.detrend(self.signals_filtradas, type='linear')
        # Normaliza la señal si se especifica
        if normalize:
            self.signals_filtradas = (self.signals_filtradas - self.signals_filtradas.mean(axis=0)) / self.signals_filtradas.std(axis=0)

        
        num_signals = self.signals_filtradas.shape[1]

            
        fig, axes = plt.subplots(num_signals, 1, figsize=(12, 8))
        N = len(self.signal_1)  # Número de muestras
        
        index_prom_tag = 0  # Inicializa el índice para las etiquetas de señales
        
        # Crear un DataFrame vacío para almacenar los resultados
        results = pd.DataFrame()


        for i in range(num_signals):
            x1 = self.signals_filtradas[:, i]
            autocorr = [pd.Series(x1).autocorr(lag) for lag in range(len(x1))]
            
            # Graficar la autocorrelación
            column = self.signals_labels[i]

            axes[i].set_xlabel('Lag')
            axes[i].set_ylabel('Autocorrelación')
            axes[i].grid(True)
            
            signal_color_t = getattr(self, f's_{index_prom_tag + 1}')  # Obtiene el color asignado a la señal actual
            axes[i].stem(range(len(autocorr)), autocorr,
                                            label=f'Autocorrelación de CH{index_prom_tag + 1}', linefmt=self.color_map[signal_color_t])

            # Configurar el título y apariencia del subgráfico
            axes[i].set_title(f'Autocorrelación de {column}', color='gray')
            axes[i].set_facecolor('none')  # Color de fondo transparente para el área del gráfico
            axes[i].spines['left'].set_color('none')  # Ocultar espina izquierda
            axes[i].spines['right'].set_color('none')  # Ocultar espina derecha
            axes[i].spines['top'].set_color('none')  # Ocultar espina superior
            axes[i].spines['bottom'].set_color('none')  # Ocultar espina inferior
            #axes[i].set_yticks([])  # Ocultar marcas del eje Y
                        
            if i < len(self.signals_labels) - 1:
                axes[i].set_xticks([])
                
            # Establecer el color de fondo de la figura principal como transparente
            fig.patch.set_facecolor('none')
            
            """# Guardar cada gráfica individualmente
            individual_fig, individual_ax = plt.subplots()
            individual_ax.set_xlabel('Lag')
            individual_ax.set_ylabel('Autocorrelación')
            individual_ax.set_title(f'Autocorrelación de {column}')
            individual_ax.grid(True)
            
            individual_ax.plt.stem(range(len(autocorr)), autocorr,
                                            label=f'CH{index_prom_tag + 1}', color=self.color_map[signal_color_t])
            
            # Incrementa el índice para pasar a la siguiente señal
            index_prom_tag += 1  

            # Configura el color de fondo del subgráfico
            individual_ax.set_facecolor('none')

            # Elimina el color de los bordes derecho y superior del subgráfico
            individual_ax.spines['right'].set_color('none')
            individual_ax.spines['top'].set_color('none')

            # Configura el color del borde inferior del subgráfico
            individual_ax.spines['bottom'].set_color('black')
            
            # Añadir leyenda
            individual_fig.legend()
            
            fig.patch.set_facecolor('black')
        
            individual_ax.set_facecolor('black')  # Cambia el color de fondo de cada gráfico individual a negro
            individual_ax.xaxis.label.set_color('gray')  # Cambia el color de las etiquetas del eje x a gris
            individual_ax.yaxis.label.set_color('gray')  # Cambia el color de las etiquetas del eje y a gris
            individual_ax.tick_params(axis='x', colors='gray')  # Cambia el color de las marcas del eje x a gris
            individual_ax.tick_params(axis='y', colors='gray')  # Cambia el color de las marcas del eje y a gris
            self.save_fig(f"{self.new_file_name}_CH{index_prom_tag + 1}_Autocorrelation")
            #individual_fig.savefig(f'Autocorrelation_{column}.svg', format='svg')
            plt.close(individual_fig)"""
            
            # Incrementa el índice para pasar a la siguiente señal
            index_prom_tag += 1  
            
            if self.save_function_flag:
                if results.empty:
                    # Si el DataFrame está vacío, agregar xf_filtered como la primera columna
                    results['Lag'] = range(len(autocorr))
                
                # Crear un DataFrame temporal con los resultados
                results[f'Autocorrelación {column}'] = autocorr
                
                # Establecer el color de fondo de la figura principal como transparente
        fig.patch.set_facecolor('none')

        # Configurar cada subgráfico individualmente
        for ax in axes.flatten():
            # Establecer el color de fondo de cada subgráfico como transparente
            ax.set_facecolor('none')

            # Ocultar las espinas (líneas de los bordes) de cada subgráfico
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Eliminar las marcas del eje Y para una apariencia más limpia
            #ax.set_yticks([])

            # Configurar la leyenda para no tener marco
            ax.legend(frameon=False)

        # Guardar la figura
        plt.tight_layout()
        self.save_fig(f"{self.new_file_name}_Autocorrelation")

        # Establecer el color de fondo de la figura principal como negro
        fig.patch.set_facecolor('black')

        # Configurar cada subgráfico para que tenga un fondo negro
        for ax in axes.flatten():
            ax.set_facecolor('black')

        # Configurar el color de los ejes y las etiquetas en gris para todos los subgráficos
        for ax in axes.flatten():
            # Establecer el color de las etiquetas del eje X e Y en gris
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')

            # Cambiar el color de los ticks del eje X a gris
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')

            # Configurar el fondo del subgráfico y las espinas del eje X en negro
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('gray')  # Establecer el color gris para el borde inferior

            # Configurar la leyenda para no tener marco y con texto en gris
            ax.legend(frameon=False, labelcolor='gray')

        # Ajustar el diseño de los gráficos para evitar superposición
        plt.tight_layout()

        # Mostrar la figura
        plt.show()

        if self.save_function_flag:
            # Guardar el DataFrame en un archivo CSV
            # Genera un timestamp y un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name = f"{self.new_file_name}_Autocorrelation"
            nombre_archivo_data = f"{name}_{timestamp}.csv"
            self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior
            # Ruta completa del archivo
            file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)
            results.to_csv(file_path, index=False)
            print(f"Datos guardados en {file_path}")


        
    def save_fig(self, name):
        """
        Guarda la figura actual en un archivo SVG con un nombre que incluye una marca de tiempo.

        Args:
            name (str): Nombre base para el archivo de la imagen.
        """

        # Obtener la marca de tiempo actual en el formato AAAAMMDDHHMMSS
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Crear el nombre del archivo incluyendo la marca de tiempo
        nombre_archivo_actualizado = f"{name}_{timestamp}.svg"

        # Construir la ruta completa para guardar la imagen en la carpeta especificada
        ruta_imagen_actualizada = os.path.join(self.carpeta_datos, nombre_archivo_actualizado)

        # Guardar la figura actual como un archivo SVG en la ruta especificada
        plt.savefig(ruta_imagen_actualizada, format='svg')

        # Imprimir un mensaje indicando que el archivo ha sido guardado
        print(f"ARCHIVO GUARDADO {nombre_archivo_actualizado}")

    def actualizar_grafico(self):
        """
        Actualiza los gráficos aplicando filtros de paso bajo y paso alto a las señales.

        Filtra las señales originales usando los cortes y filtros especificados, y actualiza
        los gráficos con las señales filtradas.
        """

        # Listas de valores de corte y filtros para paso bajo (LP) y paso alto (HP)
        cutoffs_LP = [self.cutoff_1_LP, self.cutoff_2_LP, self.cutoff_3_LP, self.cutoff_4_LP]
        cutoffs_HP = [self.cutoff_1_HP, self.cutoff_2_HP, self.cutoff_3_HP, self.cutoff_4_HP]
        filters_LP = [self.filter_1_LP, self.filter_2_LP, self.filter_3_LP, self.filter_4_LP]
        filters_HP = [self.filter_1_HP, self.filter_2_HP, self.filter_3_HP, self.filter_4_HP]

        # Aplicar filtro de paso bajo a las señales, si está habilitado
        for i in range(4):
            if filters_LP[i]:
                # Filtra la señal original usando el valor de corte de paso bajo
                filtered_signal = self.filter_LP(getattr(self, f'original_{i + 1}'), cutoffs_LP[i])
                # Actualiza la señal filtrada en la matriz de señales
                self.signals[:, i] = filtered_signal
                # Actualiza la señal filtrada en el objeto
                setattr(self, f'signal_{i + 1}', filtered_signal)
                # Desactiva el filtro de paso bajo
                setattr(self, f'filter_{i + 1}_LP', False)

        # Aplicar filtro de paso alto a las señales, si está habilitado
        for i in range(4):
            if filters_HP[i]:
                # Filtra la señal filtrada de paso bajo usando el valor de corte de paso alto
                filtered_signal = self.filter_HP(self.signals[:, i], cutoffs_HP[i])
                # Actualiza la señal filtrada en la matriz de señales
                self.signals[:, i] = filtered_signal
                # Actualiza la señal filtrada en el objeto
                setattr(self, f'signal_{i + 1}', filtered_signal)
                # Desactiva el filtro de paso alto
                setattr(self, f'filter_{i + 1}_HP', False)

        # Llamar a la función para actualizar los gráficos con las nuevas señales filtradas
        self.update_plots()

    def mover_ventana(self):
        """Actualiza los límites de la ventana de tiempo para el análisis de señales, ya sea avanzando,
        retrocediendo o manteniendo la ventana actual.

        Dependiendo de los estados `avanza`, `retrocede` o `window`, se ajustan `tiempo_inicial`
        y `tiempo_final` para definir la nueva ventana de tiempo. Luego, se calculan los índices
        correspondientes a estos tiempos y se actualizan los gráficos."""

        # Si se avanza la ventana de tiempo
        if self.avanza:
            self.tiempo_inicial = self.tiempo_inicial + self.window_time  # Incrementa el tiempo inicial
            self.tiempo_final = self.tiempo_inicial + self.window_time  # Actualiza el tiempo final

        # Si se retrocede la ventana de tiempo
        if self.retrocede:
            self.tiempo_final = self.tiempo_inicial  # El tiempo final se convierte en el tiempo inicial actual
            self.tiempo_inicial = self.tiempo_final - self.window_time  # Se calcula el nuevo tiempo inicial restando la duración de la ventana

        # Si se mantiene la ventana de tiempo actual
        if self.window:
            self.tiempo_final = self.window_time + self.tiempo_inicial  # El tiempo final se actualiza sumando la duración de la ventana al tiempo inicial

        # Calcular los índices correspondientes a los tiempos inicial y final
        self.start_a_num = round(self.tiempo_inicial * self.datos_por_s)  # Índice inicial
        self.end_a_num = round(self.tiempo_final * self.datos_por_s)  # Índice final

        # Actualizar los gráficos con los nuevos índices de tiempo
        self.update_plots()

    def update_plots(self):
        """Actualiza los gráficos de las señales y las estimaciones en los diferentes widgets de trazado.

            La función actualiza las gráficas en cuatro widgets de trazado (`plot_widget_1`, `plot_widget_2`,
            `plot_widget_3`, `plot_widget_4`) mostrando señales originales y datos de estimulación/tag.
            Cada gráfico se actualiza para el rango de tiempo definido por `start_a_num` y `end_a_num`.

            1. Actualiza las señales y datos de estimulación/tag para diferentes gráficos.
            2. Limpia los widgets de trazado antes de añadir nuevos datos.
            3. Configura etiquetas, títulos y rangos de los ejes para cada gráfico.
            """

        # Actualizar gráfico 1
        self.plot_widget_1.clear()  # Limpiar el widget de trazado
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.signal_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_1], width=self.width_line), name='Señal original')

        self.plot_widget_1.setLabel('left', 'Amplitude')
        self.plot_widget_1.setLabel('bottom', 'Time[s]')
        self.plot_widget_1.setTitle(f'{self.s_1}')
        self.plot_widget_1.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)

        # ------------------------------------------------------------------------------------- signal 2
        # Actualizar gráfico 2
        self.plot_widget_2.clear()  # Limpiar el widget de trazado
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.signal_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_2], width=self.width_line), name='Señal original')

        self.plot_widget_2.setLabel('left', 'Amplitude')
        self.plot_widget_2.setLabel('bottom', 'Time[s]')
        self.plot_widget_2.setTitle(f'{self.s_2}')
        self.plot_widget_2.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)

        # -------------------------------------------------------------------------------------
        # Actualizar gráfico 3
        self.plot_widget_3.clear()  # Limpiar el widget de trazado
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.signal_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_3], width=self.width_line), name='Señal original')

        self.plot_widget_3.setLabel('left', 'Amplitude')
        self.plot_widget_3.setLabel('bottom', 'Time[s]')
        self.plot_widget_3.setTitle(f'{self.s_3}')
        self.plot_widget_3.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_3.setYRange(self.min_amplitude_3, self.max_amplitude_3)

        # -------------------------------------------------------------------------------------
        # Actualizar gráfico 4
        self.plot_widget_4.clear()  # Limpiar el widget de trazado
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.signal_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_4], width=self.width_line), name='Señal original')


        self.plot_widget_4.setLabel('left', 'Amplitude')
        self.plot_widget_4.setLabel('bottom', 'Time[s]')
        self.plot_widget_4.setTitle(f'{self.s_4}')
        self.plot_widget_4.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_4.setYRange(self.min_amplitude_4, self.max_amplitude_4)

        # Imprimir el tamaño de los datos para guardar
        print("size data to save", self.data_copy.to_numpy().shape)


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