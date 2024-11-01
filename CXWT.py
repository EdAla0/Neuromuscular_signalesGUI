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
from scipy.signal import lfilter, butter, convolve2d
from scipy import signal
import pywt
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter, gaussian_filter

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
        loadUi('CXWT.ui', self)

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

        # Configura las opciones para el menú desplegable de canales
        options = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                   'CH 10', 'CH 11', 'CH 12']

        # Crea un diccionario que mapea cada opción del menú con su color correspondiente
        self.channel_color_map = dict(zip(options, channel_colors))

        # Configura los comboboxes para seleccionar las columnas de datos
        comboboxes = [
            self.column_combobox_1, self.column_combobox_2
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

        # Configura el QSpinBox para el tiempo de la ventana de visualización
        self.window_time.valueChanged.connect(self.update_window_time)  # Conecta al método de actualización

        # Conecta los QSpinBox de los filtros con sus funciones correspondientes para actualizar los filtros
        spinboxes = [
            (self.cutoff_s1_LP, self.actualizar_LP_fc_1), (self.cutoff_s2_LP, self.actualizar_LP_fc_2),
            (self.cutoff_s1_HP, self.actualizar_HP_fc_1), (self.cutoff_s2_HP, self.actualizar_HP_fc_2)
        ]
        for spinbox, function in spinboxes:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor con la función correspondiente

        # Conecta los botones de filtro con las funciones correspondientes para aplicar los filtros
        filter_buttons = [
            (self.btn_filter_1, self.update_filter_1), (self.btn_filter_2, self.update_filter_2)
        ]
        for button, function in filter_buttons:
            button.clicked.connect(function)  # Conecta el clic del botón con la función correspondiente

        # Configura el botón para guardar
        self.save_function.clicked.connect(self.save_active)

        # Configura las opciones para la mother Wavelet.
        mothers = ['morl', 'fbsp7-1.5-1.0', 'cmor1.5-1.0', 'cmor2.5-0.7', "shan1.5-1.0", "mexh"]
        self.stim_colors = stim_colors
        self.mother_options.addItems(mothers)  # Añade las opciones al combobox de mother Wavelets.
        self.mother_options.currentIndexChanged.connect(
            self.update_mother)  # Conecta el cambio de selección con la función `update_mother`

        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        self.name_file.textChanged.connect(self.name_file_update)

        # Configura los botones para avanzar y retroceder en las gráficas
        self.btn_avanzar.clicked.connect(self.avanzar_graficas)
        self.btn_retroceder.clicked.connect(self.retroceder_graficas)

        # Conecta los cambios en los QDoubleSpinBox para el tiempo antes y después del análisis
        """self.t_before.valueChanged.connect(self.t_before_update)
        self.t_after.valueChanged.connect(self.t_after_update)"""

        # Configura el botón para realizar el análisis de CXWT
        self.bt_go.clicked.connect(self.CXWT)

        # Configura un diccionario para asociar los PlotWidgets con sus títulos
        self.plot_widgets = {
            self.plot_widget_1: "Signal 1",
            self.plot_widget_2: "Signal 2"
        }

        # Conecta el evento de clic del mouse en los PlotWidgets a la función correspondiente
        """self.plot_widget_1.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_2.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_3.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_widget_4.scene().sigMouseClicked.connect(self.on_plot_clicked)"""

        # Inicializa la ruta del archivo y carga los datos
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)  # Extrae la carpeta del archivo de datos
        self.data = pd.read_csv(self.ruta_archivo)  # Lee los datos del archivo CSV
        self.data_copy = pd.read_csv(self.ruta_archivo)  # Crea una copia de los datos
        self.time = self.data['TIME'].to_numpy()  # Extrae el tiempo de los datos

        # Inicializa variables de estado para el análisis
        self.crossing_detected = False
        self.crossing_hysteresis_detected = False
        self.crossing_descendente = False
        self.first_up = False
        self.first_down = False
        self.hysteresis = float(values[4])  # Configura el valor de histeresis
        self.s_1 = 'CH 1'  # Columna seleccionada para Signal 1
        self.s_2 = 'CH 1'  # Columna seleccionada para Signal 2
        self.signal_1 = self.data[f'{self.s_1}'].to_numpy()  # Extrae la señal 1
        self.signal_2 = self.data[f'{self.s_2}'].to_numpy()  # Extrae la señal 2

        # Configura las señales y sus envolventes
        self.signals = self.data[
            [f'{self.s_1}', f'{self.s_2}', f'{self.s_1}', f'{self.s_2}']].to_numpy()
        self.original_1 = self.signals[:, 2]
        self.original_2 = self.signals[:, 3]
        
        # Configura el checkbox para detrend.
        self.detrend_button.stateChanged.connect(self.detrend_active)
        # Configura el checkbox para normalización.
        self.normalize_button.stateChanged.connect(self.normalize_active)
        # Configura el QSpinBox para el número de notas nNotes.
        self.nNotes.valueChanged.connect(self.update_nNotes)  # Conecta al método de actualización
        # Configura el QSpinBox para el sub muestreo.
        self.subsample_factor.valueChanged.connect(self.update_subsample_factor)  # Conecta al método de actualización
        
        # Configura el botón para aplicar el análisis unicamente en los datos mostrados en ventana.
        self.analysis_on_window_function.stateChanged.connect(self.analysis_on_window_active)

        # Configura los parámetros del análisis
        self.nNotes = 24 #nNotes (int): Número de divisiones por octava en la escala logarítmica.
        self.detrend = True #detrend (bool): Si True, elimina la tendencia lineal de las señales.
        self.normalize = True #normalize (bool): Si True, normaliza las señales.
        self.subsample_factor = 1000 #subsample_factor (int): Se tomará solo cada enésimo valor de la señal.
        self.mother = 'morl' #mother: La mother wavelet que se usará. Más información sobre cada una en: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html

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
            self.s_2: self.channel_color_map[self.s_2]
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
        self.new_file_name = "CXWT"

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
            
    def update_nNotes(self, value):
        """Actualiza el número de notas nNotes a usar en la transformada wavelet.
        Parámetros:
            value (int): Nuevo valor de nNotes."""
        self.nNotes = value  # Actualiza nNotes
        
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
        
    def update_mother(self):
        """Actualiza la selección de la mother wavelet y la configuración de color para el gráfico.

        Esta función obtiene la mother wavelet seleccionada actualmente del menú desplegable, actualiza la variable
        `mother` 

        - Obtiene el texto seleccionado del menú desplegable `mothers_options`.
        - Actualiza el atributo `mother`."""
        # Obtiene el estímulo seleccionado del menú desplegable
        self.mother = self.mother_options.currentText()

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

    def t_after_update(self, value):
        """Actualiza el tiempo después del evento.

        Parámetros:
            value (float): Nuevo valor para el tiempo después del evento."""
        self.after = value  # Actualiza el tiempo después del evento con el nuevo valor

    def t_before_update(self, value):
        """Actualiza el tiempo antes del evento.

        Parámetros:
            value (float): Nuevo valor para el tiempo antes del evento."""
        self.before = value  # Actualiza el tiempo antes del evento con el nuevo valor

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

        # Actualiza las señales del canal 1 si ha cambiado la selección
        if new_s_1 != self.s_1:
            self.s_1 = new_s_1
            self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
            self.signals[:, 0] = self.signal_1
            self.s_5 = new_s_1
            self.original_1 = self.data[f'{self.s_5}'].to_numpy()
            self.signals[:, 2] = self.original_1

        # Actualiza las señales del canal 2 si ha cambiado la selección
        if new_s_2 != self.s_2:
            self.s_2 = new_s_2
            self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
            self.signals[:, 1] = self.signal_2
            self.s_6 = new_s_2
            self.original_2 = self.data[f'{self.s_6}'].to_numpy()
            self.signals[:, 3] = self.original_2

        # Actualiza el mapa de colores para los canales
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2]
        }

        # Llama a la función para actualizar las gráficas
        self.update_plots()


    def CXWT(self):
        print("ANÁLISIS DE CXWT EN CURSO...")

        # Configuración inicial
        nNotes = self.nNotes
        detrend = self.detrend
        normalize = self.normalize
        subsample_factor = self.subsample_factor
        mother = self.mother
        # Define las etiquetas de las señales
        self.signals_labels = [self.s_1, self.s_2]

        if self.analysis_on_window:
            times = self.time[self.start_a_num: self.end_a_num]
            # Reemplaza valores NaN en las señales combinadas con 0
            x1 = np.nan_to_num(self.signal_1[self.start_a_num: self.end_a_num], nan=0.0)
            x2 = np.nan_to_num(self.signal_2[self.start_a_num: self.end_a_num], nan=0.0)
        else:
            times = self.time
            # Reemplaza valores NaN en las señales combinadas con 0
            x1 = np.nan_to_num(self.signal_1, nan=0.0)
            x2 = np.nan_to_num(self.signal_2, nan=0.0)

        if subsample_factor > 1:
            x1 = x1[::subsample_factor]
            x2 = x2[::subsample_factor]
            times = times[::subsample_factor]
        N = len(x1)

        dt = times[1] - times[0]  # Calcula el intervalo de tiempo
        

        # Elimina la tendencia lineal si se especifica
        if detrend:
            x1 = signal.detrend(x1, type='linear')
            x2 = signal.detrend(x2, type='linear')
        if normalize:
            stddev1 = x1.std()
            x1 = x1 / stddev1
            stddev2 = x2.std()
            x2 = x2 / stddev2

        # Calcula las escalas para la transformada wavelet
        dj=1.0 / nNotes
        nOctaves = int(np.log2(2 * np.floor(N / 2.0)))
        scales = 2**np.arange(1, nOctaves, dj)
        #sigma_t = 2
        #sigma_f = 20
        frequencies = pywt.scale2frequency(mother, scales) / dt

        # Define una función para calcular la CXWT
        def calculate_cwt(signal, scales):
            coef, _ = pywt.cwt(signal, scales, mother, dt)
            return coef

        # Aplica la transformada wavelet continua en paralelo
        coefs = Parallel(n_jobs=-1)(delayed(calculate_cwt)(signal, scales) for signal in [x1, x2])

        coef1, coef2 = coefs

        frequencies = pywt.scale2frequency(mother, scales) / dt

        coef12 = (coef1 * np.conj(coef2))
        scaleMatrix = np.ones([1, N]) * scales[:, None]
        
        def smoothwavelet(wave, dt, dj, scale):
            """
            Suavizado como en el apéndice de Torrence y Webster "Inter decadal changes in the ENSO-Monsoon System" 1998.
            Usado en cálculos de coherencia de wavelet.
            Solo aplicable para la wavelet Morlet.
            
            Args:
                wave (np.ndarray): La matriz de wavelets.
                dt (float): El intervalo de tiempo.
                dj (float): El paso de la escala logarítmica.
                scale (np.ndarray): El vector de escalas.
                
            Returns:
                np.ndarray: La matriz de wavelets suavizada.
            """
            n = wave.shape[1]
            twave = np.zeros_like(wave)

            npad = int(2 ** np.ceil(np.log2(n)))

            k = np.arange(1, npad // 2 + 1)
            k = k * ((2. * np.pi) / npad)
            k=np.concatenate(([0.], k, -k[(npad - 1) // 2:0:-1]))

            k2 = k**2
            snorm = scale / dt
            for ii in range(wave.shape[0]):
                F = np.exp(-0.5 * (snorm[ii]**2) * k2)  # Suavizado en el dominio del tiempo
                smooth = np.fft.ifft(F * np.fft.fft(wave[ii, :], npad))
                twave[ii, :] = smooth[:n]

            if np.isrealobj(wave):
                twave = np.real(twave)  # Hack para datos reales

            dj0 = 0.6
            dj0steps = dj0 / (dj * 2)

            part1 = np.array([[np.mod(dj0steps, 1)]])
            part2 = np.ones((2 * round(dj0steps) - 1, 1))
            part3 = np.array([[np.mod(dj0steps, 1)]])

            # Concatenar las partes del kernel
            kernel = np.vstack((part1, part2, part3))

            # Normalizar el kernel
            kernel /= (2 * round(dj0steps) - 1 + 2 * np.mod(dj0steps, 1))

            # Aplicar la convolución 2D
            swave = convolve2d(twave, kernel, mode='same') # Convolución en el dominio del tiempo"""
            return swave
        
        # Aplicar suavizado
        S1 = smoothwavelet((np.abs(coef1) **2) / scaleMatrix, dt,dj , scales)
        S2 = smoothwavelet((np.abs(coef2) **2) / scaleMatrix, dt, dj, scales)
        S12 = smoothwavelet((coef12 / scaleMatrix), dt, dj, scales)
        CXWT = np.abs(coef12)*(np.abs(S12)**2) / (S1 * S2)
        """ #Aplica filtro uniforme
        S1 = uniform_filter((np.abs(coef1)**2 / scaleMatrix), size=10) 
        S2 = uniform_filter((np.abs(coef2)**2 / scaleMatrix), size=10)
        S12 = uniform_filter((np.abs(coef12 / scaleMatrix)), size=10)
        CXWT = CXWT*(S12**2) / (S1 * S2)
        # Aplica un filtro gaussiawno para suavizar
        #un sigma mayor suavizará más en el eje correspondiente. Un sigma menor dará una resolución mayor en el eje correspondiente.
        S1 = gaussian_filter(((np.abs(coef1)**2) / scaleMatrix), sigma=[sigma_t,sigma_f]) 
        S2 = gaussian_filter(((np.abs(coef2)**2) / scaleMatrix), sigma=[sigma_t,sigma_f])
        S12 = gaussian_filter((np.abs(coef12 / scaleMatrix)), sigma=[sigma_t,sigma_f])
        CXWT = CXWT*(S12**2) / (S1 * S2) """

        # Calcula el cono de influencia
        f0 = 2 * np.pi
        cmor_coi = 1.0 / np.sqrt(2)
        cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = cmor_flambda * cmor_coi * dt * coi
        coif = 1.0 / coi

        def summarize_CXWT(CXWT, frequencies, times):
            """
            Proporciona un resumen numérico de la cohrrelación cruzada de wavelets.

            Parameters:
            - correlation (2D array): Coherencia de wavelets.


            Returns:
            - summary (dict): Resumen de coherencia de wavelets con valores promedio y frecuencia dominante.
            """
            avg_CXWT = np.mean(CXWT)
            max_CXWT_freq_index = np.argmax(np.mean(CXWT, axis=1))
            max_CXWT_freq = frequencies[max_CXWT_freq_index]
            
            summary = {
                'average_CXWT': avg_CXWT,
                'dominant_frequency': max_CXWT_freq,
                'CXWT_matrix': CXWT
            }

            return summary
        CXWT_summary = summarize_CXWT(CXWT, frequencies, times)
        print(CXWT_summary)

        # Imprimir el resumen
        print("Resumen de CXWT:")
        print(f"Promedio de CXWT: {CXWT_summary['average_CXWT']:.4f}")
        print(f"Frecuencia Dominante: {CXWT_summary['dominant_frequency']:.4f} Hz")
        # Graficar la correlación cruzada y guardar la figura como SVG
        fig, ax = plt.subplots()
        # Calcula la fase de la CXWT
        phase_data = np.angle(CXWT)
        phase_data[CXWT <= 0.90] = np.nan

        # Define la densidad de muestreo para las flechas de fase (ajusta según sea necesario)
        arrow_density = (50, 50)  # Por ejemplo, cada 10 puntos en tiempo y frecuencia
        
        def spectrogram_plot(z, times, frequencies, coif, cmap=None, norm=Normalize(), ax=None, colorbar=True, phase_data=None, arrow_density=(10, 10)):
            """
            Grafica el espectrograma de coherencia cruzada.
            
            Args:
                z (ndarray): Matriz de coherencia cruzada.
                times (ndarray): Vector de tiempos.
                frequencies (ndarray): Vector de frecuencias.
                coif (ndarray): Cono de influencia.
                cmap (str or Colormap): Mapa de colores para la gráfica.
                norm (Normalize): Normalización para los colores.
                ax (Axes): Eje sobre el cual dibujar la gráfica.
                colorbar (bool): Si True, añade una barra de colores a la gráfica.
                savefig (str or None): Nombre del archivo para guardar la figura. Si es None, no guarda la figura.

            Returns:
                Axes: Eje con la gráfica del espectrograma.
            """
            if cmap is None:
                cmap = get_cmap('Greys')
            elif isinstance(cmap, str):
                cmap = get_cmap(cmap)

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()

            # Crea la malla para la gráfica
            xx, yy = np.meshgrid(times, frequencies)
            ZZ = z
            
            # Dibuja el espectrograma
            im = ax.pcolor(xx, yy, ZZ, norm=norm, cmap=cmap)
            ax.contour(xx, yy, ZZ, levels=[0.9], colors='k', linestyles='dashed')
            ax.plot(times, coif)
            ax.fill_between(times, coif, step="mid", alpha=0.4)
            
            # Añade la barra de colores si se especifica
            if colorbar:
                # Definir los ejes de la colorbar con un ancho ajustado
                cbaxes = inset_axes(ax, width="2%", height="90%", loc=4, bbox_to_anchor=(0, 0.1, 0.961, 0.9),  bbox_transform=fig.transFigure)#bbox_transform=ax.transAxes) 

                # Crear la colorbar
                cbar = fig.colorbar(im, cax=cbaxes, orientation='vertical')
                # Cambiar el color de los números del colorbar a gris
                cbar.ax.yaxis.set_tick_params(color='gray')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='gray')
                
            ax.set_xlim(times.min(), times.max())
            ax.set_ylim(frequencies.min(), min(frequencies.max(), 2.5))
            phase_data[z <= 0.90] = np.nan
            if phase_data is not None:
                arrow_size=30*.03/np.mean(arrow_density)
                arrow_head_size=120/np.mean(arrow_density)
                phs_dt = max(round(len(times) / arrow_density[0]), 1)
                phs_dp = max(round(len(frequencies) / arrow_density[1]), 1)
                tidx = np.arange(max(phs_dt // 2, 0), len(times), phs_dt)
                pidx = np.arange(max(phs_dp // 2, 0), len(frequencies), phs_dp)

                X, Y = np.meshgrid(times[tidx], frequencies[pidx])
                U = arrow_size * np.cos(phase_data[pidx][:, tidx])
                V = arrow_size * np.sin(phase_data[pidx][:, tidx])
                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, headwidth=arrow_head_size)

            return ax
        
        spectrogram_plot(CXWT, times, frequencies, coif, cmap='turbo',norm = Normalize(), ax=ax, phase_data=phase_data, arrow_density=arrow_density)
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Frecuencia (Hz)')
        ax.set_title('CXWT', color='gray')

        # Configurar la apariencia del subgráfico
        ax.set_facecolor('none')  # Color de fondo transparente para el área del gráfico
        ax.spines['left'].set_color('none')  # Ocultar espina izquierda
        ax.spines['right'].set_color('none')  # Ocultar espina derecha
        ax.spines['top'].set_color('none')  # Ocultar espina superior
        ax.spines['bottom'].set_color('none')  # Ocultar espina inferior
        #ax.set_xticks([])
        ax.legend(frameon=False, labelcolor='gray')
        ax.grid(color = "gray")

                
        # Establecer el color de fondo de la figura principal como transparente
        fig.patch.set_facecolor('none')

        # Guarda la figura del análisis de pearsons_correlation
        plt.tight_layout()
        self.save_fig(f"{self.new_file_name}_CXWT")

        # Establecer el color de fondo de la figura principal como negro
        fig.patch.set_facecolor('black')

        # Configurar cada subgráfico para que tenga un fondo negro
        ax.set_facecolor('black')

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
            
            # Crear un DataFrame con la matriz de cross coherence wavelet
            results = pd.DataFrame(CXWT, index=frequencies, columns=times)
            # Añadir las frecuencias como una nueva columna en el DataFrame
            results.insert(0, 'Frequencies', frequencies)
            
            # Inserta una nueva columna 'DATA' llena de None
            results.insert(len(results.columns), f'DATA', None)

            # Asigna valores a la nueva columna 'DATA' con información de frecuencias de corte y estimulación
            results.at[0, 'DATA'] = f"cut off LP  {self.s_1}: {self.cutoff_1_LP}"
            results.at[1, 'DATA'] = f"cut off LP  {self.s_2}: {self.cutoff_2_LP}"
            results.at[2, 'DATA'] = f"Number of notes: {nNotes}"
            results.at[3, 'DATA'] = f"Detrend: {detrend}"
            results.at[4, 'DATA'] = f"Normalization: {normalize}"
            results.at[5, 'DATA'] = f"Mother wavelet: {mother}"
            # Genera un timestamp y un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name = f"{self.new_file_name}_CXWT_{self.s_1} - {self.s_2}"
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
        cutoffs_LP = [self.cutoff_1_LP, self.cutoff_2_LP]
        cutoffs_HP = [self.cutoff_1_HP, self.cutoff_2_HP]
        filters_LP = [self.filter_1_LP, self.filter_2_LP]
        filters_HP = [self.filter_1_HP, self.filter_2_HP]

        # Aplicar filtro de paso bajo a las señales, si está habilitado
        for i in range(2):
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
        for i in range(2):
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