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
        loadUi('BINS_GUI_con_HP.ui', self)

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

        # Configura las opciones para los estímulos
        stims = ['STIM 1', 'STIM 2', 'STIM 3']
        self.stim_colors = stim_colors
        self.stims_options.addItems(stims)  # Añade las opciones al combobox de estímulos
        self.stims_options.currentIndexChanged.connect(
            self.update_stim)  # Conecta el cambio de selección con la función `update_stim`
        self.stim_color_map = dict(
            zip(stims, self.stim_colors))  # Crea un diccionario que mapea cada estímulo con su color

        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        self.name_file.textChanged.connect(self.name_file_update)

        # Configura los botones para avanzar y retroceder en las gráficas
        self.btn_avanzar.clicked.connect(self.avanzar_graficas)
        self.btn_retroceder.clicked.connect(self.retroceder_graficas)

        # Conecta los cambios en los QDoubleSpinBox para el tiempo antes y después del análisis
        self.t_before.valueChanged.connect(self.t_before_update)
        self.t_after.valueChanged.connect(self.t_after_update)

        # Configura el botón para realizar el análisis de bins
        self.bt_go.clicked.connect(self.analisis_bins)

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
        self.stim_plot_1 = self.data['STIM 1'].to_numpy()  # Extrae el primer estímulo
        self.stim_plot_2 = self.data['STIM 2'].to_numpy()  # Extrae el segundo estímulo
        self.stim_plot_3 = self.data['STIM 3'].to_numpy()  # Extrae el tercer estímulo
        self.new_stim = 'STIM 1'
        self.stim = self.data[self.new_stim].to_numpy()  # Extrae el estímulo seleccionado

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
        self.stim_map = {
            self.new_stim: self.stim_color_map[self.new_stim],
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
        self.before = 1
        self.after = 1
        self.new_file_name = "BINS"

        # Actualiza el gráfico inicial
        self.actualizar_grafico()

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
    def name_file_update(self, new_file_name):
        """Actualiza el nombre del archivo de análisis."""
        self.new_file_name = new_file_name

    def update_stim(self):
        """Actualiza la selección de estímulo y la configuración de color para el gráfico.

        Esta función obtiene el estímulo seleccionado actualmente del menú desplegable, actualiza la variable
        `new_stim` y ajusta el atributo `stim` con los datos correspondientes del estímulo seleccionado.
        También actualiza el mapeo de colores para el estímulo y actualiza los gráficos.

        - Obtiene el texto seleccionado del menú desplegable `stims_options`.
        - Actualiza el atributo `stim` con los datos del estímulo seleccionado.
        - Actualiza el mapeo de colores (`stim_map`) para reflejar el nuevo estímulo.
        - Llama a `update_plots()` para actualizar la visualización gráfica en base a la nueva configuración."""
        # Obtiene el estímulo seleccionado del menú desplegable
        self.new_stim = self.stims_options.currentText()
        # Actualiza los datos del estímulo con los datos correspondientes del nuevo estímulo
        self.stim = self.data[self.new_stim].to_numpy()

        # Actualiza el mapeo de colores para el estímulo
        self.stim_map = {
            self.new_stim: self.stim_color_map[self.new_stim],
        }
        # Llama a la función para actualizar los gráficos con la nueva configuración
        self.update_plots()

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
        self.data_copy[f'{self.new_stim}'] = self.stim

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
        df.at[4, 'DATA'] = self.new_stim

        # Guarda el DataFrame modificado de vuelta al mismo archivo CSV
        df.to_csv(file_path, index=False)

        print(f"Datos guardados en {file_path}")

    def analisis_bins(self):
        print("ANÁLISIS DE BINS EN CURSO...")
        # Paso 1: Guarda los datos actuales si la bandera 'save_function_flag' está activada
        if self.save_function_flag:
            # Guarda los datos actuales y los inserta en un archivo con nombre especificado
            self.save_and_insert_data(f"{self.new_file_name}_BINS_DATA")

        # Paso 2: Configuración inicial
        # Define las etiquetas de las señales
        self.signals_labels = [self.s_1, self.s_2, self.s_3, self.s_4]
        # Define el tiempo antes y después del evento de estimulación
        s_antes = self.before
        s_despues = self.after

        # Combina las señales en una matriz
        self.signals_filtradas = np.column_stack((self.signal_1, self.signal_2, self.signal_3, self.signal_4))
        # Reemplaza valores NaN en las señales combinadas con 0
        self.signals_filtradas = np.nan_to_num(self.signals_filtradas, nan=0.0)

        # Paso 3: Normalización de las señales
        # Inicializa una matriz para las señales normalizadas
        self.normalized_signal = np.zeros_like(self.signals_filtradas)
        max_values = []  # Lista para almacenar los valores máximos normalizados
        min_values = []  # Lista para almacenar los valores mínimos normalizados

        for i in range(self.signals_filtradas.shape[1]):
            # Obtiene el valor máximo de la señal actual (ignorando NaNs)
            max_val = np.nanmax(self.signals_filtradas[:, i])
            # Obtiene el valor mínimo para la señal actual
            min_val = getattr(self, f'min_amplitude_{i + 1}')
            # Calcula los valores normalizados para el máximo y el mínimo
            y_max_norm = ((getattr(self, f'max_amplitude_{i + 1}')) - min_val) / (max_val - min_val)
            y_min_norm = ((getattr(self, f'min_amplitude_{i + 1}')) - min_val) / (max_val - min_val)

            max_values.append(y_max_norm)  # Almacena el valor máximo normalizado
            min_values.append(y_min_norm)  # Almacena el valor mínimo normalizado

            # Verifica si max_val o min_val son NaN
            if np.isnan(max_val) or np.isnan(min_val):
                raise ValueError(
                    f"max_val o min_val son NaN para la señal {i + 1}. max_val: {max_val}, min_val: {min_val}")

            # Normaliza la señal si min_val no es igual a max_val
            if min_val != max_val:
                self.normalized_signal[:, i] = (self.signals_filtradas[:, i] - min_val) / (max_val - min_val)
            else:
                self.normalized_signal[:, i] = self.signals_filtradas[:,
                                               i]  # Asigna la señal original si max_val == min_val

        # Paso 4: Extracción y procesamiento de datos alrededor de eventos de estimulación tipo 1
        datos_stim_1 = []
        ref_in_window_1 = []  # Lista para almacenar las referencias en la ventana de análisis

        # Itera sobre los índices donde la estimulación es igual a 1
        for indice in np.where(self.stim == 1)[0]:
            referencia = indice
            # Define el rango de datos antes y después del evento
            datos_before = referencia - round(s_antes * self.datos_por_s)
            datos_after = referencia + round(s_despues * self.datos_por_s)
            ref_in_window_1.append(referencia - datos_before)

            # Verifica que el rango de datos esté dentro de los límites válidos
            if datos_before < 0 or datos_after > len(self.time) - 1:
                continue
            # Almacena los datos segmentados para el evento de estimulación
            datos_stim_1.append(self.normalized_signal[datos_before:datos_after, :])

        datos_stim_1 = np.array(datos_stim_1)

        if datos_stim_1.size > 0 and not np.isnan(datos_stim_1).any():
            # Calcula el promedio de las señales para eventos tipo 1
            promedio_stim_1 = np.mean(datos_stim_1, axis=0)
            prom_ref_1 = round(np.mean(ref_in_window_1))  # Promedio de las referencias
            t_tag = np.linspace(-s_antes, s_despues, len(promedio_stim_1))  # Eje temporal
        else:
            promedio_stim_1 = np.array([])
            print("promedio_stim_1 está vacío o contiene valores NaN")

        # Paso 5: Extracción y procesamiento de datos alrededor de eventos de estimulación tipo 2
        datos_stim_2 = []
        ref_in_window_2 = []  # Lista para almacenar las referencias en la ventana de análisis

        # Itera sobre los índices donde la estimulación es igual a 2
        for indice in np.where(self.stim == 2)[0]:
            referencia = indice
            # Define el rango de datos antes y después del evento
            datos_before = referencia - round(s_antes * self.datos_por_s)
            datos_after = referencia + round(s_despues * self.datos_por_s)
            ref_in_window_2.append(referencia - datos_before)

            # Verifica que el rango de datos esté dentro de los límites válidos
            if datos_before < 0 or datos_after > len(self.time) - 1:
                continue
            # Almacena los datos segmentados para el evento de estimulación
            datos_stim_2.append(self.normalized_signal[datos_before:datos_after, :])

        datos_stim_2 = np.array(datos_stim_2)

        if datos_stim_2.size > 0 and not np.isnan(datos_stim_2).any():
            # Calcula el promedio de las señales para eventos tipo 2
            promedio_stim_2 = np.mean(datos_stim_2, axis=0)
            prom_ref_2 = round(np.mean(ref_in_window_2))  # Promedio de las referencias
            t_stim = np.linspace(-s_antes, s_despues, len(promedio_stim_2))  # Eje temporal
        else:
            promedio_stim_2 = np.array([])
            print("promedio_stim_2 está vacío o contiene valores NaN")

        # Paso 6: Creación y configuración de gráficos
        # Crea una figura con 2 filas y 4 columnas de subgráficos
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))

        # Verifica si el array promedio_stim_1 tiene elementos y no contiene valores NaN
        if promedio_stim_1.size > 0 and not np.isnan(promedio_stim_1).any():
            index_prom_tag = 0  # Inicializa el índice para las etiquetas de señales
            for label in self.signals_labels:  # Itera sobre las etiquetas de las señales
                ymin = min_values[index_prom_tag]  # Obtiene el valor mínimo normalizado para la señal actual
                y_max = max_values[index_prom_tag]  # Obtiene el valor máximo normalizado para la señal actual
                signal_color_t = getattr(self, f's_{index_prom_tag + 1}')  # Obtiene el color asignado a la señal actual

                # Grafica el promedio de la señal para los eventos de estimulación tipo 1
                axs[0, index_prom_tag].plot(t_tag, promedio_stim_1[:, index_prom_tag],
                                            label=f'CH{index_prom_tag + 1} STIM', color=self.color_map[signal_color_t])

                # Agrega una línea vertical en el punto de referencia de tiempo medio para los eventos de estimulación
                axs[0, index_prom_tag].axvline(x=t_tag[prom_ref_1], color='r', linestyle='--', label='REF')

                # Configura el título del subgráfico con el nombre de la nueva estimulación y la cantidad de datos utilizados
                axs[0, index_prom_tag].set_title(f'{self.new_stim}; n={len(datos_stim_1)}', color='gray')

                # Configura el color de fondo del subgráfico
                axs[0, index_prom_tag].set_facecolor('none')

                # Elimina el color de los bordes izquierdo, derecho y superior del subgráfico
                axs[0, index_prom_tag].spines['left'].set_color('none')
                axs[0, index_prom_tag].spines['right'].set_color('none')
                axs[0, index_prom_tag].spines['top'].set_color('none')

                # Configura el color del borde inferior del subgráfico
                axs[0, index_prom_tag].spines['bottom'].set_color('black')

                # Establece los límites del eje y del subgráfico con los valores mínimo y máximo normalizados
                axs[0, index_prom_tag].set_ylim(ymin, y_max)

                # Elimina las marcas en el eje x e y del subgráfico
                axs[0, index_prom_tag].set_xticks([])
                axs[0, index_prom_tag].set_yticks([])

                # Incrementa el índice para pasar a la siguiente señal
                index_prom_tag += 1
        else:
            # Imprime un mensaje si el array promedio_stim_1 está vacío o contiene valores NaN
            print("promedio_stim_1 está vacío o contiene valores NaN")

        # Graficar datos promedio para STIM 2
        if promedio_stim_2.size > 0 and not np.isnan(promedio_stim_2).any():
            index_prom_stim = 0  # Inicializa el índice para las etiquetas de señales
            for label in self.signals_labels:  # Itera sobre las etiquetas de las señales
                ymin = min_values[index_prom_stim]  # Obtiene el valor mínimo normalizado para la señal actual
                y_max = max_values[index_prom_stim]  # Obtiene el valor máximo normalizado para la señal actual
                signal_color_s = getattr(self, f's_{index_prom_stim + 1}')  # Obtiene el color asignado a la señal actual

                # Grafica el promedio de la señal para los eventos de estimulación tipo 2
                axs[1, index_prom_stim].plot(t_stim, promedio_stim_2[:, index_prom_stim], label=f'CH{index_prom_stim + 1} STIM',
                                             color=self.color_map[signal_color_s])

                # Agrega una línea vertical en el punto de referencia de tiempo medio para los eventos de estimulación
                axs[1, index_prom_stim].axvline(x=t_stim[prom_ref_2], color='r', linestyle='--', label='REF')

                # Configura el título del subgráfico con el nombre de la nueva estimulación y la cantidad de datos utilizados
                axs[1, index_prom_stim].set_title(f'{self.new_stim}; n={len(datos_stim_2)}', color='gray')

                # Configura el color de fondo del subgráfico
                axs[1, index_prom_stim].set_facecolor('none')

                # Elimina el color de los bordes izquierdo, derecho y superior del subgráfico
                axs[1, index_prom_stim].spines['left'].set_color('none')
                axs[1, index_prom_stim].spines['right'].set_color('none')
                axs[1, index_prom_stim].spines['top'].set_color('none')

                # Configura el color del borde inferior del subgráfico
                axs[1, index_prom_stim].spines['bottom'].set_color('black')

                # Configura la etiqueta del eje x del subgráfico
                axs[1, index_prom_stim].set_xlabel('Time [s]')

                # Establece los límites del eje y del subgráfico con los valores mínimo y máximo normalizados
                axs[1, index_prom_stim].set_ylim(ymin, y_max)

                # Elimina las marcas en el eje y del subgráfico
                axs[1, index_prom_stim].set_yticks([])

                # Incrementa el índice para pasar a la siguiente señal
                index_prom_stim += 1
        else:
            # Imprime un mensaje si el array promedio_stim_2 está vacío o contiene valores NaN
            print("promedio_stim_2 está vacío o contiene valores NaN")

        # Guarda la figura del análisis de bins
        self.save_fig(f"{self.new_file_name}_BINS")
        fig.patch.set_facecolor('black')  # Configura el color de fondo de la figura

        # Configura el color de fondo y las propiedades de los ejes
        for ax in axs.flatten():
            ax.set_facecolor('black')  # Establece el color de fondo del subgráfico a negro
            ax.xaxis.label.set_color('gray')  # Configura el color de la etiqueta del eje x a gris
            ax.yaxis.label.set_color('gray')  # Configura el color de la etiqueta del eje y a gris
            ax.tick_params(axis='x', colors='gray')  # Configura el color de las marcas del eje x a gris
            ax.tick_params(axis='y', colors='gray')  # Configura el color de las marcas del eje y a gris
            ax.spines['bottom'].set_color('gray')  # Configura el color del borde inferior del subgráfico a gris

        # Paso 7: Crear figura con registros originales y datos de estimulación
        fig, axs = plt.subplots(4, 1, figsize=(12, 6))
        index_graph = 0

        # Itera a través de cada etiqueta de señal en self.signals_labels
        for label in self.signals_labels:
            # Obtiene la señal correspondiente y otros atributos necesarios
            signal = getattr(self, f'signal_{index_graph + 1}')  # Obtiene la señal correspondiente
            cut_off_LP = getattr(self,
                                 f'cutoff_{index_graph + 1}_LP')  # Obtiene la frecuencia de corte de filtro paso bajo
            stim = getattr(self, f'new_stim')  # Obtiene la nueva estimulación
            signal_color = getattr(self, f's_{index_graph + 1}')  # Obtiene el color de la señal
            ymin = getattr(self, f'min_amplitude_{index_graph + 1}')  # Obtiene el valor mínimo de la amplitud
            ymax = getattr(self, f'max_amplitude_{index_graph + 1}')  # Obtiene el valor máximo de la amplitud

            # Grafica la señal original y la estimulación
            axs[index_graph].plot(self.time[self.start_a_num: self.end_a_num],
                                  signal[self.start_a_num: self.end_a_num], label=f'{label}/LP={cut_off_LP}Hz',
                                  color=self.color_map[signal_color])  # Grafica la señal filtrada
            axs[index_graph].plot(self.time[self.start_a_num: self.end_a_num],
                                  self.stim[self.start_a_num: self.end_a_num], label=f'{stim}',
                                  color=self.stim_map[self.new_stim])  # Grafica la señal de estimulación
            axs[index_graph].plot(self.time[self.start_a_num: self.end_a_num],
                                  self.data[f'TAG OUT'].to_numpy()[self.start_a_num: self.end_a_num], label=f'TAG',
                                  color='gray')  # Grafica la señal TAG OUT
            axs[index_graph].set_title(f'{label}', color='gray')  # Configura el título del gráfico
            axs[index_graph].set_facecolor('none')  # Configura el color de fondo del gráfico
            axs[index_graph].spines['left'].set_color('none')  # Oculta el borde izquierdo del gráfico
            axs[index_graph].spines['right'].set_color('none')  # Oculta el borde derecho del gráfico
            axs[index_graph].spines['top'].set_color('none')  # Oculta el borde superior del gráfico
            axs[index_graph].spines['bottom'].set_color('gray')  # Configura el color del borde inferior del gráfico
            axs[index_graph].set_yticks([])  # Oculta las marcas del eje y
            axs[index_graph].set_ylim(ymin, ymax)  # Configura el rango del eje y

            # Configura el eje x solo para el gráfico en la parte inferior
            if index_graph < len(self.signals_labels) - 1:
                axs[index_graph].set_xticks([])  # Oculta las marcas del eje x para todos los gráficos excepto el último
            if index_graph == len(self.signals_labels) - 1:
                axs[index_graph].set_xlabel('Time [s]')  # Configura la etiqueta del eje x solo para el último gráfico
            index_graph += 1  # Incrementa el índice para el siguiente gráfico
        # Itera sobre todos los ejes de la figura para configurar sus propiedades
        for ax in axs.flatten():
            ax.legend(frameon=False, labelcolor='gray')

        # Guarda la figura de los registros
        self.save_fig(f"{self.new_file_name}_SIGNALS_BINS")  # Llama al método save_fig para guardar la figura con un nombre específico

        # Configura el color de fondo de la figura
        fig.patch.set_facecolor('black')  # Cambia el color de fondo de toda la figura a negro

        # Configura el color de los ejes y etiquetas en gris
        for ax in axs.flatten():
            ax.set_facecolor('black')  # Cambia el color de fondo de cada gráfico individual a negro
            ax.xaxis.label.set_color('gray')  # Cambia el color de las etiquetas del eje x a gris
            ax.yaxis.label.set_color('gray')  # Cambia el color de las etiquetas del eje y a gris
            ax.tick_params(axis='x', colors='gray')  # Cambia el color de las marcas del eje x a gris
            ax.set_facecolor('black')  # Asegura que el color de fondo de cada gráfico sea negro
            ax.spines['bottom'].set_color('gray')  # Cambia el color del borde inferior del gráfico a gris
            ax.legend(frameon=False, labelcolor='gray')

        plt.tight_layout()  # Ajusta automáticamente el diseño de la figura para evitar solapamientos entre los gráficos
        plt.show()  # Muestra la figura generada en pantalla

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

        # Actualizar las señales de estimulación y etiqueta para los gráficos
        self.tag_graph_1 = np.where(self.stim == 1, self.stim, 0)
        self.stim_graph_1 = np.where(self.stim == 2, self.stim, 0)

        self.tag_graph_2 = np.where(self.stim_plot_1 == 1, self.stim, 0)
        self.stim_graph_2 = np.where(self.stim_plot_1 == 2, self.stim, 0)
        self.tag_graph_3 = np.where(self.stim_plot_2 == 1, self.stim, 0)
        self.stim_graph_3 = np.where(self.stim_plot_2 == 2, self.stim, 0)
        self.tag_graph_4 = np.where(self.stim_plot_3 == 1, self.stim, 0)
        self.stim_graph_4 = np.where(self.stim_plot_3 == 2, self.stim, 0)

        # Actualizar gráfico 1
        self.plot_widget_1.clear()  # Limpiar el widget de trazado
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.signal_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_1], width=self.width_line), name='Señal original')
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line,style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_1.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line))

        self.plot_widget_1.setLabel('left', 'Amplitude')
        self.plot_widget_1.setLabel('bottom', 'Time[s]')
        self.plot_widget_1.setTitle(f'{self.s_1}')
        self.plot_widget_1.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)

        # ------------------------------------------------------------------------------------- signal 2
        # Actualizar gráfico 2
        self.plot_widget_2.clear()  # Limpiar el widget de trazado
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.signal_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_2], width=self.width_line), name='Señal original')
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line,style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_2.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line))

        self.plot_widget_2.setLabel('left', 'Amplitude')
        self.plot_widget_2.setLabel('bottom', 'Time[s]')
        self.plot_widget_2.setTitle(f'{self.s_2}')
        self.plot_widget_2.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)

        # -------------------------------------------------------------------------------------
        # Actualizar gráfico 3
        self.plot_widget_3.clear()  # Limpiar el widget de trazado
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.signal_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_3], width=self.width_line), name='Señal original')

        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_3.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line))

        self.plot_widget_3.setLabel('left', 'Amplitude')
        self.plot_widget_3.setLabel('bottom', 'Time[s]')
        self.plot_widget_3.setTitle(f'{self.s_3}')
        self.plot_widget_3.setXRange(self.tiempo_inicial, self.tiempo_final)
        self.plot_widget_3.setYRange(self.min_amplitude_3, self.max_amplitude_3)

        # -------------------------------------------------------------------------------------
        # Actualizar gráfico 4
        self.plot_widget_4.clear()  # Limpiar el widget de trazado
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.signal_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.color_map[self.s_4], width=self.width_line), name='Señal original')

        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_1[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_2[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[0], width=self.width_line))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line,style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_3[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[1], width=self.width_line))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.stim_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget_4.plot(self.time[self.start_a_num: self.end_a_num], self.tag_graph_4[self.start_a_num: self.end_a_num], pen=pg.mkPen(color=self.stim_colors[2], width=self.width_line))

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