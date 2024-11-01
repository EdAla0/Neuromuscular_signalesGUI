import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
from matplotlib import pyplot as plt
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from scipy.signal import lfilter, butter, filtfilt
import csv


class MainWindow(QMainWindow):
    def __init__(self, archivo, channel_colors, stim_colors, values):
        """
        Inicializa una instancia de la ventana principal de la aplicación.

        Configura los widgets, las conexiones de señales y establece las variables necesarias
        para la visualización y el procesamiento de datos.

        :param archivo: Ruta del archivo CSV que contiene los datos. Se espera que el archivo tenga columnas de tiempo y varias señales.
        :param channel_colors: Lista de colores para los canales. Cada color se asignará a un canal específico para su visualización.
        :param stim_colors: Lista de colores para las señales de estimulación. Cada color se asignará a una señal de estimulación específica.
        :param values: Valores de configuración para la ventana. Este parámetro parece no ser utilizado en el código proporcionado.
        """

        super().__init__()
        # Carga el archivo de interfaz de usuario .ui generado con Qt Designer
        loadUi('RECT_E_INT_GUI_con_HP.ui', self)

        # Configuración del widget central y layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)  # Agrega el widget Base al layout principal
        central_widget.setLayout(layout)

        # Widgets para trazar diferentes señales y sus envolventes
        self.plot_widget_1 = pg.PlotWidget()  # Primer widget para señales
        self.Graph_layout_1.addWidget(self.plot_widget_1)  # Agrega al layout correspondiente

        self.plot_widget_2 = pg.PlotWidget()  # Segundo widget para señales
        self.Graph_layout_2.addWidget(self.plot_widget_2)  # Agrega al layout correspondiente

        self.plot_widget_3 = pg.PlotWidget()  # Tercer widget para envolventes
        self.Graph_layout_3.addWidget(self.plot_widget_3)  # Agrega al layout correspondiente

        self.plot_widget_4 = pg.PlotWidget()  # Cuarto widget para envolventes
        self.Graph_layout_6.addWidget(self.plot_widget_4)  # Agrega al layout correspondiente

        # Diccionario que mapea cada PlotWidget con su título
        self.plot_widgets = {
            self.plot_widget_1: "Signal 1",
            self.plot_widget_2: "Signal 2",
            self.plot_widget_3: "Signal 3",
            self.plot_widget_4: "Signal 4"
        }

        # Opciones para el menú desplegable de canales
        options = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                   'CH 10', 'CH 11', 'CH 12']
        # Mapea cada opción a un color correspondiente
        self.channel_color_map = dict(zip(options, channel_colors))

        # Configura los QComboBox para seleccionar columnas de datos y los conecta a las funciones correspondientes
        comboboxes = [self.column_combobox_1, self.column_combobox_2, self.column_combobox_3, self.column_combobox_5,
            self.save_in_1, self.save_in_2, self.save_in_3, self.save_in_4
        ]

        for combobox in comboboxes:
            combobox.addItems(options)  # Agrega opciones al combobox
            combobox.currentIndexChanged.connect(
                self.channel)  # Conecta la señal de cambio de índice a la función 'channel'

        for combobox in comboboxes[4:]:
            combobox.currentIndexChanged.connect(
                self.save_in_update)  # Conecta la señal de cambio de índice a la función 'save_in_update'

        # Configura los botones para aplicar integración rectificada a las señales y los conecta a sus funciones
        buttons_signals = [
            (self.btn_rect_int_1, self.rectified_integrate_1),
            (self.btn_rect_int_2, self.rectified_integrate_2),
            (self.btn_rect_int_3, self.rectified_integrate_3),
            (self.btn_rect_int_4, self.rectified_integrate_4)
        ]

        for button, function in buttons_signals:
            button.clicked.connect(function)  # Conecta el clic del botón a la función correspondiente

        # Configura los QSpinBox para cortar frecuencias y los conecta a las funciones correspondientes
        spinboxes = [
            (self.cutoff_s1_LP, self.actualizar_LP_fc_1), (self.cutoff_s2_LP, self.actualizar_LP_fc_2),
            (self.cutoff_s3_LP, self.actualizar_LP_fc_3), (self.cutoff_s4_LP, self.actualizar_LP_fc_4),
            (self.cutoff_s1_HP, self.actualizar_HP_fc_1), (self.cutoff_s2_HP, self.actualizar_HP_fc_2),
            (self.cutoff_s3_HP, self.actualizar_HP_fc_3), (self.cutoff_s4_HP, self.actualizar_HP_fc_4)
        ]
        for spinbox, function in spinboxes:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente

        # Configura los botones para aplicar filtros y los conecta a las funciones correspondientes
        filter_buttons = [
            (self.btn_filter_1, self.update_filter_1), (self.btn_filter_2, self.update_filter_2),
            (self.btn_filter_3, self.update_filter_3), (self.btn_filter_4, self.update_filter_4)
        ]
        for button, function in filter_buttons:
            button.clicked.connect(function)  # Conecta el clic del botón a la función correspondiente

        # Configura los QSpinBox para las constantes de tiempo y los conecta a las funciones correspondientes
        time_constants = [
            (self.time_constant_1, self.update_time_constant_1), (self.time_constant_2, self.update_time_constant_2),
            (self.time_constant_3, self.update_time_constant_3), (self.time_constant_4, self.update_time_constant_4)
        ]
        for spinbox, function in time_constants:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente

        # Configura los QSpinBox para el rango de valores Y y los conecta a las funciones correspondientes
        y_ranges = [
            (self.y_max_1, self.y_min_1, self.update_y_range_1), (self.y_max_2, self.y_min_2, self.update_y_range_2),
            (self.y_max_3, self.y_min_3, self.update_y_range_3), (self.y_max_4, self.y_min_4, self.update_y_range_4)
        ]
        for ymax, ymin, function in y_ranges:
            ymax.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente
            ymin.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente

        # Configura los QSpinBox para los valores de ganancia y los conecta a las funciones correspondientes
        gains = [
            (self.gain_1, self.update_gain_1), (self.gain_2, self.update_gain_2),
            (self.gain_3, self.update_gain_3), (self.gain_4, self.update_gain_4)
        ]
        for spinbox, function in gains:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente

        # Configura los QSpinBox para los valores de desplazamiento y los conecta a las funciones correspondientes
        offsets = [
            (self.offset_1, self.update_offset_1), (self.offset_2, self.update_offset_2),
            (self.offset_3, self.update_offset_3), (self.offset_4, self.update_offset_4)
        ]
        for spinbox, function in offsets:
            spinbox.valueChanged.connect(function)  # Conecta el cambio de valor a la función correspondiente

        # Configura el botón de guardado y conecta su clic a la función correspondiente
        self.save_function.clicked.connect(self.save_active)

        # Configura los QSpinBox para los tiempos de inicio y fin y conecta sus cambios a las funciones correspondientes
        self.t_inic_a.valueChanged.connect(self.update_t_inic_a)
        self.t_fin_a.valueChanged.connect(self.update_t_fin_a)

        # Configura las opciones para las señales de estimulación y conecta la selección a la función correspondiente
        stims = ['STIM 1', 'STIM 2', 'STIM 3']
        self.stim_colors = stim_colors
        self.stims_options.addItems(stims)  # Agrega las opciones al combobox de estimulación
        self.stims_options.currentIndexChanged.connect(
            self.update_stim)  # Conecta la selección a la función correspondiente
        self.stim_color_map = dict(zip(stims, self.stim_colors))  # Mapea las señales de estimulación a sus colores

        # Configura botones para avanzar y retroceder gráficos, y conecta sus clics a las funciones correspondientes
        self.btn_window_time.valueChanged.connect(self.update_window_time)
        self.btn_avanzar.clicked.connect(self.avanzar_graficas)
        self.btn_retroceder.clicked.connect(self.retroceder_graficas)
        self.btn_go.clicked.connect(self.update_go)
        self.name_file.textChanged.connect(self.name_file_update)

        # Mapea los PlotWidgets con sus títulos y conecta la señal de clic del mouse a la función correspondiente
        self.plot_widget_map = dict(zip(self.plot_widgets, ["Signal 1", "Signal 2", "Signal 3", "Signal 4"]))
        for plot_widget in self.plot_widgets:
            plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)

        # Configuración inicial de variables
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)  # Extrae la carpeta del archivo de datos
        self.data = pd.read_csv(self.ruta_archivo)  # Lee los datos del archivo CSV
        self.data_copy = pd.read_csv(self.ruta_archivo)  # Crea una copia de los datos
        self.time = self.data['TIME'].to_numpy()  # Extrae la columna de tiempo
        self.stim_plot_1 = self.data['STIM 1'].to_numpy()  # Extrae la columna de estimulación 1
        self.stim_plot_2 = self.data['STIM 2'].to_numpy()  # Extrae la columna de estimulación 2
        self.stim_plot_3 = self.data['STIM 3'].to_numpy()  # Extrae la columna de estimulación 3
        self.new_stim = 'STIM 1'
        self.stim = self.data[self.new_stim].to_numpy()  # Extrae la columna de estimulación actual

        # Configuración inicial de parámetros para visualización y filtrado
        self.window_time = 10
        self.start_a = 0
        self.end_a = self.window_time
        self.t_inic_a.setValue(int(self.start_a))
        self.t_fin_a.setValue(int(self.end_a))
        self.datos_por_s = round(1 / (self.time[1] - self.time[0]))  # Calcula los datos por segundo
        self.start_a_num = round(self.start_a * self.datos_por_s)
        self.end_a_num = round(self.end_a * self.datos_por_s)
        self.btn_window_time.setValue(int(self.window_time))
        self.min_amplitude_1 = 0
        self.max_amplitude_1 = 1
        self.min_amplitude_2 = 0
        self.max_amplitude_2 = 1
        self.min_amplitude_3 = 0
        self.max_amplitude_3 = 1
        self.min_amplitude_4 = 0
        self.max_amplitude_4 = 1

        # Estado inicial de las variables de navegación y filtro
        self.avanza = False
        self.retrocede = False
        self.window = False
        self.save_function_flag = False

        # Configuración de filtros y parámetros predeterminados
        self.before = 1
        self.after = 1
        self.fs = 3300  # Frecuencia de muestreo
        self.cutoff_1_LP = 1000  # Frecuencia de corte para el filtro paso bajo 1
        self.cutoff_2_LP = 1000  # Frecuencia de corte para el filtro paso bajo 2
        self.cutoff_3_LP = 1000  # Frecuencia de corte para el filtro paso bajo 3
        self.cutoff_4_LP = 1000  # Frecuencia de corte para el filtro paso bajo 4
        self.cutoff_1_HP = 0  # Frecuencia de corte para el filtro paso alto 1
        self.cutoff_2_HP = 0  # Frecuencia de corte para el filtro paso alto 2
        self.cutoff_3_HP = 0  # Frecuencia de corte para el filtro paso alto 3
        self.cutoff_4_HP = 0  # Frecuencia de corte para el filtro paso alto 4
        self.time_c_1 = 200 / 1000  # Constante de tiempo para el filtro 1 (en segundos)
        self.time_c_2 = 200 / 1000  # Constante de tiempo para el filtro 2 (en segundos)
        self.time_c_3 = 200 / 1000  # Constante de tiempo para el filtro 3 (en segundos)
        self.time_c_4 = 200 / 1000  # Constante de tiempo para el filtro 4 (en segundos)
        self.Ts = self.time[1] - self.time[0]  # Intervalo de tiempo entre muestras
        self.new_gain_1 = 1
        self.new_gain_2 = 1
        self.new_gain_3 = 1
        self.new_gain_4 = 1
        self.new_offset_1 = 0
        self.new_offset_2 = 0
        self.new_offset_3 = 0
        self.new_offset_4 = 0
        self.filter_1_LP = False
        self.filter_2_LP = False
        self.filter_3_LP = False
        self.filter_4_LP = False
        self.filter_1_HP = False
        self.filter_2_HP = False
        self.filter_3_HP = False
        self.filter_4_HP = False
        self.s_1 = 'CH 1'  # Columna seleccionada para Signal 1
        self.s_2 = 'CH 1'  # Columna seleccionada para Signal 2
        self.s_3 = 'CH 1'  # Columna seleccionada para Signal 3
        self.s_4 = 'CH 1'  # Columna seleccionada para Signal 4
        self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
        self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
        self.signal_3 = self.data[f'{self.s_3}'].to_numpy()
        self.signal_4 = self.data[f'{self.s_4}'].to_numpy()

        self.s_1_save = 'CH 1'  # Columna seleccionada para guardar Signal 1
        self.s_2_save = 'CH 1'  # Columna seleccionada para guardar Signal 2
        self.s_3_save = 'CH 1'  # Columna seleccionada para guardar Signal 3
        self.s_4_save = 'CH 1'  # Columna seleccionada para guardar Signal 4

        # Crea un array con todas las señales seleccionadas
        self.signals = self.data[
            [f'{self.s_1}', f'{self.s_2}', f'{self.s_3}', f'{self.s_4}', f'{self.s_1}', f'{self.s_2}', f'{self.s_3}',
             f'{self.s_4}']].to_numpy()
        self.original_1 = self.signals[:, 4]  # Señal original 1
        self.original_2 = self.signals[:, 5]  # Señal original 2
        self.original_3 = self.signals[:, 6]  # Señal original 3
        self.original_4 = self.signals[:, 7]  # Señal original 4

        # Colores predeterminados para diferentes elementos gráficos
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2],
            self.s_3: self.channel_color_map[self.s_3],
            self.s_4: self.channel_color_map[self.s_4]
        }

        self.stim_map = {
            self.new_stim: self.stim_color_map[self.new_stim],
        }
        self.width_line = 2
        self.width_crossing = 1
        self.original_color = '#00FFFF'  # Color para las señales originales
        self.filtered_color = '#5B5B5B'  # Color para las señales filtradas
        self.stim_color = 'y'  # Color para las señales de estimulación
        self.up_color = 'm'  # Color para las subidas
        self.down_color = '#FFFFFF'  # Color para las bajadas
        self.threshold_color = 'g'  # Color para el umbral
        self.new_file_name = "RECT_E_INT"  # Nombre del archivo de salida

        # Actualiza el gráfico con los valores iniciales
        self.actualizar_grafico()

    def save_active(self):
        """
        Activa o desactiva la función de guardado.

        Esta función alterna el estado de la bandera de guardado (`save_function_flag`).
        Si la bandera está desactivada (`False`), se activa (`True`). Si ya está activada,
        se desactiva. Esta función se utiliza para controlar si los datos deben ser guardados
        en el archivo.

        La función no realiza ninguna acción específica de guardado en sí misma; su propósito
        es cambiar el estado de una bandera que puede ser utilizada por otras funciones para
        decidir si deben guardar datos o no.

        Ejemplo de uso:
        - Llamar a `save_active()` para alternar entre activar o desactivar la función de guardado.
        """
        if not self.save_function_flag:
            self.save_function_flag = True
            # print("ACTIVO SAVE")
        else:
            self.save_function_flag = False
            # print("DESACTIVO SAVE")

    def name_file_update(self, new_file_name):
        """
        Actualiza el nombre del archivo de salida.

        Esta función establece el nuevo nombre de archivo en el atributo `new_file_name`.
        Es útil para actualizar el nombre del archivo antes de guardar los datos, de manera
        que se puedan generar archivos con nombres diferentes según sea necesario.

        :param new_file_name: El nuevo nombre del archivo de salida.
            - Tipo: str
            - Descripción: El nombre del archivo que se usará para guardar los datos.

        Ejemplo de uso:
        - Llamar a `name_file_update("nuevo_nombre_archivo.csv")` para cambiar el nombre del archivo
          de salida a "nuevo_nombre_archivo.csv".
        """
        self.new_file_name = new_file_name

    def update_t_inic_a(self, value):
        """
        Actualiza el tiempo inicial del intervalo de análisis.

        Esta función ajusta el tiempo de inicio (`start_a`) del intervalo de análisis basado en
        el valor proporcionado. Calcula la nueva duración del intervalo de análisis (`window_time`)
        como la diferencia entre el tiempo de fin (`end_a`) y el tiempo de inicio. Luego, actualiza
        los widgets de la interfaz de usuario relacionados y reinicia los indicadores de navegación.

        :param value: Nuevo valor para el tiempo inicial del intervalo de análisis.
            - Tipo: float
            - Descripción: El tiempo inicial en segundos para el intervalo de análisis.

        La función también actualiza la visualización del gráfico para reflejar los cambios realizados.

        Ejemplo de uso:
        - Llamar a `update_t_inic_a(5.0)` para establecer el tiempo inicial en 5 segundos y
          ajustar la visualización en consecuencia.
        """
        self.start_a = value  # Valor en segundos
        self.window_time = self.end_a - self.start_a  # Calcula la nueva duración del intervalo
        self.t_inic_a.setValue(int(self.start_a))  # Actualiza el widget del tiempo inicial
        self.btn_window_time.setValue(int(self.window_time))  # Actualiza el widget de la duración del intervalo
        self.avanza = False  # Reinicia el indicador de avance
        self.retrocede = False  # Reinicia el indicador de retroceso
        self.window = False  # Reinicia el estado de la ventana
        self.actualizar_grafico()  # Actualiza el gráfico con los nuevos valores

    def update_t_fin_a(self, value):
        """
        Actualiza el tiempo final del intervalo de análisis.

        Esta función ajusta el tiempo de fin (`end_a`) del intervalo de análisis basado en el valor
        proporcionado. Calcula la nueva duración del intervalo de análisis (`window_time`) como la
        diferencia entre el tiempo de fin y el tiempo de inicio (`start_a`). Luego, actualiza los
        widgets de la interfaz de usuario relacionados y reinicia los indicadores de navegación.

        :param value: Nuevo valor para el tiempo final del intervalo de análisis.
            - Tipo: float
            - Descripción: El tiempo final en segundos para el intervalo de análisis.

        La función también actualiza la visualización del gráfico para reflejar los cambios realizados.

        Ejemplo de uso:
        - Llamar a `update_t_fin_a(10.0)` para establecer el tiempo final en 10 segundos y
          ajustar la visualización en consecuencia.
        """
        self.end_a = value  # Valor en segundos
        self.window_time = self.end_a - self.start_a  # Calcula la nueva duración del intervalo
        self.t_fin_a.setValue(int(self.end_a))  # Actualiza el widget del tiempo final
        self.btn_window_time.setValue(int(self.window_time))  # Actualiza el widget de la duración del intervalo
        self.avanza = False  # Reinicia el indicador de avance
        self.retrocede = False  # Reinicia el indicador de retroceso
        self.window = False  # Reinicia el estado de la ventana
        self.actualizar_grafico()  # Actualiza el gráfico con los nuevos valores

    def update_stim(self):
        """
        Actualiza la señal de estimulación y su color asociado en el gráfico.

        Esta función realiza los siguientes pasos:
        1. Obtiene el nuevo nombre de la señal de estimulación seleccionado en el menú desplegable (`stims_options`).
        2. Actualiza la variable `self.stim` con los datos de la señal de estimulación correspondiente
           desde el archivo de datos (`self.data`).
        3. Actualiza el mapa de colores (`self.stim_map`) asociando el nuevo nombre de la señal de estimulación
           con su color correspondiente (`self.stim_color_map`).
        4. Llama a la función `update_plots` para actualizar la visualización de los gráficos con la nueva señal de estimulación.

        La función asegura que la interfaz gráfica refleje los cambios realizados en la selección de señales de estimulación,
        permitiendo al usuario ver los datos de la señal de estimulación seleccionada en el gráfico.

        Ejemplo de uso:
        - Si el usuario selecciona "STIM 2" en el menú desplegable de señales de estimulación, la función actualizará
          `self.stim` con los datos correspondientes a "STIM 2", y ajustará el color del gráfico según la configuración
          del color para "STIM 2".

        Nota:
        - La función `update_plots` debe ser implementada para que los cambios en la visualización se reflejen correctamente
          en los gráficos.
        """
        self.new_stim = self.stims_options.currentText()  # Obtiene el nuevo nombre de la señal de estimulación
        self.stim = self.data[
            self.new_stim].to_numpy()  # Actualiza la variable de datos con la nueva señal de estimulación
        self.stim_map = {self.new_stim: self.stim_color_map[
            self.new_stim]}  # Actualiza el mapa de colores para la señal de estimulación
        self.update_plots()  # Actualiza la visualización de los gráficos

    def update_y_range_1(self):
        """
        Actualiza el rango del eje Y para el primer gráfico.

        Esta función realiza los siguientes pasos:
        1. Obtiene el valor mínimo del rango del eje Y desde el widget `y_min_1`.
        2. Obtiene el valor máximo del rango del eje Y desde el widget `y_max_1`.
        3. Establece el nuevo rango del eje Y en `plot_widget_1` usando los valores obtenidos.

        Esto permite ajustar el rango vertical del primer gráfico de acuerdo con los valores proporcionados
        por el usuario, asegurando que el gráfico muestre los datos dentro del rango especificado.

        Ejemplo de uso:
        - Si el usuario establece el rango Y mínimo en -10 y máximo en 10, el gráfico en `plot_widget_1` se ajustará
          para mostrar solo los valores dentro de este rango.

        Nota:
        - Esta función debe ser conectada a los cambios en los widgets `y_min_1` y `y_max_1` para actualizar el gráfico en tiempo real.
        """
        self.min_amplitude_1 = self.y_min_1.value()  # Obtiene el valor mínimo del widget
        self.max_amplitude_1 = self.y_max_1.value()  # Obtiene el valor máximo del widget
        self.plot_widget_1.setYRange(self.min_amplitude_1, self.max_amplitude_1)  # Establece el nuevo rango en el gráfico

    def update_y_range_2(self):
        self.min_amplitude_2 = self.y_min_2.value()  # Obtiene el valor mínimo del widget
        self.max_amplitude_2 = self.y_max_2.value()  # Obtiene el valor máximo del widget
        self.plot_widget_2.setYRange(self.min_amplitude_2, self.max_amplitude_2)  # Establece el nuevo rango en el gráfico

    def update_y_range_3(self):
        self.min_amplitude_3 = self.y_min_3.value()  # Obtiene el valor mínimo del widget
        self.max_amplitude_3 = self.y_max_3.value()  # Obtiene el valor máximo del widget
        self.plot_widget_3.setYRange(self.min_amplitude_3, self.max_amplitude_3)  # Establece el nuevo rango en el gráfico

    def update_y_range_4(self):
        self.min_amplitude_4 = self.y_min_4.value()  # Obtiene el valor mínimo del widget
        self.max_amplitude_4 = self.y_max_4.value()  # Obtiene el valor máximo del widget
        self.plot_widget_4.setYRange(self.min_amplitude_4, self.max_amplitude_4)  # Establece el nuevo rango en el gráfico

    def update_go(self):
        """Actualiza y muestra gráficos de señales procesadas y sus correspondientes configuraciones de filtro.

        Esta función realiza las siguientes acciones:

        1. **Guardar datos**:
           - Si el indicador `save_function_flag` está activado (`True`), se llama al método `save_and_insert_data` para
           guardar los datos con un nombre de archivo basado en `new_file_name`.

        2. **Actualizar etiquetas**:
           - Se actualizan las etiquetas para las señales (`signals_labels`), los filtros de paso bajo (`fc_LP_labels`)
           y los filtros de paso alto (`fc_HP_labels`) usando los atributos correspondientes.

        3. **Configuración de gráficos**:
           - Se crea una figura con cuatro subgráficos dispuestos en una columna.
           - Para cada señal en `signals_labels`:
             - Se recuperan las señales, filtros y configuraciones relacionadas desde los atributos del objeto.
             - Se grafican las señales originales, filtradas (con ganancia y desplazamiento aplicados), estímulos, y una
             señal de referencia (`TAG OUT`).
             - Se configuran las propiedades visuales de cada subgráfico, incluyendo títulos, colores de fondo, límites
             del eje Y, y la visibilidad de las marcas y espinas del gráfico.
             - Se ajusta el rango del eje X para mostrar solo la ventana de tiempo definida por `start_a` y `end_a`.

        4. **Guardar y mostrar gráficos**:
           - Los gráficos se muestran usando `plt.show()`.
           - Comentarios indican lugares donde se podría guardar la figura en formato SVG con diferentes configuraciones
            de fondo, aunque estas líneas están actualmente comentadas.

        Ejemplo de uso:
        - Llamar a esta función actualizará los gráficos con los datos actuales y mostrará una ventana con las gráficas
        actualizadas, permitiendo la visualización de las señales procesadas, los filtros aplicados, y los estímulos en diferentes configuraciones.

        Nota:
        - Asegúrate de que todos los atributos usados en esta función estén correctamente definidos y inicializados antes
        de llamar a `update_go`.
        """
        # Guardar los datos si el flag está activado
        if self.save_function_flag:
            self.save_and_insert_data(f"{self.new_file_name}_RECT_INT_DATA")

        # Actualizar etiquetas de señales y filtros
        self.signals_labels = [f'{self.s_1}', f'{self.s_2}', f'{self.s_3}', f'{self.s_4}']
        self.fc_LP_labels = [f'{self.cutoff_1_LP}', f'{self.cutoff_2_LP}', f'{self.cutoff_3_LP}', f'{self.cutoff_4_LP}']
        self.fc_HP_labels = [f'{self.cutoff_1_HP}', f'{self.cutoff_2_HP}', f'{self.cutoff_3_HP}', f'{self.cutoff_4_HP}']

        # Crear una figura y ejes para los gráficos
        fig, axs = plt.subplots(4, 1, figsize=(12, 6))

        index_graph = 0
        # Graficar las señales para cada etiqueta en signals_labels
        for label in self.signals_labels:
            # Obtener datos y configuraciones para cada gráfico
            signal = getattr(self, f'signal_{index_graph + 1}')
            signal_original = getattr(self, f'original_{index_graph + 1}')
            cut_off_LP = getattr(self, f'cutoff_{index_graph + 1}_LP')
            cut_off_HP = getattr(self, f'cutoff_{index_graph + 1}_HP')

            time_constant = getattr(self, f'time_c_{index_graph + 1}')
            gain = getattr(self, f'new_gain_{index_graph + 1}')
            offset = getattr(self, f'new_offset_{index_graph + 1}')
            stim = getattr(self, f'new_stim')
            signal_color = getattr(self, f's_{index_graph + 1}')
            ymin = getattr(self, f'min_amplitude_{index_graph + 1}')
            ymax = getattr(self, f'max_amplitude_{index_graph + 1}')

            # Graficar las señales y configuraciones en el subgráfico correspondiente
            axs[index_graph].plot(self.time, signal_original, label=f'ORIGINAL {label}', color=self.filtered_color)
            axs[index_graph].plot(self.time, ((signal * gain) + offset),
                                  label=f'{label}/LP={cut_off_LP}Hz/HP={cut_off_HP}Hz/Tc={time_constant}ms/Gain={gain}',
                                  color=self.color_map[signal_color])
            axs[index_graph].plot(self.time, self.stim, label=f'{stim}', color=self.stim_map[self.new_stim])
            axs[index_graph].plot(self.time, self.data[f'TAG OUT'].to_numpy(), label=f'TAG', color='gray')

            # Configurar el título y apariencia del subgráfico
            axs[index_graph].set_title(f'{label}', color='gray')
            axs[index_graph].set_facecolor('none')  # Color de fondo transparente para el área del gráfico
            axs[index_graph].spines['left'].set_color('none')  # Ocultar espina izquierda
            axs[index_graph].spines['right'].set_color('none')  # Ocultar espina derecha
            axs[index_graph].spines['top'].set_color('none')  # Ocultar espina superior
            axs[index_graph].spines['bottom'].set_color('none')  # Ocultar espina inferior
            axs[index_graph].set_yticks([])  # Ocultar marcas del eje Y

            # Configurar el rango del eje Y
            axs[index_graph].set_ylim(ymin, ymax)

            if index_graph < len(self.signals_labels) - 1:
                axs[index_graph].set_xticks([])

            # Incrementar el índice del gráfico para pasar al siguiente subgráfico en la próxima iteración
            index_graph += 1

        # Establecer el color de fondo de la figura principal como transparente
        fig.patch.set_facecolor('none')

        # Configurar cada subgráfico individualmente
        for ax in axs.flatten():
            # Establecer el color de fondo de cada subgráfico como transparente
            ax.set_facecolor('none')

            # Ocultar las espinas (líneas de los bordes) de cada subgráfico
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Eliminar las marcas del eje Y para una apariencia más limpia
            ax.set_yticks([])

            # Configurar la leyenda para no tener marco
            ax.legend(frameon=False)

        # Ajustar el rango del eje X para todos los subgráficos
        for ax in axs.flatten():
            ax.set_xlim(self.start_a, self.end_a)

        # Guardar la figura
        self.save_fig(f"{self.new_file_name}_RECT_INT_ANALYSIS")

        # Establecer el color de fondo de la figura principal como negro
        fig.patch.set_facecolor('black')

        # Configurar cada subgráfico para que tenga un fondo negro
        for ax in axs.flatten():
            ax.set_facecolor('black')

        # Configurar el color de los ejes y las etiquetas en gris para todos los subgráficos
        for ax in axs.flatten():
            # Establecer el color de las etiquetas del eje X e Y en gris
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')

            # Cambiar el color de los ticks del eje X a gris
            ax.tick_params(axis='x', colors='gray')

            # Configurar el fondo del subgráfico y las espinas del eje X en negro
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('gray')  # Establecer el color gris para el borde inferior

            # Configurar la leyenda para no tener marco y con texto en gris
            ax.legend(frameon=False, labelcolor='gray')

        # Ajustar el diseño de los gráficos para evitar superposición
        plt.tight_layout()

        # Mostrar la figura
        plt.show()

    def actualizar_LP_fc_1(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la primera señal y activa el filtro.

        Parámetros:
        value (float): Nuevo valor de la frecuencia de corte del filtro pasa bajo (LP) para la primera señal."""
        self.cutoff_1_LP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 1
        self.filter_1_LP = True  # Activa el filtro pasa bajo para la señal 1
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_2(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la segunda señal y activa el filtro."""
        self.cutoff_2_LP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 2
        self.filter_2_LP = True  # Activa el filtro pasa bajo para la señal 2
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_3(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la tercera señal y activa el filtro."""
        self.cutoff_3_LP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 3
        self.filter_3_LP = True  # Activa el filtro pasa bajo para la señal 3
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_LP_fc_4(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la cuarta señal y activa el filtro."""
        self.cutoff_4_LP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 4
        self.filter_4_LP = True  # Activa el filtro pasa bajo para la señal 4
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_HP_fc_1(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la primera señal y activa el filtro.

        Parámetros:
        value (float): Nuevo valor de la frecuencia de corte del filtro pasa bajo (LP) para la primera señal."""
        self.cutoff_1_HP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 1
        self.filter_1_HP = True  # Activa el filtro pasa bajo para la señal 1
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_HP_fc_2(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la segunda señal y activa el filtro."""
        self.cutoff_2_HP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 2
        self.filter_2_HP = True  # Activa el filtro pasa bajo para la señal 2
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_HP_fc_3(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la tercera señal y activa el filtro."""
        self.cutoff_3_HP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 3
        self.filter_3_HP = True  # Activa el filtro pasa bajo para la señal 3
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def actualizar_HP_fc_4(self, value):
        """Actualiza el valor del corte del filtro pasa bajo para la cuarta señal y activa el filtro."""
        self.cutoff_4_HP = value  # Establece el nuevo valor de frecuencia de corte LP para la señal 4
        self.filter_4_HP = True  # Activa el filtro pasa bajo para la señal 4
        self.actualizar_grafico()  # Llama a la función para actualizar el gráfico con el nuevo filtro

    def update_filter_1(self):
        """Activa el filtro pasa alto para la primera señal y actualiza el gráfico.

        La función establece el atributo `filter_1_HP` en `True`, indicando que el filtro pasa alto (HP) para
        la primera señal está activado. Luego, llama a la función `actualizar_grafico()` para actualizar la visualización
        del gráfico con el filtro activo."""
        self.filter_1_HP = True  # Activa el filtro pasa alto para la señal 1
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro

    def update_filter_2(self):
        """Activa el filtro pasa alto para la segunda señal y actualiza el gráfico."""
        self.filter_2_HP = True  # Activa el filtro pasa alto para la señal 2
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro

    def update_filter_3(self):
        """Activa el filtro pasa alto para la tercera señal y actualiza el gráfico."""
        self.filter_3_HP = True  # Activa el filtro pasa alto para la señal 3
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro

    def update_filter_4(self):
        """Activa el filtro pasa alto para la cuarta señal y actualiza el gráfico."""
        self.filter_4_HP = True  # Activa el filtro pasa alto para la señal 4
        self.actualizar_grafico()  # Actualiza el gráfico para reflejar los cambios en el filtro

    def update_time_constant_1(self, value):
        """Actualiza la constante de tiempo para la primera señal.

        La función convierte el valor de entrada (en milisegundos) a segundos dividiendo por 1000 y lo asigna al atributo `time_c_1`.

        Args:
            value (int or float): Valor de la constante de tiempo en milisegundos."""
        self.time_c_1 = value / 1000  # Convierte el valor de milisegundos a segundos

    def update_time_constant_2(self, value):
        """Actualiza la constante de tiempo para la segunda señal.

        La función convierte el valor de entrada (en milisegundos) a segundos dividiendo por 1000 y lo asigna al
        atributo `time_c_2`.

        Args:
            value (int or float): Valor de la constante de tiempo en milisegundos."""
        self.time_c_2 = value / 1000  # Convierte el valor de milisegundos a segundos

    def update_time_constant_3(self, value):
        """Actualiza la constante de tiempo para la tercera señal."""
        self.time_c_3 = value / 1000  # Convierte el valor de milisegundos a segundos

    def update_time_constant_4(self, value):
        """Actualiza la constante de tiempo para la cuarta señal."""
        self.time_c_4 = value / 1000  # Convierte el valor de milisegundos a segundos

    def update_gain_1(self, value):
        """Actualiza el valor de ganancia 1 y refresca las gráficas.

           Args:
               value (float): El nuevo valor de la ganancia 1."""
        self.new_gain_1 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_gain_2(self, value):
        """Actualiza el valor de ganancia 2 y refresca las gráficas."""
        self.new_gain_2 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_gain_3(self, value):
        """Actualiza el valor de ganancia 3 y refresca las gráficas."""
        self.new_gain_3 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_gain_4(self, value):
        """Actualiza el valor de ganancia 4 y refresca las gráficas."""
        self.new_gain_4 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_offset_1(self, value):
        """Actualiza el valor del offset 1 y refresca las gráficas.

        Args:
            value (float): El nuevo valor del offset 1."""
        self.new_offset_1 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_offset_2(self, value):
        """Actualiza el valor del offset 2 y refresca las gráficas."""
        self.new_offset_2 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_offset_3(self, value):
        """Actualiza el valor del offset 3 y refresca las gráficas."""
        self.new_offset_3 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_offset_4(self, value):
        """Actualiza el valor del offset 4 y refresca las gráficas."""
        self.new_offset_4 = value
        # Llama a la función que actualiza las gráficas para reflejar el nuevo valor.
        self.update_plots()

    def update_window_time(self, value):
        """Actualiza el valor del tiempo de la ventana y configura los estados para mover la ventana.

        Args:
            value (float): El nuevo valor del tiempo de la ventana."""
        self.window_time = value
        self.avanza = False
        self.retrocede = False
        self.window = True
        # Llama a la función que mueve la ventana para reflejar el nuevo valor.
        self.mover_ventana()

    def avanzar_graficas(self):
        """Configura el estado para avanzar las gráficas."""
        self.avanza = True
        self.retrocede = False
        self.window = False
        # Llama a la función que mueve la ventana para avanzar las gráficas.
        self.mover_ventana()

    def retroceder_graficas(self):
        """Configura el estado para retroceder las gráficas."""
        self.retrocede = True
        self.avanza = False
        self.window = False
        # Llama a la función que mueve la ventana para retroceder las gráficas.
        self.mover_ventana()

    def save_in_update(self):
        """Guarda los valores actuales seleccionados en los widgets de guardado y actualiza las gráficas."""
        # Obtiene el texto actual de los widgets de guardado y los guarda en las variables correspondientes.
        self.s_1_save = self.save_in_1.currentText()
        self.s_2_save = self.save_in_2.currentText()
        self.s_3_save = self.save_in_3.currentText()
        self.s_4_save = self.save_in_4.currentText()
        # Llama a la función que actualiza las gráficas para reflejar los nuevos valores guardados.
        self.update_plots()

    def channel(self):
        """Actualiza los canales seleccionados y sus señales correspondientes."""
        # Obtiene los nuevos valores seleccionados en los combobox.
        new_s_1 = self.column_combobox_1.currentText()
        new_s_2 = self.column_combobox_2.currentText()
        new_s_3 = self.column_combobox_3.currentText()
        new_s_4 = self.column_combobox_5.currentText()

        # Si el nuevo valor del combobox 1 es diferente del actual, se actualizan las señales y los datos correspondientes.
        if new_s_1 != self.s_1:
            self.s_1 = new_s_1
            self.signal_1 = self.data[f'{self.s_1}'].to_numpy()
            self.signals[:, 0] = self.signal_1
            self.s_5 = new_s_1
            self.original_1 = self.data[f'{self.s_5}'].to_numpy()
            self.signals[:, 4] = self.original_1

        # Si el nuevo valor del combobox 2 es diferente del actual, se actualizan las señales y los datos correspondientes.
        if new_s_2 != self.s_2:
            self.s_2 = new_s_2
            self.signal_2 = self.data[f'{self.s_2}'].to_numpy()
            self.signals[:, 1] = self.signal_2
            self.s_6 = new_s_2
            self.original_2 = self.data[f'{self.s_6}'].to_numpy()
            self.signals[:, 5] = self.original_2

        # Si el nuevo valor del combobox 3 es diferente del actual, se actualizan las señales y los datos correspondientes.
        if new_s_3 != self.s_3:
            self.s_3 = new_s_3
            self.signal_3 = self.data[f'{self.s_3}'].to_numpy()
            self.signals[:, 2] = self.signal_3
            self.s_7 = new_s_3
            self.original_3 = self.data[f'{self.s_7}'].to_numpy()
            self.signals[:, 6] = self.original_3

        # Si el nuevo valor del combobox 4 es diferente del actual, se actualizan las señales y los datos correspondientes.
        if new_s_4 != self.s_4:
            self.s_4 = new_s_4
            self.signal_4 = self.data[f'{self.s_4}'].to_numpy()
            self.signals[:, 3] = self.signal_4
            self.s_8 = new_s_4
            self.original_4 = self.data[f'{self.s_8}'].to_numpy()
            self.signals[:, 7] = self.original_4

        # Actualiza el mapa de colores para los canales.
        self.color_map = {
            self.s_1: self.channel_color_map[self.s_1],
            self.s_2: self.channel_color_map[self.s_2],
            self.s_3: self.channel_color_map[self.s_3],
            self.s_4: self.channel_color_map[self.s_4]
        }

        # Guarda los nombres de los canales seleccionados.
        self.s_1_save = f"{self.s_1}"
        self.s_2_save = f"{self.s_2}"
        self.s_3_save = f"{self.s_3}"
        self.s_4_save = f"{self.s_4}"

        # Llama a la función que actualiza las gráficas para reflejar los nuevos valores.
        self.update_plots()

    def save_and_insert_data(self, name):
        """
        Guarda los datos procesados en un archivo CSV e inserta información adicional en el archivo.

        Args:
            name (str): Nombre base para el archivo CSV.

        """
        # Genera un timestamp y el nombre del archivo.
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nombre_archivo_data = f"{name}_{timestamp}.csv"
        self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior.

        # Ruta completa del archivo.
        file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)

        # Actualiza la copia de los datos con las nuevas señales y estímulos.
        self.data_copy[f'{self.new_stim}'] = self.stim
        self.data_copy[f'{self.s_1_save}'] = self.signal_1 * self.new_gain_1 + self.new_offset_1
        self.data_copy[f'{self.s_2_save}'] = self.signal_2 * self.new_gain_2 + self.new_offset_2
        self.data_copy[f'{self.s_3_save}'] = self.signal_3 * self.new_gain_3 + self.new_offset_3
        self.data_copy[f'{self.s_4_save}'] = self.signal_4 * self.new_gain_4 + self.new_offset_4
        data_to_save = self.data_copy.to_numpy()

        # Etiquetas de columnas.
        column_labels = ['TIME', 'CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                         'CH 10', 'CH 11', 'CH 12', 'TAG OUT', 'UP', 'DOWN', 'STIM 1', 'STIM 2', 'STIM 3', 'ACTIVE CYCLE',
                         'INACTIVE CYCLE', 'CYCLE TIME', 'FREQUENCY']

        # Guarda los datos en el archivo CSV.
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_labels)  # Escribir las etiquetas de columna.
            writer.writerows(data_to_save)  # Escribir las filas de datos.

        # Carga el archivo CSV en un DataFrame de pandas.
        df = pd.read_csv(file_path)
        num_columns = len(df.columns)

        # Lista de atributos para cada señal.
        signals_data = [
            (self.s_1_save, self.cutoff_1_LP, self.cutoff_1_HP, self.time_c_1, self.new_gain_1, self.new_offset_1),
            (self.s_2_save, self.cutoff_2_LP, self.cutoff_2_HP, self.time_c_2, self.new_gain_2, self.new_offset_2),
            (self.s_3_save, self.cutoff_3_LP, self.cutoff_3_HP, self.time_c_3, self.new_gain_3, self.new_offset_3),
            (self.s_4_save, self.cutoff_4_LP, self.cutoff_4_HP, self.time_c_4, self.new_gain_4, self.new_offset_4)
        ]

        # Inserta los datos de cada señal en el DataFrame.
        for idx, (s, cutoff_LP, cutoff_HP, time_c, gain, offset) in enumerate(signals_data):
            column_name = f'DATA {s}'
            df.insert(num_columns + idx, column_name, None)
            df.at[0, column_name] = f"cut off LP {cutoff_LP}"
            df.at[1, column_name] = f"cut off HP {cutoff_HP}"
            df.at[2, column_name] = f"T.C. {time_c}"
            df.at[3, column_name] = f"Gain {gain}"
            df.at[4, column_name] = f"Offset {offset}"
            df.at[5, column_name] = f"{self.new_stim}"

        # Guardar el DataFrame modificado de vuelta al mismo archivo CSV.
        df.to_csv(file_path, index=False)

        # Imprime un mensaje de confirmación.
        print(f"Datos guardados en {file_path}")

    def rectified_integrate_1(self):
        # Actualizar la señal original
        self.signals[:, 4] = self.original_1
        rect_1 = np.maximum(self.signal_1, 0)
        # Integrar la señal rectificada
        int_1 = self.integrating(rect_1, self.time_c_1)
        # Actualizar la señal en el arreglo de señales
        self.signals[:, 0] = int_1
        self.signal_1 = self.signals[:, 0]

        # Actualizar los gráficos
        self.update_plots()

    def rectified_integrate_2(self):
        # Actualizar la señal original
        self.signals[:, 5] = self.original_2  # Rectificar la señal
        rect_2 = np.maximum(self.signal_2, 0)
        # Integrar la señal rectificada
        int_2 = self.integrating(rect_2, self.time_c_2)
        # Actualizar la señal en el arreglo de señales
        self.signals[:, 1] = int_2
        self.signal_2 = self.signals[:, 1]

        # Actualizar los gráficos
        self.update_plots()

    def rectified_integrate_3(self):
        # Actualizar la señal original
        self.signals[:, 6] = self.original_3
        rect_3 = np.maximum(self.signal_3, 0)
        # Integrar la señal rectificada
        int_3 = self.integrating(rect_3, self.time_c_3)
        # Actualizar la señal en el arreglo de señales
        self.signals[:, 2] = int_3
        self.signal_3 = self.signals[:, 2]

        # Actualizar los gráficos
        self.update_plots()

    def rectified_integrate_4(self):
        # Actualizar la señal original
        self.signals[:, 7] = self.original_4
        rect_4 = np.maximum(self.signal_4, 0)
        # Integrar la señal rectificada
        int_4 = self.integrating(rect_4, self.time_c_4)
        # Normalizar la señal integrada
        # Actualizar la señal en el arreglo de señales
        self.signals[:, 3] = int_4
        self.signal_4 = self.signals[:, 3]
        # Actualizar los gráficos
        self.update_plots()

    def integrating(self, signal, tau):
        """Integra una señal usando un filtro de primer orden.

        Args:
            signal (numpy array): Señal de entrada a integrar.
            tau (float): Constante de tiempo del filtro.

        Returns:
            numpy array: Señal integrada."""
        dt = self.Ts  # Tiempo de muestreo
        integrated_signal = np.zeros_like(signal)  # Inicializa la señal integrada con ceros
        previous_value = 0  # Valor previo de la señal integrada

        # Itera sobre cada muestra de la señal
        for i in range(len(signal)):
            # Aplica la fórmula del filtro de primer orden
            integrated_signal[i] = previous_value + (signal[i] - previous_value) * dt / tau
            # Actualiza el valor previo
            previous_value = integrated_signal[i]

        return integrated_signal

    def filter_HP(self, signal, cutoff_HP):
        """Aplica un filtro pasa-altas Butterworth a la señal.

        Args:
            signal (numpy array): Señal de entrada a filtrar.
            cutoff_HP (float): Frecuencia de corte del filtro pasa-altas.

        Returns:
            numpy array: Señal filtrada."""
        order = 5  # Orden del filtro
        # Calcula los coeficientes del filtro pasa-altas Butterworth
        b, a = butter(order, cutoff_HP, btype='high', fs=self.fs)
        # Aplica el filtro a los datos
        y = lfilter(b, a, signal)
        return y

    def filter_LP(self, signal, cutoff_LP):
        """Aplica un filtro pasa-bajas Butterworth a la señal.

        Args:
            signal (numpy array): Señal de entrada a filtrar.
            cutoff_LP (float): Frecuencia de corte del filtro pasa-bajas.

        Returns:
            numpy array: Señal filtrada."""
        order = 5  # Orden del filtro
        # Calcula los coeficientes del filtro pasa-bajas Butterworth
        b, a = butter(order, cutoff_LP, btype='low', fs=self.fs)
        # Aplica el filtro a los datos
        z = lfilter(b, a, signal)
        return z

    def on_plot_clicked(self, event):
        """Maneja el evento de clic en el gráfico.

        Args:
            event (QEvent): Evento de clic."""
        modifiers = QApplication.keyboardModifiers()
        # Si se hace clic con el botón izquierdo del ratón y se mantiene presionada la tecla Ctrl
        if event.button() == Qt.MouseButton.LeftButton and (modifiers & Qt.KeyboardModifier.ControlModifier):
            pos = event.scenePos()  # Obtiene la posición del clic en la escena
            for plot_widget, title in self.plot_widgets.items():
                pos_mapped = plot_widget.plotItem.vb.mapSceneToView(pos)  # Mapea la posición a la vista del gráfico
                index_clicked = np.searchsorted(self.time,
                                                pos_mapped.x())  # Encuentra el índice correspondiente en el vector de tiempo

                if 0 <= index_clicked < len(self.time):
                    ventana_indices = 100  # Tamaño de la ventana de búsqueda
                    inicio_ventana = max(0, index_clicked - ventana_indices)
                    fin_ventana = min(len(self.time), index_clicked + ventana_indices + 1)

                    # Encuentra los índices donde self.stim es 1 o 2 dentro de la ventana
                    stim_indices = np.where((self.stim == 1) | (self.stim == 2))[0]
                    indices_ventana = stim_indices[(stim_indices >= inicio_ventana) & (stim_indices < fin_ventana)]

                    if len(indices_ventana) > 0:
                        # Si hay índices dentro de la ventana, marca el más cercano como 0 en self.stim
                        closest_index = min(indices_ventana, key=lambda x: abs(x - index_clicked))
                        self.stim[closest_index] = 0
                        self.update_plots()  # Actualiza los gráficos
                    else:
                        print("No se encontraron índices de estimulación (1 o 2) dentro de la ventana")

        # Si se hace clic con el botón izquierdo del ratón sin la tecla Ctrl
        elif event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()  # Obtiene la posición del clic en la escena
            for plot_widget, title in self.plot_widgets.items():
                pos_mapped = plot_widget.plotItem.vb.mapSceneToView(pos)  # Mapea la posición a la vista del gráfico
                index_clicked = np.searchsorted(self.time,
                                                pos_mapped.x())  # Encuentra el índice correspondiente en el vector de tiempo

                if 0 <= index_clicked < len(self.time):
                    # Agregar un 1 al vector stim en el índice correspondiente
                    self.stim[index_clicked] = 1
                    self.update_plots()  # Actualiza los gráficos

    def actualizar_grafico(self):
        """Aplica filtros de paso bajo y paso alto a las señales y actualiza los gráficos correspondientes.

        Este método realiza las siguientes acciones:
        1. Filtra las señales originales utilizando los filtros de paso bajo y paso alto, si están habilitados.
        2. Actualiza las señales almacenadas y los atributos relacionados.
        3. Actualiza los gráficos para reflejar los cambios en las señales filtradas."""
        # Listas con las frecuencias de corte para filtros de paso bajo (LP) y paso alto (HP)
        cutoffs_LP = [self.cutoff_1_LP, self.cutoff_2_LP, self.cutoff_3_LP, self.cutoff_4_LP]
        cutoffs_HP = [self.cutoff_1_HP, self.cutoff_2_HP, self.cutoff_3_HP, self.cutoff_4_HP]

        # Indicadores de si se deben aplicar los filtros de paso bajo y paso alto
        filters_LP = [self.filter_1_LP, self.filter_2_LP, self.filter_3_LP, self.filter_4_LP]
        filters_HP = [self.filter_1_HP, self.filter_2_HP, self.filter_3_HP, self.filter_4_HP]

        # Aplicar el filtro de paso bajo a cada señal si el indicador está activado
        for i in range(4):
            if filters_LP[i]:
                # Obtener la señal original correspondiente
                original_signal = getattr(self, f'original_{i + 1}')
                # Filtrar la señal original usando el filtro de paso bajo
                filtered_signal = self.filter_LP(original_signal, cutoffs_LP[i])
                # Almacenar la señal filtrada en la matriz de señales
                self.signals[:, i] = filtered_signal
                # Actualizar el atributo de la señal filtrada correspondiente
                setattr(self, f'signal_{i + 1}', filtered_signal)
                # Desactivar el indicador de filtro de paso bajo para la señal
                setattr(self, f'filter_{i + 1}_LP', False)

        # Aplicar el filtro de paso alto a cada señal si el indicador está activado
        for i in range(4):
            if filters_HP[i]:
                # Filtrar la señal ya filtrada con el filtro de paso alto
                filtered_signal = self.filter_HP(self.signals[:, i], cutoffs_HP[i])
                # Almacenar la señal filtrada en la matriz de señales
                self.signals[:, i] = filtered_signal
                # Actualizar el atributo de la señal filtrada correspondiente
                setattr(self, f'signal_{i + 1}', filtered_signal)
                # Desactivar el indicador de filtro de paso alto para la señal
                setattr(self, f'filter_{i + 1}_HP', False)

        # Actualizar los gráficos con las señales filtradas
        self.update_plots()

    def mover_ventana(self):
        """Ajusta la ventana de visualización de acuerdo con el estado de avance, retroceso o ventana fija.

        Dependiendo de los flags `self.avanza`, `self.retrocede` y `self.window`, esta función:
        - Avanza la ventana de visualización hacia adelante.
        - Retrocede la ventana de visualización hacia atrás.
        - Establece una ventana de visualización fija de tamaño `self.window_time`.

        También actualiza los valores de los controles de tiempo `t_inic_a` y `t_fin_a` y calcula los índices de datos
        correspondientes."""
        # Avanzar la ventana de visualización si `self.avanza` es True
        if self.avanza:
            self.start_a = self.start_a + self.window_time
            self.end_a = self.start_a + self.window_time
            # Actualiza los controles de tiempo con los nuevos valores
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)

        # Retroceder la ventana de visualización si `self.retrocede` es True
        if self.retrocede:
            self.end_a = self.start_a
            self.start_a = self.end_a - self.window_time
            # Actualiza los controles de tiempo con los nuevos valores
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)

        # Establecer una ventana fija si `self.window` es True
        if self.window:
            self.start_a = self.start_a
            self.end_a = self.window_time + self.start_a
            # Actualiza los controles de tiempo con los nuevos valores
            self.t_inic_a.setValue(self.start_a)
            self.t_fin_a.setValue(self.end_a)
            # Desactiva la flag de ventana fija
            self.window = False

        # Calcula los índices de datos correspondientes basados en el tiempo de inicio y fin
        self.start_a_num = round(self.start_a * self.datos_por_s)
        self.end_a_num = round(self.end_a * self.datos_por_s)

        # Actualiza los gráficos con la nueva ventana de visualización
        self.update_plots()

    def update_plots(self):
        """Actualiza los gráficos de señales y estímulos en los widgets de visualización.

        1. Actualiza las etiquetas y gráficos de estímulo.
        2. Actualiza los gráficos de las señales originales y filtradas en los widgets de visualización.
        3. Ajusta las etiquetas, rangos y títulos de los gráficos."""
        # Actualización de gráficos de estímulo y etiquetas
        stim_tags = [(self.stim, self.stim_plot_1), (self.stim_plot_1, self.stim_plot_2),
                     (self.stim_plot_2, self.stim_plot_3), (self.stim_plot_3, self.stim)]

        # Asigna los datos de estímulo para los gráficos de etiqueta y estímulo
        for i, (stim, stim_plot) in enumerate(stim_tags):
            setattr(self, f'tag_graph_{i + 1}', np.where(stim_plot == 1, self.stim, 0))
            setattr(self, f'stim_graph_{i + 1}', np.where(stim_plot == 2, self.stim, 0))

        # Definición de los widgets y sus parámetros
        plot_widgets = [self.plot_widget_1, self.plot_widget_2, self.plot_widget_3, self.plot_widget_4]
        signals = [self.signal_1, self.signal_2, self.signal_3, self.signal_4]
        originals = [self.original_1, self.original_2, self.original_3, self.original_4]
        gains = [self.new_gain_1, self.new_gain_2, self.new_gain_3, self.new_gain_4]
        offsets = [self.new_offset_1, self.new_offset_2, self.new_offset_3, self.new_offset_4]
        titles = [self.s_1, self.s_2, self.s_3, self.s_4]
        min_amplitudes = [self.min_amplitude_1, self.min_amplitude_2, self.min_amplitude_3, self.min_amplitude_4]
        max_amplitudes = [self.max_amplitude_1, self.max_amplitude_2, self.max_amplitude_3, self.max_amplitude_4]

        # Actualiza cada widget de gráfico con las señales correspondientes
        for i, widget in enumerate(plot_widgets):
            widget.clear()  # Limpia el gráfico actual del widget

            # Grafica la señal original (no filtrada) en el intervalo de tiempo seleccionado
            widget.plot(self.time[self.start_a_num: self.end_a_num], originals[i][self.start_a_num: self.end_a_num],
                        pen=pg.mkPen(color=self.filtered_color, width=self.width_line))

            # Grafica la señal filtrada con ganancia y offset aplicados
            widget.plot(self.time[self.start_a_num: self.end_a_num],
                        ((signals[i][self.start_a_num: self.end_a_num] * gains[i]) + offsets[i]),
                        pen=pg.mkPen(color=self.color_map[titles[i]], width=self.width_line), name='Señal original')

            # Grafica las etiquetas y estímulos para cada señal
            for j in range(1, 5):
                widget.plot(self.time[self.start_a_num: self.end_a_num],
                            getattr(self, f'stim_graph_{j}')[self.start_a_num: self.end_a_num],
                            pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line,
                                         style=QtCore.Qt.PenStyle.DashLine))
                widget.plot(self.time[self.start_a_num: self.end_a_num],
                            getattr(self, f'tag_graph_{j}')[self.start_a_num: self.end_a_num],
                            pen=pg.mkPen(color=self.stim_map[self.new_stim], width=self.width_line))

            # Configura las etiquetas y rangos de los ejes del gráfico
            widget.setLabel('left', 'Amplitude [V]')
            widget.setLabel('bottom', 'Time [s]')
            widget.setTitle(f'{titles[i]}')
            widget.setXRange(self.start_a, self.end_a)
            widget.setYRange(min_amplitudes[i], max_amplitudes[i])

    def save_fig(self, name):
        """Guarda la figura actual como un archivo SVG con un nombre basado en el timestamp actual.

        Parámetros:
        - name: Nombre base para el archivo de la figura.

        El archivo se guarda en la carpeta especificada por self.carpeta_datos."""
        # Obtener la marca de tiempo actual en el formato YYYYMMDDHHMMSS
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Crear el nombre del archivo con el nombre base y el timestamp
        nombre_archivo_actualizado = f"{name}_{timestamp}.svg"

        # Construir la ruta completa para guardar la imagen
        ruta_imagen_actualizada = os.path.join(self.carpeta_datos, nombre_archivo_actualizado)

        # Guardar la figura actual en la ruta especificada como un archivo SVG
        plt.savefig(ruta_imagen_actualizada, format='svg')

        # Imprimir un mensaje confirmando que el archivo ha sido guardado
        print(f"ARCHIVO GUARDADO {nombre_archivo_actualizado}")


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

