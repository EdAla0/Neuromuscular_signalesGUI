import os
import sys
import time
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QMessageBox, QInputDialog, QFileDialog
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QSemaphore, QTime, QTimer, QMutex
from PyQt6.uic import loadUi
import pyqtgraph as pg
import nidaqmx
import numpy as np
import logging
import csv
import pandas as pd
import subprocess
import matplotlib
matplotlib.use('Qt5Agg')


class SaveThread(QObject):
    finished = pyqtSignal()  # Señal para indicar que la ejecución del código ha terminado

    def __init__(self):
        super().__init__()
        self.stopped = False # Flag para indicar si el hilo se ha detenido
        self.counter = 1 # Contador para numerar los archivos guardados
        self.folder_name = "" # Nombre de la carpeta seleccionada
        self.base_name = "" # Nombre base del archivo
        self.filename = "" # Nombre del archivo completo
        self.csv_file_path = "" # Ruta del archivo CSV
        self.saved_files = [] # Lista de archivos guardados
        self.column_labels = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                              'CH 10', 'CH 11', 'CH 12', 'TAG OUT', 'UP', 'DOWN', 'STIM 1', 'STIM 2', 'STIM 3']
        self.select_folder_and_base_file() # Método para seleccionar la carpeta y archivo base

    def select_folder_and_base_file(self):
        """Selecciona la carpeta de destino y el archivo base."""
        # Abre un diálogo para seleccionar la carpeta de destino
        self.folder_name = QFileDialog.getExistingDirectory(None, "Seleccionar carpeta de destino")

        # Si no se selecciona ninguna carpeta (el usuario cancela el diálogo), retorna sin hacer nada
        if not self.folder_name:
            return
        # Abre un diálogo para seleccionar o crear un archivo base dentro de la carpeta seleccionada
        base_file, _ = QFileDialog.getSaveFileName(None, "Seleccionar o crear archivo base", self.folder_name)

        # Si no se selecciona ningún archivo (el usuario cancela el diálogo), retorna sin hacer nada
        if not base_file:
            return

        # Extrae el nombre base del archivo (sin extensión) y lo guarda en `self.base_name`
        self.base_name = os.path.splitext(os.path.basename(base_file))[0] # Obtiene el nombre base del archivo

    def generate_filename(self):
        """Genera un nuevo nombre de archivo basado en el contador."""
        self.filename = os.path.join(self.folder_name, f"{self.base_name}_{self.counter}.csv")
        self.counter += 1
        self.saved_files.append(os.path.basename(self.filename)) # Agrega el archivo generado a la lista

    def stop(self):
        """Detiene el hilo y realiza el procesamiento posterior."""
        self.stopped = True
        thread_a.terminate()
        self.perform_post_processing()

    def perform_post_processing(self):
        """Realiza el procesamiento posterior a los archivos guardados."""

        # Itera sobre cada archivo guardado en la lista `saved_files`
        for saved_file in self.saved_files:
            # Construye la ruta completa del archivo
            file_path = os.path.join(self.folder_name, saved_file)

            # Lee el archivo CSV en un DataFrame de pandas
            df = pd.read_csv(file_path)

            # Inserta la columna de tiempo
            # `thread_a.sample_time` se asume que es el incremento de tiempo entre muestras
            increment = thread_a.sample_time
            df.insert(0, 'TIME', [0] + [increment * i for i in range(1, len(df))])

            # Calcula los ciclos activos e inactivos
            # Un ciclo activo ocurre entre 'DOWN' y 'UP', y un ciclo inactivo entre 'UP' y 'DOWN'
            df['ACTIVE CYCLE'] = df.apply(lambda row: self.calculate_cycle_duration(row, df, 'DOWN', 'UP'), axis=1)
            df['INACTIVE CYCLE'] = df.apply(lambda row: self.calculate_cycle_duration(row, df, 'UP', 'DOWN'), axis=1)

            # Calcula el tiempo de ciclo y la frecuencia
            # Encuentra los índices donde 'UP' es 1 (indica el inicio de un ciclo)
            up_indices = df[df['UP'] == 1].index

            # Calcula la diferencia de tiempo entre inicios consecutivos de ciclos
            delta_cycle = [df.at[up_indices[i + 1], 'TIME'] - df.at[up_indices[i], 'TIME'] if i < len(up_indices) - 1 else None for i in range(len(up_indices))]

            # Calcula la diferencia de tiempo entre inicios consecutivos de ciclos
            df['CYCLE TIME'] = pd.Series(delta_cycle, index=up_indices)

            # Calcula la frecuencia como el inverso del tiempo promedio de ciclo
            frequency_value = 1 / df['CYCLE TIME'].mean()

            # Inserta la frecuencia en la primera fila de la columna 'FREQUENCY'
            df.at[0, 'FREQUENCY'] = frequency_value

            # Llena las demás filas de la columna 'FREQUENCY' con None
            df['FREQUENCY'] = df['FREQUENCY'].where(df.index == 0, None)

            # Guarda el archivo CSV con los datos procesados
            df.to_csv(file_path, index=False)

            # Imprime un mensaje indicando que los datos han sido insertados en el archivo procesado
            print("DATOS INSERTADOS", saved_file)

    def calculate_cycle_duration(self, row, df, start_label, end_label):
        """Calcula la duración del ciclo entre dos eventos."""
        if row[start_label] == 1:
            index_start = row.name
            index_end = df.loc[:index_start, end_label][::-1].idxmax()
            return row['TIME'] - df.at[index_end, 'TIME']
        return None

    def save_data(self, data, cross_up, cross_down, stim_1, stim_2, stim_3):
        """Guarda los datos en un archivo CSV."""
        try:
            # Verifica si se ha seleccionado una carpeta
            if not self.folder_name:
                print("Por favor, seleccione una carpeta para guardar los archivos.")
                return

            # Si el hilo no ha sido detenido y hay un nombre de archivo disponible
            if not self.stopped and self.filename:
                # Abre el archivo en modo de añadir ('a')
                with open(self.filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    # Si el archivo está vacío, escribe la fila de etiquetas de columna
                    if csvfile.tell() == 0:
                        writer.writerow(self.column_labels)

                    # Transpone los datos y los combina en una sola lista
                    transposed_data = list(map(list, zip(*data, cross_up, cross_down, stim_1, stim_2, stim_3)))

                    # Escribe las filas de datos en el archivo CSV
                    writer.writerows(transposed_data)

                    # Emite una señal indicando que se ha terminado de guardar los datos
                    self.finished.emit()

            # Si el hilo ha sido detenido, emite la señal de terminado
            if self.stopped:
                self.finished.emit()

        except Exception as e:
            # Registra cualquier error que ocurra durante la ejecución del guardado de datos
            logging.error(f'Error en el hilo SaveData: {e}')

# Inicializa la interfaz
class VentanaPrincipal(QMainWindow):

    def __init__(self, number_of_samples, samp_per_iteration, tmax, tiempo, sample_rate):
        super().__init__()

        # Carga la interfaz de usuario desde el archivo .ui
        loadUi('PRINCIPAL_GUI_CON_TH_BAJADA.ui', self)

        # Llama a la función para detectar dispositivos NIDAQ y obtener el nombre del dispositivo
        self.dev_name = None
        self.detect_nidaq_devices()

        # Configura el diseño del widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)
        central_widget.setLayout(layout)

        # Conecta los controles de amplitud máxima y mínima a la función de actualización de rango en Y
        self.SB_Amplitude_max.valueChanged.connect(self.update_y_range)
        self.SB_Amplitude_min.valueChanged.connect(self.update_y_range)

        # Conecta los controles de ejes X y tasa de muestreo a sus funciones de actualización correspondientes
        self.SB_X_axis.valueChanged.connect(self.update_number_of_samples)
        self.SB_sample_rate.valueChanged.connect(self.update_sample_rate)

        # Conecta los controles de histéresis ascendente y descendente a sus funciones de actualización
        self.SB_Hysteresis_asc.valueChanged.connect(self.update_hysteresis_up)
        self.hysteresis_up = 0.0
        self.SB_Hysteresis_desc.valueChanged.connect(self.update_hysteresis_down)
        self.hysteresis_down = 0.0

        # Conecta el control del umbral a su función de actualización
        self.SB_Threshold.valueChanged.connect(self.update_threshold)
        self.threshold = 0.0

        # Conecta el control de hold off a su función de actualización
        self.SB_Hold_off.valueChanged.connect(self.update_hold_off)

        # Conecta el control de delay a su función de actualización
        self.SB_Delay.valueChanged.connect(self.update_delay)

        # Opciones del menú desplegable de canales
        options = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Channel 5", "Channel 6",
                   "Channel 7", "Channel 8", "Channel 9", "Channel 10", "Channel 11", "Channel 12", "Channel 13"]

        self.channel_colors = ['#23BAC4', '#DAA520', '#E69DFB','#F08080','#FF0000' , '#00FF00','#7FFFD4' ,'#FF9900' ,'#DA70D6', '#E7D40A', '#F08080', '#FF689D', '#FFFFFF']
        self.stim_color_1 = '#FF9900'
        self.stim_color_2 = '#E7D40A'
        self.stim_color_3 = '#F08080'
        self.threshold_color = '#FFFFFF'
        self.line_up_color = '#FFFFFF'
        self.line_down_color = '#FFFFFF'
        self.stim_colors = [self.stim_color_1, self.stim_color_2, self.stim_color_3, self.threshold_color]

        # Agrega las opciones al menú desplegable y conecta la señal de cambio de índice a sus funciones
        self.Amount.addItems(options)
        self.Amount.currentIndexChanged.connect(self.reset_crosses_count)
        self.Amount.currentIndexChanged.connect(self.clear_lines_and_indices)

        # Configura el gráfico principal y lo agrega al diseño
        self.main_graph = CanvasGraph(self.channel_colors, self.stim_color_1, self.stim_color_2, self.stim_color_3,
                                      self.threshold_color, self.line_up_color, self.line_down_color, number_of_samples,
                                      samp_per_iteration, tmax, tiempo, sample_rate)
        self.Graph_layout.addWidget(self.main_graph)

        # Conecta los controles de offset a sus funciones de actualización en el gráfico principal
        self.offset_1.valueChanged.connect(self.main_graph.update_offset_1)
        self.offset_2.valueChanged.connect(self.main_graph.update_offset_2)
        self.offset_3.valueChanged.connect(self.main_graph.update_offset_3)
        self.offset_4.valueChanged.connect(self.main_graph.update_offset_4)
        self.offset_5.valueChanged.connect(self.main_graph.update_offset_5)
        self.offset_6.valueChanged.connect(self.main_graph.update_offset_6)
        self.offset_7.valueChanged.connect(self.main_graph.update_offset_7)
        self.offset_8.valueChanged.connect(self.main_graph.update_offset_8)
        self.offset_9.valueChanged.connect(self.main_graph.update_offset_9)
        self.offset_10.valueChanged.connect(self.main_graph.update_offset_10)
        self.offset_11.valueChanged.connect(self.main_graph.update_offset_11)
        self.offset_12.valueChanged.connect(self.main_graph.update_offset_12)

        # Configura el botón de análisis y su conexión ----------------------------------------------------------
        self.bt_go.clicked.connect(self.abrir_dialogo_archivo)
        # Configura el botón de conversión ATF a CSV ----------------------------------------------------------
        self.bt_atf.clicked.connect(self.abrir_dialogo_archivo_ATF)

        # Configura el menú desplegable de análisis y conecta las señales de cambio de índice a sus funciones
        self.selected_analysis_1 = 0 #Se inician en la primer opción
        self.selected_analysis_2 = 0 #Se inician en la primer opción

        self.options_analysis = ["Bins", "Cycle duration and phases", "FFT", "Autocorrelation", "Pearson correlation", "Spike triggered averaging", "Cross Wavelet XWT", "Wavelet coherence CWT", "Coherent power CXWT", "PCA"]
        
        self.options_analysis_2 = ["None", "Bins", "Cycle duration and phases"]
        
        self.Analysis_1.addItems(self.options_analysis)
        # Conectar la señal currentIndexChanged a la función correspondiente
        self.Analysis_1.currentIndexChanged.connect(self.update_Analysis_1)

        self.Analysis_2.addItems(self.options_analysis_2)
        # Conectar la señal currentIndexChanged a la función correspondiente
        self.Analysis_2.currentIndexChanged.connect(self.update_Analysis_2)

        # Conecta el botón de rectificado e integrado a su función
        self.bt_explore.clicked.connect(self.rectificado_int_update)

        # Configura el loop de botones de radio para actualizar el conteo máximo
        # de cruces en el protocolo("Loop")
        for i in range(1, 11):
            radio_button = getattr(self, f"Loop_{i}")
            radio_button.setChecked(False)
            radio_button.clicked.connect(lambda _, value=i: self.update_crosses_count_max(value))

        # Configura y alterna los botones de los canales para la visibilidad de las señales.
        i = 0
        while i < 13:
            button = getattr(self, f'ENG_{i + 1}')
            button.clicked.connect(lambda _, index=i: self.toggle_channel_visibility(index))
            self.set_channel_visibility(i, False)
            i += 1

        # Inicializa las variables de estimulación y sus botones de conexión
        self.stim_1 = False
        self.stim_2 = False
        self.stim_3 = False

        # Diccionario para almacenar los botones activados por cada estimulador
        self.estimuladores_botones_activos = {1: [], 2: [], 3: []}

        # Inicializar el valor de crosses_count_max en 1
        self.crosses_count_max = 1

        # Incializar la bandera de guardado de datos en "False"
        self.saving = False

        # Botones para la generación del pulso TTL en el protocolo
        self.bt_stimulation_1.clicked.connect(self.stim_active)
        self.bt_stimulation_2.clicked.connect(self.stim_active)
        self.bt_stimulation_3.clicked.connect(self.stim_active)

        # Conecta el botón de stop a su función
        self.bt_stop.clicked.connect(self.stop_trig)

        # Inicializa variables de los parámetros para la detección de cruce
        self.channel_index = 0
        self.threshold = 0
        self.hold_off = 0
        self.delay_time = 0

        # Configura el temporizador usado en la función del guardado de los datos
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.time_inic = None


    def stim_active(self):
        """
            Alterna el estado de activación de los botones de estimulación (stim_1, stim_2, stim_3).
            Llama a una función para actualizar el estado de estimulación en un hilo.
            """
        # Obtiene el botón que envió la señal
        sender = self.sender()

        # Alterna el estado booleano de stim_1 si el botón presionado es bt_stimulation_1
        if sender == self.bt_stimulation_1:
            self.stim_1 = not self.stim_1
        # Alterna el estado booleano de stim_2 si el botón presionado es bt_stimulation_2
        elif sender == self.bt_stimulation_2:
            self.stim_2 = not self.stim_2
        # Alterna el estado booleano de stim_3 si el botón presionado es bt_stimulation_3
        elif sender == self.bt_stimulation_3:
            self.stim_3 = not self.stim_3

        # Actualiza el estado de estimulación en el hilo thread_index con los nuevos valores de stim_1, stim_2 y stim_3
        thread_index.stim_active_update(self.stim_1, self.stim_2, self.stim_3)

    def stop_trig(self):
        # Detiene el temporizador
        self.timer.stop()
        # Detiene el temporizador
        save_thread.stop()

    def save_data(self):
        """Inicia o detiene el proceso de guardado de datos basado en el estado de `self.saving`."""
        # Si no se está guardando actualmente, inicia el proceso de guardado
        if not self.saving:
            # Genera un nuevo nombre de archivo para guardar los datos
            save_thread.generate_filename()
            # Actualiza el hilo thread_a para comenzar a guardar datos
            thread_a.save_data_update(True)
            # Actualiza el hilo thread_index para comenzar a guardar datos
            thread_index.save_data_update(True)
            # Conecta la señal save_data del hilo thread_a a la función save_data del hilo save_thread
            thread_a.save_data.connect(save_thread.save_data)
            # Cambia el texto del botón bt_save a "Stop saving"
            self.bt_save.setText("Stop saving")
            # Inicia el temporizador para actualizar el tiempo cada segundo
            self.timer.start(1000)
            # Registra el tiempo actual como el tiempo inicial
            self.time_inic = QTime.currentTime()
            # Actualiza el tiempo mostrado en el display
            self.update_time()
            # Cambia el estado de self.saving a True
            self.saving = True

        # Si ya se está guardando, detiene el proceso de guardado
        else:
            # Detiene el temporizador
            self.timer.stop()

            # Actualiza el hilo thread_a para detener el guardado de datos
            thread_a.save_data_update(False)

            # Actualiza el hilo thread_index para detener el guardado de datos
            thread_index.save_data_update(False)

            # Cambia el estado de self.saving a False
            self.saving = False

            # Cambia el texto del botón bt_save a "Save data"
            self.bt_save.setText("Save data")
    def update_time(self):
        """Actualiza el display del tiempo en el LCD (`Tempo_lcd`) con el tiempo transcurrido desde `self.time_inic`."""
        # Si el tiempo inicial está definido
        if self.time_inic is not None:
            # Obtiene el tiempo actual
            current_time = QTime.currentTime()
            # Calcula la diferencia en segundos entre el tiempo inicial y el tiempo actual
            time_diff = self.time_inic.secsTo(current_time)
            # Convierte la diferencia de tiempo en una cadena con el formato "hh:mm:ss"
            time_str = QTime(0, 0).addSecs(time_diff).toString("hh:mm:ss")
            # Muestra la cadena de tiempo en el display LCD
            self.Tempo_lcd.display(time_str)

    def detect_nidaq_devices(self):
        """ Detecta dispositivos NIDAQmx conectados al sistema local.
        Si se detectan dispositivos, muestra una lista para que el usuario seleccione uno.
        Actualiza los atributos dev_name y dev_model con el dispositivo seleccionado."""

        try:
            # Obtiene el sistema NIDAQmx local
            system = nidaqmx.system.System.local()
            # Obtiene la lista de dispositivos conectados al sistema
            devices = system.devices

            # Si no se encuentran dispositivos, muestra una advertencia y sale de la función
            if not devices:
                QMessageBox.warning(self, "Advertencia", "No se encontraron dispositivos NIDAQmx conectados.")
                return

            # Crea una lista de nombres de dispositivos con sus tipos de producto
            device_names = [f"{device.name} - {device.product_type}" for device in devices]

            # Muestra un cuadro de diálogo para que el usuario seleccione un dispositivo de la lista
            selected_device_name, ok = QInputDialog.getItem(self, "Seleccionar dispositivo", "Dispositivos NIDAQmx:",
                                                            device_names, 0, False)

            # Si el usuario selecciona un dispositivo (ok es True)
            if ok:
                # Obtiene el índice del dispositivo seleccionado en la lista
                selected_device_index = device_names.index(selected_device_name)

                # Obtiene el dispositivo seleccionado usando el índice
                selected_device = devices[selected_device_index]

                # Actualiza los atributos dev_name y dev_model con el nombre y tipo del dispositivo seleccionado
                self.dev_name = selected_device.name
                self.dev_model = selected_device.product_type

            else:
                # Si el usuario no selecciona ningún dispositivo, muestra una advertencia
                QMessageBox.warning(self, "Advertencia", "No se seleccionó ningún dispositivo.")

        except Exception as e:
            # Si ocurre un error, muestra un mensaje de error con la descripción del error
            QMessageBox.critical(self, "Error", f"Error al buscar dispositivos NIDAQmx: {str(e)}")

    def update_number_of_samples(self, value):
        """Actualiza el número de muestras en diferentes componentes."""
        # Actualiza el número de muestras en el objeto thread_a
        thread_a.set_number_of_samples(value)

        # Actualiza el número de muestras en el gráfico principal
        self.main_graph.set_number_of_samples(value)

        # Actualiza el número de muestras en el objeto thread_index
        thread_index.set_number_of_samples(value)

    def update_y_range(self):
        """Actualiza el rango de amplitud en el gráfico principal."""
        # Obtiene el valor mínimo de amplitud desde el control SB_Amplitude_min
        min_amplitude = self.SB_Amplitude_min.value()
        # Obtiene el valor máximo de amplitud desde el control SB_Amplitude_max
        max_amplitude = self.SB_Amplitude_max.value()
        # Establece el rango de amplitud en el gráfico principal
        self.main_graph.set_y_range(min_amplitude, max_amplitude)

    def toggle_channel_visibility(self, channel_index):
        """Alterna la visibilidad de un canal en el gráfico principal."""
        # Verifica si el botón de radio para el canal está activado
        is_checked = self.radio_is_active(channel_index)
        # Establece la visibilidad del canal en el gráfico principal
        self.set_channel_visibility(channel_index, is_checked)

    def set_channel_visibility(self, channel_index, visible):
        """Establece la visibilidad de un canal específico en el gráfico principal."""
        # Obtiene la curva correspondiente al canal en el gráfico principal
        curve = self.main_graph.curves[channel_index]
        # Establece la visibilidad de la curva
        curve.setVisible(visible)

    def radio_is_active(self, radio_index):
        """Verifica si el botón de radio para un canal específico está activado."""
        # Obtiene el botón de radio correspondiente al canal
        button = getattr(self, f'ENG_{radio_index + 1}')
        # Devuelve True si el botón de radio está activado, False en caso contrario
        return button.isChecked()

    def update_threshold(self, value):
        """Actualiza el umbral y lo comunica a los componentes relevantes."""
        # Actualiza el valor del umbral en la instancia actual
        self.threshold = value
        # Actualiza el valor del umbral en el objeto thread_a
        thread_a.update_threshold(value)
    def update_sample_rate(self, sample_rate):
        """Actualiza la tasa de muestreo y lo comunica a los componentes relevantes."""
        # Verifica que la tasa de muestreo sea mayor a 0
        if sample_rate > 0:
            # Establece la tasa de muestreo en el objeto thread_a
            thread_a.set_sample_rate(sample_rate)
            # Establece la tasa de muestreo en el gráfico principal
            self.main_graph.set_sample_rate(sample_rate)
            # Establece la tasa de muestreo en el objeto thread_index
            thread_index.set_sample_rate(sample_rate)

    def update_hysteresis_up(self, hysteresis):
        """Actualiza el valor de la histéresis ascendente."""
        self.hysteresis_up = hysteresis

    def update_hysteresis_down(self, hysteresis_d):
        """Actualiza el valor de la histéresis descendente."""
        self.hysteresis_down = hysteresis_d

    def update_lcd(self, crosses_count):
        """Actualiza el display LCD con el número de cruces detectados."""
        self.LCD_cycles_detected.display(crosses_count)

    def update_Analysis_1(self, index):
        """Actualiza la selección del primer análisis."""
        self.selected_analysis_1 = index

    def update_Analysis_2(self, index):
        """Actualiza la selección del segundo análisis."""
        self.selected_analysis_2 = index

    def update_Analysis_3(self, index):
        """Actualiza la selección del tercer análisis."""
        self.selected_analysis_3 = index

    def abrir_dialogo_archivo(self):
        """Abre un cuadro de diálogo para seleccionar un archivo CSV y ejecuta los análisis seleccionados."""
        # Abre el cuadro de diálogo para seleccionar un archivo
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos CSV (*.csv)")

        # Verifica si se seleccionó un archivo
        if archivo:
            # Actualiza la ruta del archivo en todos los análisis
            self.ruta_archivo = archivo
            # Ejecuta los análisis seleccionados utilizando la ruta del archivo
            self.execute_selected_analyses()
            
    def abrir_dialogo_archivo_ATF(self):
        """Abre un cuadro de diálogo para seleccionar un archivo ATF para su conversión a CSV."""
        # Abre el cuadro de diálogo para seleccionar un archivo
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos ATF (*.atf)")

        # Verifica si se seleccionó un archivo
        if archivo:
            # Actualiza la ruta del archivo en todos los análisis
            self.ruta_archivo = archivo
            
            # Crea una lista de valores que serán utilizados como parámetros para los análisis.
            self.values = [self.channel_index, self.threshold, self.hold_off, self.delay_time, self.hysteresis_up, self.hysteresis_down]
            
            self.open_Read_ATF(self.values, self.channel_colors, self.stim_colors)


    def execute_selected_analyses(self):
        """Ejecuta los análisis seleccionados en función de los índices seleccionados para los análisis 1 y 2."""
        # Crea una lista de valores que serán utilizados como parámetros para los análisis.
        self.values = [self.channel_index, self.threshold, self.hold_off, self.delay_time, self.hysteresis_up, self.hysteresis_down]

        # Verificar la selección del primer análisis
        if self.selected_analysis_1 is not None:

            # Si el análisis 1 está seleccionado y es 0 (análisis de bins)
            if self.selected_analysis_1 == 0:
                self.open_bins_analysis(self.values, self.channel_colors, self.stim_colors)

            # Si el análisis 1 está seleccionado y es 1 (análisis de duración de ciclo y fases)
            elif self.selected_analysis_1 == 1:
                
                self.open_phase_and_cycle_duration_analysis(self.values, self.channel_colors, self.stim_colors)
            
            # Si el análisis 2 está seleccionado (FFT)
            elif self.selected_analysis_1 == 2:
                
                self.open_FFT(self.values, self.channel_colors, self.stim_colors)
            
            # Si el análisis 3 está seleccionado (Autocorrelation)
            elif self.selected_analysis_1 == 3:
                
                self.open_Autocorrelation(self.values, self.channel_colors, self.stim_colors)
                
            # Si el análisis 4 está seleccionado (Pearson correlation)
            elif self.selected_analysis_1 == 4:
                
                self.open_Pearson_correlation(self.values, self.channel_colors, self.stim_colors)
                
            # Si el análisis 5 está seleccionado (Spike triggered averaging)
            elif self.selected_analysis_1 == 5:
                
                self.open_Spike_triggered_averaging(self.values, self.channel_colors, self.stim_colors)
                
            # Si el análisis 6 está seleccionado (Cross wavelet XWT)
            elif self.selected_analysis_1 == 6:
                
                self.open_Cross_wavelet(self.values, self.channel_colors, self.stim_colors)
            
            # Si el análisis 7 está seleccionado (Wavelet coherence CWT)
            elif self.selected_analysis_1 == 7:
                
                self.open_Wavelet_coherence(self.values, self.channel_colors, self.stim_colors)
                
            # Si el análisis 8 está seleccionado (Coherent power CXWT)
            elif self.selected_analysis_1 == 8:
                
                self.open_Coherent_power(self.values, self.channel_colors, self.stim_colors)
                
            # Si el análisis 9 está seleccionado (PCA)
            elif self.selected_analysis_1 == 9:
                
                self.open_PCA(self.values, self.channel_colors, self.stim_colors)



        # Verificar la selección del segundo análisis
        if (self.selected_analysis_1 == 1 or self.selected_analysis_1 == 2) and self.selected_analysis_2 != 0 and self.selected_analysis_2 is not None:

            # Solo ejecutar el análisis del segundo índice si es diferente del primero
            if not self.selected_analysis_1 == self.selected_analysis_2:

                # Si el análisis 2 está seleccionado y es 0 (análisis de bins)
                if self.selected_analysis_2 == 0:
                    self.open_bins_analysis(self.values, self.channel_colors, self.stim_colors)

                # Si el análisis 2 está seleccionado y es 1 (análisis de duración de ciclo y fases)
                elif self.selected_analysis_2 == 1:
                    self.open_phase_and_cycle_duration_analysis(self.values, self.channel_colors, self.stim_colors)
        else:
            print("Ejecutando sólo analisís 1.")

    def rectificado_int_update(self):
        """Abre un cuadro de diálogo para seleccionar un archivo y ejecuta un script de análisis con los parámetros proporcionados."""

        # Actualiza la lista de valores con parámetros actuales
        self.values = [self.channel_index, self.threshold, self.hold_off, self.delay_time, self.hysteresis_up,  self.hysteresis_down]

        # Abre el cuadro de diálogo para seleccionar un archivo CSV
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos CSV (*.csv)")

        # Verifica si se seleccionó un archivo
        if archivo:
            # Actualiza la ruta del archivo en todos los análisis
            self.ruta_archivo = archivo
            print("Abriendo rectificado e integrado...")

            # Ruta al script de análisis que se ejecutará
            script_path_explore = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\RECT_E_INT_con_HP.py"

            # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
            subprocess.run([sys.executable, script_path_explore, self.ruta_archivo, *map(str, self.channel_colors),
                            *map(str, self.stim_colors), *map(str, self.values)])
    
    
    def open_Read_ATF(self, values, channel_colors, stim_colors):
        #Ejecuta la conversión de ATF a CSV.
        print("Abriendo conversión de ATF a CSV...")
        

        # Ruta al script de análisis de conversión CSV que se ejecutará
        script_path_atf = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\Read_ATF.py"
        
        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_atf, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        

        
    def open_bins_analysis(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de bins utilizando un script externo.
        print("Abriendo análisis de bins...")

        # Ruta al script de análisis de bins que se ejecutará
        script_path_bins = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\BINS.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_bins, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        

    def open_phase_and_cycle_duration_analysis(self, values, channel_colors, stim_colors):
        """Ejecuta el análisis de duración de ciclo y fases utilizando un script externo."""
        print("Abriendo análisis de fases circulares...")

        # Ruta al script de análisis de duración de ciclo y fases que se ejecutará
        script_path_phase = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\DURACIÓN_CICLO_Y_FASES_2.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_phase, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        
    def open_FFT(self, values, channel_colors, stim_colors):
        #Ejecuta la Transformada rápida de Fourier utilizando un script externo.
        print("Abriendo análisis de Transformada rápida de Fourier...")

        # Ruta al script de análisis de Transformada rápida de Fourier que se ejecutará
        script_path_FFT = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\FFT.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_FFT, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
    
    def open_Autocorrelation(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de Autocorrelation utilizando un script externo.
        print("Abriendo análisis de Autocorrelation...")

        # Ruta al script de análisis de Autocorrelation que se ejecutará
        script_path_Autocorrelation = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\Autocorrelation.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_Autocorrelation, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
    
             
    def open_Pearson_correlation(self, values, channel_colors, stim_colors):
        #Ejecuta la correlación de Pearson utilizando un script externo.
        print("Abriendo análisis de Pearson correlation...")

        # Ruta al script de análisis de Pearson correlation que se ejecutará
        script_path_Pearson_correlation = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\Pearson_Correlation.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_Pearson_correlation, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
    
    def open_Spike_triggered_averaging(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de STA utilizando un script externo.
        print("Abriendo análisis de Spike triggered averaging...")

        # Ruta al script de análisis de STA que se ejecutará
        script_path_STA = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\STA.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_STA, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        
    def open_Cross_wavelet(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de Cross wavelet utilizando un script externo.
        print("Abriendo análisis de Cross wavelet...")

        # Ruta al script de análisis de Cross wavelet que se ejecutará
        script_path_Cross_wavelet = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\Cross_wavelet.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_Cross_wavelet, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        
        
    def open_Wavelet_coherence(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de Wavelet coherence utilizando un script externo.
        print("Abriendo análisis de Wavelet coherence...")

        # Ruta al script de análisis de Cross wavelet que se ejecutará
        script_path_Wavelet_coherence = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\Coherence_wavelet.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_Wavelet_coherence, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
    
    def open_Coherent_power(self, values, channel_colors, stim_colors):
        #Ejecuta el análisis de Coherent power utilizando un script externo.
        print("Abriendo análisis de Coherent power...")

        # Ruta al script de análisis de Cross wavelet que se ejecutará
        script_path_Coherent_power = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\CXWT.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_Coherent_power, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])
        
    def open_PCA(self, values, channel_colors, stim_colors):
        #Ejecuta el PCA utilizando un script externo.
        print("Abriendo PCA...")

        # Ruta al script de análisis de PCA que se ejecutará
        script_path_PCA = r"C:\Users\eduar\OneDrive\Documentos\Documentos para SS UPIITA\Entregables\GUIs_Sahian\PCA_.py"

        # Ejecuta el script como un proceso independiente pasando los parámetros necesarios
        subprocess.run(
            [sys.executable, script_path_PCA, self.ruta_archivo, *map(str, channel_colors), *map(str, stim_colors),
             *map(str, values)])

    def reset_crosses_count(self, index):
        """Reinicia el contador de cruces y actualiza la interfaz con el nuevo índice de canal."""

        # Actualiza el índice de canal
        self.channel_index = index

        # Reinicia el contador de cruces a 0 en el hilo thread_index
        thread_index.crosses_count = 0

        # Actualiza la referencia de data_p en el hilo thread_a con el nuevo índice de canal
        thread_a.update_data_p_index(self.channel_index)

        # Actualiza la pantalla LCD con el contador de cruces reiniciado
        self.update_lcd(thread_index.crosses_count)

    def update_hold_off(self, hold_off_time):
        """Actualiza el tiempo de espera antes de permitir otro cruce de umbral."""

        # Actualiza la variable hold_off con el tiempo proporcionado
        self.hold_off = hold_off_time

        # Convierte el tiempo de espera de milisegundos a segundos
        self.hold_off_time = hold_off_time / 1000

        # Actualiza el tiempo de espera en el hilo hold_off_thread
        hold_off_thread.update_hold_off_time(self.hold_off_time)

    def update_delay(self, delay_time):
        """Actualiza el tiempo de retraso para la señal de estímulo."""

        # Convierte el tiempo de retraso de milisegundos a segundos
        self.delay_time = delay_time / 1000

        # Actualiza el tiempo de retraso en los hilos thread_pulse y thread_index
        thread_pulse.update_delay_time(self.delay_time)
        thread_index.update_delay_time(self.delay_time)

    def clear_lines_and_indices(self, index):
        """Elimina todas las líneas verticales del gráfico y reinicia los índices de cruce."""

        # Eliminar todas las líneas del gráfico
        for line in self.main_graph.vertical_lines:
            self.main_graph.Graph_layout.removeItem(line)

        # Limpia las listas de índices y líneas verticales
        self.main_graph.cross_indices.clear()
        self.main_graph.vertical_lines.clear()

    def update_crosses_count_max(self, value):
        """Actualiza el número máximo de cruces permitidos y ajusta la interfaz en consecuencia."""

        # Actualiza el número máximo de cruces
        self.crosses_count_max = value

        # Actualiza el número máximo de cruces en el hilo thread_index
        thread_index.update_crosses_max(value)

        # Actualiza el estado de los botones según el valor de crosses_count_max
        for stim_number in range(1, 4):  # Considerando que hay 3 estimuladores
            for button_number in range(1, 11):  # 10 botones por estimulador
                button_name = f"Stim_{stim_number}_{button_number}"
                button = getattr(self, button_name)

                # Habilita o deshabilita los botones según el número máximo de cruces
                if button_number <= self.crosses_count_max:
                    button.setEnabled(True)
                else:
                    button.setEnabled(False)
                    button.setChecked(False)

                # Conecta la señal clicked del botón a la función correspondiente
                button.clicked.connect(lambda checked, s=stim_number, n=button_number: self.toggle_button_activation(s, n))

        # Llamar a toggle_button_activation para actualizar el estado de los botones
        self.toggle_button_activation(stim_number, button_number)

    def toggle_button_activation(self, stim_number, button_number):
        """Activa o desactiva un botón de estímulo y actualiza la lista de botones activos para cada estimulador."""

        # Construye el nombre del botón con el formato "Stim_<stim_number>_<button_number>"
        button_name = f"Stim_{stim_number}_{button_number}"

        # Obtiene el botón correspondiente usando el nombre construido
        button = getattr(self, button_name)

        # Verifica si el botón está habilitado
        if button.isEnabled():
            # Si el botón está marcado y su número es menor o igual al número máximo de cruces permitidos
            if button.isChecked() and button_number <= self.crosses_count_max:
                # Verifica que el número del botón no exceda el número máximo de cruces permitidos para el estimulador actual
                if button_number <= self.crosses_count_max:
                    # Verifica si el número del botón no está ya en la lista de botones activos del estimulador
                    if button_number not in self.estimuladores_botones_activos[stim_number]:
                        # Agrega el número del botón a la lista de botones activos del estimulador
                        self.estimuladores_botones_activos[stim_number].append(button_number)
                else:
                    # Si el número del botón excede el número máximo de cruces permitidos, desmarca el botón
                    button.setChecked(False)
            else:
                # Si el botón está desmarcado o su número excede el número máximo de cruces permitidos
                # Quita el número del botón de la lista de botones activos del estimulador si está presente
                if button_number in self.estimuladores_botones_activos[stim_number]:
                    self.estimuladores_botones_activos[stim_number].remove(button_number)

            # Ordena los números de los botones activos del estimulador de manera ascendente
            self.estimuladores_botones_activos[stim_number].sort()

        # Asegura que ningún botón activo tenga un número mayor al número máximo de cruces permitidos
        for stim_number in range(1, 4):
            self.estimuladores_botones_activos[stim_number] = [val for val in
                                                               self.estimuladores_botones_activos[stim_number] if
                                                               val <= self.crosses_count_max or button.setChecked(False)]

        # Actualiza los estimuladores activos en el hilo thread_index
        thread_index.update_stims(self.estimuladores_botones_activos)

class ThreadA(QThread):
    """Hilo designado para la adquisición y detección de cruces"""
    # Definición de señales que el hilo emitirá para comunicar con la interfaz u otros componentes
    data_generated = pyqtSignal(np.ndarray, int)
    cross_x_up = pyqtSignal(int)
    cross_x_down = pyqtSignal(int)
    send_index = pyqtSignal(int, int, float, float)
    save_data = pyqtSignal(np.ndarray, list, list, list, list, list)

    def __init__(self, sample_rate, tmax, dev_name, number_of_samples, samp_per_iteration,
                 semaphore, mutex, samp_per_channel, tiempo):
        super().__init__()

        # Inicialización de parámetros y atributos de la clase
        self.dev_name = dev_name # Nombre del dispositivo NIDAQ
        self.sample_rate = sample_rate # Frecuencia de muestreo
        self.samp_per_channel = samp_per_channel # Número de muestras por canal
        self.tmax = tmax # Tiempo máximo de muestreo
        self.number_of_samples = number_of_samples # Número total de muestras
        self.samples_per_iteration = samp_per_iteration # Número de muestras por iteración
        self.tiempo = tiempo # Array de tiempos (eje x)

        # Inicialización del array para almacenar datos, inicialmente lleno de ceros
        self.data_array_zeros = np.zeros((13, self.number_of_samples))

        # Inicialización de banderas y contadores
        self.crossing_detected = False # Bandera para detectar cruce
        self.crossing_hysteresis_detected = False # Bandera para detectar histéresis
        self.primer_cruce_encontrado = False # Bandera para detectar el primer cruce
        self.crosses_count = 0 # Contador de cruces

        # Inicialización de otros parámetros
        self.data_p_index = 0 # Índice del dato
        self.delay_time = 0 # Tiempo de retardo
        self.threshold = 0 # Umbral de detección
        self.hold_off_active = False # Bandera de hold-off
        self.current_index = 0 # Índice actual
        self.sample_time = self.tmax / self.number_of_samples # Tiempo entre muestras
        self.data_per_s = round(1 / self.sample_time) # Datos por segundo
        self.t = np.arange(len(self.data_array_zeros[0])) # Array de tiempo

        self.task = None # Inicialización de la tarea NIDAQ
        self.semaphore = semaphore # Semáforo para sincronización de hilos
        self.mutex = mutex # Mutex para exclusión mutua
        self.save_data_active = False # Bandera para activar/desactivar guardado de datos

        # Inicialización de vectores de guardado, llenos de ceros
        self.cross_up_save = np.zeros(self.number_of_samples)
        self.cross_down_save = np.zeros(self.number_of_samples)
        self.stim_1_save = np.zeros(self.number_of_samples)
        self.stim_2_save = np.zeros(self.number_of_samples)
        self.stim_3_save = np.zeros(self.number_of_samples)
        self.time_end = 0

    def set_sample_rate(self, new_sample_rate):
        """Actualiza la frecuencia de muestreo y ajusta otros parámetros relacionados con la adquisición de datos."""

        # Actualiza la frecuencia de muestreo
        self.sample_rate = new_sample_rate
        # Actualiza el número de muestras por canal al nuevo valor de frecuencia de muestreo
        self.samp_per_channel = new_sample_rate
        # Calcula el número total de muestras basándose en el tiempo máximo y la nueva frecuencia de muestreo
        self.number_of_samples = int(self.tmax * self.sample_rate)
        # Calcula el número de muestras por iteración
        self.samples_per_iteration = int(self.number_of_samples / self.sample_rate)
        # Re-inicializa el array de datos con ceros con las nuevas dimensiones
        self.data_array_zeros = np.zeros((13, self.number_of_samples))
        # Re-inicializa los vectores de guardado con ceros
        self.cross_up_save = np.zeros(self.number_of_samples)
        self.cross_down_save = np.zeros(self.number_of_samples)
        self.stim_1_save = np.zeros(self.number_of_samples)
        self.stim_2_save = np.zeros(self.number_of_samples)
        self.stim_3_save = np.zeros(self.number_of_samples)
        # Reinicia el índice actual a 0
        self.current_index = 0
        # Ajusta el número de muestras a la nueva frecuencia de muestreo
        self.set_number_of_samples(self.tmax)
    def save_data_update(self, save_data_active):
        """Actualiza la bandera que indica si los datos deben ser guardados o no."""
        # Actualiza la bandera para activar o desactivar el guardado de datos
        self.save_data_active = save_data_active

    def initialize_task(self):
        """Inicializa la tarea de adquisición de datos utilizando nidaqmx."""

        try:
            if self.dev_name is None:
                raise ValueError("No DAQ device specified.")
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(f"{self.dev_name}/ai0:12", min_val=-10.0, max_val=10.0)
            # Configura la frecuencia de muestreo y el modo de adquisición continua
            self.task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                             samps_per_chan=self.samp_per_channel)
        except nidaqmx.errors.DaqError as e:
            print(f"DAQ Error: {e}")
        except ValueError as ve:
            print(ve)
        
        """# Crea una nueva tarea de nidaqmx
        self.task = nidaqmx.Task()
        # Añade un canal de voltaje para la adquisición de datos (canales ai0 a ai12)
        self.task.ai_channels.add_ai_voltage_chan(f"{self.dev_name}/ai0:12", min_val=-10.0, max_val=10.0)"""
        
        

    def update_data_p_index(self, index):
        self.data_p_index = index

    def update_hold_off(self, hold_off_flag):
        self.hold_off_active = hold_off_flag

    def update_threshold(self, threshold_value):
        self.threshold = threshold_value

    def reset_flags(self):
        self.crossing_detected = False
        self.primer_cruce_encontrado = False
        self.crossing_hysteresis_detected = False

    def set_number_of_samples(self, new_tmax):
        self.tmax = new_tmax
        #print("t max: ", self.tmax)
        self.number_of_samples = int(new_tmax * self.sample_rate)
        #print("NUMBER_SAMPLES-----------------------------", self.number_of_samples, self.sample_rate)
        #self.samples_per_iteration = int(self.number_of_samples / self.sample_rate
        self.samples_per_iteration = 20
        #print("muestras por iteración: ", self.samples_per_iteration)
        self.tiempo = np.array(np.linspace(0, new_tmax, self.number_of_samples))
        #print("tiempo: ", self.tiempo)
        self.data_array_zeros = np.zeros((13, self.number_of_samples))
        self.t = np.arange(len(self.data_array_zeros[0]))
        #print("T: ", self.t, self.tiempo[self.t[len(self.t)-1]])
        # Ajustar índices
        self.current_index = 0
        self.reset_flags()
        # Asegúrate de que cualquier operación de guardado o procesamiento se resetea
        self.cross_up_save = np.zeros(self.number_of_samples)
        self.cross_down_save = np.zeros(self.number_of_samples)
        self.stim_1_save = np.zeros(self.number_of_samples)
        self.stim_2_save = np.zeros(self.number_of_samples)
        self.stim_3_save = np.zeros(self.number_of_samples)

        print("------------------------------------------------------------------------------------------------CAMBIO DE TIEMPO A", self.tmax, new_tmax)

    def run(self):
        if self.task is None:
            self.initialize_task()

        while True:
            data_read = np.array(self.task.read(number_of_samples_per_channel=self.samples_per_iteration))
            current_end_index = (self.current_index + self.samples_per_iteration) % self.number_of_samples
            #print("current index", self.current_index)
            #print("current end index: ", current_end_index)

            #print(f"self.samples_per_iteration: {self.samples_per_iteration}")
            #print(f"data_read.shape: {data_read.shape}")

            self.semaphore.acquire()
            try:
                self.mutex.lock()
                try:
                    data_t_p = self.t[self.current_index:self.current_index + self.samples_per_iteration]
                    #print("RANGOS EN HILO A", self.current_index, self.current_index + self.samples_per_iteration)
                    #print("DATA_T_P", data_t_p)
                    data = np.array(data_read[self.data_p_index, :])
                    promedio_p = np.mean(data)
                    self.detect_crosses(promedio_p, data, data_t_p)
                    self.data_generated.emit(data_read, self.current_index)
                    if self.save_data_active and current_end_index == 0:
                        self.save_data.emit(data_read, self.cross_up_save, self.cross_down_save, self.stim_1_save,
                                            self.stim_2_save, self.stim_3_save)
                        self.cross_up_save.fill(0)
                        self.cross_down_save.fill(0)
                        self.stim_1_save.fill(0)
                        self.stim_2_save.fill(0)
                        self.stim_3_save.fill(0)

                    self.save_data.emit(data_read,
                                        self.cross_up_save[
                                        self.current_index:self.current_index + self.samples_per_iteration],
                                        self.cross_down_save[
                                        self.current_index:self.current_index + self.samples_per_iteration],
                                        self.stim_1_save[
                                        self.current_index:self.current_index + self.samples_per_iteration],
                                        self.stim_2_save[
                                        self.current_index:self.current_index + self.samples_per_iteration],
                                        self.stim_3_save[
                                        self.current_index:self.current_index + self.samples_per_iteration])
                finally:
                    self.mutex.unlock()
            finally:
                self.semaphore.release()

            self.current_index = current_end_index
            self.time_end = time.time()

    def detect_crosses(self, data_p, data, data_t_p):
        if self.hold_off_active:
            return

        if not self.crossing_detected and not self.primer_cruce_encontrado:
            primer_cruce_index = np.argmax(data_p >= self.threshold)
            self.primer_cruce_encontrado = True
            self.crossing_detected = True
            self.crossing_hysteresis_detected = False

        if self.crossing_detected and self.primer_cruce_encontrado and not self.crossing_hysteresis_detected:
            if data_p >= self.threshold + window.hysteresis_up:
                first_index = np.argmax(data >= self.threshold + window.hysteresis_up)
                #print("INDICES DE TIEMPO EN CRUCES", data_t_p)
                #print("DATOS", data)
                self.value_up = data_t_p[first_index]
                #print("CRUCE ASCENDENTE", self.value_up,"-----------------------------------")
                #print("RANGOS", data_t_p)
                start_time = time.time()
                # print("EN COMPARAR------------", start_time)
                thread_index.run(self.value_up, start_time)
                self.cross_x_up.emit(self.value_up)
                hold_off_thread.start()
                self.hold_off_active = True
                self.crossing_hysteresis_detected = True
                if self.save_data_active:
                    self.cross_up_save[self.value_up] = 1

        if self.crossing_detected and self.primer_cruce_encontrado and self.crossing_hysteresis_detected and not self.hold_off_active:
            if data_p < self.threshold - window.hysteresis_down:
                first_index_down = np.argmax(data <= self.threshold)
                self.value_down = data_t_p[first_index_down]
                #print("CRUCE DESCENDENTE", self.value_down)
                self.cross_x_down.emit(self.value_down)
                if self.save_data_active:
                    self.cross_down_save[self.value_down] = 1
                self.reset_flags()
class Thread_index(QThread):
    crosses_detected = pyqtSignal(int)
    tag_1 = pyqtSignal(int)
    tag_2 = pyqtSignal(int)
    tag_3 = pyqtSignal(int)
    Stim_1 = pyqtSignal(int)
    Stim_2 = pyqtSignal(int)
    Stim_3 = pyqtSignal(int)

    def __init__(self, sample_rate, number_of_samples, tmax):
        super().__init__()
        self.tmax = tmax
        self.sample_rate = sample_rate
        self.number_of_samples = number_of_samples
        self.sample_time = self.tmax / self.number_of_samples
        self.data_per_s = round(1 / self.sample_time)
        self.delay_time = 0
        self.crosses_count = 0
        self.tag_1_active = False
        self.tag_2_active = False
        self.tag_3_active = False
        self.stim_1_apply = False
        self.stim_2_apply = False
        self.stim_3_apply = False
        self.S_1 = 0
        self.S_2 = 0
        self.S_3 = 0
        self.crosses_count_max = 0
        self.save_data_active = False
        self.estimuladores_botones_activos = {1: [], 2: [], 3: []}

    def stim_active_update(self, stim_1, stim_2, stim_3):
        self.stim_1_apply = stim_1
        self.stim_2_apply = stim_2
        self.stim_3_apply = stim_3

    def update_stims(self, estimuladores_botones_activos):
        self.estimuladores_botones_activos = estimuladores_botones_activos
        self.tag_1_active = any(self.estimuladores_botones_activos[1])
        self.tag_2_active = any(self.estimuladores_botones_activos[2])
        self.tag_3_active = any(self.estimuladores_botones_activos[3])

    def save_data_update(self, save_data_active):
        self.save_data_active = save_data_active

    def reset_flags(self):
        self.tag_1_active = False
        self.tag_2_active = False
        self.tag_3_active = False
        self.S_1 = 0
        self.S_2 = 0
        self.S_3 = 0

    def set_sample_rate(self, new_sample_rate):
        self.sample_rate = new_sample_rate
        self.samples_per_channel = new_sample_rate
        self.set_number_of_samples(self.tmax)

    def set_number_of_samples(self, new_tmax):
        self.tmax = new_tmax
        self.number_of_samples = int(new_tmax * self.sample_rate)
        self.sample_time = self.tmax / self.number_of_samples
        self.data_per_s = round(1 / self.sample_time)

    def update_crosses_count(self, crosses_count):
        self.crosses_count = crosses_count

    def update_delay_time(self, delay_time):
        self.delay_time = delay_time

    def update_crosses_max(self, crosses_count_max):
        self.crosses_count_max = crosses_count_max

    def run(self, value_up, start_time):
        try:
            #print("EN INDEX", start_time)
            if self.crosses_count >= self.crosses_count_max:
                self.crosses_count = 0
            self.crosses_count += 1
            self.crosses_detected.emit(self.crosses_count)

            index_t = (value_up + round(self.delay_time * self.data_per_s)) % self.number_of_samples

            stim_signals = {
                1: (self.stim_1_apply, self.tag_1, 'port1/line0', 'S_1', 'tag_1_active', 'stim_1_save', window.main_graph.update_tag_1),
                2: (self.stim_2_apply, self.tag_2, 'port1/line1', 'S_2', 'tag_2_active', 'stim_2_save', window.main_graph.update_tag_2),
                3: (self.stim_3_apply, self.tag_3, 'port1/line2', 'S_3', 'tag_3_active', 'stim_3_save', window.main_graph.update_tag_3)
            }

            if self.save_data_active:
                self.save_data(index_t)

            for stim_number, vector in self.estimuladores_botones_activos.items():
                if self.crosses_count in vector:
                    self.handle_stimulus(stim_signals[stim_number], index_t, start_time)

            self.reset_flags()
        except Exception as e:
            print(f"Error en PulseThread: {e}")

    def handle_stimulus(self, stim_signal, index_t, start_time):
        apply_stim, tag_signal, device_channel, S_attr, tag_active_attr, save_attr, update_function = stim_signal

        if not getattr(self, tag_active_attr) and getattr(self, S_attr) == 0:
            setattr(self, tag_active_attr, True)
            setattr(self, S_attr, 1)

        if apply_stim:
            thread_pulse.start()
            thread_pulse.send_ttl_pulse(device_channel, start_time)
            if self.save_data_active:
                getattr(thread_a, save_attr)[index_t] = 2
        else:
            if self.save_data_active:
                getattr(thread_a, save_attr)[index_t] = 1.5

        tag_signal.emit(index_t)

    def save_data(self, index_t):
        if any(self.estimuladores_botones_activos[1]):
            thread_a.stim_1_save[index_t] = 1
        if any(self.estimuladores_botones_activos[2]):
            thread_a.stim_2_save[index_t] = 1
        if any(self.estimuladores_botones_activos[3]):
            thread_a.stim_3_save[index_t] = 1
class PulseThread(QThread):
    tag_signal = pyqtSignal(int)  # Asegúrate de definir esto si no está definido

    def __init__(self, dev_name):
        super().__init__()
        self.dev_name = dev_name
        self.delay_time = 0
        self.device_channel = ''
        self.start_time = 0
        self.index_t = 0

    def update_delay_time(self, delay_time):
        self.delay_time = delay_time

    def send_ttl_pulse(self, device_channel, start_time):
        self.device_channel = device_channel
        new_delay = self.delay_time - (time.time() - start_time)
        if new_delay < 0:
            new_delay = 0
        QTimer.singleShot(int(new_delay * 1000), self.trigger_pulse)

    def trigger_pulse(self):
        with nidaqmx.Task() as ttl_task:
            ttl_task.do_channels.add_do_chan(f"{self.dev_name}/{self.device_channel}")
            ttl_task.write(True, timeout=0.00001)
            ttl_task.write(False, timeout=0.00001)
            #print("Dentro de trigger",time.time() - self.start_time)
            #print("Pulso enviado correctamente")
class Hold_off(QThread):
    hold_off_finished = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.hold_off_time = 0

    def update_hold_off_time(self, hold_off_time):
        self.hold_off_time = hold_off_time

    def run(self):
        QTimer.singleShot(int(self.hold_off_time * 1000), self.finish_hold_off)

    def finish_hold_off(self):
        self.hold_off_finished.emit(False)

class CanvasGraph(QWidget):
    def __init__(self, channel_colors, stim_color_1, stim_color_2, stim_color_3, threshold_color, line_up_color, line_down_color, number_of_samples, samp_per_iteration, tmax, tiempo, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.tiempo = tiempo
        self.tmax = tmax
        self.samples_per_iteration = samp_per_iteration
        self.number_of_samples = number_of_samples
        self.data_array_zeros = np.zeros((13, number_of_samples))
        self.cross_up = np.zeros(number_of_samples)
        self.cross_down = np.zeros(number_of_samples)
        self.Stim_1 = np.zeros(number_of_samples)
        self.Stim_2 = np.zeros(number_of_samples)
        self.Stim_3 = np.zeros(number_of_samples)
        self.zeros = np.zeros(samp_per_iteration)
        self.current_index = 0
        self.New_value_up = False
        self.New_value_down = False
        self.tag_1_value = False
        self.tag_2_value = False
        self.tag_3_value = False
        self.value_S_1 = 0
        self.value_S_2 = 0
        self.value_S_3 = 0
        self.value_up = 0
        self.value_down = 0

        self.Graph_layout = pg.PlotWidget(enableOpenGL=True)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.Graph_layout)
        self.setLayout(self.layout)

        self.Graph_layout.getPlotItem().getAxis('bottom').enableAutoSIPrefix(False)
        self.Graph_layout.getPlotItem().getAxis('left').enableAutoSIPrefix(False)
        self.Graph_layout.getPlotItem().getAxis('right').enableAutoSIPrefix(False)

        self.curves = [self.Graph_layout.plot(pen=pg.mkPen(color, width=1), name=f'Canal {i + 1}') for i, color in enumerate(channel_colors)]

        self.Stim_1_line = self.Graph_layout.plot(pen=pg.mkPen(color=stim_color_1, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.Stim_2_line = self.Graph_layout.plot(pen=pg.mkPen(color=stim_color_2, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.Stim_3_line = self.Graph_layout.plot(pen=pg.mkPen(color=stim_color_3, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.threshold_line = self.Graph_layout.plot(pen=pg.mkPen(color=threshold_color, width=1))
        self.cross_line_up = self.Graph_layout.plot(pen=pg.mkPen(color=line_up_color, width=1))
        self.cross_line_down = self.Graph_layout.plot(pen=pg.mkPen(color=line_down_color, width=1))

        self.Graph_layout.setRange(yRange=[0, 5])

        self.Graph_layout.setLabel('bottom', text='Tiempo[s]')
        self.Graph_layout.setLabel('left', text='Amplitud [V]')

        self.legend = self.Graph_layout.addLegend(offset=(100, 2))

        for curve in self.curves:
            self.legend.addItem(curve, curve.opts['name'])
        self.vertical_lines = []
        self.cross_indices = []
        self.legend.setRotation(0)

        for i in range(len(channel_colors)):
            setattr(self, f'offset_{i + 1}', 0)

        self.tmax_changed = False

    def save_data_update(self, bool):
        self.save_data_active = bool

    def set_y_range(self, min_value, max_value):
        self.Graph_layout.setRange(yRange=[min_value, max_value])

    def set_sample_rate(self, new_sample_rate):
        self.sample_rate = new_sample_rate
        self.number_of_samples = int(self.tmax * self.sample_rate)
        self.samples_per_iteration = int(self.number_of_samples / self.sample_rate)  # Este valor debe ser igual al número de muestras por segundo

        print(f"Nuevo sample_rate: {self.sample_rate}")
        print(f"Nuevo number_of_samples: {self.number_of_samples}")
        print(f"Nuevo samples_per_iteration: {self.samples_per_iteration}")

        self.tiempo = np.linspace(0, self.tmax, self.number_of_samples)
        self.data_array_zeros = np.zeros((13, self.number_of_samples))
        self.cross_up = np.zeros(self.number_of_samples)
        self.cross_down = np.zeros(self.number_of_samples)
        self.Stim_1 = np.zeros(self.number_of_samples)
        self.Stim_2 = np.zeros(self.number_of_samples)
        self.Stim_3 = np.zeros(self.number_of_samples)
        self.zeros = np.zeros(self.samples_per_iteration)
        self.set_number_of_samples(self.tmax)

    def update_x_up(self, value_up):
        self.value_up = value_up
        self.New_value_up = True

    def update_x_down(self, value_down):
        self.value_down = value_down
        self.New_value_down = True

    def update_tag_1(self, value_S_1):
        self.value_S_1 = value_S_1
        self.tag_1_value = True

    def update_tag_2(self, value_S_2):
        self.value_S_2 = value_S_2
        self.tag_2_value = True

    def update_tag_3(self, value_S_3):
        self.value_S_3 = value_S_3
        self.tag_3_value = True

    def update_offset_1(self, value):
        self.offset_1 = value
        #print("OFFSET 1", value)

    def update_offset_2(self, value):
        self.offset_2 = value

    def update_offset_3(self, value):
        self.offset_3 = value

    def update_offset_4(self, value):
        self.offset_4 = value
        #print("OFFSET 4", value)

    def update_offset_5(self, value):
        self.offset_5 = value

    def update_offset_6(self, value):
        self.offset_6 = value

    def update_offset_7(self, value):
        self.offset_7 = value
        #print("OFFSET 7", value)

    def update_offset_8(self, value):
        self.offset_8 = value

    def update_offset_9(self, value):
        self.offset_9 = value

    def update_offset_10(self, value):
        self.offset_10 = value

    def update_offset_11(self, value):
        self.offset_11 = value
        #print("OFFSET 11", value)

    def update_offset_12(self, value):
        self.offset_12 = value

    def set_number_of_samples(self, new_tmax):
        self.tmax = new_tmax
        self.number_of_samples = int(new_tmax * self.sample_rate)
        #self.samples_per_iteration = int(self.number_of_samples / self.sample_rate)
        self.samples_per_iteration = 20
        #print("number_of_samples", self.number_of_samples)
        #print("MUESTRAS POR ITE", self.samples_per_iteration)
        self.zeros = np.zeros(self.samples_per_iteration)
        #print("ZEROS", len(self.zeros))
        self.tiempo = np.array(np.linspace(0, new_tmax, self.number_of_samples))
        #print("tiempo", len(self.tiempo))
        self.data_array_zeros = np.zeros((13, self.number_of_samples))
        #print("data_array_zeros", self.data_array_zeros.shape)
        self.cross_up = np.zeros(self.number_of_samples)
        self.cross_down = np.zeros(self.number_of_samples)
        self.Stim_1 = np.zeros(self.number_of_samples)
        self.Stim_2 = np.zeros(self.number_of_samples)
        self.Stim_3 = np.zeros(self.number_of_samples)
        #self.current_index = 0
        self.tmax_changed = True
        print("-----------------------------------------------------------------------------CAMBIO DE TIEMPO GRAPH",
              self.tmax, new_tmax)


    def update_plot(self, data, current_index):
        self.current_index = current_index
        if len(data[0]) == self.samples_per_iteration:
            min_length = min(len(self.tiempo), self.number_of_samples)
            vectors = [self.cross_up, self.cross_down, self.Stim_1, self.Stim_2, self.Stim_3]
            tiempo_s = self.tiempo[:min_length]
            data_all = self.data_array_zeros[:, :min_length]
            subsample_factor = 1
            tiempo_T_s = tiempo_s[::subsample_factor]
            data_T_s = data_all[:, ::subsample_factor]

            current_end_index = (self.current_index + self.samples_per_iteration) % min_length
            #if self.tmax_changed:
                #current_end_index = (0 + self.samples_per_iteration) % min_length
                #print("INDICES ACTUALIZADOS EN GRAPH", self.current_index, current_end_index)

                #self.tmax_changed = False
            #print("plot")
            #print(f"current_index: {self.current_index}")
            #print(f"current_end_index: {current_end_index}")
            #print("CURRENT INDEX EN GRAPH", self.current_index)

            #self.data_array_zeros[:, current_end_index:current_end_index + self.samples_per_iteration] = data
            self.data_array_zeros[:, self.current_index:self.current_index + self.samples_per_iteration] = data

            for array in vectors:
                array[self.current_index:self.current_index + self.samples_per_iteration] = self.zeros
                #print("RANGOS GRAPH", self.current_index, current_end_index)

            #print("RANGOS GRAPH", self.current_index, current_end_index)
            if self.New_value_up: #and self.value_up <= self.number_of_samples:
                self.cross_up[self.value_up] = 1
                #print(f"VALOR DE SUBIDA GRAPH {self.value_up}-------------------")
                #print("RANGOS GRAPH", self.current_index, current_end_index)
                self.New_value_up = False

            if self.New_value_down: #and self.value_down <= self.number_of_samples:
                self.cross_down[self.value_down] = -1
                self.New_value_down = False

            if self.tag_1_value and self.value_S_1 < self.number_of_samples and self.current_index <= self.value_S_1 < current_end_index:
                self.Stim_1[self.value_S_1] = 10
                self.tag_1_value = False

            if self.tag_2_value and self.value_S_2 < self.number_of_samples and self.current_index <= self.value_S_2 < current_end_index:
                self.Stim_2[self.value_S_2] = 10
                self.tag_2_value = False

            if self.tag_3_value and self.value_S_3 < self.number_of_samples and self.current_index <= self.value_S_3 < current_end_index:
                self.Stim_3[self.value_S_3] = 10
                self.tag_3_value = False

            threshold_line = np.full_like(tiempo_s, window.threshold)
            self.cross_line_up.setData(tiempo_s, self.cross_up[:min_length])
            self.cross_line_down.setData(tiempo_s, self.cross_down[:min_length])
            self.Stim_1_line.setData(tiempo_s, self.Stim_1[:min_length])
            self.Stim_2_line.setData(tiempo_s, self.Stim_2[:min_length])
            self.Stim_3_line.setData(tiempo_s, self.Stim_3[:min_length])
            self.threshold_line.setData(tiempo_s, threshold_line)

            offsets = [self.offset_1, self.offset_2, self.offset_3, self.offset_4, self.offset_5,self.offset_6,
                       self.offset_7, self.offset_8, self.offset_9, self.offset_10, self.offset_11, self.offset_12, self.offset_13]  # Ejemplos de offsets para cada curva

            for i, curve in enumerate(self.curves):
                if curve.isVisible():
                    curve.setData(x=tiempo_T_s, y=data_T_s[i, :] + offsets[i])

            #self.current_index = current_end_index



if __name__ == "__main__":
    app = QApplication(sys.argv)
    sample_rate = 3300
    #print("sample_rate inicial", sample_rate)
    samp_per_channel = sample_rate #cuántas muestras se almacenan en el buffer antes de que la aplicación las lea
    #print("samp_per_channel inicial", samp_per_channel)
    tmax = 20
    #print("tmax inicial", tmax)
    number_of_samples = tmax * sample_rate # Valor para 20s a una fs=3.3KHz
    #print("number_of_samples inicial", number_of_samples)
    samp_per_iteration = int(number_of_samples/sample_rate) #Samples_per_channel
    #print("samp_per_iteration inicial", samp_per_iteration)
    tiempo = np.array(np.linspace(0, tmax, number_of_samples))
    #print(f"{len(tiempo)} tiempo inicial", tiempo)

    window = VentanaPrincipal(number_of_samples, samp_per_iteration, tmax, tiempo, sample_rate)

    # Inicializar semáforo y mutex
    semaphore = QSemaphore(1)
    mutex = QMutex()

    # Pasar las instancias de semáforo y mutex a los hilos
    thread_a = ThreadA(sample_rate, tmax, window.dev_name, number_of_samples, samp_per_iteration,
                       semaphore, mutex, samp_per_channel, tiempo)
    thread_pulse = PulseThread(window.dev_name)
    hold_off_thread = Hold_off()
    thread_index = Thread_index(sample_rate, number_of_samples, tmax)
    save_thread = SaveThread()

    thread_a.data_generated.connect(window.main_graph.update_plot)
    hold_off_thread.hold_off_finished.connect(thread_a.update_hold_off)
    thread_index.crosses_detected.connect(window.update_lcd)
    thread_a.cross_x_up.connect(window.main_graph.update_x_up)
    thread_a.cross_x_down.connect(window.main_graph.update_x_down)
    thread_index.tag_1.connect(window.main_graph.update_tag_1)  # index_t
    thread_index.tag_2.connect(window.main_graph.update_tag_2)  # index_t
    thread_index.tag_3.connect(window.main_graph.update_tag_3)  # index_t
    # Conectar el botón de guardar a la función correspondiente
    window.bt_save.clicked.connect(window.save_data)

    thread_a.start()
    thread_index.start()

    window.show()
    app.exec()
