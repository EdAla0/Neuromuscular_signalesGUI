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
        loadUi('Read_ATF.ui', self)

        # Crea y configura el widget central y su layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)
        central_widget.setLayout(layout)


        # Configura las opciones para el menú desplegable de canales
        options = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9',
                   'CH 10', 'CH 11', 'CH 12']


        # Configura los comboboxes para seleccionar las columnas de datos
        comboboxes = [
            self.column_combobox_1, self.column_combobox_2, self.column_combobox_3, self.column_combobox_4,
            self.column_combobox_5, self.column_combobox_6, self.column_combobox_7, self.column_combobox_8,
            self.column_combobox_9, self.column_combobox_10, self.column_combobox_11, self.column_combobox_12
        ]
        for combobox in comboboxes:
            combobox.addItems(options)  # Añade las opciones al combobox
            combobox.currentIndexChanged.connect(
                self.channel)  # Conecta el cambio de selección con la función `channel`


        self.stim_colors = stim_colors
        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        self.name_file.textChanged.connect(self.name_file_update)

        # Configura el botón para realizar la conversión de ATF a CSV
        self.bt_go.clicked.connect(self.read_atf)


        # Inicializa la ruta del archivo y carga los datos
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)  # Extrae la carpeta del archivo de datos
        

        # Toma los nombres de las columnas
        with open(self.ruta_archivo, 'r') as file:
            self.lines = file.readlines()

        # Encontrar la línea de encabezados de datos
        self.data_start_index = None
        self.signal_labels = []
        for i, line in enumerate(self.lines):
            if line.startswith('"Signals='):
                self.signal_labels = line.strip().replace('"', '').split('\t')[1:]
            if line.startswith('"Time'):
                self.data_start_index = i
                break
        
        # Si no se encuentra la línea de encabezados, lanzar un error
        if self.data_start_index is None:
            raise ValueError("No se encontró la línea de encabezados de datos")
        
        
        signal_labels = self.signal_labels
        
        # Asignar los nombres de signal_labels a los QLabels y vaciar los QLabels restantes
        for index in range(12):
            label_name = f"label_{index + 1}"
            label_widget = getattr(self, label_name)
            
            if index < len(signal_labels):
                # Asignar el nombre de la señal correspondiente
                label_widget.setText(signal_labels[index])
            else:
                # Vaciar el QLabel si no hay más señales
                label_widget.setText("No signal")

        self.s_1 = 'CH 1'  # Columna seleccionada para Signal 1
        self.s_2 = 'CH 1'  # Columna seleccionada para Signal 2
        self.s_3 = 'CH 1'  # Columna seleccionada para Signal 3
        self.s_4 = 'CH 1'  # Columna seleccionada para Signal 4
        self.s_5 = 'CH 1'  # Columna seleccionada para Signal 5
        self.s_6 = 'CH 1'  # Columna seleccionada para Signal 6
        self.s_7 = 'CH 1'  # Columna seleccionada para Signal 7
        self.s_8 = 'CH 1'  # Columna seleccionada para Signal 8
        self.s_9 = 'CH 1'  # Columna seleccionada para Signal 9
        self.s_10 = 'CH 1'  # Columna seleccionada para Signal 10
        self.s_11 = 'CH 1'  # Columna seleccionada para Signal 11
        self.s_12 = 'CH 1'  # Columna seleccionada para Signal 12
        
        directory, file_name = os.path.split(self.ruta_archivo)
        self.new_file_name  = os.path.splitext(file_name)[0]



    def channel(self):
        """Actualiza las señales de los canales seleccionados y las asigna a las variables correspondientes.
            Luego, actualiza el mapa de colores y las gráficas."""

        # Obtiene las nuevas selecciones de los comboboxes de canales
        self.s_1 = self.column_combobox_1.currentText()
        self.s_2 = self.column_combobox_2.currentText()
        self.s_3 = self.column_combobox_3.currentText()
        self.s_4 = self.column_combobox_4.currentText()
        self.s_5 = self.column_combobox_5.currentText()
        self.s_6 = self.column_combobox_6.currentText()
        self.s_7 = self.column_combobox_7.currentText()
        self.s_8 = self.column_combobox_8.currentText()
        self.s_9 = self.column_combobox_9.currentText()
        self.s_10 = self.column_combobox_10.currentText()
        self.s_11 = self.column_combobox_11.currentText()
        self.s_12 = self.column_combobox_12.currentText()
        
    def name_file_update(self, new_file_name):
        """Actualiza el nombre del archivo de análisis."""
        self.new_file_name = new_file_name



    def read_atf(self):
        """
        Parameters:
        
        -ruta del archivo

        Returns:
        - dominant_frequencies_dict (dict): Un diccionario que mapea los nombres de las columnas de señal
                                            a las frecuencias dominantes correspondientes.
        
        """
    
        print("Conversión de ATF a CSV EN CURSO...")
        file_path = self.ruta_archivo
        lines = self.lines
        signal_labels = self.signal_labels
        data_start_index = self.data_start_index
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Encontrar la línea de encabezados de datos
        data_start_index = None
        signal_labels = []
        for i, line in enumerate(lines):
            if line.startswith('"Signals='):
                signal_labels = line.strip().replace('"', '').split('\t')[1:]
            if line.startswith('"Time'):
                data_start_index = i
                break
        
        # Si no se encuentra la línea de encabezados, lanzar un error
        if data_start_index is None:
            raise ValueError("No se encontró la línea de encabezados de datos")

        # Leer los encabezados de datos
        header_line = lines[data_start_index].strip().replace('"', '')
        headers = header_line.split('\t')

        # Leer los datos tabulares
        data = []
        for line in lines[data_start_index + 1:]:
            split_line = line.strip().split('\t')
            if len(split_line) == len(headers):
                data.append(split_line)
            else:
                print(f"El largo de la linea '{line.strip()}' no corresponde a la longitud de la de encabezados.")

        # Crear un DataFrame de pandas
        df = pd.DataFrame(data, columns=headers)

        # Convertir los datos a tipos numéricos si es posible
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Reemplazar los elementos de signal_labels con self.s_1, self.s_2, ..., self.s_12
        for index in range(len(signal_labels)):
            if index < 12:  # Solo hasta el número de s_ disponible
                signal_labels[index] = getattr(self, f's_{index + 1}')

        # Asignar las etiquetas de señales a las columnas del DataFrame
        df.columns = ["TIME"] + signal_labels
        
        # Paso 1: Crear la lista completa de nombres de columnas que deben estar en el DataFrame
        ch_columns = [f'CH {i}' for i in range(1, 13)]

        # Paso 2: Crear un DataFrame nuevo con las columnas en el orden correcto y las vacías si es necesario
        # Conservar la columna "TIME" del DataFrame original
        df_new = pd.DataFrame()
        df_new['TIME'] = df['TIME']  # Conservar la columna TIME

        for ch in ch_columns:
            if ch in signal_labels:
                # Si el CH está en signal_labels, copiar los datos correspondientes
                col_index = signal_labels.index(ch) + 1  # +1 porque el índice 0 es TIME
                df_new[ch] = df.iloc[:, col_index]
            else:
                # Si el CH no está en signal_labels, crear una columna vacía
                df_new[ch] = pd.NA
        
        name = f"{self.new_file_name}"
        nombre_archivo_data = f"{name}.csv"
        self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior
        # Ruta completa del archivo
        file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)

        # Guardar el DataFrame como CSV en la misma ruta que el archivo ATF
        df_new.to_csv(file_path, index=False)

        # Imprimir un mensaje indicando que el archivo ha sido guardado
        print(f"ARCHIVO GUARDADO {file_path}")

        return df






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