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
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QInputDialog, QFileDialog
import pyqtgraph as pg
from scipy.signal import lfilter, butter
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA


class MainWindow(QMainWindow):
    def __init__(self, archivo, channel_colors, stim_colors, values):
        """Inicializa una instancia de la clase, configurando la interfaz gráfica y los datos necesarios para el análisis de señales.

        Parámetros:
        archivo (str): Ruta del archivo CSV que contiene los datos de señales.
"""
        super().__init__()
        # Carga la interfaz gráfica desde el archivo .ui
        loadUi('PCA_.ui', self)

        # Crea y configura el widget central y su layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.Base)
        central_widget.setLayout(layout)
        


        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        #self.name_file.textChanged.connect(self.name_file_update)

        # Configura el botón para realizar la conversión de ATF a CSV
        #self.bt_go.clicked.connect(self.read_atf)
        
        # Configura el botón parar agregar la columna selecionada e ir al siguiente CSV ----------------------------------------------------------
        self.bt_next.clicked.connect(self.next_file)


        # Inicializa la ruta del archivo y carga los datos
        self.ruta_archivo = archivo
        self.carpeta_datos = os.path.dirname(self.ruta_archivo)  # Extrae la carpeta del archivo de datos
        
        self.save_function_flag = False
        
        # Lee el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(self.ruta_archivo)
        
        # Obtener el head (encabezados y 5 primeras filas)
        df_head = df.head()
        
        # Convertir el DataFrame en HTML con estilo de tabla
        csv_html = df_head.to_html(index=False)
        
        # Mostrar en el QLabel
        self.label_csv_actual.setText(csv_html)
            
        # Guarda los nombres de las columnas en una lista
        columns = df.columns.tolist()

        
        self.column_combobox.addItems(columns)  # Añade las columnas del csv al combobox
        self.column_combobox.currentIndexChanged.connect(
            self.choose_column)  # Conecta el cambio de selección con la función `choose_column`
        
        self.selected_column = columns[0] # Columna seleccionada para ser la feature agregada   
        self.selected_column_values = df[self.selected_column]     
        
        directory, file_name = os.path.split(self.ruta_archivo)
        self.new_file_name  = os.path.splitext(file_name)[0]
        
        # Configura el botón para realizar el análisis de PCA
        self.bt_go.clicked.connect(self.PCA_function)
        
        # Configura el botón para guardar
        self.save_function.clicked.connect(self.save_active)
        
        # Conecta el cambio de texto en el campo de nombre del archivo con la función correspondiente
        self.name_file.textChanged.connect(self.name_file_update)
        
#--- Nombre y guardado        
    def name_file_update(self, new_file_name):
        """Actualiza el nombre del archivo de análisis."""
        self.new_file_name = new_file_name
        
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
            
#--- Selección de features

    def choose_column(self):
        """Actualiza las columnas del csv seleccionado en el comobobox"""

        # Obtiene las nuevas selecciones de los comboboxes de canales
        self.selected_column = self.column_combobox.currentText()        
        
        # Obtener los valores de la nueva columna seleccionada del DataFrame original
        df = pd.read_csv(self.ruta_archivo)  # Vuelve a leer el CSV para obtener los datos actualizados
        
        if self.selected_column in df.columns:
            # Actualizar los valores de la columna seleccionada
            self.selected_column_values = df[self.selected_column]
            
            # Añadir la columna seleccionada al feature_df
            if hasattr(self, 'feature_df'):
                self.feature_df[self.selected_column] = self.selected_column_values
            else:
                self.feature_df = pd.DataFrame({self.selected_column: self.selected_column_values})
            
            # Actualizar el QLabel con el nuevo DataFrame en formato HTML
            df_html = self.feature_df.to_html(index=False)
            self.label_features.setText(df_html)
        else:
            QMessageBox.warning(self, "Advertencia", "La columna seleccionada no se encuentra en el archivo CSV.")

    
    def next_file(self):
        if hasattr(self, 'feature_df'):
            print("feature_df ya existe.")
        else:
            print("feature_df no existe, se creará ahora.")
            self.feature_df = pd.DataFrame()  # Crear un DataFrame vacío si no existe

        self.feature_df[self.selected_column] = self.selected_column_values
        # Convertir el DataFrame en HTML con estilo de tabla
        df_html = self.feature_df.to_html(index=False)
        
        # Mostrar en el QLabel
        self.label_features.setText(df_html)
        
        """Abrir un cuadro de diálogo para seleccionar un archivo CSV y ejecuta los análisis seleccionados."""
        # Abre el cuadro de diálogo para seleccionar un archivo
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos CSV (*.csv)")

        # Verifica si se seleccionó un archivo
        if archivo:
            # Actualiza la ruta del archivo en todos los análisis
            self.ruta_archivo = archivo
            self.label_csv_update
        
    
    def label_csv_update(self):
        """Actualiza df, df_head, csv_html y el QLabel que muestra el CSV."""
        
        # Verifica si ya existe una ruta de archivo
        if hasattr(self, 'ruta_archivo'):
            try:
                # Lee el archivo CSV en un DataFrame de pandas
                df = pd.read_csv(self.ruta_archivo)
                
                # Obtener el head (encabezados y 5 primeras filas)
                df_head = df.head()
                
                # Convertir el DataFrame en HTML con estilo de tabla
                csv_html = df_head.to_html(index=False)
                
                # Mostrar en el QLabel
                self.label_csv_actual.setText(csv_html)
                
                # Actualizar el combobox con los nuevos nombres de columnas
                columns = df.columns.tolist()
                self.column_combobox.clear()  # Limpia el combobox actual
                self.column_combobox.addItems(columns)  # Añade las nuevas columnas
                
                # Actualiza la columna seleccionada por defecto
                self.selected_column = columns[0]  
                self.selected_column_values = df[self.selected_column]  # Actualiza los valores de la columna seleccionada

                print("CSV actualizado exitosamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al leer el archivo CSV: {str(e)}")
        else:
            QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún archivo.")

# --- Comienza Parte de Análisis


    def PCA_function(self):
        print("PCA EN CURSO...")
        
        # Drop rows with missing values.
        # Save DataFrame in variable `feature_signals`.
        feature_signals = self.feature_df.dropna(axis=0).reset_index(drop = True)
        
        # Usar kdeplot solo si hay suficiente dispersión
        def custom_kde(x, y, **kwargs):
            if x.var() != 0 and y.var() != 0:
                sns.kdeplot(x=x, y=y, **kwargs)
        
        # Create a pairplot of the data to show the relationship between pairs of independent variables.
        
        if len(feature_signals) < 10:
            num_bins = len(feature_signals)
        else:
            num_bins = int(np.ceil(np.log2(len(feature_signals)) + 1))
        
        # Usar una paleta de colores brillante
        palette = sns.color_palette("bright")        
        g = sns.pairplot(feature_signals, hue="Signal", diag_kind="hist", diag_kws={"bins": num_bins}, palette = palette)
        # Colorear los histogramas según la clase ("Signal")
        classes = feature_signals["Signal"].unique()  # Obtener las clases (A, B, C, etc.)
        colors = dict(zip(classes, palette))  # Asociar cada clase a un color de la paleta

        # Iterar sobre los ejes de los histogramas diagonales y colorearlos
        for ax, feature in zip(g.diag_axes, feature_signals.columns[1:]):
            for class_value in classes:
                subset = feature_signals[feature_signals["Signal"] == class_value]
     

        # Cambiar el color de los textos de la leyenda a blanco
        for text in g._legend.texts:
            text.set_color("gray")  # Cambiar el color de la leyenda a blanco
        g._legend.get_title().set_color("gray")

        # Añade el kdeplot en la parte inferior del gráfico
        g.map_lower(custom_kde, fill=True, color=".2", warn_singular=False)
        g.map_lower(custom_kde, levels=4, color=".2", warn_singular=False)
        # Agregar líneas de regresión a la parte superior del gráfico
        g.map_upper(sns.regplot, scatter_kws={"s": 10}, line_kws={"color": "red"})
        
        # Configurar la figura
        fig = plt.gcf()  # Obtener la figura de pairplot
        fig.set_size_inches(10, 6)  # Tamaño de la figura

        # Cambiar el título de la figura
        fig.suptitle('Pairplot among features', color='gray')

        # Configurar fondo transparente y espinas para todos los subgráficos
        for ax in g.axes.flatten():
            if ax is not None:
                ax.set_facecolor('black')  # Fondo negro para los subgráficos
                ax.xaxis.label.set_color('gray')  # Etiquetas del eje X en gris
                ax.yaxis.label.set_color('gray')  # Etiquetas del eje Y en gris
                ax.tick_params(axis='x', colors='gray')  # Ticks del eje X en gris
                ax.tick_params(axis='y', colors='gray')  # Ticks del eje Y en gris
                ax.spines['bottom'].set_color('gray')  # Bordes en gris
                ax.spines['top'].set_color('none')  # Ocultar espina superior
                ax.spines['right'].set_color('none')  # Ocultar espina derecha
                ax.spines['left'].set_color('none')  # Ocultar espina izquierda
                ax.grid(color='gray')  # Color de la cuadrícula

        # Establecer el color de fondo de la figura principal
        fig.patch.set_facecolor('black')
        
        # Colocar la leyenda de forma más visible
        plt.legend(title='Clase', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title_fontsize='medium', frameon=False)
    

        # Ajustar el diseño para evitar superposiciones
        plt.tight_layout()

        # Guardar la figura.
        #plt.savefig(f"{self.new_file_name}_s", facecolor=fig.get_facecolor())
        self.save_fig(f"{self.new_file_name}_Pairplot_among_features")

        # Mostrar la figura
        plt.show()

        
        # **Note:**Ya que signal no es un feature, no se escala. 
        #
        # Excluir `Signal` de X
        self.signals_labels = feature_signals['Signal']
        features_df = feature_signals.drop(['Signal'], axis=1)

        
        def apply_pca(features_df, min_var=0.95, top_features = 3):
            """
            Aplica PCA a las características escaladas y visualiza los resultados.

            Args:
            feature_df (pd.DataFrame): DataFrame con las características originales.

            Returns:
            PCA: Objeto PCA ajustado.
            """

            # Escalar las características
            #feature_columns = features_df.columns.drop('Signal') #elimina la etiqueta "signal" de la lista de nombres de columnas (df.columns) 
            feature_columns = features_df.columns
            features_scaled = StandardScaler().fit_transform(features_df[feature_columns])
    
            
            # Aplicar PCA
            pca = PCA()
            principal_components = pca.fit_transform(features_scaled)

            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            num_components = np.argmax(cumulative_variance >= min_var) + 1
            
            
            loadings = pca.components_.T
            component_names = [f'Componente Principal {i+1}' for i in range(len(explained_variance))]
            
            # Imprimir resultados del PCA
            print(f"Número de componentes para explicar al menos el {min_var * 100}% de la varianza: {num_components}")
            for idx, var in enumerate(explained_variance[:num_components], start=1):
                print(f'{component_names[idx-1]}: Varianza explicada = {var:.4f}')
                sorted_loadings = sorted(zip(loadings[:, idx-1], feature_columns), key=lambda x: abs(x[0]), reverse=True)
                print(f'{top_features} características más influyentes en {component_names[idx-1]}: '
                    f'{", ".join([f"{name} ({loading:.4f})" for loading, name in sorted_loadings[:top_features]])}')
            
            # Insertar los nombres de las señales en la primera columna del DataFrame de componentes principales
            features_df.insert(0, 'Signal', self.signals_labels)
            
            if self.save_function_flag:
                # Guardar el DataFrame en un archivo CSV
                
                # Obtener los loading scores (los coeficientes de las variables originales)
                loading_scores = pd.DataFrame(pca.components_, 
                                            columns=features_df[feature_columns].columns, 
                                            index=[f'PC{i+1}' for i in range(len(pca.components_))])
                # Insertar los nombres de las características en la primera columna del DataFrame de loading scores
                loading_scores.T.insert(0, 'Feature', features_df[feature_columns].columns)

                # Crear DataFrame para guardar los resultados
                df_pca = pd.DataFrame(principal_components, 
                                    columns=[f'PC{i+1}' for i in range(len(principal_components[0]))])
                
                # Insertar los nombres de las señales en la primera columna del DataFrame de componentes principales
                df_pca.insert(0, 'Signal', self.signals_labels)

                # Crear una lista con los valores para la fila: 'Explained Variance' en la columna 'Signal' y las varianzas en las demás columnas
                explained_variance_row = ['Explained Variance'] + explained_variance.tolist()

                # Usar pd.DataFrame para añadir la fila al DataFrame original
                df_pca.loc[len(df_pca)] = explained_variance_row
                
                # Genera un timestamp y un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                name = f"{self.new_file_name}_PCA_loading_scores"
                nombre_archivo_data = f"{name}_{timestamp}.csv"
                self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior
                # Ruta completa del archivo
                file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)
                loading_scores.to_csv(file_path, index=False)
                print(f"Datos guardados en {file_path}")
                # Genera un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
                name = f"{self.new_file_name}_PCA"
                nombre_archivo_data = f"{name}_{timestamp}.csv"
                self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior
                # Ruta completa del archivo
                file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)
                df_pca.to_csv(file_path, index=False)
                print(f"Datos guardados en {file_path}")
                # Genera un nombre de archivo único basado en el nombre proporcionado y el timestamp actual
                name = f"{self.new_file_name}_PCA_features"
                nombre_archivo_data = f"{name}_{timestamp}.csv"
                self.nombre_archivo = nombre_archivo_data  # Guarda el nombre del archivo para su uso posterior
                # Ruta completa del archivo
                file_path = os.path.join(self.carpeta_datos, nombre_archivo_data)
                features_df.to_csv(file_path, index=False)
                print(f"Datos guardados en {file_path}")
            
            # Visualización 3D
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            for signal in features_df['Signal'].unique():
                indices_to_keep = features_df['Signal'] == signal
                ax.scatter(principal_components[indices_to_keep, 0], 
                        principal_components[indices_to_keep, 1], 
                        principal_components[indices_to_keep, 2], 
                        label=signal)
            
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_zlabel('Componente Principal 3')
            plt.title('PCA de señales neuromusculares en 3D', color='gray')

            # Configurar la apariencia del subgráfico
            ax.set_facecolor('none')  # Color de fondo transparente para el área del gráfico
            ax.spines['left'].set_color('none')  # Ocultar espina izquierda
            ax.spines['right'].set_color('none')  # Ocultar espina derecha
            ax.spines['top'].set_color('none')  # Ocultar espina superior
            ax.spines['bottom'].set_color('none')  # Ocultar espina inferior
            #ax.set_xticks([])
            ax.legend(frameon=False)
            ax.grid(color = "gray")

                    
            # Establecer el color de fondo de la figura principal como transparente
            fig.patch.set_facecolor('none')

            # Guarda la figura del análisis de PCA
            plt.tight_layout()
            self.save_fig(f"{self.new_file_name}_PCA")

            # Establecer el color de fondo de la figura principal como negro
            fig.patch.set_facecolor('black')

            # Configurar cada subgráfico para que tenga un fondo negro
            ax.set_facecolor('black')

            # Establecer el color de las etiquetas del eje X e Y en gris
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.set_zlabel('Componente Principal 3', color='gray')
            

            # Cambiar el color de los ticks del eje X a gris
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.tick_params(axis='z', colors='gray')

            # Configurar el fondo del subgráfico y las espinas del eje X en negro
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('gray')  # Establecer el color gris para el borde inferior

            # Configurar la leyenda para no tener marco y con texto en gris
            ax.legend(frameon=False, labelcolor='gray')

            # Ajustar el diseño de los gráficos para evitar superposición
            plt.tight_layout()

            # Mostrar la figura
            plt.show()
            return pca
        
        pca_result = apply_pca(features_df)    






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