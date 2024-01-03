# -*- coding: utf-8 -*-
"""
Clasificacion movimientos entrenamiento de fuerza tren inferior

@author: vicguzher y celllarod
"""
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

plt.style.use("seaborn") # estilo de gráficas

#%% Creación del archivo con todos los datos
if not os.path.exists('dataset\datos_completos.csv'):
    # Especifica la carpeta donde se encuentran tus archivos CSV
    carpeta = 'dataset'
    
    # Lista para almacenar todos los marcos de datos
    marcos_de_datos = []
    
    # Itera sobre los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.csv'):
            # Crea la ruta completa al archivo
            ruta_completa = os.path.join(carpeta, archivo)
            
            # Lee el archivo CSV y añade el DataFrame a la lista
            df = pd.read_csv(ruta_completa)
            marcos_de_datos.append(df)
    
    # Concatena todos los DataFrames en uno solo
    df_final = pd.concat(marcos_de_datos, ignore_index=True)
    
    # Guarda el DataFrame resultante en un nuevo archivo CSV
    df_final.to_csv('dataset\datos_completos.csv', index=False, float_format='%.6f')
    
#%%  Etiquetas de las actividades

LABELS = ['Extension rodilla',
             'Curl femoral',
             'Abduccion cadera',
             'Aduccion cadera',
             'Patada de gluteo']

# El número de pasos dentro de un segmento de tiempo
TIME_PERIODS = 60

# Los pasos a dar de un segmento al siguiente; si este valor es igual a
# TIME_PERIODS, entonces no hay solapamiento entre los segmentos
STEP_DISTANCE = 5

# al haber solapamiento aprovechamos más los datos

#%% cargamos los datos

# column_names = ['tiempo',
#                     'tipo',
#                     'accel_x',
#                     'accel_y',
#                     'accel_z',
#                     'gyros_x',
#                     'gyros_y',
#                     'gyros_z',
#                     'edad',
#                     'sexo'
#                     ]

df = pd.read_csv("dataset\datos_completos.csv")


print(df.info())

#%% Datos que tenemos

print(df.shape)

#%% Procesamiento de los datos

# eliminamos el ; del eje z 
#df['z-axis'].replace(regex=True, inplace=True, to_replace=r';',
      #value=r'')

#%% convertimos a flotante

# def convert_to_float(x):
#     try:
#         return float(x)
#     except:
#         return np.nan

# df['z-axis'] = df['z-axis'].apply(convert_to_float)


#%% Eliminamos entradas que contengan Nan --> ausencia de datos

#df.dropna(axis=1, how='any', inplace=True)

#%% Mostramos los primeros datos

print(df.head())

#%% Mostramos los últimos

print(df.tail())

#%% Visualizamos la cantidad de datos que tenemos
# de cada actividad 

actividades = df['tipo'].value_counts()
plt.bar(range(len(actividades)), actividades.values)
plt.xticks(range(len(actividades)), actividades.index)

#%% visualizamos 

def dibuja_datos_aceleracion(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["accel_x"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["accel_y"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["accel_z"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

for actividad in np.unique(df['tipo']):
    subset = df[df['tipo'] == actividad][:30]
    dibuja_datos_aceleracion(subset, actividad)

#%% Codificamos la actividad de manera numérica

from sklearn import preprocessing

LABEL = 'ActivityEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df[LABEL] = le.fit_transform(df['tipo'].values.ravel())



print(df.head())

#%% Normalizamos los datos

df["accel_x"] = (df["accel_x"] - min(df["accel_x"].values)) / (max(df["accel_x"].values) - min(df["accel_x"].values))
df["accel_y"] = (df["accel_y"] - min(df["accel_y"].values)) / (max(df["accel_y"].values) - min(df["accel_y"].values))
df["accel_z"] = (df["accel_z"] - min(df["accel_z"].values)) / (max(df["accel_z"].values) - min(df["accel_z"].values))
df["gyros_x"] = (df["gyros_x"] - min(df["gyros_x"].values)) / (max(df["gyros_x"].values) - min(df["gyros_x"].values))
df["gyros_y"] = (df["gyros_y"] - min(df["gyros_y"].values)) / (max(df["gyros_y"].values) - min(df["gyros_y"].values))
df["gyros_z"] = (df["gyros_z"] - min(df["gyros_z"].values)) / (max(df["gyros_z"].values) - min(df["gyros_z"].values))


#%% Representamos para ver que se ha hecho bien

plt.figure(figsize=(5,5))
plt.plot(df["accel_x"].values[:30])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")

#%% Disión datos den entrenamiento y test

# df_test = df[df['user-id'] > 28]
# df_train = df[df['user-id'] <= 28]
#%%
#Realizamos el one-hote econding para los datos de salida



# Se divide el Dataset en Train y Test
# Se barajan antes de dividirlo (shuffle=True)
# Se dividen de forma que entrenamiento y test estén balanceados (stratify=y)
# Se dividen de forma aleatoria, pero siempre la misma (random_state=int)
y=df.loc[:, "tipo"].to_numpy()

df_train, df_test = train_test_split(df, test_size=0.2,stratify=None,shuffle=False,random_state=5)

print("Entrenamiento", df_train.shape)
print("Test", df_test.shape)

#%% comprobamos cual ha sido la división

print("Entrenamiento", df_train.shape[0]/df.shape[0])
print("Test", df_test.shape[0]/df.shape[0])

#%% Creamos las secuencias

from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleraciones
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['accel_x'].values[i: i + time_steps]
        ys = df['accel_y'].values[i: i + time_steps]
        zs = df['accel_z'].values[i: i + time_steps]
        xg = df['gyros_x'].values[i: i + time_steps]
        yg = df['gyros_y'].values[i: i + time_steps]
        zg = df['gyros_z'].values[i: i + time_steps]
        # Lo etiquetamos como la actividad más frecuente 
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xs, ys, zs, xg, yg, zg])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

#%% observamos la nueva forma de los datos (80, 3)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

#%% datos de entrada de la red neuronal

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

#%% transformamos los datos a flotantes

x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
y_train_hot = cat_encoder.fit_transform(y_train.reshape(len(y_train),1))
y_train = y_train_hot.toarray()

#%% RED NEURONAL


epochs = 70
batch_size = 5
filters = 128
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=filters/2, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()
#%% Compilamps el modelo y entrenamos

from tensorflow.keras import callbacks

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath='best_model_padel.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    callbacks.EarlyStopping(monitor='acc', patience=1)
]

# fit network       
history = train_log = model.fit(x_train, 
                      y_train, 
                      validation_split=0.2, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      verbose=True)

#%% visualizamos el entrenamiento
import matplotlib.pyplot as plt

# Suponiendo que `history` contiene los datos de entrenamiento y validación de tu modelo

# Precisión
plt.figure(figsize=(12, 6))

# Plot de la precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'r--', label='Training Loss')
plt.plot(history.history['val_loss'], 'b--', label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Training Epoch')
plt.legend()

plt.tight_layout()
plt.show()

#%% Evaluamos el modelo

cat_encoder = OneHotEncoder()
y_test_hot = cat_encoder.fit_transform(y_test.reshape(len(y_test),1))
y_test = y_test_hot.toarray()
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


# test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

# print("Test accuracy", test_accuracy)
# print("Test loss", test_loss)




#%%
# Print confusion matrix for training data
from sklearn.metrics import classification_report
y_pred_train = model.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train, axis=1)
print(classification_report(max_y_train, max_y_pred_train))

#%%
import seaborn as sns
from sklearn import metrics

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model.predict(x_test)
# Toma la clase con la mayor probabilidad a partir de las predicciones de la prueba
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))

