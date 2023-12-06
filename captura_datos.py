import time
import statistics as stats
import os.path
from time import sleep
import pandas as pd
from sense_hat import SenseHat
import cv2
import matplotlib.pyplot as plt

sense = SenseHat()

t = []
t_sample = []
accel_x, accel_y, accel_z = [], [], []
gyros_x, gyros_y, gyros_z = [], [], []
t_ini = time.time()
t_ant = t_ini
muestras = 400
accel_average = []

ejercicio = ["Extension rodilla",
             "Curl femoral",
             "Abduccion cadera",
             "Aduccion cadera",
             "Patada de gluteo"]

sexos = ["Hombre", "Mujer"]

# Variables a configurar
edad = 24
sexo = sexos[0]
tipo_ejercicio = ejercicio[0]

# CSV a generar
archivo = "datos"
tipo_archivo = ".csv"
bucle = True


# Configurar la visualización en tiempo real
plt.ion()
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('Datos de Aceleración y Giroscopio en Tiempo Real')
ax[0].set_ylabel('Aceleración (Gs)')
ax[1].set_ylabel('Giroscopio (rad/s)')
ax[1].set_xlabel('Tiempo (s)')

# Bucle para tomar las muestras hasta que se pulse el botón
while bucle:
    t_actual = time.time()

    # Lectura de la Aceleración en Gs
    acceleration = sense.get_accelerometer_raw()
    accel_x.append(acceleration['x'])
    accel_y.append(acceleration['y'])
    accel_z.append(acceleration['z'])

    # Lectura del giroscopio en rad/s
    gyroscope = sense.get_gyroscope_raw()
    gyros_x.append(gyroscope['x'])
    gyros_y.append(gyroscope['y'])
    gyros_z.append(gyroscope['z'])

    # Tiempo
    t.append(t_actual - t_ini)
    t_sample.append(t_actual - t_ant)

    t_ant = t_actual

    # Actualizar gráficos en tiempo real
    ax[0].cla()
    ax[1].cla()

    ax[0].plot(t, accel_x, label='Aceleración X')
    ax[0].plot(t, accel_y, label='Aceleración Y')
    ax[0].plot(t, accel_z, label='Aceleración Z')
    ax[0].legend()

    ax[1].plot(t, gyros_x, label='Giroscopio X')
    ax[1].plot(t, gyros_y, label='Giroscopio Y')
    ax[1].plot(t, gyros_z, label='Giroscopio Z')
    ax[1].legend()

    plt.pause(0.01)

# Esperar 100 milisegundos y verificar si la tecla 'q' fue presionada
    key = cv2.waitKey(100) & 0xFF

    # Si la tecla 'q' fue presionada, salir del bucle
    if key == ord("q"):
        print("Tecla 'q' presionada. Deteniendo la ejecución.")
        break

# Detener la visualización al finalizar
plt.ioff()
plt.show()

# Crear un DataFrame con los datos
df = pd.DataFrame({
    'tiempo': t,
    'tipo': tipo_ejercicio,
    'accel_x': accel_x,
    'accel_y': accel_y,
    'accel_z': accel_z,
    'gyros_x': gyros_x,
    'gyros_y': gyros_y,
    'gyros_z': gyros_z,
    'edad': edad,
    'sexo': sexo
})

# El siguiente código comprueba que no hay ningún archivo con su nombre para no sobrescribirlo
intentos = 0
while os.path.isfile(archivo + tipo_archivo):
    print("El archivo", archivo, "ya existe")
    intentos += 1
    if intentos > 1:
        archivo = archivo[:-1]
    archivo = archivo + str(intentos)

print("Se guardará el archivo con nombre: ", archivo)

# Crea un archivo csv y guarda los datos
df.to_csv(archivo + tipo_archivo, index=False, float_format='%.6f')


