import time
import statistics as stats
import os.path
from time import sleep
import pandas as pd
from sense_hat import SenseHat
sense=SenseHat()

t=[]
t_sample=[]
accel_x, accel_y, accel_z = [], [], []
gyros_x, gyros_y, gyros_z = [], [], []
t_ini=time.time()
t_ant=t_ini
muestras=400 # muestras que tomamos
accel_average=[]

ejercicio=["Extension rodilla",
           "Curl femoral",
           "Abduccion cadera",
           "Aduccion cadera",
           "Patada de gluteo"]

sexos=["Hombre","Mujer"]

# Variables a configurar
edad=24
sexo=sexos[0]
tipo_ejercicio=ejercicio[0]

# CSV a generar
archivo="datos" # nombre del archivo
tipo_archivo=".csv" # extensión 
bucle=True
# Bucle para tomar las muestras hasta que se pulse el boton
while(bucle):
    events = sense.stick.get_events()

    for event in events:
        if event.direction == 'middle':
            bucle=False

    t_actual=time.time()

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
    t.append(t_actual-t_ini)
    t_sample.append(t_actual-t_ant)
    
    t_ant=t_actual
    
sense=SenseHat()
O = [255,0,0]
X = [255,255,0]

UNO = [
  O, O, O, O, O, O, O, O,
  O, O, O, O, O, O, O, O,
  O, O, O, O, O, O, O, O,
  O, O, O, X, X, O, O, O,
  O, O, O, X, X, O, O, O,
  O, O, O, O, O, O, O, O,
  O, O, O, O, O, O, O, O,
  O, O, O, O, O, O, O, O,
  ]

# fin de la toma de muestras
sense.set_pixels(UNO) 
sleep(10)
sense.clear()

print("Rate: ",int(1/float(format(stats.mean(t_sample),"f")))," Hz")


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

#El siguiente código comprueba que no hay ningun archivo con su nombre para no sobreescribirlo
intentos=0
while(os.path.isfile(archivo+tipo_archivo)):
    #En caso de que exista un fichero con dicho nombre, le pone un numero al final (consecutivo)
    print("El archivo",archivo,"ya existe")
    intentos+=1
    if(intentos>1):
        archivo=archivo[:-1]
    archivo=archivo+str(intentos)

print("Se guardará el archivo con nombre: ",archivo)

# Crea un archivo csv y guarda los datos
df.to_csv(archivo+tipo_archivo, index=False, float_format='%.6f')



