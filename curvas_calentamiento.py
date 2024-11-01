#%% Templogs / Transiciones de Fase / Calores especificos 
'''
Analizo templogs
'''
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from datetime import datetime,timedelta
import matplotlib as mpl
from scipy.interpolate import CubicSpline,PchipInterpolator
#%%
def lector_templog_rugged(directorio,rango_T_fijo=True):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura y plotea el log completo 
    '''
    data = pd.read_csv(directorio,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
        
    temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2= pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    t = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return t,temp_CH1


def lector_templog_raytec(directorio):
    '''
    Busca archivo .txt en el directorio especificado.
    Extrae los datos de timestamp y temperatura de la tercera columna.
    Retorna arrays timestamp, temperatura y plotea el log completo si es necesario.
    
    Argumentos:
    directorio -- la ruta del archivo .txt
    '''
    
    # Cargar el archivo .txt, omitiendo las primeras tres líneas de encabezado
    data = pd.read_csv(directorio, sep='\t', header=2,
                       names=('Timestamp', 'Temp_CH1'), 
                       usecols=(0, 1), decimal='.', engine='python')
    
    # Convertir la columna de temperatura a un arreglo numpy de floats
    temp_CH1 = pd.Series(data['Temp_CH1']).to_numpy(dtype=float)
    
    # Crear el array de timestamps
    timestamp = np.array([datetime.strptime(date.strip(), '%d/%m/%Y %H:%M:%S.%f') for date in data['Timestamp']])
    t = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    
    return t, temp_CH1


path_test=os.path.join(os.getcwd(),'10-30-24_4-37-23_pm_Logfile.txt')

time,Temp=lector_templog_raytec(path_test)
plt.plot(time,Temp)
#%% 
paths_IR = glob(os.path.join('*.txt'))
paths_Rugged = glob(os.path.join('*.csv'))

t_IR_1,T_IR_1=lector_templog_raytec(paths_IR[0])
t_IR_2,T_IR_2=lector_templog_raytec(paths_IR[1])
t_IR_3,T_IR_3=lector_templog_raytec(paths_IR[2])

t_Rugged_1,T_Rugged_1=lector_templog_rugged(paths_Rugged[0])
t_Rugged_2,T_Rugged_2=lector_templog_rugged(paths_Rugged[1])
t_Rugged_3,T_Rugged_3=lector_templog_rugged(paths_Rugged[2])
#%% recorto al maximo valor
indx_max_IR_1 = np.nonzero(T_IR_1==max(T_IR_1))
t_IR_1=t_IR_1[indx_max_IR_1[0][0]:]
t_IR_1-=t_IR_1[0]
T_IR_1=T_IR_1[indx_max_IR_1[0][0]:]

indx_max_IR_2 = np.nonzero(T_IR_2==max(T_IR_2))
t_IR_2=t_IR_2[indx_max_IR_2[0][0]:]
t_IR_2-=t_IR_2[0]
T_IR_2=T_IR_2[indx_max_IR_2[0][0]:]

indx_max_IR_3 = np.nonzero(T_IR_3==max(T_IR_3))
t_IR_3=t_IR_3[indx_max_IR_3[0][0]:]
t_IR_3-=t_IR_3[0]
T_IR_3=T_IR_3[indx_max_IR_3[0][0]:]

indx_max_Rugged_1 = np.nonzero(T_Rugged_1==max(T_Rugged_1))
t_Rugged_1=t_Rugged_1[indx_max_Rugged_1[0][0]:]
t_Rugged_1-=t_Rugged_1[0]
T_Rugged_1=T_Rugged_1[indx_max_Rugged_1[0][0]:]

indx_max_Rugged_2 = np.nonzero(T_Rugged_2==max(T_Rugged_2))
t_Rugged_2=t_Rugged_2[indx_max_Rugged_2[0][0]:]
t_Rugged_2-=t_Rugged_2[0]
T_Rugged_2=T_Rugged_2[indx_max_Rugged_2[0][0]:]

indx_max_Rugged_3 = np.nonzero(T_Rugged_3==max(T_Rugged_3))
t_Rugged_3=t_Rugged_3[indx_max_Rugged_3[0][0]:]
t_Rugged_3-=t_Rugged_3[0]
T_Rugged_3=T_Rugged_3[indx_max_Rugged_3[0][0]:]

# %% 2x1
fig,(ax,ax1)=plt.subplots(nrows=2,figsize=(10,8),constrained_layout=True)
ax.plot(t_Rugged_1,T_Rugged_1,'.-',label='Rugged_1')
ax.plot(t_Rugged_2,T_Rugged_2,'.-',label='Rugged_2')
ax.plot(t_Rugged_3,T_Rugged_3,'.-',label='Rugged_3')

ax1.plot(t_IR_1,T_IR_1,'.-',label='IR_1')
ax1.plot(t_IR_2,T_IR_2,'.-',label='IR_2')
ax1.plot(t_IR_3,T_IR_3,'.-',label='IR_3')

ax.set_title('Rugged',loc='left')
ax1.set_title('Raytec IR',loc='left')
for a in [ax,ax1]:
    a.grid()
    a.set_ylabel('T (°C)')
    a.legend()
ax1.set_xlabel('t (s)')
plt.savefig('Rugged_vs_Raytec.png',dpi=300)
plt.show()

#%% 3x1
fig,(ax,ax1,ax2)=plt.subplots(nrows=3,figsize=(10,12),constrained_layout=True,sharex=True)
ax.plot(t_IR_1,T_IR_1,'.-',label='IR_1')
ax.plot(t_Rugged_1,T_Rugged_1,'.-',label='Rugged_1')

ax1.plot(t_IR_2,T_IR_2,'.-',label='IR_2')
ax1.plot(t_Rugged_2,T_Rugged_2,'.-',label='Rugged_2')

ax2.plot(t_IR_3,T_IR_3,'.-',label='IR_3')
ax2.plot(t_Rugged_3,T_Rugged_3,'.-',label='Rugged_3')

ax.set_title('Comparativa sensores',loc='left')
for a in [ax,ax1,ax2]:
    a.grid()
    a.set_ylabel('T (°C)')
    a.legend(loc='lower left')
ax2.set_xlabel('t (s)')
# plt.savefig('sar_vs_Temp.png',dpi=300)
plt.show()

# %% interpolo para restar 

t_max_common_1 = min(t_IR_1[-1], t_Rugged_1[-1])
indices_IR = np.where(t_IR_1 <= t_max_common_1)
t_IR_1_recortado = t_IR_1[indices_IR]
T_IR_1_recortado = T_IR_1[indices_IR]
indices_Rugged = np.where(t_Rugged_1 <= t_max_common_1)
t_Rugged_1_recortado = t_Rugged_1[indices_Rugged]
T_Rugged_1_recortado = T_Rugged_1[indices_Rugged]

t_max_common_2 = min(t_IR_2[-1], t_Rugged_2[-1])
indices_IR = np.where(t_IR_2 <= t_max_common_2)
t_IR_2_recortado = t_IR_2[indices_IR]
T_IR_2_recortado = T_IR_2[indices_IR]
indices_Rugged = np.where(t_Rugged_2 <= t_max_common_2)
t_Rugged_2_recortado = t_Rugged_2[indices_Rugged]
T_Rugged_2_recortado = T_Rugged_2[indices_Rugged]

t_max_common_3 = min(t_IR_3[-1], t_Rugged_3[-1])
indices_IR = np.where(t_IR_3 <= t_max_common_3)
t_IR_3_recortado = t_IR_3[indices_IR]
T_IR_3_recortado = T_IR_3[indices_IR]
indices_Rugged = np.where(t_Rugged_3 <= t_max_common_3)
t_Rugged_3_recortado = t_Rugged_3[indices_Rugged]
T_Rugged_3_recortado = T_Rugged_3[indices_Rugged]

fig,(ax,ax1,ax2)=plt.subplots(nrows=3,figsize=(10,12),constrained_layout=True,sharex=True)
ax.plot(t_IR_1_recortado,T_IR_1_recortado,'.-',label='IR_1')
ax.plot(t_Rugged_1_recortado,T_Rugged_1_recortado,'.-',label='Rugged_1')

ax1.plot(t_IR_2_recortado,T_IR_2_recortado,'.-',label='IR_2')
ax1.plot(t_Rugged_2_recortado,T_Rugged_2_recortado,'.-',label='Rugged_2')

ax2.plot(t_IR_3_recortado,T_IR_3_recortado,'.-',label='IR_3')
ax2.plot(t_Rugged_3_recortado,T_Rugged_3_recortado,'.-',label='Rugged_3')

ax.set_title('Comparativa sensores',loc='left')
for a in [ax,ax1,ax2]:
    a.grid()
    a.set_ylabel('T (°C)')
    a.legend(loc='lower left')
ax2.set_xlabel('t (s)')
# plt.savefig('sar_vs_Temp.png',dpi=300)
plt.show()
#%%

interp_func = PchipInterpolator(t_full, T_full)
        t_interp = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
        T_interp = np.round(interp_func(t_interp), 2)