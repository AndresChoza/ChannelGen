import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import prod, size
from numpy.core.function_base import linspace

""" #Espacio lineal equi-espaciado
x = np.linspace(-15,15,200) 

print(x_normal)
print("////////////////////////////////////////////")
print(x)

sinc_sig = np.sinc(x)
print(sinc_sig)
#sinc_sig-=3 #Esto desplaza la amplitud
#print(sinc_sig) """

""" plt.plot(x+5,sinc_sig) #Restar a x en el plot "desplaza" la señal sinc
plt.show() """

######################
Fs = 20e6 
Ts = 1/Fs

#Cost 207 Model (Tux)
delay = np.array([0,0.217,0.512,0.514,0.517,0.674,0.882,1.230,1.287,1.311,1.349,1.533,1.535,1.1622,1.818,1.836,1.884,1.943,2.048,2.140])*1e-6
pw_db = np.array([-5.7,-7.6,-10.1,-10.2,-10.2,-11.5,-13.4,-16.3,-16.9,-17.1,-17.4,-19,-19,-19.8,-21.5,-21.6,-22.1,-22.6,-23.5,-24.3])
pw_lineal = 10**(pw_db/10)

""" print(np.size(delay))
print(np.size(pw_db)) """

TauMax = delay[-1] #Retardo Máximo
#print(TauMax)

M=np.size(delay)
L=TauMax*Fs

#Generar Matriz RNG de Paths
#Distribución Normal con Media 0 y Varianza (Potencia) Unitaria
mu = 0 #Media 
var = potencia = 0.5 #Potencia 
sigma = np.square(var) #Desviación Std
x_q = np.random.normal(mu,sigma,M) #Digamos que esta es la ponderación de las sincs que vas a generar
x_i = np.random.normal(mu,sigma,M)*1j
x_normal = x_q + x_i  #Se podría decir que estos son los factores de atenuación? (1.3.1 Matz)

#print(x_normal)

#Generar Sincs
L = int(np.ceil(L))
L = L-1
space = linspace(0,L-1,L) #Linspace usando L es eje en muestras, necesita estar en tiempo
t = space*Ts #Eje de Tiempo
#print(space)
#sinc_test = np.sinc(space-0.5) 

#Necesitamos hacer "M" sincs retrasadas según la variable de delay, en este caso son 20 sincs
""" sinc_1 = pw_lineal[1]*np.sinc(space-delay[1])
sinc_2 = pw_lineal[2]*np.sinc(space-delay[2]) """

ML_Matrix = np.array(np.zeros(shape=(L,M)))
for i in range(M):
    ML_Matrix[:,i] = np.sqrt(pw_lineal[i])*np.sinc((t-(delay[i]))*Fs) #Restar a 't' en la sinc es desplazar, multiplicar por Fs es ponderar
    #plt.plot(t,ML_Matrix[:,i])
#print((ML_Matrix))

producto = ML_Matrix@x_normal
print(np.size(producto))
print(L)
print(x_normal)
#print(producto)

plt.plot(t,(producto))

""" t = linspace(-5,5,1000)
sink = np.sinc(t*Fs) #Ponderar cruces por 0
plt.plot(t,sink) """

#plt.plot(space,ML_Matrix[:,2],'r--',space,ML_Matrix[:,3],'b--')
plt.show()