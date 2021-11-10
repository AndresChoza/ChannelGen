import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import prod, size
from numpy.core.function_base import linspace
import cmath
import sys
np.set_printoptions(threshold=sys.maxsize)

from numpy.random.mtrand import f

######################
Fs = 20e6 
Ts = 1/Fs

#Cost 207 Model (Tux)
delay = np.array([0,0.217,0.512,0.514,0.517,0.674,0.882,1.230,1.287,1.311,1.349,1.533,1.535,1.1622,1.818,1.836,1.884,1.943,2.048,2.140])*1e-6
pw_db = np.array([-5.7,-7.6,-10.1,-10.2,-10.2,-11.5,-13.4,-16.3,-16.9,-17.1,-17.4,-19,-19,-19.8,-21.5,-21.6,-22.1,-22.6,-23.5,-24.3])
pw_lineal = 10**(pw_db/10)

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

##############################JAKES###########################
#Introducimos el filtro FIR de distribución Jakes a cada uno de los coeficientes de x_q y x_i 
Fs = 5e5
N1=20001
N=2001
f_lin = (np.linspace(-Fs/2,Fs/2,N1)) #O bien define tu linspace como complejo (?)
Fmax = 1500
#test = np.sqrt(1-((f_lin/Fmax)**2)) 
#print(test)
Sc_LambdaT = (1/((np.pi)*Fmax*(np.sqrt(1-((np.complex64(f_lin)/Fmax)**2))+0.0000001)))  #np.sqrt no puede lidiar con complejos, solo cmath.sqrt
print(Sc_LambdaT[10000]) #En teoría sí funcionó lo de hacer complejo el linspace

#Sc_LambdaT(abs(f_lin)>=Fmax)=0;
#Necesitamos la parte real y luego todo fuera de Fmax debe ser 0 
Sc_LambdaT = np.real(Sc_LambdaT)
Sc_LambdaT[abs(f_lin)>=Fmax] = 0


print(type(Sc_LambdaT[10000]))

#print(np.real(Sc_LambdaT[abs(f_lin)>=Fmax]))

print(Sc_LambdaT[1])
print(size(Sc_LambdaT))
#plt.plot(f_lin,(Sc_LambdaT)) #Todo bien, EN ALGUN PUNTO HAY VALORES IMAG
#plt.show()

Hf = np.sqrt(Sc_LambdaT)

#plt.plot(f_lin,(Hf)) #Todo bien
#plt.show()

Hf_FL = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Hf)))

#plt.plot(f_lin,(Hf_FL)) #Todo bien
#plt.show()

window = np.hamming(N)
window = np.pad(window, (9000,9000), 'constant', constant_values=(0,0))

hfw = Hf_FL*window

#plt.plot(f_lin,(hfw)) #XD
#plt.show()

hfw = hfw[hfw != 0]

print(np.real(hfw))

hfw = hfw / np.linalg.norm(hfw)

plt.plot((hfw)) #XD sigue habiendo valores imag
plt.show()

# x_normal = x_q + x_i  #Se podría decir que estos son los factores de atenuación? (1.3.1 Matz)

# #print(x_normal)

# #Generar Sincs
# L = int(np.ceil(L))
# L = L-1
# space = linspace(0,L-1,L) #Linspace usando L es eje en muestras, necesita estar en tiempo
# t = space*Ts #Eje de Tiempo
# #print(space)
# #sinc_test = np.sinc(space-0.5) 

# #Necesitamos hacer "M" sincs retrasadas según la variable de delay, en este caso son 20 sincs
# ML_Matrix = np.array(np.zeros(shape=(L,M)))
# for i in range(M):
#     ML_Matrix[:,i] = np.sqrt(pw_lineal[i])*np.sinc((t-(delay[i]))*Fs) #Restar a 't' en la sinc es desplazar, multiplicar por Fs es ponderar
#     plt.plot(t,ML_Matrix[:,i])
# #print((ML_Matrix))

# producto = ML_Matrix@x_normal 
# print(np.size(producto))
# print(L)
# #print(x_normal)
# print(producto)

# #plt.plot(t,(producto))

# #plt.plot(space,ML_Matrix[:,2],'r--',space,ML_Matrix[:,3],'b--')
# plt.show()