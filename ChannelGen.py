import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import prod, size
from numpy.core.function_base import linspace
import cmath
import sys
np.set_printoptions(threshold=sys.maxsize)

from numpy.random.mtrand import f
from scipy import signal

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
N=20
L=TauMax*Fs

#Generar Matriz RNG de Paths
#Distribución Normal con Media 0 y Varianza (Potencia) Unitaria
mu = 0 #Media 
var = potencia = 1 #Potencia 
sigma = np.square(var) #Desviación Std
#Estos dos representan un solo path, necesitamos M de estos, no estos de M tamaño

x_q = np.zeros((M,N))
x_i = np.zeros((M,N))

for k in range(M):
    x_q[k,:] = np.random.normal(mu,sigma,N) #Digamos que esta es la ponderación de las sincs que vas a generar
    x_i[k,:] = np.random.normal(mu,sigma,N)

x_i = x_i*1j

##############################JAKES###########################
#Introducimos el filtro FIR de distribución Jakes a cada uno de los coeficientes de x_q y x_i 
Fs_J = 5e5
N1=20001
N_Hamming=2001
f_lin = (np.linspace(-Fs_J/2,Fs_J/2,N1)) #O bien define tu linspace como complejo (?)
Fmax = 1500
#test = np.sqrt(1-((f_lin/Fmax)**2)) 
#print(test)
Sc_LambdaT = (1/((np.pi)*Fmax*(np.sqrt(1-((np.complex64(f_lin)/Fmax)**2))+0.0000001)))  #np.sqrt no puede lidiar con complejos, solo cmath.sqrt
#print(Sc_LambdaT[10000]) #En teoría sí funcionó lo de hacer complejo el linspace

#Sc_LambdaT(abs(f_lin)>=Fmax)=0;
#Necesitamos la parte real y luego todo fuera de Fmax debe ser 0 
Sc_LambdaT = np.real(Sc_LambdaT)
Sc_LambdaT[abs(f_lin)>=Fmax] = 0


#print(type(Sc_LambdaT[10000]))

#print(np.real(Sc_LambdaT[abs(f_lin)>=Fmax]))

#print(Sc_LambdaT[1])
#print(size(Sc_LambdaT))
#plt.plot(f_lin,(Sc_LambdaT)) #Todo bien, EN ALGUN PUNTO HAY VALORES IMAG
#plt.show()

Hf = np.sqrt(Sc_LambdaT)

#plt.plot(f_lin,(Hf)) #Todo bien
#plt.show()

Hf_FL = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Hf)))

#plt.plot(f_lin,(Hf_FL)) #Todo bien
#plt.show()

window = np.hamming(N_Hamming)
window = np.pad(window, (9000,9000), 'constant', constant_values=(0,0))

hfw = Hf_FL*window

#plt.plot(f_lin,(hfw)) #XD
#plt.show()

hfw = hfw[hfw != 0]

#print(np.real(hfw))

hfw = hfw / np.linalg.norm(hfw) #Normalizamos

#plt.plot((hfw)) #XD sigue habiendo valores imag
#plt.show()

### Respuesta en Frec del Filtro FIR
#w, h = signal.freqz(hfw)

#Sin freqz
# hfw_z = np.pad(hfw, (0,size(hfw)),'constant',constant_values=(0,0))
# freq = np.fft.fft(hfw_z)
# ???

#fig = plt.figure()
#plt.title('Digital filter frequency response')
# ax1 = fig.add_subplot(111)

# plt.plot(freq)
#plt.plot((w*Fs)/2*np.pi, (20 * np.log10(abs(h))), 'b')
# plt.ylabel('Amplitude [dB]', color='b')
# plt.xlabel('Frequency [rad/sample]')
#plt.show()

# Aplicamos filtro FIR a ambas variables
#Necesitamos condiciones iniciales para poder inyectarlas en la siguiente función

#x_qf = signal.lfilter(hfw,1,x_q)
#x_if = signal.lfilter(hfw,1,x_i)

##############Jakes############

#Necesitamos crear una matriz grande donde metamos todos los paths (Real+Imag) 
x_iq = np.zeros((M,N),dtype='complex_')

#Para cada valor de M vamos a filtrar un vector fila de tamaño N, luego sumar I + Q
#Ese resultado lo metemos en x_iq 
#Recordemos que lo importante es hacer toda la matriz NxM sin perder el estado 

#Creamos estados iniciales para I y Q, funcioens lfilter_zi tiran valores del tamaño de hfw menos 1 
#es decir de 2000, son demasiados
zi_i = np.zeros(size(hfw)-1)
zi_q = np.zeros(size(hfw)-1)

#Ademas de variables que sostengan el estado
zf_i = 0
zf_q = 0

#Necesitamos un ciclo que dure 'M' donde se filtre, se sume y se ingrese en x_iq 
#ademas de conservar estados

#print(size(hfw))

for i in range(M):
    xfilt_i, zf_i = signal.lfilter(hfw,1,x_i[i,:],zi=zi_i)
    xfilt_q, zf_q = signal.lfilter(hfw,1,x_q[i,:],zi=zi_q)

    x_iq[:,i] = xfilt_i + xfilt_q #Estos son los path 

    #print(x_iq[:,i])

    zi_i = zf_i
    zi_q = zf_q

#xfilt_i, zf_i = signal.lfilter(hfw,1,x_i[1,:],zi=zi_i) #Asi si entiende, olvidabamos poner zi=...

# x_normal = x_q + x_i  #Se podría decir que estos son los factores de atenuación? (1.3.1 Matz)
# x_normal_f = x_qf + x_if

# #print(x_normal)

############################################

#Generar Sincs
L = int(np.ceil(L))
L = L-1
space = linspace(0,L-1,L) #Linspace usando L es eje en muestras, necesita estar en tiempo
t = space*Ts #Eje de Tiempo
#print(space)
#sinc_test = np.sinc(space-0.5) 

#Necesitamos hacer "M" sincs retrasadas según la variable de delay, en este caso son 20 sincs
ML_Matrix = np.array(np.zeros(shape=(L,M)))
for i in range(M):
    ML_Matrix[:,i] = np.sqrt(pw_lineal[i])*np.sinc((t-(delay[i]))*Fs) #Restar a 't' en la sinc es desplazar, multiplicar por Fs es ponderar
    plt.plot(t,ML_Matrix[:,i])
#print((ML_Matrix))

taps = np.zeros((L,M),dtype='complex_')

for k in range(M):
    taps[:,k] = ML_Matrix@(x_iq[k,:])

# producto = ML_Matrix@x_normal #M-paths a L-taps
print(taps)
#print(L)
# #print(x_normal)
# print(producto)

plt.plot(t,(taps))

# #plt.plot(space,ML_Matrix[:,2],'r--',space,ML_Matrix[:,3],'b--')
plt.show()