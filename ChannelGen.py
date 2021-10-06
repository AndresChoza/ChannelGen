import matplotlib.pyplot as plt
import numpy as np

#Espacio lineal equi-espaciado
x = np.linspace(-15,15,200) 

#Distribución Normal con Media 0 y Varianza (Potencia) alguna
mu = 0 #Media 
var = potencia = 1 #Potencia 
sigma = np.square(var) #Desviación Std
x_normal = np.random.normal(mu,sigma,200)
print(x_normal)
print("////////////////////////////////////////////")
print(x)

sinc_sig = np.sinc(x)
print(sinc_sig)
#sinc_sig-=3 #Esto desplaza la amplitud
#print(sinc_sig)

plt.plot(x+5,sinc_sig) #Restar a x en el plot "desplaza" la señal sinc
plt.show()
#Test2
