import matplotlib.pyplot as plt
import numpy as np

def gamma_schedule(t):
    start_gamma=15000
    if t<=iterations/2.:
        return start_gamma
    else:
        return start_gamma*(1-(t-iterations/2.)/(iterations/2.))

def gamma_schedule2(t):
    end_gamma=15000
    if t<=iterations/2.:
        return 0
    else:
        return end_gamma*((t-iterations/2.)/(iterations/2.))

iterations=100
t=np.linspace(0,iterations,num=100)
y=np.zeros_like(t)
ind=0
for i in t:
    y[ind]=gamma_schedule2(i)
    ind+=1

plt.plot(t,y)
plt.show()
