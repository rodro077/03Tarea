# -*- coding: utf-8 -*-
'''
P2
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

'''
Condiciones iniciales de x y z
'''
A0=[10,10,10]

'''
Defino la funcion lorentz con sus valores iniciales
'''
def lorenz(t,A):
    return [sigma*(A[1]-A[0]), A[0]*(rho-A[2])-A[1],A[0]*A[1]-beta*A[2]]

sigma=10 ; rho=28 ; beta=8/3 ; t0=1e-3
r = ode(lorenz)
r.set_integrator('dopri5')
r.set_initial_value(A0)

t = 10000
t_values = np.linspace(t0, 10* np.pi,t)
x = np.zeros(t)
y = np.zeros(t)
z = np.zeros(t)

for i in range(len(t_values)):
    r.integrate(t_values[i])
    x[i], y[i] , z[i]= r.y

fig = plt.figure(4)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

plt.title("$ \ Atractor \ de \ Lorenz$",fontsize=25)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.plot(x,y,z,'b')
plt.savefig("fig4.png")
plt.show()


