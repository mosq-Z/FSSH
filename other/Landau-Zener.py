
import numpy as np
from math import *
import matplotlib.pyplot as plt

from scipy.linalg import expm

hbar = 1.0



def H_di(x):

    # 模型一
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    if x > 0:
        V11 = A*(1-e**(-B*x))
    else :
        V11 = -A*(1-e**(B*x))
    V22 = -V11
    V12 = V21 = C*e**(-D*x**2)
    return np.array([[V11,V12],[V21,V22]])

def d_H_di(x):
    dx = 1e-8
    H_p = H_di(x-dx)
    H_m = H_di(x+dx)
    return (H_m - H_p)/(2*dx)

H_dia = H_di(0)
M_Quality = 2000

k_input = np.linspace(8.0,32.0,500)
E_up = np.linalg.eigh(H_dia)[0][0] - (-0.01)
# print(E_up)
k_eff = ((k_input**2/(2*M_Quality) - E_up)*2*M_Quality)**0.5
# print(k_eff)
v = k_eff/M_Quality
d_V11 = d_H_di(0)[0,0]
d_V22 = d_H_di(0)[1,1]
V12 = H_dia[0,1]

ans = np.array([exp(-2*pi*V12**2/((d_V11-d_V22)*i)) for i in v])

for i in range(len(ans)):
    print(k_input[i],ans[i])

plt.scatter(k_input,ans,label='L-Z')
plt.legend()
plt.grid()
plt.show()



