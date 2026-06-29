import numpy as np
from math import *
import matplotlib.pyplot as plt

from scipy.linalg import expm

hbar = 1.0


# 根据坐标x文献里面透热哈密顿
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
    
    # # 模型二
    # A = 0.1
    # B = 0.28
    # E0 = 0.05
    # C = 0.015
    # D = 0.06
    # V11 = 0
    # V22 = -A*e**(-B*x**2)+E0
    # V12 = V21 = C*e**(-D*x**2)
    

    # 模型三
    # A = 6e-4
    # B = 0.1
    # C = 0.9
    # V11 = A
    # V22 = -A
    # if x < 0:
    #     V12 = B*e**(C*x)
    # else:
    #     V12 = B*(2-e**(-C*x))
    # V21 = V12

    return np.array([[V11,V12],[V21,V22]])

    
def stable_eigh_first(mat):
    eig_vals, U = np.linalg.eigh(mat)
    n = U.shape[1]
    tol = 0
    for i in range(n):
        vec = U[:, i]
        # 找到第一个有效非零分量作为相位基准
        idx = np.where(np.abs(vec) > tol)[0][0]
        coeff = vec[idx]
        # 基准为负，整列向量取反，统一符号
        if np.real(coeff) < 0:
            U[:, i] *= -1
    return eig_vals, U

# 计算变换矩阵
def U(H):
    U = stable_eigh_first(H)[1]
    return U

# 计算绝热哈密顿
def H_ad(H):
    return np.linalg.eigh(H)[0]

# 计算透热哈密顿对于x导数
def dH_di_dx(x,dx=1e-7):
    H_p = H_di(x-dx)
    H_m = H_di(x+dx)
    return (H_m - H_p)/(2*dx)

# 根据公式计算d12，就是NAC非绝热耦合常数
# (这里按照我的矩阵变换使用eigh做的，得到矩阵E1小于E2，从小到大排列，d12做出来是个复数),算两遍就算两遍吧
def d12(x):
    H_dia = H_di(x)
    U_ = U(H_dia)
    E1,E2 = H_ad(H_dia)
    dH_di_dx_ = dH_di_dx(x)
    ans = (U_.conj().T @ dH_di_dx_ @ U_)[0,1] / (E2-E1)
    return ans

# 根据当前位置x，所在面C，计算核的受力F，及计算绝热哈密顿矢量对于x导数的相反数         豆师傅让我改成1e-7
def F(x,C,dx=1e-7):
    H_p = H_ad(H_di(x-dx))
    H_m = H_ad(H_di(x+dx))
    return (-(H_m - H_p)/(2*dx))[1-C]

# 根据当前动量k，坐标x，质量m，受力F，计算dilta_t时间后原子位置
def x_change_by_t(k,m,x,F,del_t):
    x_new = x + k*del_t/m + 0.5*(F/m)*del_t**2  #新坐标
    return x_new

# 根据当前动量k,受力F，计算dilta_t时间后的动量
def k_change_by_t(k,F,del_t):
    k_new = k+F*del_t
    return k_new

# 根据动量，质量，面归属，绝热哈密顿量给出粒子总能能量
def E_total(k,m,C,H_ad):
    E = H_ad[1-C]
    return k**2/(2*m) + E


# 计算当前有效哈密顿量H_eff
def H_eff(H_ad,k,m,d12):
    v = k/m
    E1,E2 = H_ad
    H11 = E1
    H22 = E2
    H12 = -v*d12*(1j*hbar)
    H21 = v*d12*(1j*hbar)
    return np.array([[H11,H12],[H21,H22]])

# 根据当有效哈密顿，和之前的振幅向量，计算dilta_t时间后的振幅向量 (这里我用的是自己推到的方法)
def C_change_by_t(H_eff,C_t,del_t):
    ans = expm (-1j/hbar*H_eff*del_t) @ C_t
    return ans

# 论文中使用的RKG方法
#RKG方法中的ODE方程，和要带入的每层有效哈密顿（与t相关）和要带入的每层振幅向量有关，k,m,x,F是当前点常数
def f_t_c(k,m,x,F,
          f_del_t,c):
    x_f = x_change_by_t(k,m,x,F,f_del_t)
    H_di_f = H_di(x_f)
    H_ad_f = H_ad(H_di_f)
    d12_f = d12(x_f)
    k_f = k_change_by_t(k,F,f_del_t)
    H_eff_f = H_eff(H_ad_f,k_f,m,d12_f)
    ans = (-1j/hbar)*H_eff_f @ c
    return ans
# 根据给出的RKG公式计算dilta_t后的振幅向量
def C_change_by_t_RKG(k,m,x,F,H_eff_now,C_t,del_t):
    k1 = del_t*(-1j/hbar)*H_eff_now @ C_t
    # print(f'k1{k1}')
    k2 = del_t*f_t_c(k,m,x,F,del_t/2,C_t+k1/2)
    # print(f'k2{k2}')
    k3 = del_t*f_t_c(k,m,x,F,del_t/2,C_t+k1/2+(2**0.5-1)*k2)
    # print(f'k3{k3}')
    k4 = del_t*f_t_c(k,m,x,F,del_t,C_t+(1-2**0.5/2)*k2+(2**0.5/2)*k3)
    # print(f'k4{k4}')
    ans = C_t+1/6*(k1+(2-2**0.5)*k2+(2+2**0.5)*k3+k4)
    # print(f'ans{ans}')
    return ans

    

# 根据振幅向量，坐标，计算跃迁概率g12，g21
def g12(C,x,k,m,del_t):
    d21_ = -d12(x).conjugate()
    c1,c2 = C
    v = k/m
    g12_ = del_t*(-2*((c1*c2.conjugate()*v*d21_).real))  /  (abs(c1)**2)    
    return g12_

def g21(C,x,k,m,del_t):
    d12_ = d12(x)
    c1,c2 = C
    v = k/m
    g21_ = del_t*(-2*((c2*c1.conjugate()*v*d12_).real))  /  (abs(c2)**2)
    return g21_

# 根据当前动量，质量，绝热哈密顿向量，现在面归属，计算跃迁后动量，如果是负值返回-1
def k_aft(k,m,H_ad,C):
    E1,E2 = H_ad
    if C == 0:  #刚从低能面跳到高能面
        E_new = (k**2/(2*m)) - (E2-E1)
        if E_new >= 0:
            k_new = (2*E_new*m)**0.5  *  np.sign(k)
        else:
            k_new = -10086
    else:
        E_new = (k**2/(2*m)) + (E2-E1)
        if E_new >= 0:
            k_new = (2*E_new*m)**0.5  *  np.sign(k)
        else:
            k_new = -10086
    return k_new
















# (这里按照我的矩阵变换使用eigh做的，得到矩阵E1小于E2，从小到大排列，d12做出来是个复数)
# 绝热哈密顿量矩阵元V，非绝热耦合常数随坐标x图像
# x = np.linspace(-10,10,500)
# y1 = [np.linalg.eigh(H_di(i))[0][0] for i in x]
# y2 = [np.linalg.eigh(H_di(i))[0][1] for i in x]
# y3 = [d12(i) for i in x]

# plt.plot(x,y1,label='V1')
# plt.plot(x,y2,label='V2')
# plt.plot(x,y3,label='d12')

# plt.legend()
# plt.grid()
# plt.show()


# # 我发现eigh函数演我
# x = np.linspace(-10,10,500)
# # y1 = [dH_di_dx(i)[0][0] for i in x]
# # y2 = [dH_di_dx(i)[0][1] for i in x]
# # y3 = [dH_di_dx(i)[1][1] for i in x]
# y4 = [U(H_di(i))[0][0] for i in x]
# y5 = [U(H_di(i))[0][1] for i in x]
# y6 = [U(H_di(i))[1][0] for i in x]
# y7 = [U(H_di(i))[1][1] for i in x]
# # plt.plot(x,y1,label='V11')
# # plt.plot(x,y2,label='V12')
# # plt.plot(x,y3,label='V22')
# # plt.plot(x,y4,label='U(i)[0][0]')
# plt.plot(x,y5,label='U(i)[0][1]')
# # plt.plot(x,y6,label='U(i)[1][0]')
# plt.plot(x,y7,label='U(i)[1][1]')

# plt.legend()
# plt.grid()
# plt.show()












# 输出测试
# H_dia = H_di(3)
# print(H_dia)
# U = U(H_dia)
# print(U)
# H_adi_1,H_adi_2 = H_ad(H_dia)
# print(H_adi_1,H_adi_2)


# 透热哈密顿量矩阵元V随坐标x图像
# x = np.linspace(-10,10,500)
# y1 = [H_di(i)[0,0] for i in x]
# y2 = [H_di(i)[0,1] for i in x]
# y3 = [H_di(i)[1,0] for i in x]
# y4 = [H_di(i)[1,1] for i in x]
# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,y4)
# plt.grid()
# plt.show()


# 绝热哈密顿量矩阵元V随坐标x图像
# x = np.linspace(-10,10,500)
# y1 = [np.linalg.eigh(H_di(i))[0][0] for i in x]
# y2 = [np.linalg.eigh(H_di(i))[0][1] for i in x]
# plt.plot(x,y1,label='V00')
# plt.plot(x,y2,label='V11')
# plt.legend()
# plt.grid()
# plt.show()



