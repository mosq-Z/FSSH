import numpy as np
from math import *
import matplotlib.pyplot as plt
from math1 import H_di,U,H_ad,F,x_change_by_t,k_change_by_t,d12,H_eff,C_change_by_t,g12,g21,k_aft,C_change_by_t_RKG,E_total
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

class FSSH():
    
    # 类变量 最大模拟步数，最大距离，单个动量上生成模拟次数，最终模拟结果
    Step_Max = 400000
    X_Max = 5
    Number_of_simulations = 3000
    M_Quality = 2000
    Dleta_T = 1e-1

    Simulation_Results_Plot = np.empty((0, 6))

    # 根据初始位置，初始动量，质量，初始绝热太分布,最大步数   新建演化过程
    def __init__(self,k_Momentum,
                 x_Localtion = -X_Max,
                 C_initial = np.array([1,0]),
                 step_max = Step_Max,
                 x_max = X_Max,
                 m_Quality = M_Quality,delta_t = Dleta_T):

        # 质量，帧长
        self.m_Quality = m_Quality
        self.delta_t = delta_t
        


        # 步骤III更新参数
        # 面归属(代表是否在低能态上)，
        self.C_evolution_t = self.C_evolution_t0 = int(C_initial[0])
        
        # 步骤II更新参数
        # 初始化绝热态分布，
        # 位置，
        # 动量，
        self.x_Localtion_t = x_Localtion
        self.C_t = C_initial
        self.k_Momentum_t = k_Momentum

        # 仅依赖核坐标x参数
        # 透热哈密顿矩阵，
        # 变换矩阵，
        # 绝热哈密顿向量，
        # 绝热耦合常数NAC
        self.H_di_t = self.update_H_di(x_Localtion)
        self.U_t0 = self.U_t = self.update_U(self.H_di_t)
        self.H_ad_t = self.update_H_ad(self.H_di_t)
        self.d12_t = self.update_d12_t(self.x_Localtion_t)
        
        # 依赖核坐标x，动量k
        # 有效哈密顿量
        self.H_eff_t = self.update_H_eff_t(self.H_ad_t,self.k_Momentum_t,self.m_Quality,self.d12_t)

        # 依赖核坐标x，面归属C
        # 核受力情况,
        self.F_t = self.update_F(self.x_Localtion_t,self.C_evolution_t0)
        
        
        self.g12_t = 0
        self.g21_t = 0
        self.g_total = 0

        # 当前步数,最大步数,最大位移
        self.step = 0
        self.step_max = step_max
        self.x_max = x_max
        
        # 是否发生跃迁
        self.if_shift = False
        
        #当前体系总能量（动能+势能）/上一时刻体系总能量
        self.E_total  = self.update_E_total(self.k_Momentum_t,self.m_Quality,self.C_evolution_t,self.H_ad_t)

    ####################################### 更新参数函数 #######################################

    #更新透热哈密顿量
    def update_H_di(self,x):
        return H_di(x)
    
    #更新变换矩阵
    def update_U(self,H):
        return U(H)
    
    #更新绝热哈密顿量
    def update_H_ad(self,H):
        return H_ad(H)
        
    #更新核受力情况
    def update_F(self,x,C):
        return F(x,C)

    #更新核坐标
    def update_x_Localtion_t(self,k,m,x,F,del_t):
        return x_change_by_t(k,m,x,F,del_t)
    
    #更新动量
    def update_k_Momentum_t(self,k,F,del_t):
        return k_change_by_t(k,F,del_t)
    
    #更新绝热耦合常数d12
    def update_d12_t(self,x):
        return d12(x)
    
    #更新有效哈密顿矩阵
    def update_H_eff_t(self,H_ad,k,m,d12):
        return H_eff(H_ad,k,m,d12)

    #更新绝热态分布k,m,x,F,H_eff_now,C_t,del_t
    def update_C_t(self,k,m,x,F,H_eff,C_t,del_t):
        return C_change_by_t_RKG(k,m,x,F,H_eff,C_t,del_t)
    

    #更新跃迁概率
    def update_g12_t(self,C,x,k,m,del_t):
        return g12(C,x,k,m,del_t)
    def update_g21_t(self,C,x,k,m,del_t):
        return g21(C,x,k,m,del_t)

    #跃迁后削减后动量
    def k_after(self,k,m,H_ad,C):
        return k_aft(k,m,H_ad,C)
    
    #更新体系总能量
    def update_E_total(self,k,m,C,H_ad):
        return E_total(k,m,C,H_ad)

    ####################################### 更新参数函数 #######################################




    ####################################### 步进操作函数 #######################################

    #步骤二：原子按照经典理论运动，轨迹积分得到绝热态分布
    def step_II(self):


        #具体思路是先更新核坐标(k,m,x,F,del_t),更新绝热态分布(k,m,x,F,H_eff_now,C_t,del_t),然后参数覆盖的顺序还得想一下:x,C,k然后更新仅和这三位依赖的参数，不包含面归属，受力，总能量
        self.x_Localtion_t = self.update_x_Localtion_t(self.k_Momentum_t,
                                                       self.m_Quality,
                                                       self.x_Localtion_t,
                                                       self.F_t,
                                                       self.delta_t)
        
        self.k_Momentum_t = self.update_k_Momentum_t(self.k_Momentum_t,
                                                     self.F_t,
                                                     self.delta_t)




        self.H_di_t = self.update_H_di(self.x_Localtion_t)
        self.U_t  = self.update_U(self.H_di_t)
        self.H_ad_t = self.update_H_ad(self.H_di_t)
        self.d12_t = self.update_d12_t(self.x_Localtion_t)

        self.H_eff_t = self.update_H_eff_t(self.H_ad_t,self.k_Momentum_t,self.m_Quality,self.d12_t)


        f_F_t = self.update_F(self.x_Localtion_t,self.C_evolution_t)  #这里要先用一下这个点的核受力，因为下面计算c要用，并不更新类变量影响下面的计算
        # 根据这个点新的参数，更新绝热态分布
        self.C_t = self.update_C_t(self.k_Momentum_t,
                                   self.m_Quality,
                                   self.x_Localtion_t,
                                   f_F_t,
                                   self.H_eff_t,
                                   self.C_t,
                                   self.delta_t)
        
        # self.print_all_params_full()

        # self.C_evolution_t =    面归属和受力，目前没必要更新
        # self.F_t = 
        
        self.E_total = self.update_E_total(self.k_Momentum_t,self.m_Quality,self.C_evolution_t,self.H_ad_t)
        


    #步骤三：计算切换概率g，生成随机数，并判断是否切换面
    def step_III(self):
        if self.C_evolution_t == 1:
            #update_g12_t(self,C,x,p,m,del_t):

            self.g_total = self.g_total + self.g12_t
            self.g12_t = self.update_g12_t(self.C_t,self.x_Localtion_t,self.k_Momentum_t,self.m_Quality,self.delta_t)

            # #测试
            # with open("test_out", "a", encoding="utf-8") as f:
            #     f.write(f"self.g12_t = {self.g12_t}\n")
            #     f.write(f"self.g_total = {self.g_total}\n")
            # print(self.g12_t)

            xi = np.random.rand()
            if self.g12_t > xi :#and  not self.if_shift:    #只切换一次实验？
                self.C_evolution_t = 0
            else:
                pass

        else :
            self.g21_t = self.update_g21_t(self.C_t,self.x_Localtion_t,self.k_Momentum_t,self.m_Quality,self.delta_t)
            xi = np.random.rand()
            if self.g21_t > xi :#and  not self.if_shift:
                self.C_evolution_t = 1
            else:
                pass
        



    #步骤四：计算切换概率g，生成随机数，并判断是否切换面
    def step_IV(self):
        if self.C_evolution_t == self.C_evolution_t0:
            pass

        else:
            # 计算需要削减的动量
            k_af = self.k_after(self.k_Momentum_t,self.m_Quality,self.H_ad_t,self.C_evolution_t)
            if k_af == -10086:
                #撤回态切换
                self.C_evolution_t = self.C_evolution_t0
            else:
                #态切换成功，不撤回，并更新
                self.k_Momentum_t = k_af
                self.if_shift = True
                

        # 这里要更新依赖面归属的核受力
        self.F_t = self.update_F(self.x_Localtion_t,self.C_evolution_t)
        # 覆盖上一时刻的面归属
        self.C_evolution_t0 = self.C_evolution_t
            
        self.E_total = self.update_E_total(self.k_Momentum_t,self.m_Quality,self.C_evolution_t,self.H_ad_t)

    # 打包三个步骤，并循环至最高次数
    def start_simulation(self):

        with open("test_out", "a", encoding="utf-8") as f:
        #     f.write(f"第{self.step}次步进结果\n")
        #     f.write(f"k = {self.k_Momentum_t}  E = {self.H_ad_t[1-self.C_evolution_t]}\n")
        #     f.write(f'第{self.step}步结束时体系总能量为{self.E_total}\n')
            # f.write(f'{self.x_Localtion_t}  {self.k_Momentum_t}\n')
            f.write( f"{self.step}\t{self.C_t[0].real}\t{self.C_t[0].imag}\t{self.C_t[1].real}\t{self.C_t[1].imag}\t{self.x_Localtion_t}\t{self.C_evolution_t}\t{self.F_t}\t{self.k_Momentum_t}\t{self.g12_t}\n")
    

        # print(f'第{self.step}次步进结果')
        # print(f'self.C_evolution_t = {self.C_evolution_t}  self.x_Localtion_t = {self.x_Localtion_t}')
        # self.print_all_params_full()

        # print(f'第{self.step}步结束时体系总能量为{self.E_total}')


        while True:
            self.step += 1
            self.step_II()
            self.step_III()
            self.step_IV()
            with open("test_out", "a", encoding="utf-8") as f:
                f.write( f"{self.step}\t{self.C_t[0].real}\t{self.C_t[0].imag}\t{self.C_t[1].real}\t{self.C_t[1].imag}\t{self.x_Localtion_t}\t{self.C_evolution_t}\t{self.F_t}\t{self.k_Momentum_t}\t{self.g12_t}\n")
            # with open("test_out", "a", encoding="utf-8") as f:
            #     f.write(f"第{self.step}次步进结果\n")
            #     f.write(f"k = {self.k_Momentum_t}  E = {self.H_ad_t[1-self.C_evolution_t]}  x = {self.x_Localtion_t}\n")
            #     f.write(f'第{self.step}步结束时体系总能量为{self.E_total}\n')
                # f.write(f'{self.x_Localtion_t}  {self.k_Momentum_t}\n')
            # print(f'第{self.step}次步进结果')
            # print(f'self.C_evolution_t = {self.C_evolution_t}  self.x_Localtion_t = {self.x_Localtion_t}')
            # self.print_all_params_full()

            # print(f'第{self.step}步结束时体系总能量为{self.E_total}')

            if self.if_stop_simulation():
                break
    
    ####################################### 步进操作函数 #######################################


    ####################################### 结果判定函数 #######################################
    # 判断该条件下的粒子是否停止模拟
    def if_stop_simulation(self):
        ans = False
        if self.x_Localtion_t > self.x_max:
            ans = True
        if self.x_Localtion_t < - self.x_max:
            ans = True
        if self.step >= self.step_max:
            ans = True
        return ans

    # 根据当前条件给出模拟的结果判定
    def simulation_result(self):
        ans = np.array([10086,10086])
        # 第二位 1代表透射，0代反射
        if self.x_Localtion_t > 0:
            ans[1] = 1
        else:
            ans[1] = 0
        # 第一位 0代表低能态，1代表高能态
        if self.C_evolution_t == 1:
            ans[0] = 0
        else:
            ans[0] = 1
        return ans

    

    ####################################### 结果判定函数 #######################################


    ####################################### 批量操作函数 #######################################
    
    # 在一个动量上生成一定数量的模拟，并将统计结果打包成np向量，放入最终结果列表中
    # @classmethod
    # def Create_simulations_k(cls,k):
    #     result_k = np.array([[k,0.0,0.0,0.0,0.0,0.0]])  #这四个位置分别代表低态透射，低态反射，高态透射，高台衍射,其他
    #     for i in range(cls.Number_of_simulations):
    #         simu = cls(k)
    #         simu.start_simulation()
    #         result = simu.simulation_result()
    #         if result[0] == 0 and result[1] == 1:
    #             result_k[0,1] += 1
    #         elif result[0] == 0 and result[1] == 0:
    #             result_k[0,2] += 1
    #         elif result[0] == 1 and result[1] == 1:
    #             result_k[0,3] += 1
    #         elif result[0] == 1 and result[1] == 0:
    #             result_k[0,4] += 1
    #         else:
    #             result_k[0,5] += 1
    #         # print(f'以完成在动量{k}处的第{i+1}次模拟')
    #         # print(result_k)


    #     # 换算成概率
    #     for i in range(1,6):
    #         result_k[0, i] = result_k[0, i] / cls.Number_of_simulations
    #         # print(result_k)
        
    #     cls.Simulation_Results_Plot = np.vstack([cls.Simulation_Results_Plot, result_k])


    #这是我让豆大人帮我重写的并行代码，可以拉满CPU的使用
    @staticmethod
    def _run_single_trajectory(args):
        # 自动解包 (k, cl
        k, cls = args
        simu = cls(k)
        simu.start_simulation()
        return simu.simulation_result()

    @classmethod
    def Create_simulations_k(cls, k):
        n_sim = cls.Number_of_simulations

        # 构造参数列表：每个元素是 (k, cls)
        args_list = [(k, cls) for _ in range(n_sim)]

        # 多进程并行（100% Windows 兼容）
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(cls._run_single_trajectory, args_list))  

        # ======================
        # 下面是你原来的统计代码
        # ======================
        result_k = np.array([[k, 0.0, 0.0, 0.0, 0.0, 0.0]])

        for res in results:
            if res[0] == 0 and res[1] == 1:
                result_k[0, 1] += 1
            elif res[0] == 0 and res[1] == 0:
                result_k[0, 2] += 1
            elif res[0] == 1 and res[1] == 1:
                result_k[0, 3] += 1
            elif res[0] == 1 and res[1] == 0:
                result_k[0, 4] += 1
            else:
                result_k[0, 5] += 1

        # 转概率
        for i in range(1, 6):
            result_k[0, i] /= n_sim

        cls.Simulation_Results_Plot = np.vstack([cls.Simulation_Results_Plot, result_k])


    # 在不同动量上执行上一个函数，形成结果矩阵
    @classmethod
    def Create_simulations_from_k1_to_k2(cls,k1,k2,gap):

        print(f'开始执行动量{k1}到{k2}处模拟，间隔为{gap}')
        k = k1
        while True:
            cls.Create_simulations_k(k)
            print(f'以完成在动量{k}处的{cls.Number_of_simulations}全部模拟')
            print(cls.Simulation_Results_Plot)
            k+= gap
            if k>k2:
                break
        
    # 根据输入文件test_input在列表中的数值执行上一个函数，形成结果矩阵
    @classmethod
    def Create_simulations_from_input(cls):
        filename="test_input"
        print(f'开始执行根据动量列表test_out输入的模拟')
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                # 去掉空行、空格、换行
                line = line.strip()
                if not line or line[0] == '#': #设置一些注释
                    continue
                else:
                    k = float(line)
                    cls.Create_simulations_k(k)
                    print(f'以完成在动量{k}处的全部{cls.Number_of_simulations}模拟')
                    print(cls.Simulation_Results_Plot)


    

    ####################################### 批量操作函数 #######################################



    ####################################### 变量调节函数 #######################################

    # 可以调节类变量 最大模拟步数，最大距离，单个动量上生成模拟次数

    @classmethod
    def Set_Step_Max(cls,x):
        cls.Step_Max = int(x)

    @classmethod
    def Set_X_Max(cls,x):
        cls.X_Max = float(x)

    @classmethod
    def Set_Number_of_simulations(cls,x):
        cls.Number_of_simulations = int(x)
    
    ####################################### 变量调节函数 #######################################



    ####################################### 状态打印函数 #######################################
    def print_all_params_full(self):  #(豆)
        print("=" * 90)
        print("完整参数表")
        print("=" * 90)

        # 你的所有参数（按你截图里的顺序）
        params = [
            ("C_evolution_t", self.C_evolution_t),
            ("C_t", self.C_t),
            ("x_Localtion_t", self.x_Localtion_t),
            ("k_Momentum_t", self.k_Momentum_t),
            ("H_di_t", self.H_di_t),
            ("U_t", self.U_t),
            ("H_ad_t", self.H_ad_t),
            ("F_t", self.F_t),
            ("d12_t", self.d12_t),
            ("H_eff_t", self.H_eff_t),
        ]

        # 逐个打印，保证完整输出
        for name, value in params:
            print(f"🔹 {name}:")
            
            # 完整输出 numpy 数组（矩阵/向量）
            if isinstance(value, np.ndarray):
                # 关闭科学计数法/省略，完整打印每一个元素
                with np.printoptions(precision=6, suppress=False, threshold=np.inf, linewidth=200):
                    print(value)
            
            # 完整输出复数
            elif isinstance(value, complex):
                print(f"{value.real:.50f} + {value.imag:.50f}j")
            
            # 标量（数字）
            else:
                print(f"{value:.50f}")
            
            print("-" * 90)

        print("=" * 90)

    # 对矩阵绘图
    @classmethod
    def Draw_Simulation_Results_Plot(cls):

        x = cls.Simulation_Results_Plot[:, 0]  
        y1 = cls.Simulation_Results_Plot[:, 1]  
        y2 = cls.Simulation_Results_Plot[:, 2]  
        y3 = cls.Simulation_Results_Plot[:, 3]  
        y4 = cls.Simulation_Results_Plot[:, 4]  

        plt.plot(x,y1,label='T_L')
        plt.plot(x,y2,label='R_L')
        plt.plot(x,y3,label='T_H')
        plt.plot(x,y4,label='R_H')

        plt.legend()
        plt.grid()
        plt.show()

    ####################################### 状态打印函数 #######################################

    



    
    


    
































