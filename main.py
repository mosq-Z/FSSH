import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
from fssh_class import FSSH
import numpy as np


# # 测试单次步骤2  这里的语法后来被我改过了，必须的输入只有一个动量
# # #def __init__(self,x_Localtion,k_Momentum,C_initial):新建一个对象
# FSSH1 = FSSH(-3,4.5,np.array([1,0]))
# # FSSH1.print_all_params_full()
# FSSH1.step_II()
# ans = abs(FSSH1.C_t[0])**2+abs(FSSH1.C_t[1])**2
# print("{0:.50f}".format(ans))
# # FSSH1.print_all_params_full()

# FSSH1 = FSSH(-3,30,np.array([1,0]))
# FSSH1.Start_simulation()


# # 批量化测试
# FSSH.Set_Number_of_simulations(100)
# FSSH.Create_simulations_from_k1_to_k2(0,35,1)
# FSSH.Draw_Simulation_Results_Plot()
# print(FSSH.Simulation_Results_Plot)



# FSSH1 = FSSH(30)
# FSSH1.start_simulation()

# if __name__ == "__main__":

#     FSSH.Set_Number_of_simulations(20)
#     FSSH.Create_simulations_from_k1_to_k2(0,35,1)
#     FSSH.Draw_Simulation_Results_Plot()
#     print(FSSH.Simulation_Results_Plot)



# # 测试输入文件
if __name__ == "__main__":

    FSSH.Set_Number_of_simulations(1)
    FSSH.Create_simulations_from_input()
    FSSH.Draw_Simulation_Results_Plot()
    print(FSSH.Simulation_Results_Plot)

# if __name__ == "__main__":

#     FSSH.Set_Number_of_simulations(1)
#     FSSH.Create_simulations_from_k1_to_k2(8.4,8.4,1)
#     # FSSH.Draw_Simulation_Results_Plot()
#     print(FSSH.Simulation_Results_Plot)
