import math 
import numpy as np

# k_first = 5.875
# k_10 = 19.831
# dilia_k = (k_10-k_first)/10

# 根据当前动量算出对应波包的动量展宽sigma   
def sigma_by_k(k):
    return k/20

# 根据3sigsma原则和提供的k列表找出动量k的两侧要纳入运算的k个数，返回动量列表间隔和个数
def find_k_id(lis_k,sigma):
    k1 = lis_k[1]
    k2 = lis_k[2]
    gap = k2-k1
    k_id = int(3*sigma/gap)-1
    return gap,k_id

# 根据高斯分布和3sigema生成对应的权重列表
def gen_c_list(gap,k_id,sigma):
    c_list = []
    for i in range(k_id*2+1):
        c = math.exp(-(abs(k_id-i)*gap)**2/(2*sigma**2))
        c_list.append(c)
        sum_c = sum(c_list)
        c_list = [i/sum_c for i in c_list]
    return c_list

# 判断该动量值左右有没有充足空间进行展宽
def if_can_ES(k_id_now,k_id,len_lis):
    flag = False
    if k_id_now - k_id >= 0 and k_id_now + k_id < len_lis:
        flag = True
    return flag

# 读取读取数据并封装
filename1="sat_input"
row_list = []
with open(filename1, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = [float(num) for num in line.split("\t")]
        row_list.append(row)
res = np.array(row_list)

# 提取一下参数
k_lis_0 = res[:,0]
len_lis = len(k_lis_0)

# 便利一遍k看看从哪个下标到哪个下标可以求系综,顺便生成对应权重列表组成列表
first_flag = 0
last_flag = 0
c_list_list = []

for i in range(len_lis):
    k_i = k_lis_0[i]
    sigma_k = sigma_by_k(k_i)
    gap_i,k_id_i = find_k_id(k_lis_0,sigma_k)
    
    if if_can_ES(i,k_id_i,len_lis):
        c_list = gen_c_list(gap_i,k_id_i,sigma_k)
        c_list_list.append([k_id_i,c_list])  #顺便把他的半径也传一下
        if first_flag == 0:
            first_flag = i
            # 记录第一个可以ES的下标
    else:
        if first_flag != 0 and last_flag == 0:
            last_flag = i
            # 记录末尾第一个不可以ES的下标


# 放入动量行
row_list_ES = []
row_list_ES.append(k_lis_0.tolist()[first_flag:last_flag])

for i in range(4):
    ans_lis = res[:,i+1]
    ans_ES_lis = []
    for j in range(first_flag,last_flag):
        ans_ES = 0
        k_id_j,c_list_j = c_list_list[j-first_flag][0],c_list_list[j-first_flag][1]
        for ic in range(k_id_j*2+1):
            ans_ES += c_list_j[ic]*ans_lis[j-k_id_j+ic]
        ans_ES_lis.append(ans_ES)
    row_list_ES.append(ans_ES_lis)
res_ES = np.array(row_list_ES).T
print(res_ES)

filename2="sat_out"

with open("sat_out", "w", encoding="utf-8") as f:
    h,l = res_ES.shape
    for i in range(h):
        f.write( f"{res_ES[i,0]} {res_ES[i,1]} {res_ES[i,2]} {res_ES[i,3]} {res_ES[i,4]}\n")
    print(f"结果已经写入{filename2}")









    
        
        
    
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
