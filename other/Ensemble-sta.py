import math 
import numpy as np

# k_first = 5.875
# k_10 = 19.831
# dilia_k = (k_10-k_first)/10

sigma = 1.4


# 根据3sigema原则找出第一要系综统计k的下标
def find_k_id(lis_k):
    k1 = lis_k[1]
    k2 = lis_k[2]
    gap = k2-k1
    ans = int(3*sigma/gap)-1
    return gap,ans

# 根据高斯分布和3sigema生成对应的权重列表
c_list = []
def gen_c_list(gap,k_id):
    for i in range(k_id*2+1):
        c = math.exp(-(abs(k_id-i)*gap)**2/(2*sigma**2))
        c_list.append(c)


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

# 初始化
k_lis_0 = res[:,0]
gap,k_id  = find_k_id(k_lis_0)
gen_c_list(gap,k_id)
sum_c = sum(c_list)
len_lis = len(k_lis_0)
#归一化
c_list = [i/sum_c for i in c_list]

row_list_ES = []
row_list_ES.append(k_lis_0.tolist()[k_id:len_lis-k_id])

for i in range(4):
    ans_lis = res[:,i+1]
    ans_ES_lis = []
    for j in range(k_id,len_lis-k_id):
        ans_ES = 0
        for ic in range(k_id*2+1):
            ans_ES += c_list[ic]*ans_lis[j-k_id+ic]
        ans_ES_lis.append(ans_ES)
    row_list_ES.append(ans_ES_lis)
res_ES = np.array(row_list_ES).T
print(res_ES)

filename2="sat_out"

with open("sat_out", "w", encoding="utf-8") as f:
    h,l = res_ES.shape
    for i in range(h):
        f.write( f"{res_ES[i,0]}\t{res_ES[i,1]}\t{res_ES[i,2]}\t{res_ES[i,3]}\t{res_ES[i,4]}\n")
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
