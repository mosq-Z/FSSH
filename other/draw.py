import numpy as np
import matplotlib.pyplot as plt
import re
import sys

print("请粘贴你的完整矩阵数据，粘贴完成后：")
print("Windows：Ctrl+Z 然后回车结束输入")
print("Mac/Linux：Ctrl+D 结束输入")
input_text = sys.stdin.read()

# 清洗：去除括号、换行、制表符
clean_str = re.sub(r'[\[\]\n\t]', ' ', input_text)
# 正则同时匹配：整数、正负小数、科学计数格式
num_list = re.findall(r'[-+]?\d+\.?\d*(?:e[+-]?\d+)?', clean_str)
nums = np.array(num_list, dtype=float)

# 每行6列还原矩阵
n_col = 6
n_row = len(nums) // n_col
arr = nums.reshape(n_row, n_col)

# 按第一列升序排序
Simulation_Results_Plot = arr[np.argsort(arr[:, 0])]

# 第二列加上第五列
Simulation_Results_Plot[:, 2] += Simulation_Results_Plot[:, 4]

print("===== 按首列排序后矩阵 =====")
for row in Simulation_Results_Plot:
    print(' '.join(map(str, row)))

# 绘图
x = Simulation_Results_Plot[:, 0]
y1 = Simulation_Results_Plot[:, 1]
y2 = Simulation_Results_Plot[:, 2]
y3 = Simulation_Results_Plot[:, 3]

plt.figure(figsize=(10, 6))
plt.scatter(x, y1, label='T_L')
# 如需同时画其他曲线取消下面注释即可
# plt.scatter(x, y2, label='R_L')
# plt.scatter(x, y3, label='T_H')

plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlabel('X')
plt.ylabel('Value')
plt.tight_layout()
plt.show()