import math
import random
def count_partial_orders(n):
dp = [[O for _ in rango(n+ 1)] for _ in range(n+ 1)]
# 自反性和空集的分for i in range(n+1):
dp(i][]=1
dpli][i]=1
for i in range(2, n+ 1):
for i in range(1, i):
for k in range(j+1):
dplili] += dpli - jllk]* dpili-k]
#返回n个元素的偏
return sum(dp[ng]l:-1])
n = int(input（请输入一个整数n:))
print(F”偏序关系个数 C((n))为：(countL partialL_orders(n)”)