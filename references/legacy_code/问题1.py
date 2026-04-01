from pulp import *

# 定义问题
prob = LpProblem("Maximize y", LpMaximize)

# 定义变量
x1 = LpVariable("x1", 0, None, LpInteger)
x2 = LpVariable("x2", 0, None, LpInteger)
x3 = LpVariable("x3", 0, None, LpInteger)
x4 = LpVariable("x4", 0, None, LpInteger)
y1 = LpVariable("y1", 0, 1, LpInteger)
y2 = LpVariable("y2", 0, 1, LpInteger)
y3 = LpVariable("y3", 0, 1, LpInteger)
y4 = LpVariable("y4", 0, 1, LpInteger)

# 定义目标函数
prob += 2000*x1 + 3000*x2 + 5000*x3 + 6000*x4

# 定义约束条件
prob += 100*x1 + 140*x2 + 200*x3 + 320*x4 <= 2400
prob += x1 >= y1
prob += x2 >= y2
prob += x3 >= y3
prob += x4 >= y4
prob += y1 + y2 + y3 + y4 >= 3

# 求解
prob.solve()

# 输出结果
print("Status:", LpStatus[prob.status])
print("Optimal Solution to the problem: ", value(prob.objective))
print("Individual decision_variables: ")
for v in prob.variables():
    print(v.name, "_", v.varValue)