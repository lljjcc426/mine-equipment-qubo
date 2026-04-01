import kaiwu as kw

if __name__ == '__main__':
    kw.license.init(user_id="39882039024955394", sdk_code="PWqPf0lIaM65ZdD6WMurCk7K0C5wId")
import numpy as np

# 输入的图矩阵取反
matrix = -np.array([
                [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 1, 1, 0 ,1, 0],
                [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]])
output = kw.cim.simulator(
            matrix,
            pump = 0.7,
            noise = 0.01,
            laps = 50,
            dt = 0.1,
            normalization = 0.3,
            iterations = 10)
opt = kw.sampler.optimal_sampler(matrix, output, 0)
best = opt[0][0]
max_cut = (np.sum(-matrix)-np.dot(-matrix,best).dot(best))/4
print("The obtained max cut is "+str(max_cut)+".")
