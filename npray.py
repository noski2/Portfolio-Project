import numpy as np

arr = np.random.uniform(size = 1500)

sum = 0
for x in arr:
    sum = sum + x

print(sum)

print(arr.sum())