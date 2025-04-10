# coding=utf-8
import numpy as np
import sys
sys.path.append('D:\\AI\\mycuda\\cudaWrap\\Debug\\')
import pefile

pyd_file = r'D:\AI\mycuda\cudaWrap\Debug\addCuda.pyd'
pe = pefile.PE(pyd_file)

for entry in pe.DIRECTORY_ENTRY_IMPORT:
    print(entry.dll.decode('utf-8'))

# from ctypes import CDLL
# dll = CDLL(r'D:\AI\mycuda\cudaWrap\Debug\addCuda.pyd')
# print(dll.PyInit_addCuda())

import addCuda

# 初始化数组
size = 10
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
b = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
c = np.zeros(size, dtype=np.int32)

# 调用 CUDA
addCuda.addWithCuda(c, a, b, size)

print("Result:", c)
