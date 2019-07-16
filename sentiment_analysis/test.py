#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : test.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/3/18 10:18
import numpy as np
import pandas as pd




def process_tags(mat):
    r = mat.shape[0]
    c = mat.shape[1]
    ret = np.zeros((r, c * 4))
    for i in range(r):
        for j in range(c):
            ret[i, 4 * j + 1 - mat[i, j]] = 1
    return ret


def restore_tags(mat):
    r = mat.shape[0]
    c = mat.shape[1] // 4
    ret = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            lt = list(mat[i, 4 * j:4 * j + 4])
            maxind = lt.index(max(lt))
            ret[i, j] = 1 - maxind
    return ret


y = [[-2,0,1,-1]]
y = np.array(y)
print(y)

y_ = process_tags(y)
print(y_)
print(len(y_[0]))










