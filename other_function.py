import numpy as np
import math

def equal(galLabels, probLabels):
    binaryLabels = []
    for i in range(len(galLabels)):
        for j in range(len(probLabels)):
            if galLabels[i] == probLabels[j]:
                binaryLabels.append(1)
            else:
                binaryLabels.append(0)
    return np.asarray(binaryLabels).reshape((len(galLabels), len(probLabels)))

def greater_equal(genScore, thresholds):
    temp_4 = []
    for i in range(len(genScore)):
        for j in range(len(thresholds)):
            if genScore[i] >= thresholds[j]:
                temp_4.append(1)
            else:
                temp_4.append(0)
    return np.asarray(temp_4).reshape((len(genScore), len(thresholds)))

def logical_and_3d(T1, T2):
    arr_1 = []
    for ind_0, ind_1 in zip(T1, T2):
        arr = []
        for j in ind_0[0]:
            arr.append([j & i for i in ind_1])
        arr_1.append(np.transpose(arr))
    return arr_1

def remove_elem(mass, isZeroFAR, isOneFAR):
    temp_ind = []
    for i in enumerate(np.logical_and(np.logical_not(isZeroFAR), np.logical_not(isOneFAR)).astype(int).tolist()):
        if i[1] == 0:
            temp_ind.append(mass[i[0]])
    for i in temp_ind:
        for j in enumerate(mass):
            if j[1] == i:
                mass[j[0]] = 0
        # mass.remove(i)
    return mass

def insert_elem(mass, temp, isZeroFAR, isOneFAR):
    temp_ind = []
    for i in enumerate(np.logical_and(np.logical_not(isZeroFAR), np.logical_not(isOneFAR)).astype(int).tolist()):
        if i[1] == 1:
            temp_ind.append(i[0])
    for i, j in zip(temp_ind, range(len(temp))):
        mass[i] = temp[j]
    # for i, j in zip(temp_ind, range(len(temp))):
    #     if j == i:
    #         mass[i] = temp[j]
    return mass

def sub2ind(array_shape, rows, cols):
    ind = [i + (array_shape[0]*j) for i, j in zip(range(len(rows)), range(len(cols)))]
    # for i in range(len(ind)):
    #     if ind[i] < 0:
    #         ind[i] = -1
    #     elif ind[i] >= array_shape[0] * array_shape[1]:
    #         ind[i] = -1
    return ind

def ind2sub(array_shape, ind):
    temp = [i[0] for i in enumerate(ind) if i[1] <= 0]
    for i in temp:
        for j in range(len(ind)):
            if j == i:
                ind[j] = -1
    temp1 = [i[0] for i in enumerate(ind) if i[1] >= array_shape[0] * array_shape[1]]
    for i in temp1:
        for j in range(len(ind)):
            if j == i:
                ind[j] = -1
    # rows = [int(i) / array_shape[1] for i in ind]
    rows = [int(i) // array_shape[0] for i in ind]
    cols = [i % array_shape[1] for i in ind]
    return [rows, cols]

def less_equal(probRanks, rankPoints):
    temp_4 = []
    for i in range(len(rankPoints)):
        for j in range(len(probRanks)):
            if probRanks[j] <= rankPoints[i]:
                temp_4.append(1)
            else:
                temp_4.append(0)
    return np.asarray(temp_4).reshape((len(rankPoints), len(probRanks)))

def argsort(S, genScore):
    temp_1 = []
    for i, j in zip(S, genScore):
        k = 0
        for ind in enumerate(i):
            if ind[1] == j:
                k = ind[0]
        temp_1.append(k)
    return sorted(temp_1)

def count_args(*args):
    count = 0
    for i in args:
        if not (i is None):
            count += 1
    return count

def bsxfun(view_fun, a, b):
    labels = []
    if str(view_fun) == "ge":
        # temp[:] = [0 if elem == 0 else 1 for elem in temp] # to compare list and number
        for i in a:
            if a[i] >= b[i]:
                labels.append(1)
            else:
                labels.append(0)
        return labels
    if str(view_fun) == "le":
        for i in a:
            if a[i] == b[i]:
                labels.append(1)
            else:
                labels.append(0)
        return labels
    if str(view_fun) == "and":
        labels = [i and j for i, j in zip(a, b)]
        return labels
    if str(view_fun) == "minus":
        return np.subtract(a, b)

def isempty(X):
    return True if np.size(X) == 0 else False

def normr(x):
    return np.divide(x, np.linalg.norm(x, axis=1, keepdims=True))

def isequal(x,y):
    for i in x:
        if x[i] == y[i]:
            return True
        else:
            return False