# from PCA import PCA
import time
import numpy as np
from other_function import *
from EvalROC import EvalROC
from ismember import ismember
import hdf5storage
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def GridSearch(trainX, trainY, testX, testY):
    candiPcaDims = [i for i in range(100, 600, 100)]
    candiLambda = [10**i for i in range(-4, 1)]
    veriFarPoints = sorted(np.concatenate((np.kron([math.pow(10, i) for i in range(-8, 0)],
                                                   [i for i in range(1, 10)]), np.array([0, 1]))))
    savePCAfile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\debug\\W_pca.mat"
    data_pca = hdf5storage.loadmat(savePCAfile)
    W = data_pca['W_pca']
    # # подкачка результата работы функции PCA.py
    # # W = PCA(trainX)
    trainX = np.dot(trainX, W)
    testX = np.dot(testX, W)
    hst = plt.hist(trainY, bins=np.arange(max(trainY)))
    temp = [int(i) for i in hst[0].tolist()]
    classIndex = [i[0] for i in enumerate(temp) if i[1] >= 2]
    sampleIndex, trainY = ismember(np.asarray(trainY), np.asarray(classIndex))
    sampleIndex = [int(i) for i in np.squeeze(sampleIndex).tolist()]
    trainY = trainY.tolist()
    trainY = [trainY[i[0]] for i in enumerate(sampleIndex) if i[1] == 1]
    trainX = [trainX[i[0]][:] for i in enumerate(sampleIndex) if i[1] == 1]
    numTrainSamples = len(trainY)
    Y = np.zeros((numTrainSamples, len(classIndex))).astype(int)
    temp2 = [j * np.shape(Y)[0] + i for i, j in zip([i for i in range(numTrainSamples)], trainY)]
    Y = Y.tolist()
    rows = []

    for i in temp2:
        ind = i // np.shape(Y)[0]
        if ind <= np.shape(Y)[0]:
            rows.append(ind)

    for i, j in zip(enumerate(Y), rows):
        Y[i[0]][j] = 1

    R = np.dot(np.transpose(trainX), trainX)
    Z = np.dot(np.transpose(trainX), Y)
    numPara1 = len(candiPcaDims)
    numPara2 = len(candiLambda)
    auc = np.zeros((numPara1, numPara2))

    for i in range(numPara1):
        d = int(candiPcaDims[i])
        for j in range(numPara2):
            W = lstsq(R[:d, :d] + np.dot(candiLambda[j], np.dot(numTrainSamples, np.eye(d))), Z[:d, :])
            W = W[0]
            X = np.dot(testX[:, :d], W)
            X = normr(X)
            score = np.dot(X, np.transpose(X))
            VR = EvalROC(score, np.squeeze(testY).tolist(), [], veriFarPoints)[0]
            VR = VR.tolist()
            auc[i][j] = (sum(VR)/len(VR))
        print(f'{i}th iteration success complete')

    plt.figure(0)
    plt.semilogx(candiPcaDims, auc, linewidth=2)
    plt.grid(True)
    plt.xlabel("PCA Dimension")
    plt.ylabel("AUC")
    plt.title("AUC performance w.r.t. different PCA dimensions")
    plt.legend(['lambda = 0.0001', 'lambda = 0.001',
                'lambda = 0.01', 'lambda = 0.1', 'lambda = 1'])
    plt.show()

    bestAuc = np.max(auc)
    index = np.argmax(auc)
    temp1 = ind2sub(np.shape(auc), [index])
    r, c = temp1[0][0], temp1[1][0]
    pcaDims = candiPcaDims[r]
    lamb = candiLambda[c]
    return [pcaDims, lamb]

input_data = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\debug\\arg1_4.mat"
data = hdf5storage.loadmat(input_data)
trainX, trainLabels, testX, testLabels = data['trainX'], data['trainLabels'], \
                                         data['testX'], data['testLabels']
output = GridSearch(trainX, trainLabels, testX, testLabels)
print(output)