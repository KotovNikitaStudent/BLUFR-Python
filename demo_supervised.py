import os
# from PCA import PCA
import numpy as np
from other_function import *
from ismember import ismember
from GridSearch import GridSearch
from EvalROC import EvalROC
from OpenSetROC import OpenSetROC
import matplotlib.pyplot as plt
import hdf5storage
from scipy.linalg import lstsq
import time

feaFile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\data\\lfw.mat"
configFile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\config\lfw\\blufr_lfw_config.mat"
outDir = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\result"
savePCAfile_super = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\result\\savePCAfile_super.mat"
outMatFile = os.path.join(outDir, "result_lfw_supervised.mat")
outLogFile = os.path.join(outDir, "result_lfw_supervised.txt")

veriFarPoints = sorted(np.concatenate((np.kron([math.pow(10, i) for i in range(-8, 0)],
                                               [i for i in range(1, 10)]), np.array([0, 1]))))
osiFarPoints = sorted(np.concatenate((np.kron([math.pow(10, i) for i in range(-4, 0)],
                                              [i for i in range(1, 10)]), np.array([0, 1]))))
rankPoints = [i for i in range(1, 10, 1)] + [i for i in range(10, 110, 10)]
reportVeriFar = 0.001
reportOsiFar = 0.01
reportRank = 1
timer_start = time.time()
print('Load data...\n')
mat = hdf5storage.loadmat(configFile)
data = hdf5storage.loadmat(feaFile)
data_pca = hdf5storage.loadmat(savePCAfile_super)
Descriptors = np.sqrt(data['Descriptors'].astype(float))
labels = np.squeeze(mat['labels']).tolist()
trainX = np.asarray([Descriptors[i-1, :] for i in np.squeeze(mat['devTrainIndex']).tolist()])
trainLabels = np.asarray([labels[i] for i in np.squeeze(mat['devTrainIndex']).tolist()])
testX = np.asarray([Descriptors[i-1, :] for i in np.squeeze(mat['devTestIndex']).tolist()])
testLabels = np.asarray([labels[i] for i in np.squeeze(mat['devTestIndex']).tolist()])

# temp_1 = GridSearch(trainX, trainLabels, testX, testLabels)
# pcaDims, lambda_1 = temp_1[0], temp_1[1]
data_grid = hdf5storage.loadmat("D:\Точка_Зрения\Проект_по_турникету\BLUFR\\debug\\temp_grid.mat")
pcaDims, lambda_1 = data_grid['pcaDims'], data_grid['lambda']

pcaDims = int(np.squeeze(pcaDims))
lambda_1 = float(np.squeeze(lambda_1))

numTrials = np.shape(mat['testIndex'])[0]
numVeriFarPoints = len(veriFarPoints)
VR = np.zeros((numTrials, numVeriFarPoints))
veriFAR = np.zeros((numTrials, numVeriFarPoints))
numOsiFarPoints = len(osiFarPoints)
numRanks = len(rankPoints)
DIR = np.zeros((numRanks, numOsiFarPoints, numTrials))
osiFAR = np.zeros((numTrials, numOsiFarPoints))

_, veriFARIndex = ismember(np.asarray(reportVeriFar), np.asarray(veriFarPoints))
_, osiFarIndex = ismember(np.asarray(reportOsiFar), np.asarray(osiFarPoints))
_, rankIndex = ismember(np.asarray(reportRank), np.asarray(rankPoints))

print('Evaluation with 10 trials\n')

for t in range(numTrials):
    print(f'Process the {t}th trial...\n')
    trainX = [Descriptors[i-1, :] for i in np.squeeze(mat['trainIndex'][t][0]).tolist()]
    trainLabels = [labels[i-1] for i in np.squeeze(mat['trainIndex'][t][0]).tolist()]
    testX = [Descriptors[i-1, :] for i in np.squeeze(mat['testIndex'][t][0]).tolist()]
    testLabels = [labels[i-1] for i in np.squeeze(mat['testIndex'][t][0]).tolist()]
    # W = PCA(trainX)
    W = data_pca['W']
    trainX = np.dot(trainX, W[:, :pcaDims])
    testX = np.dot(testX, W[:, :pcaDims])
    hst = plt.hist(trainLabels, bins=np.arange(max(trainLabels)))
    temp = [int(i) for i in hst[0].tolist()]
    classIndex = [i[0] for i in enumerate(temp) if i[1] >= 2]
    sampleIndex, trainLabels = ismember(np.asarray(trainLabels), np.asarray(classIndex))
    sampleIndex = list(map(int, sampleIndex.tolist()))
    trainLabels = trainLabels.tolist()
    sampleIndex = sampleIndex[:np.shape(trainLabels)[0]]
    trainLabels = [trainLabels[i[0]] for i in enumerate(sampleIndex) if i[1] == 1]
    trainX = [trainX[i[0]][:] for i in enumerate(sampleIndex) if i[1] == 1]
    numTrainSamples = len(trainLabels)
    Y = np.zeros((numTrainSamples, len(classIndex)))
    temp2 = [j * np.shape(Y)[0] + i for i, j in zip([i for i in range(numTrainSamples)], trainLabels)]
    Y = Y.tolist()
    rows = []

    for i in temp2:
        ind = i // np.shape(Y)[0]
        if ind <= np.shape(Y)[0]:
            rows.append(ind)

    for i, j in zip(enumerate(Y), rows):
        Y[i[0]][j] = 1

    W = lstsq(np.dot(np.transpose(trainX), trainX) + np.dot(lambda_1, np.dot(numTrainSamples, np.eye(pcaDims))), np.dot(np.transpose(trainX), Y))
    W = W[0]
    testX = np.dot(testX, W)
    testX = normr(testX)
    score = np.dot(testX, np.transpose(testX))
    temp_EvalROC = EvalROC(score, testLabels, [], veriFarPoints)
    VR[t, :], veriFAR[t, :] = temp_EvalROC[0], temp_EvalROC[1]
    _, gIdx = ismember(np.squeeze(mat['galIndex'][t][0]).tolist(), np.squeeze(mat['testIndex'][t][0]).tolist())
    _, pIdx = ismember(np.squeeze(mat['probIndex'][t][0]).tolist(), np.squeeze(mat['testIndex'][t][0]).tolist())
    gIdx = gIdx.tolist()
    pIdx = pIdx.tolist()
    osiFarPoints = np.asarray(osiFarPoints)
    temp_OpenSetRoc = OpenSetROC(np.asarray([[score[i][j] for j in pIdx] for i in gIdx]),
                                 np.asarray([testLabels[i] for i in gIdx]),
                                 np.asarray([testLabels[i] for i in pIdx]),
                                 osiFarPoints)
    DIR[:, :, t], osiFAR[t, :] = temp_OpenSetRoc[0], temp_OpenSetRoc[1]
    print('Verification:\n')
    print(f'FAR = {reportVeriFar * 100} %: VR = {VR[t, veriFARIndex] * 100} %\n')
    print('Open-set Identification:\n')
    print(f'Rank = {reportRank}, FAR = {reportOsiFar * 100} %: DIR = {DIR[rankIndex, osiFarIndex, t] * 100} %\n')

veriFARIndex = int(veriFARIndex[0])
rankIndex = int(rankIndex[0])
osiFarIndex = int(osiFarIndex[0])
meanVeriFAR = np.mean(veriFAR, axis=0).tolist()
meanVR = np.mean(VR, axis=0).tolist()
stdVR = np.std(VR, axis=0).tolist()
reportMeanVR = meanVR[veriFARIndex]
reportStdVR = stdVR[veriFARIndex]
meanOsiFAR = np.mean(osiFAR, axis=0).tolist()
meanDIR = np.mean(DIR, axis=2).tolist()
stdDIR = np.std(DIR, axis=2).tolist()
reportMeanDIR = meanDIR[rankIndex][osiFarIndex]
reportStdDIR = stdDIR[rankIndex][osiFarIndex]
fusedVR = np.dot(abs(np.subtract(meanVR, stdVR)), 100).tolist()
reportVR = np.dot(abs(np.subtract(reportMeanVR, reportStdVR)), 100).tolist()
fusedDIR = np.dot(abs(np.subtract(meanDIR, stdDIR)), 100).tolist()
reportDIR = np.dot(abs(np.subtract(reportMeanDIR, reportStdDIR)), 100).tolist()

print(f'Evaluation time: {time.time() - timer_start} sec')
str1 = 'Verification:\n'
str1 = f'{str1} FAR = {reportVeriFar * 100} %: VR = {reportVR}\n'
str1 = f'Open-set Identification {str1}'
str1 = f'{str1} Rank = {reportRank}, FAR = {reportOsiFar * 100} %: DIR = {reportDIR}\n'
print('The fused (mu-sigma) performance:\n')
print(str1)

# fout = open(outLogFile, 'w')
# fout.write(str1)
# fout.close()

plt.figure(1)
plt.semilogx(np.dot(meanVeriFAR, 100).tolist(), fusedVR, linewidth=2)
plt.grid(True)
plt.xlabel("False Accept Rate (%)")
plt.ylabel("Verification Rate (%)")
plt.title('Face Verification ROC Curve')
plt.show()

plt.figure(2)
plt.semilogx(np.dot(meanOsiFAR, 100), fusedDIR[rankIndex][:], linewidth=2)
plt.grid(True)
plt.xlabel("False Accept Rate (%)")
plt.ylabel("Detection and Identification Rate (%)")
plt.title(f"Open-set Identification ROC Curve at Rank {reportOsiFar * 100} %")
plt.show()

temp_6 = []
for i in fusedDIR:
    for j in enumerate(i):
        if j[0] == osiFarIndex:
            temp_6.append(j[1])

plt.figure(3)
plt.semilogx(rankPoints, temp_6, linewidth=2)
plt.grid(True)
plt.xlabel("Rank")
plt.ylabel("Detection ans Identification Rate (%)")
plt.title(f"Open-set Identification CMC Curve at FAR {reportOsiFar * 100} %")
plt.show()

# hdf5storage.savemat(outMatFile, {'reportVeriFar': reportVeriFar,
#                                  'reportOsiFar': reportOsiFar,
#                                  'reportRank': reportRank,
#                                  'reportVR': reportVR,
#                                  'reportDIR': reportDIR,
#                                  'meanVeriFAR': meanVeriFAR,
#                                  'fusedVR': fusedVR,
#                                  'meanOsiFAR': meanOsiFAR,
#                                  'fusedDIR': fusedDIR,
#                                  'rankPoints': rankPoints,
#                                  'rankIndex': rankIndex,
#                                  'osiFarIndex': osiFarIndex}, appendmat=True)