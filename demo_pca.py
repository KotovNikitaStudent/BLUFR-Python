import os
import time
import hdf5storage
# from PCA import PCA
from other_function import *
from ismember import ismember
from OpenSetROC import OpenSetROC
from EvalROC import EvalROC
import matplotlib.pyplot as plt

feaFile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\data\\lfw.mat"
configFile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\config\lfw\\blufr_lfw_config.mat"
outDir = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\result"
outMatFile = os.path.join(outDir, "result_lfw_pca.mat")
outLogFile = os.path.join(outDir, "result_lfw_pca.txt")
savePCAfile = "D:\Точка_Зрения\Проект_по_турникету\BLUFR\\result\\savePCAfile.mat"

veriFarPoints = sorted(np.concatenate((np.kron([math.pow(10, i) for i in range(-8, 0)],
                                               [i for i in range(1, 10)]), np.array([0, 1]))))
osiFarPoints = sorted(np.concatenate((np.kron([math.pow(10, i) for i in range(-4, 0)],
                                              [i for i in range(1, 10)]), np.array([0, 1]))))
rankPoints = [i for i in range(1, 10, 1)] + [i for i in range(10, 110, 10)]
reportVeriFar = 0.001
reportOsiFar = 0.01
reportRank = 1
pcaDims = 400
timer_start = time.time()
print('Load data...\n')
mat = hdf5storage.loadmat(configFile)
data = hdf5storage.loadmat(feaFile)
data_pca = hdf5storage.loadmat(savePCAfile)
Descriptors = np.sqrt(data['Descriptors'].astype(float))
numTrials = len(mat['testIndex'])
numVeriFarPoints = len(veriFarPoints)
VR = np.zeros((numTrials, numVeriFarPoints))
veriFAR = np.zeros((numTrials, numVeriFarPoints))
numOsiFarPoints = len(osiFarPoints)
numRanks = len(rankPoints)
DIR = np.zeros((numRanks, numOsiFarPoints, numTrials))
osiFAR = np.zeros((numTrials, numOsiFarPoints))
_, veriFarIndex = ismember(np.asarray(reportVeriFar), np.asarray(veriFarPoints))
_, osiFarIndex = ismember(np.asarray(reportOsiFar), np.asarray(osiFarPoints))
_, rankIndex = ismember(np.asarray(reportRank), np.asarray(rankPoints))

print('Evalution with 10 trials\n')

# for t in range(numTrials):
for t in range(1):
    print(f'Process the {t} trial\n')
    # подкачка результата работы функции PCA.py
    # X = [Descriptors[i-1, :] for i in np.squeeze(mat['trainIndex'][t][0]).tolist()]
    # W = PCA(X, pcaDims)
    W = data_pca['W']
    X = [Descriptors[i-1, :] for i in np.squeeze(mat['testIndex'][t][0]).tolist()]
    X = np.dot(X, W[:, :pcaDims])
    X = normr(X)
    score = np.dot(X, np.transpose(X))
    testLabels = [np.squeeze(mat['labels']).tolist()[i-1] for i in np.squeeze(mat['testIndex'][t][0]).tolist()]
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
    print(f'FAR = {reportVeriFar * 100} %: VR = {VR[t, veriFarIndex] * 100} %\n')
    print('Open-set Identification:\n')
    print(f'Rank = {reportRank}, FAR = {reportOsiFar * 100} %: DIR = {DIR[rankIndex, osiFarIndex, t] * 100} %\n')

veriFarIndex = int(veriFarIndex[0])
rankIndex = int(rankIndex[0])
osiFarIndex = int(osiFarIndex[0])
meanVeriFAR = np.mean(veriFAR, axis=0).tolist()
meanVR = np.mean(VR, axis=0).tolist()
stdVR = np.std(VR, axis=0).tolist()
reportMeanVR = meanVR[veriFarIndex]
reportStdVR = stdVR[veriFarIndex]
meanOsiFAR = np.mean(osiFAR, axis=0).tolist()
meanDIR = np.mean(DIR, axis=2).tolist()
stdDIR = np.std(DIR, axis=2).tolist()
reportMeanDIR = meanDIR[rankIndex][osiFarIndex]
reportStdDIR = stdDIR[rankIndex][osiFarIndex]
fusedVR = np.dot(abs(np.subtract(meanVR, stdVR)), 100).tolist()
reportVR = np.dot(abs(np.subtract(reportMeanVR, reportStdVR)), 100).tolist()
fusedDIR = np.dot(abs(np.subtract(meanDIR, stdDIR)), 100).tolist()
reportDIR = np.dot(abs(np.subtract(reportMeanDIR, reportStdDIR)), 100).tolist()

print(f'Evalution time: {time.time() - timer_start} sec')
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
plt.title(f'Open-set Identification ROC Curve at Rank {reportRank}')
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
#                                  'reportVR': reportVR,
#                                  'reportDIR': reportDIR,
#                                  'meanVeriFAR': meanVeriFAR,
#                                  'fusedVR': fusedVR,
#                                  'meanOsiFAR': meanOsiFAR,
#                                  'fusedDIR': fusedDIR,
#                                  'rankPoints': rankPoints,
#                                  'rankIndex': rankIndex,
#                                  'osiFarIndex': osiFarIndex}, appendmat=True)