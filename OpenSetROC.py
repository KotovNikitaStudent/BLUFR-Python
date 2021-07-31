from other_function import *

def OpenSetROC(score=None, galLabels=None, probLabels=None, farPoints=None, rankPoints=None):
    nargin = count_args(score, galLabels, probLabels, farPoints, rankPoints)
    if nargin < 5 or isempty(rankPoints):
        rankPoints = [i for i in range(1, 10, 1)] + [i for i in range(10, 110, 10)]
        for i in rankPoints:
            if i > len(galLabels):
                del i

    # galLabels, probLabels = galLabels.tolist(), probLabels.tolist()
    binaryLabels = equal(galLabels, probLabels)
    t = np.any(binaryLabels, axis=0)
    genProbIndex = np.where(t == True)
    impProbIndex = np.where(t == False)
    Ngen = int(np.shape(genProbIndex)[1])
    Nimp = int(np.shape(impProbIndex)[1])
    farPoints = np.asarray(farPoints)

    if nargin < 4 or isempty(farPoints):
        falseAlarms = [i for i in range(Nimp)]
    else:
        if np.any(np.where(farPoints < 0)) or np.any(np.where(farPoints > 1)):
            print("FAR should be in the range (0,1)")
        falseAlarms = np.round(np.dot(farPoints, Nimp))

    impProbIndex = np.squeeze(impProbIndex).tolist()
    impScore = [score[:, i] for i in impProbIndex]
    impScore = [np.max(i) for i in impScore]
    impScore = np.flip(np.sort(impScore, axis=0), axis=0)
    S = np.asarray([score[:, i] for i in np.squeeze(np.asarray(genProbIndex)).tolist()]).tolist()
    sortedIndex = np.transpose(np.flip(np.argsort(S), axis=1))
    M = np.transpose(np.asarray([binaryLabels[:, i] for i in np.squeeze(np.asarray(genProbIndex)).tolist()])).tolist()
    S = np.transpose(S)

    for i, j in zip(range(len(S)), [[ind for ind, val in enumerate(i) if val == 0] for i in M]):
        for k in j:
            S[i][k-1] = -math.inf

    genScore = [max(i) for i in np.transpose(S)]
    genGalIndex = argsort(np.transpose(S), genScore)
    probRanks = np.where(np.transpose([np.equal(i, genGalIndex).astype(int).tolist() for i in sortedIndex.tolist()]))[1].tolist()
    isZeroFAR = [1 if i == 0 else 0 for i in falseAlarms]
    isOneFAR = [1 if i == Nimp else 0 for i in falseAlarms]
    thresholds = np.zeros(len(falseAlarms))
    falseAlarms = list(map(int, falseAlarms.tolist()))
    thresholds = list(map(int, thresholds.tolist()))
    temp = [impScore[i] for i in remove_elem(falseAlarms, isZeroFAR, isOneFAR)]
    thresholds = insert_elem(thresholds, temp, isZeroFAR, isOneFAR)
    highGenScore = [i for i in genScore if i > impScore[0]]

    if isempty(highGenScore):
        for i in [i[0] for i in enumerate(isZeroFAR) if i[1] == 1]:
            thresholds[i] = impScore[0] + math.sqrt(2.2204*10**-16)
    else:
        for i in [i[0] for i in enumerate(isZeroFAR) if i[1] == 1]:
            thresholds[i] = (impScore[0] + min(highGenScore)) / 2

    for i in [i[0] for i in enumerate(isOneFAR) if i[1] == 1]:
        thresholds[i] = min(impScore[-1], min(genScore)) - math.sqrt(2.2204*10**-16)

    T1 = greater_equal(genScore, thresholds)
    T2 = np.transpose(less_equal(probRanks, rankPoints))
    T1 = T1.reshape(Ngen, 1, np.shape(T1)[1])
    T = logical_and_3d(T1, T2)
    DIR = np.mean(T, axis=0)
    FAR = np.divide(falseAlarms, Nimp)
    return [DIR, FAR, thresholds]