from other_function import *

def EvalROC(score=None, galLabels=None, probLabels=None, farPoints=None):
    nargin = count_args(score, galLabels, probLabels, farPoints)
    scoreMask = np.tril(np.ones((np.shape(score)[0], np.shape(score)[1])), -1)
    if nargin < 3 or isempty(probLabels):
        probLabels = galLabels

    if not np.shape(galLabels)[0] == 1:
        galLabels = np.transpose(galLabels)

    if not np.shape(probLabels)[0] == 1:
        probLabels = np.transpose(probLabels)

    # binaryLabels = np.equal(galLabels, probLabels).astype(int)
    binaryLabels = equal(galLabels, probLabels)

    if not (np.size(binaryLabels) == np.size(score)):
        print("the size of labels is not same as the size of the score matrix")

    temp_1, temp_2 = [], []

    for i, j in zip(range(len(score)), [[j[0] for j in enumerate(i) if j[1] == 1] for i in scoreMask]):
        for k in j:
            temp_1.append(score[i][k])

    for i, j in zip(range(len(binaryLabels)), [[j[0] for j in enumerate(i) if j[1] == 1] for i in scoreMask]):
        for k in j:
            temp_2.append(binaryLabels[i][k])

    score = temp_1
    binaryLabels = temp_2
    genScore = [score[i[0]] for i in enumerate(binaryLabels) if i[1] == 1]
    impScore = [score[i[0]] for i in enumerate(binaryLabels) if i[1] == 0]
    Nimp = np.shape(impScore)[0]
    farPoints = np.asarray(farPoints)

    if nargin < 4 or isempty(farPoints):
        falseAlarms = [i for i in range(Nimp)]
    else:
        if np.any(np.where(farPoints < 0)) or np.any(np.where(farPoints > 1)):
            print("FAR should be in the range (0,1)")
        falseAlarms = np.round(np.dot(farPoints, Nimp))

    falseAlarms = np.squeeze(falseAlarms).astype(int).tolist()
    impScore = np.flip(np.sort(impScore, axis=0), axis=0)
    isZeroFAR = [1 if i == 0 else 0 for i in falseAlarms]
    isOneFAR = [1 if i == Nimp else 0 for i in falseAlarms]
    thresholds = np.zeros(len(falseAlarms))
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

    thresholds = np.squeeze([i.tolist() for i in thresholds])
    FAR = [i / Nimp for i in falseAlarms]
    VR = np.mean(greater_equal(genScore, thresholds), axis=0)
    return [VR, FAR, thresholds]