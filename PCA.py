import time
from other_function import *

def PCA(X=None, pcaDims=None):
    nargin = count_args(X, pcaDims)
    n, d = np.shape(X)[0], np.shape(X)[1]

    if nargin == 1:
        pcaDims = np.min(d, n) - 1

    if pcaDims <= 0 or pcaDims >= min(d, n):
        print("Incorrect pcaDims")

    print('Start to train PCA')
    t0 = time.time()
    mu = np.mean(X)
    X = bsxfun("minus", X, mu)

    if n >= d:
        C = np.transpose(X) * X
    else:
        C = X * np.transpose(X) * X

    W, D = np.linalg.eigh(C)
    latent = np.diag(D)
    latent, index = np.sort(latent), np.argsort(latent)
    W = W[:, index]
    W = W[:, 1:pcaDims]
    latent = latent[1:pcaDims]

    if n < d:
        W = np.transpose(X) * W * np.diag(1 / np.sqrt(latent))

    S = X * W
    print('second '.format(time.time() - t0))
    return [W, S, latent]