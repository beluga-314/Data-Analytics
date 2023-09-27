import numpy as np
import numpy.linalg as lg
import scipy.stats as stats
from matplotlib import pyplot as plt
import os

def import_data(dir):
    Data = []
    # Removes the row if there is no data in any column
    with open(dir, 'r') as file:
        next(file)
        for row in file:
            values = [float(x) for x in row.strip().split()[1:49]]
            if len(values) == 48:
                Data.append(values)
    Data = np.array(Data)
    return Data

def buildD(rows):
    D = np.zeros((rows, 4))
    for row in range(rows):
        if row < rows * 1 / 4:
            D[row] = np.array([0, 1, 0, 0])
        elif row < rows * 2 / 4:
            D[row] = np.array([1, 0, 0, 0])
        elif row < rows * 3 / 4:
            D[row] = np.array([0, 0, 0, 1])
        else:
            D[row] = np.array([0, 0, 1, 0])
    return D

def buildN(rows):
    N = np.zeros((rows, 4))
    for row in range(rows):
        if row < rows * 1 / 4:
            N[row] = np.array([1, 0, 0, 1])
        elif row < rows * 2 / 4:
            N[row] = np.array([1, 0, 1, 0])
        elif row < rows * 3 / 4:
            N[row] = np.array([0, 1, 0, 1])
        else:
            N[row] = np.array([0, 1, 1, 0])
    return N

def findFStat(Data, D, N):
    rankD = lg.matrix_rank(D)
    rankN = lg.matrix_rank(N)
    n = Data.shape[1]
    FStat = []
    Identity = np.eye(n)
    NTranspose = np.transpose(N)
    DTranspose = np.transpose(D)
    X = np.zeros((1, Data.shape[1]))
    XTranspose = np.transpose(X)
    DOFNum = rankD - rankN
    DOFDen = n - rankD
    for row in Data:
        X = row
        XTranspose = np.transpose(X)
        QNum = Identity - np.matmul(N, np.matmul(lg.pinv(np.matmul(NTranspose, N)), NTranspose))
        Numerator = np.matmul(XTranspose, np.matmul(QNum, X))
        QDen = Identity - np.matmul(D, np.matmul(lg.pinv(np.matmul(DTranspose, D)), DTranspose))
        Denominator = np.matmul(XTranspose, np.matmul(QDen, X))
        FStat.append(DOFDen / DOFNum * (Numerator/ (Denominator + 1e-10) - 1))
    return FStat, DOFNum, DOFDen

def pvalue(FStat, DOFNum, DOFDen):
    p_value = []
    for f in FStat:
        p_value.append(1 - stats.f.cdf(f, DOFNum, DOFDen))
    return p_value

def main():
    Data = import_data('../data/Raw Data_GeneSpring.txt')
    D = buildD(Data.shape[1])
    N = buildN(Data.shape[1])
    FStat, DOFNum, DOFDen = findFStat(Data, D, N)
    p_value = pvalue(FStat, DOFNum, DOFDen)
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    plt.figure()
    plt.hist(p_value, edgecolor = 'k', bins = 50)
    plt.xlabel("p-values")
    plt.ylabel("Frequency")
    plt.title('Histogram of p-values')
    plt.savefig('../plots/hist.png')
    del os.environ['QT_QPA_PLATFORM']
    return

main()