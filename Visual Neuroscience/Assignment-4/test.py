import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress, gmean, gamma, kstest

plot_dir = '../plots/'

def import_data(dir, type = ''):
    data = pd.read_csv(dir)
    for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    if type == 'firingrate':
        delete_cols = [19,21,23,25,27,29]
        data = data.drop(data.columns[delete_cols], axis=1)
        data = data.drop(data.index[:2])
    return data

def behIndex(searchtime):
    BehIdx = []
    for _, column_data in searchtime.items():
        BehIdx.append(column_data.iloc[2:].sum())
    for i in range(len(BehIdx)):
        if i <= 17:
            BehIdx[i] /= 72
        else:
            BehIdx[i] /= 144
        BehIdx[i] -= 328
        BehIdx[i] = 1 / BehIdx[i]
    return BehIdx

def l1distance(firingrate):
    L1Distance = []
    for i in range(1, len(firingrate.columns), 2):
        x = (firingrate.iloc[:,i] - firingrate.iloc[:,i - 1]).abs().sum()
        L1Distance.append(x)
        L1Distance.append(x)
    for i in range(len(L1Distance)):
        if i < 6:
            L1Distance[i] /= 174
        elif i >= 6 and i < 12:
            L1Distance[i] /= 122
        elif i >= 12 and i < 18:
            L1Distance[i] /= 114
        else:
            L1Distance[i] /= 128
    return L1Distance

def relentropy(firingrate):
    RelEntropy = []
    for i in range(1, len(firingrate.columns), 2):
        if i < 6:
            lambda0 = firingrate.iloc[:,i - 1].iloc[:174]
            lambda1 = firingrate.iloc[:,i].iloc[:174]
        elif i >= 6 and i < 12:
            lambda0 = firingrate.iloc[:,i - 1].iloc[:122]
            lambda1 = firingrate.iloc[:,i].iloc[:122]
        elif i >= 12 and i < 18:
            lambda0 = firingrate.iloc[:,i - 1].iloc[:114]
            lambda1 = firingrate.iloc[:,i].iloc[:114]
        else:
            lambda0 = firingrate.iloc[:,i - 1].iloc[:128]
            lambda1 = firingrate.iloc[:,i].iloc[:128]
        x = 0
        y = 0
        for l0, l1 in zip(lambda0, lambda1):
            if l0 > 1 / (2 * 24 * 250):
                x += max(l0 * np.log((2 * 24 * 250 * l0- 1)/ (2*24*250 * l1 + 1)) - l0 + l1, 0)
            else:
                x += l1
            if l1 > 1 / (2 * 24 * 250):
                y += max(l1 * np.log((2 * 24 * 250 * l1 - 1)/ (2*24*250 * l0 + 1)) - l1 + l0, 0)
            else:
                y += l0
        RelEntropy.append(x / len(lambda0))
        RelEntropy.append(y / len(lambda0))
    return RelEntropy

def AMGM(searchtime, relent):
    product = []
    for i, j in zip(searchtime, relent):
        product.append(j / i)
    return np.mean(product)/gmean(product)

def findAlpha(searchtime):
    search = searchtime.iloc[:, :12]
    search = search.drop(search.index[:2])
    Means = []
    Stds = []
    for i in range(len(search.columns)):
        col = search.iloc[:,i].iloc[:72] - 328
        Means.append(col.mean())
        Stds.append(np.std(col))
    _,_,_,slope = plotter(Means, Stds, xlbl = 'Means', ylbl = 'Stds', name = 'Stds vs Means', ttl= 'Stds vs Means')
    return 1 / slope ** 2

def findRate(searchtime):
    Means = []
    search = searchtime.iloc[:, 12:24]
    search = search.drop(search.index[:2])
    for i in range(len(search.columns)):
        col = search.iloc[:,i].iloc[:72] - 328
        if i < 6:
            Means.append(col[0:36].mean())
        else:
            Means.append(col[0:72].mean())
    return Means

def findKSstat(searchtime, alpha, Rates):
    search = searchtime.iloc[:, 12:24]
    search = search.drop(search.index[:2])
    KSStat = []
    KSPvalue = []
    for i in range(len(search.columns)):
        if i < 6:
            col = search.iloc[:,i].iloc[:72] - 328
            arr = np.array(col[36:72])
        else:
            col = search.iloc[:,i].iloc[:144] - 328
            arr = np.array(col[72:144])
        sorted_arr = np.sort(arr)
        ecdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        gcdf = gamma.cdf(sorted_arr, a = alpha, scale = 1 / Rates[i])
        plt.figure()
        plt.plot(sorted_arr, ecdf, label='ECDF')
        plt.plot(sorted_arr, gcdf, label='GCDF')
        plt.xlabel('samples')
        plt.ylabel('ECDF and GCDF')
        plt.title('ECDF and GCDF vs samples')
        plt.legend()
        plt.savefig(plot_dir + 'gcdf'+str(i)+'.png')
        ks_stat, ks_pval = kstest(ecdf, gcdf)
        KSStat.append(ks_stat)
        KSPvalue.append(ks_pval)
    return KSStat, KSPvalue

def plotter(X, Y, xlbl = '', ylbl = '', name = '', ttl = ''):
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    line = slope * np.array(X) + intercept
    plt.figure()
    plt.scatter(X, Y, label='Data points')
    plt.plot(X, line, color='red', label='Fitted line')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(ttl)
    plt.legend()
    plt.savefig(plot_dir + name)
    return r_value, p_value, std_err , slope

if __name__ == "__main__":
    searchtime = import_data('../data/02_data_visual_neuroscience_searchtimes.csv', type = 'searchtime')
    firingrate = import_data('../data/02_data_visual_neuroscience_firingrates.csv', type = 'firingrate')
    BehaviourIndex = behIndex(searchtime)
    RelativeEntropy = relentropy(firingrate)
    L1Dist = l1distance(firingrate)
    l1_r_val, l1_p_val, l1_std, _ = plotter(L1Dist, BehaviourIndex, xlbl = 'L1 Distance', ylbl = 'Behavioural Index', name = 'bl1.png', ttl = 'Behavioural Index vs L1 Distance')
    re_r_val, re_p_val, re_std, _ = plotter(RelativeEntropy, BehaviourIndex, xlbl = 'Relative Entropy', ylbl = 'Behavioural Index', name = 'bre.png', ttl = 'Behavioural Index vs Relative Entropy')
    AMbyGML1dist = AMGM(BehaviourIndex, L1Dist)
    AMbyGMrelent = AMGM(BehaviourIndex, RelativeEntropy)
    alpha = findAlpha(searchtime)
    Rates = alpha / findRate(searchtime)
    KSStat, KSPval = findKSstat(searchtime, alpha, Rates)
    print('\n')
    print("Behavioural Index vs L1Distance")
    print('r2value = {}, p1value = {}, stdvalue = {}'.format(l1_r_val**2, l1_p_val, l1_std))
    print('AM / GM = ', AMbyGML1dist, '\n')
    print("Behavioural Index vs Relative Entropy")
    print('r2value = {}, p1value = {}, stdvalue = {}'.format(re_r_val**2, re_p_val, re_std))
    print('AM / GM = ', AMbyGMrelent, '\n')
    print('alpha = ', alpha)
    print('Rate Parameters:', Rates)
    print('KS stats: ', KSStat)
    print('KS pvalues: ', KSPval)
    print('\n')

