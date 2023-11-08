import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

def import_data(data_dir):
    data = pd.read_csv(data_dir)
    unwanted = ['Other', 'Total Doses Administered', 'Recovered', 'Second Dose Administered', 'Deceased']
    data = data.drop(columns=unwanted)
    data['Date'] = pd.to_datetime(data['Date'])
    # dates taken from 8th March for running averages 
    full_data = data[data['Date'] >= '2021-03-08']
    data = data[(data['Date'] >= '2021-03-08') & (data['Date'] <= '2021-04-26')]
    
    # reset the index
    full_data = full_data.reset_index(drop=True)
    data = data.reset_index(drop=True)
    # Calculate differences and convert to integer
    data['Confirmed'] = data['Confirmed'].diff().drop(0).astype(int)
    data['Tested'] = data['Tested'].diff().drop(0).astype(int)
    # drop the first row with NaN values
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data.to_numpy(), full_data.to_numpy()

def RunningAverages(ModelData, OriginalData):

    # Confirmed Running Average
    ConRunAvg = np.convolve(ModelData[:, 1], np.ones(7) / 7, mode='valid')[1:]

    # Tested Running Average
    Tested = OriginalData[:, [0,2]]
    SecCol = Tested[:,1]
    for i in range(len(SecCol) - 1, 7, -1):
        SecCol[i] = (SecCol[i] - SecCol[i-7]) / 7
    last_date = Tested[-1, 0]
    date_range = pd.date_range(start=last_date + pd.DateOffset(days=1), end='2021-12-31')

    # Calculate the number of days for extrapolation
    days_to_extrapolate = (date_range[-1] - last_date).days

    # Create an array with extrapolated dates and rolling average values
    extrapolated_dates = np.array([last_date + pd.DateOffset(days=i) for i in range(1, days_to_extrapolate + 1)])
    extrapolated_values = np.full(days_to_extrapolate, SecCol[-1])

    # Concatenate the extrapolated data to the original 'tested' data
    Tested = np.vstack([Tested, np.column_stack((extrapolated_dates, extrapolated_values))])[8:]

    # First Dose Running Average
    First = ModelData[:, [0,3]]
    SecCol = First[:,1]

    for i in range(len(SecCol) - 1, 6, -1):
        SecCol[i] = (SecCol[i] - SecCol[i-7]) / 7
    last_date = First[-1, 0]
    date_range = pd.date_range(start=last_date + pd.DateOffset(days=1), end='2021-12-31')
    days_to_extrapolate = (date_range[-1] - last_date).days
    extrapolated_dates = np.array([last_date + pd.DateOffset(days=i) for i in range(1, days_to_extrapolate + 1)])
    extrapolated_values = np.full(days_to_extrapolate, 200000)
    First = np.vstack([First, np.column_stack((extrapolated_dates, extrapolated_values))])[7:]

    # Ground Truth
    GrouTruth = OriginalData[:, 1]
    GrouTruth = (GrouTruth - np.roll(GrouTruth, 7))/ 7
    GrouTruth= GrouTruth[8:].astype(int)
    return ConRunAvg, Tested, First, GrouTruth

class SEIRV():
    def __init__(self, beta, S0, E0, I0, R0, CIR0, Totaldays, RunningAvgs, waning = True):
        self.days = Totaldays
        self.S = np.zeros(Totaldays)
        self.E = np.zeros(Totaldays)
        self.I = np.zeros(Totaldays)
        self.R = np.zeros(Totaldays)
        self.e = np.zeros(Totaldays)
        self.CIR0 = CIR0
        self.S[0] = S0
        self.E[0] = E0
        self.I[0] = I0
        self.R[0] = R0
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.N = 7e7
        self.gamma = 1/5
        self.beta = beta
        self.alpha = 1/5.8
        self.epsilon = 0.66
        self.ConvRunAvg, self.Tested, self.First, self.GrTruth = RunningAvgs
        self.waning = waning

    def calculate_delta_w(self, day):
        if day <= 30:
            delW = self.R0 / 30
        elif day >= 180:
            if self.waning:
                delW = self.R[day - 180] + self.epsilon * self.First[day - 180][1]
            else:
                delW = 0
        else:
            delW = 0

        return delW
    
    def update_arrays(self, day, delW):
        self.S[day + 1] = self.S[day] - self.beta * self.S[day] * self.I[day] / self.N - self.epsilon * self.First[day][1] + delW
        self.E[day + 1] = self.E[day] + self.beta * self.S[day] * self.I[day] / self.N - self.alpha * self.E[day]
        self.I[day + 1] = self.I[day] + self.alpha * self.E[day] - self.gamma * self.I[day]
        self.R[day + 1] = self.R[day] + self.gamma * self.I[day] + self.epsilon * self.First[day][1] - delW

    def calculate_rolling_averages(self):
        for day in range(self.days):
            avg_count = 0
            for prev_day in range(day, day - 7, -1):
                if prev_day >= 0:
                    self.avgS[day] += self.S[prev_day]
                    self.avgE[day] += self.E[prev_day]
                    self.avgI[day] += self.I[prev_day]
                    self.avgR[day] += self.R[prev_day]
                    avg_count += 1
            self.avgS[day] = self.avgS[day] / avg_count
            self.avgE[day] = self.avgE[day] / avg_count
            self.avgI[day] = self.avgI[day] / avg_count
            self.avgR[day] = self.avgR[day] / avg_count

    def calculate_effective_reproduction_number(self):
        for day in range(self.days):
            CIR = self.CIR0 * self.Tested[0][1] / self.Tested[day][1]
            self.e[day] = self.avgE[day] / CIR

    def generate_time_series(self):
        for day in range(self.days - 1):
            delW = self.calculate_delta_w(day)
            self.update_arrays(day, delW)
        self.avgS = np.zeros(self.days)
        self.avgE = np.zeros(self.days)
        self.avgI = np.zeros(self.days)
        self.avgR = np.zeros(self.days)

        self.calculate_rolling_averages()
        self.calculate_effective_reproduction_number()

    def calculate_loss(self):
        self.generate_time_series()
        self.e = self.alpha * self.e
        avge = np.zeros(self.days)
        for i in range(self.days):
            count = 0
            for j in range(i, i - 7, -1):
                if j >= 0:
                    count += 1
                    avge[i] += self.e[j]
                else:
                    break
            avge[i] /= count
        loss = 0
        for i in range(self.days):
            loss += (math.log(self.ConvRunAvg[i]) - math.log(avge[i])) ** 2
        loss /= self.days
        return loss
    
    def gradient(self):  
        loss = self.calculate_loss()
        self.beta += 0.01
        loss_beta = self.calculate_loss()
        self.beta -= 0.01
        self.S0 += 1
        loss_S0 = self.calculate_loss()
        self.S0 -= 1
        self.E0 += 1
        loss_E0 = self.calculate_loss()
        self.E0 -= 1
        self.I0 += 1
        loss_I0 = self.calculate_loss()
        self.I0 -= 1
        self.R0 += 1
        loss_R0 = self.calculate_loss()
        self.R0 -= 1
        self.CIR0 += 0.1
        loss_CIR0 = self.calculate_loss()
        self.CIR0 -= 0.1
        return (loss_beta - loss) / 0.01 , loss_S0 - loss , loss_E0 - loss, loss_I0 - loss, loss_R0 - loss, (loss_CIR0 - loss) / 0.1, loss
    
    def optimize_params(self, threshold=30000):
        stepsize = 0.01
        iterations = 0
        Loss = []
        while iterations <= threshold:
            # stepsize = 1 / (iterations + 1)
            dbeta, dS0, dE0, dI0, dR0, dCIR0, loss = self.gradient()
            Loss.append(loss)
            if iterations % 5000 == 0:
                print('iter: {}, loss: {}'.format(iterations, loss))
            iterations += 1

            # if iterations == threshold or loss < 0.001:
            #     print('\nprogram terminating...')
            #     print('iter: {}, loss: {}'.format(iterations, loss))
            #     break
            self.beta -= dbeta * stepsize
            self.S0 -= dS0 * stepsize
            self.E0 -= dE0 * stepsize
            self.I0 -= dI0 * stepsize
            self.R0 -= dR0 * stepsize
            self.CIR0 -= dCIR0 * stepsize
        self.plot_loss(Loss)
        
        print('Training Ends after 30,000 iterations....\n')

    def plot_loss(self,Loss):
        plt.figure(figsize = (9, 7))
        plt.plot(Loss)
        plt.title('Loss vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig('../plots/loss.jpg')

    def future(self, beta, closed_loop = True):
        new_cases = []
        for day in range(self.days - 1):
            if closed_loop and day % 7 == 1 and day >= 7:
                avg_cases = 0
                for i in range(7):
                    CIR = self.CIR0 * self.Tested[0][1] / self.Tested[day - i][1]
                    avg_cases += self.alpha * self.E[day - i] / CIR
                avg_cases /= 7
                if avg_cases < 10000:
                    self.beta = beta
                elif avg_cases < 25000:
                    self.beta = beta * 2 / 3
                elif avg_cases < 100000:
                    self.beta = beta / 2 
                else:
                    self.beta = beta / 3
            delW = self.calculate_delta_w(day)
            self.update_arrays(day, delW)
            CIR = self.CIR0 * self.Tested[0][1] / self.Tested[day][1] 
            new_cases.append(self.alpha * self.E[day])
        self.avgS = np.zeros(self.days)
        self.avgE = np.zeros(self.days)
        self.avgI = np.zeros(self.days)
        self.avgR = np.zeros(self.days)
        self.calculate_rolling_averages()
        for day in range(self.days):
            CIR = self.CIR0 * self.Tested[0][1] / self.Tested[day][1] 
            self.e[day] = self.avgE[day] / CIR 
        return self.avgS, new_cases
    
    def plotSEIR(self, plot_dir, title, savename):
        plt.figure(figsize = (9, 7))
        plt.plot(self.S, label = 'Susceptible')
        plt.plot(self.E, label = 'Exposed')
        plt.plot(self.I, label = 'Infected')
        plt.plot(self.R, label = 'Recovered')
        plt.legend()
        plt.title(title)
        if title == 'SIER with Immunity waning':  
            plt.xlabel('200 days from 16-03-2021')
        else:
            plt.xlabel('180 days from 16-03-2021')
        plt.ylabel('Number of People')
        plt.savefig(plot_dir + savename)

def plotFutureCases(Params, BetaFractions, plot_dir, susceptible = False):
    beta = Params[0]
    RestParams = Params[1:]
    plt.figure(figsize = (9, 7))
    for fract in BetaFractions:
        f = fract.split('/')
        if len(f) > 1:
            f = int(f[0]) / int(f[1])
        else:
            f = int(f[0])
        seirv = SEIRV(*([beta * f] + RestParams), 290, RunningAverages(ModelData, OriginalData))
        if susceptible:
            S,_ = seirv.future(beta, closed_loop=False)
            plt.plot(S[42:] / N, label = fract + '* beta, Open Loop')
        else:
            _,new_cases = seirv.future(beta, closed_loop=False)
            plt.plot(new_cases[42:], label = fract + '* beta, Open Loop')
    seirv = SEIRV(*([beta * f] + RestParams), 290, RunningAverages(ModelData, OriginalData))
    if susceptible:
        S,_ = seirv.future(beta, closed_loop=True)
        plt.plot(S[42:] / N, label = 'beta, Closed Loop')
    else:
        _,new_cases = seirv.future(beta, closed_loop=True)
        plt.plot(new_cases[42:], label = 'beta, Closed Loop')
    if not susceptible:
        plt.plot(GrTruth[42:], label = 'Actual Reported Cases')
        plt.title('Open-Loop & Closed-Loop Predictions till 31st Dec 2021')
    else:
        plt.title('Open-Loop & Closed-Loop Predictions till 31st Dec 2021 and Actual Reported Cases till 20th Sep 2021')
    plt.legend()
    plt.xlabel('From 27th April 2021 to 31st Dec 2021')
    if susceptible:
        plt.ylabel('Fraction of Susceptible People')
        plt.savefig(plot_dir + 'suceptible.jpg')
    else:
        plt.ylabel('New Cases Everyday')
        plt.savefig(plot_dir + 'newcases.jpg')



if __name__ == "__main__":

    plot_dir = '../plots/'
    ModelData, OriginalData= import_data('../COVID19_data.csv')
    print(ModelData, OriginalData)
    ConRunAvg, TestRunAvg, FirstRunAvg, GrTruth = RunningAverages(ModelData, OriginalData)

    # Initializations & constraints
    N = 7e7
    beta0 = 0.45
    S0 = 0.7 * N
    E0 = 0.001 * N
    I0 = 0.001 * N
    R0 = 0.298 * N
    CIR0 = 15

    InitialParams = [beta0, S0, E0, I0, R0, CIR0]
    print('Initial Parameters:\n')
    print('             beta0_init = {}\n \
            S0_init = {}\n \
            E0_init = {}\n \
            I0_init = {}\n \
            R0_init = {}\n \
            CIR0_init = {}'.format(InitialParams[0], InitialParams[1], InitialParams[2], InitialParams[3], InitialParams[4], InitialParams[5]))
    print('\n Training Begins....')
    seirv = SEIRV(*InitialParams, 42, RunningAverages(ModelData, OriginalData), True)
    seirv.optimize_params()
    FinalParams = [seirv.beta, seirv.S0, seirv.E0, seirv.I0, seirv.R0, seirv.CIR0]
    print('Final Parameters:\n')
    print('             beta0 = {}\n \
            S0 = {}\n \
            E0 = {}\n \
            I0 = {}\n \
            R0 = {}\n \
            CIR0 = {}'.format(FinalParams[0], FinalParams[1], FinalParams[2], FinalParams[3], FinalParams[4], FinalParams[5]))

    seirv = SEIRV(*FinalParams, 200, RunningAverages(ModelData, OriginalData), True)
    seirv.generate_time_series()
    # seirv.plotSEIR(plot_dir, 'SIER with Immunity Waning', 'seir_immunity.jpg')


    seirv = SEIRV(*FinalParams, 200, RunningAverages(ModelData, OriginalData), False)
    seirv.generate_time_series()
    # seirv.plotSEIR(plot_dir, 'SIER without Immunity Waning', 'seir_wo_immunity.jpg')

    BetaFractions = ['1', '2/3', '1/2', '1/3']
    plotFutureCases(FinalParams, BetaFractions, plot_dir)
    plotFutureCases(FinalParams, BetaFractions, plot_dir, susceptible=True)