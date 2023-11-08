import pandas as pd
import numpy as np

class DataProcessor():
    def __init__(self, path):
        self.data = pd.read_csv(path)
    
    def getRequiredData(self, start, end):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        DataForExtrapolation = self.data[self.data['Date'] >= start]
        TrainData = DataForExtrapolation[DataForExtrapolation['Date'] <= end]
        DataForExtrapolation = DataForExtrapolation.reset_index(drop=True)
        TrainData = TrainData.reset_index(drop=True)
        TrainData = TrainData.drop(TrainData.columns[[2, 3, 4, 7, 8]], axis=1)
        DataForExtrapolation = DataForExtrapolation.drop(DataForExtrapolation.columns[[2, 3, 4, 7, 8]], axis=1)
        TrainDate = TrainData.to_numpy()[:,0]
        TrainConfirmed = TrainData['Confirmed'].values
        TrainTested = TrainData['Tested'].values
        TrainDose = TrainData['First Dose Administered'].values
        TrainConfirmed = self.getConseciff(TrainConfirmed)
        TrainTested = self.getConseciff(TrainTested)
        TrainDose = TrainDose[1:]
        TrainDate = TrainDate[1:]
        TrainData = np.column_stack((TrainDate, TrainConfirmed, TrainTested, TrainDose))
        DataForExtrapolation = DataForExtrapolation.to_numpy()
        ## Make this method to return those four arrays.. write more methods for that which use these two arrays
        return TrainData, DataForExtrapolation
    
    def getConseciff(self, arr):
        diff = np.diff(arr)
        return diff.astype(int)


d = DataProcessor('../COVID19_data.csv')
start = '2021-03-08'
end = '2021-04-26'
d.getRequiredData(start,end)