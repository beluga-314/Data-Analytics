import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        return Z_0 * (1 - np.exp(-L * X/Z_0))

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        z = Params[w - 1]
        predictions = self.get_predictions(X, z, L = Params[-1])
        return np.mean((predictions - Y) ** 2)
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    return pd.read_csv(data_path)


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """

    # Removing all the colums that are not needed
    # Only columns between 1 - 7 , 11, 24 are kept 
    cols = [i for i in range(1,8)]
    cols.extend([11, 24])
    data = data.iloc[:, cols]

    # Changing the date format
    month_dict = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}
    for index, str in data['Date'].items():
        if '/' in str:
            d, m, y = str.split('/')
        else:
            m, d, y = str.split(' ')
            m = month_dict[m]
            d,_ = d.split('-')
        data.at[index, 'Date'] = f'{d}-{m}-{y}'

    # Drop any rows with empty cells
    data = data.dropna()

    # Removing 2nd Innings rows
    data = data[data['Innings'] == 1]

    # Processing data and storing tuples of (runs, overs) for every value of wickets in hand
    ordered_pairs = []
    for wickets in range(1, data['Wickets.in.Hand'].max() + 1):
        filtered_df = data[data['Wickets.in.Hand'] == wickets]
        run_tuples = list(zip(filtered_df['Runs.Remaining'], filtered_df['Total.Overs'] - filtered_df['Over']))
        if wickets == data['Wickets.in.Hand'].max():
            Total_runs_df = data[data['Over'] == 1]
            for i in Total_runs_df['Innings.Total.Runs']:
                run_tuples.append((i, data['Total.Overs'].max()))
        ordered_pairs.append(run_tuples)

    filtered_df = data[data['Over'] == data['Total.Overs'].max()]
    ordered_pairs.append(filtered_df['Runs'])

    # Final data that is returned will be a list of 11 elements. Each element is again a list of tuples
    # For each value of Wicket.in.Hand, each of the first list will have (runs, overs) tuples and
    # the last list is used to store the runs scored in last over of 1st innings in all matches
    # to calculate approximate value of L

    return np.array(ordered_pairs, dtype=object)


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    # Tracks whether training is done or not to print the loss.
    # Avoids printing loss during training

    global TrainingDone
    TrainingDone = False
    
    def totalLoss(Params):
        model.L = Params[-1]
        model.Z0 = Params[:-1]
        return calculate_loss(model, data)

    # Find approximate L which is used to initialize the value of L during optimization
    approx_L = data[-1].mean()

    # A list of all the Z_0s and L in the end
    ParamsInitializations = [1] * 10
    ParamsInitializations.append(approx_L)

    # Optimizes over the total loss
    OptimParams = sp.optimize.minimize(totalLoss,ParamsInitializations)

    # Setting the model parameters to OptimParams
    model.Z0 = OptimParams.x[:-1]
    model.L = OptimParams.x[-1]
    TrainingDone = True
    return model

def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    overs = np.linspace(0, 50, 50)
    _, ax = plt.subplots()
    wickets = [i for i in range(1, 11)]
    for (z,w) in zip(model.Z0, wickets):
        Avg_runs = model.get_predictions(overs, z, L = model.L)
        ax.plot(overs, Avg_runs, label = 'wickets =' + str(w))
    ax.set_xlabel('Overs remaining')
    ax.set_ylabel('Average runs obtainable')
    ax.set_title('Run Production Functions')
    ax.legend()
    plt.savefig(plot_path)
    del os.environ['QT_QPA_PLATFORM']
    pass


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    params = []
    params.extend(model.Z0)
    params.append(model.L)
    print('\nModel Parameters are as follows:')
    print("\n", "Wickets in Hand      ", "Z0", "                          L")
    for i in range(len(model.Z0)):
        print("     ", i + 1, "         ", model.Z0[i], "     ", model.L)
    return params

def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    ordered_pairs = data[:-1]
    loss = 0
    samples = 0
    params = []
    params.extend(model.Z0)
    params.append(model.L)
    for i in range(len(model.Z0)):
        run_tuples = ordered_pairs[i]
        runs = np.array([pair[0] for pair in run_tuples])
        overs = np.array([pair[1] for pair in run_tuples])
        loss += model.calculate_loss(params, overs, runs, i + 1) * len(runs)
        samples += len(run_tuples)
    loss /= samples
    if TrainingDone:
        print('\nNormalized Squared Loss over all the datapoints is: ', loss, '\n')
    return loss


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model

    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model,data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)