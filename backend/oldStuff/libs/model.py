import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_PATH = 'ml_models/model.pkl'


class MLModel:
    def __init__(self):
        # Load the model from the given path
        with open(MODEL_PATH, 'rb') as model_file:
            self.model = pickle.load(model_file)
        self.X = []
        self.Y = []

    def train(self, x, y, data_path='data/10_points.csv'):
        # Prepare data set
        df = pd.read_csv(data_path)
        df_new = pd.DataFrame({
            'x': x,
            'y': y
        })
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(data_path, index=False)
        # Reformat the data
        self.X = df['x'].values.reshape(-1, 1)  # values converts it into a numpy array
        self.Y = df['y'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

        # Train the model
        linear_regressor = LinearRegression()
        linear_regressor.fit(self.X, self.Y)
        # Set and Export the model
        self.model = linear_regressor
        pickle.dump(linear_regressor, open(MODEL_PATH, 'wb'))

    def predict(self, x, model_path='data/model.pkl'):
        if isinstance(x, np.ndarray):  # Ensure correct shape
            if x.ndim == 1:
                x = x.reshape(1, -1)  # Reshape if it's a single sample (1D array)
        elif isinstance(x, list):
            x = np.array(x).reshape(1, -1)  # Convert a list to a 2D numpy array if necessary

        # Make the prediction
        y = self.model.predict(x)
        return y

    def load_csv(self, data_path='data/10_points.csv'):
        # Prepare data set
        df = pd.read_csv(data_path)
        # Reformat the data
        self.X = df['x'].values.reshape(-1, 1)  # values converts it into a numpy array
        self.Y = df['y'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        # Train the model
        self.model = self.model.fit(self.X, self.Y)
        pickle.dump(self.model, open(MODEL_PATH, 'wb'))