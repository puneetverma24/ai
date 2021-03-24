# Importing libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor


# K Nearest Neighbors Regression

class K_Nearest_Neighbors_Regressor():

    def __init__(self, K):

        self.K = K

    # Function to store training set

    def fit(self, X_train, Y_train):

        self.X_train = X_train

        self.Y_train = Y_train

        # no_of_training_examples, no_of_features

        self.m, self.n = X_train.shape

    # Function for prediction

    def predict(self, X_test):

        self.X_test = X_test

        # no_of_test_examples, no_of_features

        self.m_test, self.n = X_test.shape

        # initialize Y_predict

        Y_predict = np.zeros(self.m_test)

        for i in range(self.m_test):
            x = self.X_test[i]

            # find the K nearest neighbors from current test example

            neighbors = np.zeros(self.K)

            neighbors = self.find_neighbors(x)

            # calculate the mean of K nearest neighbors

            Y_predict[i] = np.mean(neighbors)

        return Y_predict

    # Function to find the K nearest neighbors to current test example

    def find_neighbors(self, x):

        # calculate all the euclidean distances between current test
        # example x and training set X_train

        euclidean_distances = np.zeros(self.m)

        for i in range(self.m):
            d = self.euclidean(x, self.X_train[i])

            euclidean_distances[i] = d

        # sort Y_train according to euclidean_distance_array and
        # store into Y_train_sorted

        inds = euclidean_distances.argsort()

        Y_train_sorted = self.Y_train[inds]

        return Y_train_sorted[:self.K]

    # Function to calculate euclidean distance

    def euclidean(self, x, x_train):

        return np.sqrt(np.sum(np.square(x - x_train)))


# Driver code

def main():
    # Importing dataset
    df = pd.read_csv("../data/salary_data.csv")

    X = df.iloc[:, :-1].values   # initialize value in X
    Y = df.iloc[:, 1].values     # initialize value in y


    # Splitting dataset into train and test set,we are using 1/3(20 to train , 10 to test) of data for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    # Model training

    model = K_Nearest_Neighbors_Regressor(K=2)

    model.fit(X_train, Y_train)



    # Prediction on test set

    Y_pred = model.predict(X_test)

    print("Predicted values by our model: ", np.round(Y_pred[:10], 2))
    print("Real values					 : ", Y_test[:10])

if __name__ == "__main__":
    main()
