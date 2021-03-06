"""
William Austin
Prakash Dhimal
George Mason University
CS 584 Theory and Applications of Data Mining
Semester project: Predicting the Impact of COVID-19
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import Common as common
import Utilities as util

if __name__ == '__main__':
    tsData = common.getTimeSeriesData()

    X, y = util.buildSlidingWindowTrainingSet(tsData, 31, 30)

    print("Finished building training set.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    print("Finished splitting train/test sets")
    # y_train = np.reshape(y_train, newshape=(-1, 1))
    # y_test = np.reshape(y_test, newshape=(-1, 1))

    # reg = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 4, 2), random_state=1, solver='lbfgs')
    reg = MLPRegressor(alpha=1e-07, hidden_layer_sizes=(5, 4, 3), random_state=1, solver='lbfgs')
    reg.fit(X_train, y_train)
    print("Finished training!!")

    y_predicted = reg.predict(X_test)

    print("MSE = " + str(mean_squared_error(y_test, y_predicted)))
    print("Done predicting with NN.")
