#test
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import xlrd
import numpy as np

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('test.xlsx', index_col=0)
    data_xls.to_csv('test.csv', encoding='utf-8')

def load_csv(diet_path, split_percentage):
    df = pd.read_csv(diet_path, index_col=0)
    df.dropna(inplace=True)
    df = df.reset_index()
    df = shuffle(df)
    x = df.drop(['Price','PriceUpdatedDate','FuelCode','Brand','Suburb','Address','ServiceStationName'], axis=1).values
    y = df['Price'].values

    # Split the dataset in train and test data
    # A random permutation, to split the data randomly

    split_point = int(len(x) * split_percentage)
    X_train = x[:split_point]
    y_train = y[:split_point]
    X_test = x[split_point:]
    y_test = y[split_point:]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':

    xlsx_to_csv_pd()

    X_train, y_train, X_test, y_test = load_csv("test.csv", split_percentage=0.7)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    for i in range(len(y_test)):
        print("Expected:", y_test[i], "Predicted:", y_pred[i])

    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(y_test, y_pred))

