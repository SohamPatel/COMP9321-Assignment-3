import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
import sys

def load_xlsx(xlsx_file):
    df = pd.read_excel(xlsx_file, index_col=0)
    df.dropna(inplace=True)
    df = df.reset_index()
    df = shuffle(df)
    return df


def split_df(df, not_dropping, split_percentage):
    drop_list = ['Price','PriceUpdatedDate', 'Postcode',
                'FuelCode','Brand','Suburb','Address','ServiceStationName']
    for col in not_dropping:
        if col in drop_list:
            drop_list.remove(col)
        else:
            print("I shouldn't have gone here!")

    x = df.drop(drop_list, axis=1).values
    y = df['Price'].values

    # Split the dataset in train and test data
    # A random permutation, to split the data randomly

    split_point = int(len(x) * split_percentage)
    X_train = x[:split_point]
    y_train = y[:split_point]
    X_test = x[split_point:]
    y_test = y[split_point:]

    return X_train, y_train, X_test, y_test

def train_model(not_dropping):
    model = linear_model.LinearRegression()
    x_train, y_train, x_test, y_test = split_df(df, not_dropping, 0.7)
    model.fit(x_train, y_train)

    # test model
    y_pred = model.predict(x_test)

    # The mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: %.2f\n" % mse)

    global lowest_error, combination
    if mse < lowest_error:
        lowest_error = mse
        combination = not_dropping

if __name__ == '__main__':

    # global var to determine lowest error
    lowest_error = int(sys.maxsize)
    combination = []

    df = load_xlsx("test.xlsx")

    # drop columns we will not use
    df.drop(['Suburb', 'ServiceStationName'], axis=1)

    # remove time component from dates yyyy/mm/dd hh:mm:ss:SS
    df['PriceUpdatedDate'] = df['PriceUpdatedDate'].astype(str).str.replace(r' .*', '', regex=True)

    # convert strings to ints
    le = preprocessing.LabelEncoder()
    df['Brand'] = le.fit_transform(df['Brand'].values)
    df['FuelCode'] = le.fit_transform(df['FuelCode'].values)
    df['Address'] = le.fit_transform(df['Address'].values)
    df['PriceUpdatedDate'] = le.fit_transform(df['PriceUpdatedDate'].values)

    # Use Postcode as only parameter
    print("Params: Postcode")
    train_model(['Postcode'])

    # Use Brand as only parameter
    print("Params: Brand")
    train_model(['Brand'])

    # Use FuelCode as only parameter
    print("Params: FuelCode")
    train_model(['FuelCode'])

    # Use Address as only parameter
    print("Params: Address")
    train_model(['Address'])

    # Use Date as only parameter
    print("Params: Date")
    train_model(['PriceUpdatedDate'])

    # Use Postcode & Brand as parameters
    print("Params: Postcode, Brand")
    train_model(['Postcode', 'Brand'])

    # Use Postcode & FuelCode as parameters
    print("Params: Postcode, FuelCode")
    train_model(['Postcode', 'FuelCode'])

    # Use Postcode & Address as parameters
    print("Params: Postcode, Address")
    train_model(['Postcode', 'Address'])

    # Use Postcode & Date as parameters
    print("Params: Postcode, Date")
    train_model(['Postcode', 'PriceUpdatedDate'])

    # Use Brand & FuelCode as parameters
    print("Params: Brand, FuelCode")
    train_model(['Brand', 'FuelCode'])

    # Use Brand & Address as parameters
    print("Params: Brand, Address")
    train_model(['Brand', 'Address'])

    # Use Brand & Date as parameters
    print("Params: Brand, Date")
    train_model(['Brand', 'PriceUpdatedDate'])

    # Use FuelCode & Address as parameters
    print("Params: FuelCode, Address")
    train_model(['FuelCode', 'Address'])

    # Use FuelCode & Date as parameters
    print("Params: FuelCode, Date")
    train_model(['FuelCode', 'PriceUpdatedDate'])

    # Use Address & Date as parameters
    print("Params: Date, Address")
    train_model(['PriceUpdatedDate', 'Address'])

    # Use Postcode, Brand, FuelCode as parameters
    print("Params: Postcode, Brand, FuelCode")
    train_model(['Postcode', 'Brand', 'FuelCode'])

    # Use Postcode, Brand, Address as parameters
    print("Params: Postcode, Brand, Address")
    train_model(['Postcode', 'Brand', 'Address'])

    # Use Postcode, Brand, Date as parameters
    print("Params: Postcode, Brand, Date")
    train_model(['Postcode', 'Brand', 'PriceUpdatedDate'])

    # Use Brand, FuelCode, Address as parameters
    print("Params: Brand, FuelCode, Address")
    train_model(['Brand', 'FuelCode', 'Address'])

    # Use Brand, FuelCode, Date as parameters
    print("Params: Brand, FuelCode, Date")
    train_model(['Brand', 'FuelCode', 'PriceUpdatedDate'])

    # Use FuelCode, Address, Date as parameters
    print("Params: FuelCode, Address, Date")
    train_model(['FuelCode', 'Address', 'PriceUpdatedDate'])

    # Use Postcode, Brand, FuelCode, Address as parameters
    print("Params: Postcode, Brand, FuelCode, Address")
    train_model(['Postcode', 'Brand', 'FuelCode', 'Address'])

    # Use Postcode, Brand, FuelCode, Address as parameters
    print("Params: Postcode, Brand, FuelCode, Address")
    train_model(['Postcode', 'Brand', 'FuelCode', 'Address'])

    # Use Postcode, FuelCode, Address, Date as parameters
    print("Params: Postcode, FuelCode, Address, Date")
    train_model(['Postcode', 'FuelCode', 'Address', 'PriceUpdatedDate'])

    # Use Postcode, Brand, Address, Date as parameters
    print("Params: Postcode, Brand, Address, Date")
    train_model(['Postcode', 'Brand', 'Address', 'PriceUpdatedDate'])

    # Use Postcode, Brand, FuelCode, Date as parameters
    print("Params: Postcode, Brand, FuelCode, Date")
    train_model(['Postcode', 'Brand', 'FuelCode', 'PriceUpdatedDate'])

    # Use Brand, FuelCode, Address, Date as parameters
    print("Params: Brand, FuelCode, Address, Date")
    train_model(['Brand', 'FuelCode', 'Address', 'PriceUpdatedDate'])

    # Use Postcode, Brand, Address, Date as parameters
    print("Params: Postcode, Brand, FuelCode, Address, Date")
    train_model(['Postcode', 'Brand', 'FuelCode', 'Address', 'PriceUpdatedDate'])

    print("Lowest Error Combination: " + ' '.join(combination))
