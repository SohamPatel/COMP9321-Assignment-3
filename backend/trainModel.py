import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing, linear_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.externals import joblib

dataset_file = 'Fuel_Dataset.xlsx'

def load_xlsx(xlsx_file):
    df = pd.read_excel(xlsx_file, index_col=0)
    df.dropna(inplace=True)
    df = df.reset_index()
    return df

def preprocess_data(df):
    df = shuffle(df)

    # remove time component from dates yyyy/mm/dd hh:mm:ss:SS
    df['PriceUpdatedDate'] = df['PriceUpdatedDate'].astype(str).str.replace(r' .*', '', regex=True)

    # convert strings to ints
    df['Brand'] = brand_le.fit_transform(df['Brand'].values)
    df['FuelCode'] = fuelcode_le.fit_transform(df['FuelCode'].values)
    df['PriceUpdatedDate'] = date_le.fit_transform(df['PriceUpdatedDate'].values)

    x = df.drop('Price', axis=1).values
    y = df['Price'].values

    # uncomment below when debugging, just so it loads faster
    # return x[0:int(len(x)*0.05)], y[0:int(len(y)*0.05)]
    return x, y

if __name__ == '__main__':
    drop_list = ['ServiceStationName', 'Address', 'Suburb']
    df = load_xlsx(dataset_file)
    df = df.drop(columns=drop_list)

    brand_le = preprocessing.LabelEncoder()
    fuelcode_le = preprocessing.LabelEncoder()
    date_le = preprocessing.LabelEncoder()

    # x = params, y = labels (i.e price)
    x_train, y_train = preprocess_data(df)
    #model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    #model.fit(x_train, y_train)

    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    # save the model to disk
    filename = 'prefuel_model.sav'
    joblib.dump(model, filename)

    print("COMPLETE!")
