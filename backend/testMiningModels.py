import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.utils import shuffle
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_xlsx(xlsx_file):
    df = pd.read_excel(xlsx_file, index_col=0)
    df.dropna(inplace=True)
    df = df.reset_index()
    df = shuffle(df, random_state=0)
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
    x_train, y_train, x_test, y_test = split_df(df, not_dropping, 0.7)

    # standardise x values as features vary greatly -- doesn't do anything?
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    global lowest_error, combination, highest_score, r2_combination
    global rf_lowest_error, rf_combination, rf_highest_score, rf_r2_combination
    global regression, random_forest

    if regression:
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        # test model
        y_pred = model.predict(x_test)

        print("---Linear Regression---")

        # The mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean squared error: %.2f" % mse)

        # Variance score
        # var = explained_variance_score(y_test, y_pred)
        # print("Explained variance score: %.2f\n" % var)

        # r2 score
        r2 = r2_score(y_test, y_pred)
        print("r2 score: %.2f\n" % r2)

        if mse < lowest_error:
            lowest_error = mse
            combination = not_dropping

        if r2 > highest_score:
            highest_score = r2
            r2_combination = not_dropping

    if random_forest:
        rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)

        print("---Random Forest---")

        # The mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean squared error: %.2f" % mse)

        r2 = r2_score(y_test, y_pred)
        print("r2 score: %.2f\n" % r2)

        if mse < rf_lowest_error:
            rf_lowest_error = mse
            rf_combination = not_dropping

        if r2 > rf_highest_score:
            rf_highest_score = r2
            rf_r2_combination = not_dropping


def testMoreRegression():
    x_train, y_train, x_test, y_test = split_df(df, ['Postcode', 'Brand', 'FuelCode', 'Address', 'PriceUpdatedDate'], 0.7)
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=10, random_state=21)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

if __name__ == '__main__':

    # global vars to determine lowest error and highest score
    lowest_error = int(sys.maxsize)
    combination = []
    rf_lowest_error = int(sys.maxsize)
    rf_combination = []
    highest_score = 0
    r2_combination = []
    rf_highest_score = 0
    rf_r2_combination = []

    # global vars to determine which models to use
    regression = True
    random_forest = True

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

    # BELOW attempts to calculate mean from past 3 days
    # HOWEVER - generation is VERY SLOW!
    # df['S_3'] = np.nan
    # for index, rows in df.iterrows():
    #     if not df['S_3'][index]:
    #         continue

    #     selected_df = df.query(f'Postcode == {rows["Postcode"]} and Brand == {rows["Brand"]} and FuelCode == {rows["FuelCode"]} and Address == {rows["Address"]}')
    #     # this resets index to start from 0
    #     # and stores prev index in col = 'index'
    #     selected_df.reset_index(inplace=True)

    #     # print(selected_df.index)
    #     # for i, r in selected_df.iterrows():
    #     #     print(",".join([str(r[column]) for column in selected_df]))

    #     for i in range(len(selected_df)):
    #         if i > 2:
    #             # set df with index stored in selected index S_3 value to the mean of past 3
    #             df.loc[selected_df['index'][i], 'S_3'] = selected_df['Price'][i-3:i].mean()
    #         else:
    #             df.loc[selected_df['index'][i], 'S_3'] = None

    #     # selected_df = df.query(f'Postcode == {rows["Postcode"]} and Brand == {rows["Brand"]} and FuelCode == {rows["FuelCode"]} and Address == {rows["Address"]}')
    #     # for i, r in selected_df.iterrows():
    #     #     print(",".join([str(r[column]) for column in selected_df]))
    #     # break

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

    print("===RESULTS===")
    print("---Linear Regression---")
    print("Lowest Error Combination: " + ' '.join(combination))
    print("Highest r2 Score Combination: " + ' '.join(r2_combination))
    print("---Random Forest---")
    print("Lowest Error Combination: " + ' '.join(rf_combination))
    print("Highest r2 Score Combination: " + ' '.join(rf_r2_combination))

    print()
    testMoreRegression()
