#test
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import xlrd
import numpy as np
import pprint

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('test.xlsx', index_col=0)
    data_xls.to_csv('test.csv', encoding='utf-8')

def load_csv(df, split_percentage, droped = []):
    # df = pd.read_csv(diet_path, index_col=0)
    # df.dropna(inplace=True)
    # df = df.reset_index()
    # df = shuffle(df)
    droplist = ['Price','PriceUpdatedDate','FuelCode','Brand','Suburb','Address','ServiceStationName']

    for d in droped:
        replaceValues(df, d)
        droplist.remove(d)

    x = df.drop(droplist, axis=1).values
    y = df['Price'].values


    # Split the dataset in train and test data
    # A random permutation, to split the data randomly

    split_point = int(len(x) * split_percentage)
    X_train = x[:split_point]
    y_train = y[:split_point]
    X_test = x[split_point:]
    y_test = y[split_point:]
    #df.to_csv('outClean.csv')
    return X_train, y_train, X_test, y_test

    '''Test section'''
def read_csv(diet_path):
    '''
    Given a path or a csv file it read the csv file
    And return a pandas df file with the content shuffled content
    '''
    df = pd.read_csv(diet_path, index_col=0)
    df.dropna(inplace=True)
    df = df.reset_index()
    return shuffle(df)

def getNoDupList(df,column):
    '''
    Gets a list of a df's column and removes all duplicates
    '''
    if(column == 'ServiceStationName'):
        column = df.index.tolist()
    else:
        column = df[column].tolist()
    #pprint.pprint(brands)
    noDup = list(set(column))
    return noDup

def replaceValues(df, columnTitle):
    '''
    Replace string values of a df to an integer
    '''
#    noDupList = getNoDupList(df, columnTitle)
#    for i in noDupList:
#        df[columnTitle].replace(to_replace=[i],value=noDupList.index(i),inplace=True)
    df[columnTitle] = LabelEncoder().fit_transform(df[columnTitle].tolist())
    return df

def trainingModel(df, drop = []):
    '''
    Return the mean squared error of a df
    The put values that you want to keep in the drop array
    Values can be ['Price','PriceUpdatedDate','FuelCode','Brand','Suburb','Address','ServiceStationName']
    '''
    X_train, y_train, X_test, y_test = load_csv(df, split_percentage=0.7, droped = drop)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #for i in range(len(y_test)):
        #print("Expected:", y_test[i], "Predicted:", y_pred[i])


    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

    return mean_squared_error(y_test, y_pred)

if __name__ == '__main__':

    #xlsx_to_csv_pd()

    #Finding errors
    postcodeError = list()
    postBrandError = list()
    postFuelError = list()
    brandFuelError = list()
    bankFuelUpdatePirce  = list()
    for i in range(0,50):
        df = read_csv('test.csv')
        postcodeError.append(trainingModel(df, drop = []))
        postBrandError.append(trainingModel(df, drop = ['Brand']))
        postFuelError.append(trainingModel(df, drop = ['FuelCode']))
        brandFuelError.append(trainingModel(df, drop = ['Brand','FuelCode']))
        bankFuelUpdatePirce.append(trainingModel(df, drop = ['Brand','FuelCode','PriceUpdatedDate']))

    for name, information in [("Postcode",postcodeError),
                            ("Postcode & Brand",postBrandError),
                            ("Postcode & Fuel Type",postFuelError),
                            ("PostCode & Fuel Type & Brand",brandFuelError),
                            ("PostCode & Fuel Type & Brand & UpdatePrice",bankFuelUpdatePirce)]:
        pprint.pprint(name)
        pprint.pprint("Average: " + str(np.mean(information)))
        pprint.pprint("Minimum: " + str(np.min(information)))
        pprint.pprint("Maximum: " + str(np.max(information)))
        pprint.pprint("25th Percentile: " + str(np.percentile(information, 25)))
        pprint.pprint("Median: " + str(np.median(information)))
        pprint.pprint("75th Percentile: " + str(np.percentile(information, 75)))
        pprint.pprint("Variance: " + str(np.var(information)))
        pprint.pprint("Standard Deviation: " + str(np.std(information)))
        #pprint.pprint(information)
        print()
