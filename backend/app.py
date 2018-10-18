import json
import requests
import pandas as pd
from flask import Flask, request
from flask_restplus import Resource, Api, fields, inputs, reqparse,abort
from sklearn.utils import shuffle
from sklearn import preprocessing, linear_model
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import numpy as np
from itsdangerous import SignatureExpired, JSONWebSignatureSerializer, BadSignature
from functools import wraps
from time import time

class AuthenticationToken:
    def __init__(self, secret_key, expires_in):
        self.secret_key = secret_key
        self.expires_in = expires_in
        self.serializer = JSONWebSignatureSerializer(secret_key)

    def generate_token(self, username,password):
        info = {
            'username': username,
            'password': password,
            'creation_time': time()
        }

        token = self.serializer.dumps(info)
        return token.decode()

    def validate_token(self, token):
        info = self.serializer.loads(token.encode())

        if time() - info['creation_time'] > self.expires_in:
            raise SignatureExpired("The Token has been expired; get a new token")

        return info['username']

SECRET_KRY = 'this is test for authentication token'
expires_in = 600
auth = AuthenticationToken(SECRET_KRY,expires_in)


app = Flask(__name__)
api = Api(app, authorizations={
    'API_KEY':{
        'type' :'apiKey',
        'in' : 'header',
        'name' : 'AUTH-TOKEN'

    }
},
        security='API_KEY',
        version='1.0',
        default="Fuel Price",
        title='NSW fuel prices prediction data service',
        description='Data service which provide users a prediction of how the fuel prices will change in the future so they are able to make a better judgement on when to refill their fuel tanks.',
)

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        token = request.headers.get('AUTH-TOKEN')
        if not token:
            abort(401, 'Authentication token is missing')

        try:
            user = auth.validate_token(token)
        except SignatureExpired as e:
            abort(401, e.message)
        except BadSignature as e:
            abort(401, e.message)

        return f(*args, **kwargs)

    return decorated



dataset_file = 'Fuel_Dataset.xlsx'


brand_parser = reqparse.RequestParser()
brand_parser.add_argument('postcode', type=int)

fuel_type_parser = reqparse.RequestParser()
fuel_type_parser.add_argument('postcode', type=int)
fuel_type_parser.add_argument('brand', type=str)

fuel_price_parser = reqparse.RequestParser()
fuel_price_parser.add_argument('postcode',type = int)
fuel_price_parser.add_argument('brand',type = str)
fuel_price_parser.add_argument('fueltype', type =str)

credential_model = api.model('credential', {
    'username': fields.String,
    'password': fields.String
})

credential_parser = reqparse.RequestParser()
credential_parser.add_argument('username', type=str)
credential_parser.add_argument('password', type=str)

#df = pd.read_csv("UserInformation.csv", index_col=0)
#usernameList = df['username'].values

@api.route('/token')
class Token(Resource):
    @api.response(200, 'Successful')
    @api.doc(description="Generates a authentication token")
    @api.expect(credential_parser, validate=True)
    def get(self):
        args = credential_parser.parse_args()

        username = args.get('username')
        password = args.get('password')

        username = [username]
        df_username = pd.read_csv('UserInformation.csv')
        usernamelist = np.array(df_username[['username']]).tolist()
        password_df = df_username.query(f'username == {username}')

        if password_df.empty or username not in usernamelist:
            output_response = {
                "message :"f'username/password is incorrect'
            }
            return output_response, 401
        else:
            return {"token":auth.generate_token(username,password)}


@api.route('/getFuelPredictions')
class FuelPrice(Resource):
    @requires_auth
    @api.expect(fuel_price_parser, validate=True)
    @api.response(404, 'Not found')
    @api.response(200, 'Successful')
    @api.doc(description="Gets all predicted prices given a postcode, type of fuel and its brand")
    def get(self):
        args = fuel_price_parser.parse_args()
        postcode = args.get('postcode')
        brand  = args.get('brand')
        fuelcode = args.get('fueltype')

        check_df = df.query(f'Postcode == {postcode} and Brand == "{brand}" and FuelCode == "{fuelcode}"')
        if check_df.empty:
            # Invalid postcode, brand and/or fuelcoded
            output_response = {
                "message" : f"Unable to predict using the given arguments! Please check if its a valid Postcode, Brand and Fuel type."
            }
            return output_response, 404

        else:
            dates = []
            test_data = []
            today_date = datetime.now()
            transform_brand = list(brand_le.transform([brand]))[0]
            transform_fuelcode = list(fuelcode_le.transform([fuelcode]))[0]
            for i in range(1, 15):
                curr_date = today_date + timedelta(days=i)
                curr_date = curr_date.strftime('%Y/%m/%d')
                dates.append(curr_date)

                # need to convert new date to label. needs to be fitted into encoder
                all_dates = list(date_le.classes_)
                all_dates.append(curr_date)
                date_le.fit_transform(all_dates)
                transform_date = list(date_le.transform([curr_date]))[0]
                test_data.append(np.array([postcode, transform_brand, transform_fuelcode, transform_date]))

            prediction_results = model.predict(np.array(test_data))

            prediction_list = []
            for i in range(len(dates)):
                entry = {
                    "date": dates[i],
                    "predicted_price": prediction_results[i]
                }
                prediction_list.append(entry)

            prediction_prices = {
                "postcode": postcode,
                "brand": brand,
                "fuelcode": fuelcode,
                "predictions": prediction_list
            }

            return prediction_prices, 200

@api.route('/getBrands')
class FuelBrands(Resource):
    @requires_auth
    @api.expect(brand_parser, validate=True)
    @api.response(404, 'Not found')
    @api.response(200, 'Successful')
    @api.doc(description="Get all brands for the given postcode")
    def get(self):
        args = brand_parser.parse_args()
        postcode = args.get('postcode')

        brand_df = df.query(f'Postcode == {postcode}')

        if brand_df.empty:
            # Invalid postcode and/or brand
            output_response = {
                "message" : f"No Brands found using the given arguments! Please check if its a valid Postcode."
            }
            return output_response, 404

        else:
            # Keep brand column only
            brand_df = brand_df[['Brand']]
            # Remove duplicate brands
            brand_df = brand_df.drop_duplicates('Brand')

            brands = list(brand_df['Brand'])
            return brands, 200

@api.route('/getFuelTypes')
class FuelTypes(Resource):
    @requires_auth
    @api.expect(fuel_type_parser, validate=True)
    @api.response(404, 'Not found')
    @api.response(200, 'Successful')
    @api.doc(description="Get all fuel types for the given postcode and brand")
    def get(self):
        args = fuel_type_parser.parse_args()
        postcode = args.get('postcode')
        brand = args.get('brand')

        fueltype_df = df.query(f'Postcode == {postcode} and Brand == "{brand}"')
        print(fueltype_df.empty)
        if fueltype_df.empty:
            # Invalid postcode and/or brand
            output_response = {
                "message" : f"No Fuel Types found using the given arguments! Please check if its a valid Postcode and Brand."
            }
            return output_response, 404

        else:
            # Keep fuel type column only
            fueltype_df = fueltype_df[['FuelCode']]
            # Remove duplicate fuel types
            fueltype_df = fueltype_df.drop_duplicates('FuelCode')

            fuel_types = list(fueltype_df['FuelCode'])
            return fuel_types, 200

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
    # return x[:int(len(x)*0.05)], y[:int(len(y)*0.05)]
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

    # run the application
app.run(debug=True, port=5100)