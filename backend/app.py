import json
import requests
import pandas as pd
from flask import Flask, request
from flask_restplus import Resource, Api, fields, inputs, reqparse

app = Flask(__name__)
api = Api(app,
        version='1.0',
        default="Fuel Price",
        title='NSW fuel prices prediction data service',
        description='Data service which provide users a prediction of how the fuel prices will change in the future so they are able to make a better judgement on when to refill their fuel tanks.',
)

brand_parser = reqparse.RequestParser()
brand_parser.add_argument('postcode', type=int)

fuel_type_parser = reqparse.RequestParser()
fuel_type_parser.add_argument('postcode', type=int)
fuel_type_parser.add_argument('brand', type=str)

fuel_price_parser = reqparse.RequestParser()
fuel_price_parser.add_argument('postcode',type = int)
fuel_price_parser.add_argument('brand',type = str)
fuel_price_parser.add_argument('fueltype', type =str)

drop_list = ['ServiceStationName', 'Address', 'Suburb']
df = pd.read_csv('test.csv')

@api.route('/fuelprice/<int:postcode>/<string:fueltype>/<string:brand>')
class FuelPrice(Resource):
    @api.response(404, 'Not found')
    @api.response(200, 'Successful')
    @api.doc(description="Gets all predicted prices given a postcode, type of fuel and its brand")
    def get(self,postcode,fueltype,brand):
        return None

@api.route('/getBrands')
class FuelBrands(Resource):
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

class FuelPrice(Resource):
    @api.expect(fuel_price_parser, validate=True)
    @api.make_response(400,'Not found')
    @api.make_response(200,'Successful')
    @api.doc(description='Get all fuel price for the given postcode、brand and fuel type')
    def get(self):
        args = fuel_price_parser.parse_args()
        postcode = args.get('postcode')
        brand = args.get('brand')
        fueltype = args.get('fueltype')

        fuelprice_df = df.query(f'Postcod == {postcode} and  Brand == {brand} and fueltype = {fueltype}')
        print(fuelprice_df.empty)
        if fuelprice_df.empty:
             # Invalia postcode and/or brand and/or fueltype
            output_response = {
                "message" :f'No Fuel Price found using the given arguments! please check if it is a valid Postcode、Brand and FuelType'
            }
            return output_response,400

        else:
            # keep fuel price column only
            fuelprice_df = fuelprice_df[['Fuelprice']]
            # remove duplicate fuel types
            fuelprice_df = fuelprice_df.drop_duplicates('FuelPrice')
            fuel_prices = list(fuelprice_df['FuelPrice'])
            return fuel_prices,200




if __name__ == '__main__':
    df = df.drop(columns=drop_list)
    print (df)

    # run the application
    app.run(debug=True)
