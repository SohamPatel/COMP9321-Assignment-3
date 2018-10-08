import json
import requests
import pandas as pd
from flask import Flask, request
from flask_restplus import Resource, Api, fields, inputs, reqparse

app = Flask(__name__)
api = Api(app,
        version='1.0',
        default="Fuel Price"
        title='NSW fuel prices prediction data service',
        description='Data service which provide users a prediction of how the fuel prices will change in the future so they are able to make a better judgement on when to refill their fuel tanks.',
)

@api.route('/fuelprice/<int:postcode>/<string:fueltype>/<string:brand>')
class FuelPrice(Resource):
    @api.doc(description="Gets all predicted prices")
    def get(self,postocde,fueltype,brand):
        return None;

if __name__ == '__main__':
    # run the application
    app.run(debug=True)
