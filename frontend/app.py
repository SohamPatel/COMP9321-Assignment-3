from flask import Flask, render_template, url_for, request
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    postcode = request.args.get('postcode')
    brand = request.args.get('brand')
    fuel_type = request.args.get('type')
    print("Postcode:", postcode, "\nBrand:", brand, "\nFuel Type:", fuel_type)

    # Use these parameters to perform a GET request to Machine Learning API

    #cheapest_price = 129.54
    #cheapest_date = "17/12/2018"
    return render_template('predictions.html', postcode=postcode, brand=brand, fuel_type=fuel_type, cheapest_price=getCheapestPrice(), cheapest_date=getCheapestDate())

def getCheapestPrice():
    return 129.54;

def getCheapestDate():
    return "17/12/2018";

def getPredication():
    requests.get()
    return None;

if __name__ == '__main__':
    app.run(debug=True)
