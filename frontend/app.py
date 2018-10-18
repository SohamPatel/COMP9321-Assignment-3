from flask import Flask, render_template, url_for, request
import requests

app = Flask(__name__)
URL = "http://127.0.0.1:5100"

def sendRequest(url, params):
    #url = "{url}/getBrands?{args}".format(url = URL, args = urllib.parse.urlencode(args))
    try:
        r = requests.get(url = url, params = params )
        return r.json()
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print("Turn on API")
        #sys.exit(1)
    return None;

def getBrand(postcode):
    args = {"postcode" : postcode}
    return sendRequest("{}/getBrands".format(URL),args)

def getFuelTypes(postcode, brand):
    args = {"postcode" : postcode, "brand" : brand}
    return sendRequest("{}/getFuelTypes".format(URL), args)

def getFuelPredictions(postcode, brand, fuel):
    args = {"postcode" : postcode, "brand" : brand, "fueltype" : fuel}
    return sendRequest("{}/getFuelPredictions".format(URL), args)

def getCheapestPrice():
    return 129.54;

def getCheapestDate():
    return "17/12/2018";

def getPredication():
    return None;


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
    data = requests.get(f'http://127.0.0.1:5100/getFuelPredictions?postcode={postcode}&brand={brand}&fueltype={fuel_type}').json()
    predictions = data['predictions']

    cheapest_price = "%.2f" % round(float(predictions[0]['predicted_price']), 2)
    cheapest_date = predictions[0]['date']

    for prediction in predictions:
        prediction['predicted_price'] = "%.2f" % round(float(prediction['predicted_price']), 2)
        if (prediction['predicted_price'] < cheapest_price):
            cheapest_price = temp_price

    return render_template('predictions.html', postcode=postcode, brand=brand, fuel_type=fuel_type, cheapest_price=cheapest_price, cheapest_date=cheapest_date, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
