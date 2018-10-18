from flask import Flask, render_template, url_for, request, jsonify
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

def getCheapest(data):
    '''
    Data is a list of dictionary
    E.g [{'date': '2018/10/19', 'predicted_price': 150.5233789566008},
         {'date': '2018/10/20', 'predicted_price': 150.53885414912406}]
    '''
    minPricedItem = min(data, key=lambda x:x['predicted_price'])

    return minPricedItem['date'], round(float(minPricedItem['predicted_price']), 2);

def getExpensive(data):
    maxPricedItem = max(data, key=lambda x:x['predicted_price'])
    return maxPricedItem['date'], round(float(maxPricedItem['predicted_price']), 2);

def roundPrice(predictions):
    for i in range(0,len(predictions)):
        predictions[i]['predicted_price'] = round(float(predictions[i]['predicted_price']),2)
    return predictions


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    postcode = request.args.get('postcode')
    brand = request.args.get('brand')
    fuel_type = request.args.get('type')
    print("Postcode:", postcode, "\nBrand:", brand, "\nFuel Type:", fuel_type)
    '''
    # Use these parameters to perform a GET request to Machine Learning API
    data = requests.get(f'http://127.0.0.1:5100/getFuelPredictions?postcode={postcode}&brand={brand}&fueltype={fuel_type}').json()
    predictions = data['predictions']

    cheapest_price = "%.2f" % round(float(predictions[0]['predicted_price']), 2)
    cheapest_date = predictions[0]['date']

    for prediction in predictions:
        prediction['predicted_price'] = "%.2f" % round(float(prediction['predicted_price']), 2)
        if (prediction['predicted_price'] < cheapest_price):
            cheapest_price = temp_price
    '''
    data = getFuelPredictions(postcode, brand, fuel_type)
    predictions = data['predictions']
    predictions = roundPrice(predictions)
    cheapest_date,cheapest_price = getCheapest(predictions)

    return render_template('predictions.html', postcode=postcode, brand=brand, fuel_type=fuel_type, cheapest_price=cheapest_price, cheapest_date=cheapest_date, predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
