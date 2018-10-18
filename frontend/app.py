from flask import Flask, render_template, url_for, request, jsonify
import requests

app = Flask(__name__)
URL = "http://127.0.0.1:5100"
api_token = None
auth_credentials = {
    'username' : 'Tony',
    'passowrd' : '123456'
}

def sendRequest(url, params):
    try:
        api_token = requests.get(url = "{}/token".format(URL), params = auth_credentials).json()

        if 'token' not in api_token: # Check if unable to retrieve token
            print("Uable to get a authorisation token. Invalid credentials.")
            return None

        api_token = api_token['token']
        r = requests.get(url = url, params = params, headers={'AUTH-TOKEN' : api_token})
        return r.json()

    except requests.exceptions.RequestException:  # This is the correct syntax
        # unable to authenticate
        return None

def getBrand(postcode):
    args = {"postcode" : postcode}
    return sendRequest(f"{URL}/getBrands", args)

def getFuelTypes(postcode, brand):
    args = {"postcode" : postcode, "brand" : brand}
    return sendRequest(f"{URL}/getFuelTypes", args)

def getFuelPredictions(postcode, brand, fuel):
    args = {"postcode" : postcode, "brand" : brand, "fueltype" : fuel}
    return sendRequest(f"{URL}/getFuelPredictions", args)

def getCheapest(data):
    '''
    Data is a list of dictionary
    E.g [{'date': '2018/10/19', 'predicted_price': 150.5233789566008},
         {'date': '2018/10/20', 'predicted_price': 150.53885414912406}]
    '''
    minPricedItem = min(data, key=lambda x:x['predicted_price'])

    return minPricedItem['date'], round(float(minPricedItem['predicted_price']), 2)

def getExpensive(data):
    maxPricedItem = max(data, key=lambda x:x['predicted_price'])
    return maxPricedItem['date'], round(float(maxPricedItem['predicted_price']), 2)

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
            cheapest_price = prediction['predicted_price']
            cheapest_price = temp_price
    '''
    data = getFuelPredictions(postcode, brand, fuel_type)
    if data:
        # Successfully retrieved predictions
        predictions = data['predictions']
        predictions = roundPrice(predictions)
        cheapest_date,cheapest_price = getCheapest(predictions)

        # Convert to datasets
        date_dataset = list(map(lambda x: x['date'], predictions))
        price_dataset = list(map(lambda x : x['predicted_price'], predictions))
        point_radius = list()
        point_colour = list()

        for price in price_dataset:
            if (price == cheapest_price):
                point_radius.append(7)
                point_colour.append('#4CAF50')
            else:
                point_radius.append(3)
                point_colour.append('#3E78C2')

    else:
        # Unsuccessful retrieval of predctions
        predictions = [{"date" : "No date found.", "predicted_price" : "No prediction available."}]
        cheapest_date = "No date found."
        cheapest_price = "No price found."

        # Convert to datasets
        date_dataset = list()
        price_dataset = list()
        point_radius = list()

    return render_template('predictions.html', postcode=postcode, brand=brand, fuel_type=fuel_type, 
                            cheapest_price=cheapest_price, cheapest_date=cheapest_date, predictions=predictions, 
                            date_dataset=date_dataset, price_dataset=price_dataset, point_radius=point_radius,
                            point_colour=point_colour)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
