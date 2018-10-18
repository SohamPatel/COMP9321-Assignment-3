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