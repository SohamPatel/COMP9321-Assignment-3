from flask import Flask, render_template, url_for, request

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

    cheapest_price = 129.54
    cheapest_date = "17/12/2018"
    return render_template('predictions.html', postcode=postcode, brand=brand, fuel_type=fuel_type, cheapest_price=cheapest_price, cheapest_date=cheapest_date)

if __name__ == '__main__':
    app.run(debug=True)