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

    return render_template('predictions.html')

if __name__ == '__main__':
    app.run(debug=True)