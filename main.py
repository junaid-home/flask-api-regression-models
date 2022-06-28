from pickle import load
from flask import Flask, request, jsonify

app = Flask(__name__)

slr = load(open("trained_slr_model.pkl", "rb"))
mlr = load(open("trained_mlr_model.pkl", "rb"))


@app.route("/slr/predict", methods=["POST"])
def slr_pridict():
    data = request.json

    rooms = int(data["rooms"])

    predicted_price = slr.predict([[rooms]])

    return jsonify({
        "Status_Code": 200,
        "Model_Name": "Simple Linear Regression",
        "Predicted_Price": predicted_price[0][0],
    }), 200


@app.route("/mlr/predict", methods=["POST"])
def mlr_predict():
    data = request.json
    bedrooms = int(data["bedrooms"])
    bathrooms = int(data["bathrooms"])
    floors = int(data["floors"])
    condition = int(data["condition"])

    predicted_price = mlr.predict(
        [[bedrooms, bathrooms, floors, condition]])

    return jsonify({
        "Status_Code": 200,
        "Model_Name": "Multiple Linear Regression",
        "Predicted_Price": predicted_price[0],
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
