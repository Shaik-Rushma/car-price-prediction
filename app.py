from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

data = pd.read_csv("car_data.csv")

data["Car_Age"] = 2026 - data["Year"]

data.drop(["Car_Name","Year"], axis=1, inplace=True)

le = LabelEncoder()

data["Fuel_Type"] = le.fit_transform(data["Fuel_Type"])
data["Selling_type"] = le.fit_transform(data["Selling_type"])
data["Transmission"] = le.fit_transform(data["Transmission"])

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    present_price = float(request.form["present_price"])
    kms = int(request.form["kms"])
    fuel = int(request.form["fuel"])
    seller = int(request.form["seller"])
    transmission = int(request.form["transmission"])
    owner = int(request.form["owner"])
    age = int(request.form["age"])

    prediction = model.predict([[
        present_price,
        kms,
        fuel,
        seller,
        transmission,
        owner,
        age
    ]])

    return render_template(
        "index.html",
        prediction_text=f"Estimated Car Price: ₹ {round(prediction[0],2)} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)