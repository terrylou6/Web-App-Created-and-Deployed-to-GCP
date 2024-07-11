from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        warehouse = request.form["warehouse (Input Integer 0-6)"]
        holiday_name = request.form["holiday_name (Input Integer 0-24)"]
        holiday = request.form["holiday (Input 0 or 1)"]
        shops_closed = request.form["shops_closed (Input 0 or 1)"]
        winter_school_holidays = request.form["winter_school_holidays (Input 0 or 1)"]
        school_holidays = request.form["school_holidays (Input 0 or 1)"]
        year = request.form["year"]
        month = request.form["month (Input Integer 1-12)"]
        day = request.form["day (Input Integer 1-31)"]
        dayofweek = request.form["dayofweek (Input Integer 0-6)"]
        is_weekend = request.form["is_weekend (Input 0 or 1)"]
        X = np.array([[float(warehouse), float(holiday_name), float(holiday), float(shops_closed), float(winter_school_holidays), float(school_holidays), float(year), float(month), float(day), float(dayofweek), float(is_weekend)]])
        pred = model.predict(X)
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
