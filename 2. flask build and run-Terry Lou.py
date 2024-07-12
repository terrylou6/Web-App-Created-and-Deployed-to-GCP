from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='C:/ProgramData/anaconda3/Lib/site-packages/flask/templates')

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        warehouse = request.form["warehouse"]
        holiday_name = request.form["holiday_name"]
        holiday = request.form["holiday"]
        shops_closed = request.form["shops_closed"]
        winter_school_holidays = request.form["winter_school_holidays"]
        school_holidays = request.form["school_holidays"]
        year = request.form["year"]
        month = request.form["month"]
        day = request.form["day"]
        dayofweek = request.form["dayofweek"]
        is_weekend = request.form["is_weekend"]
        X = np.array([[float(warehouse), float(holiday_name), float(holiday), float(shops_closed), float(winter_school_holidays), float(school_holidays), float(year), float(month), float(day), float(dayofweek), float(is_weekend)]])
        pred = model.predict(X)[0]
    return render_template("index_new.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
