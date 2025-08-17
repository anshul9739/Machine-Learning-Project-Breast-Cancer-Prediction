# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import math
# from flask import Flask, request, render_template
# import re

# app = Flask("__name__")

# @app.route("/")
# def loadPage():
#     return render_template('home.html',query="")

# @app.route("/", methods=['POST'])
# def cancerPrediction():
#     dataset_url = "breast-cancer-data.csv"
#     df = pd.read_csv(dataset_url)
    
#     inputQuery1=request.form['query1']
#     inputQuery2=request.form['query2']
#     inputQuery3=request.form['query3']
#     inputQuery4=request.form['query4']
#     inputQuery5=request.form['query5']
    
#     df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
#     features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']

#     X = df[features]
#     y = df.diagnosis
    
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
#     model = RandomForestClassifier(n_estimators=500,n_jobs=-1)
#     model.fit(X_train, y_train)
    
#     #prediction = model.predict(X_test)
    
#     data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    
#     new_df = pd.DataFrame(data, columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean'])
#     single = model.predict(new_df)
#     proba = model.predict_proba(new_df)[:,1]
    
#     if single==1:
#         output1= "The patient is diagnosed with Breast Cancer"
#         output2 = "Confidence: {}".format(proba*100)
#     else:
#         output1 = "The patient is not diagnosed with Breast Cancer"
#         output2 = ""
    
#     return render_template('home.html', output1=output1, output2=output2, query1=request.form['query1'], query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'], query5=request.form['query5'])


# app.run()




# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, request, render_template, abort
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -------------------- Config --------------------
DATA_PATH = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "smoothness_mean",
    "compactness_mean",
]

# -------------------- Train once at startup --------------------
df = pd.read_csv(DATA_PATH)

# map diagnosis to 0/1 and drop any unexpected / missing rows
df = df.copy()
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df = df.dropna(subset=FEATURES + ["diagnosis"])

X = df[FEATURES]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def load_page():
    return render_template("home.html", query="")

@app.route("/", methods=["POST"])
def cancer_prediction():
    # Get and validate inputs as floats
    try:
        vals = [
            float(request.form["query1"]),
            float(request.form["query2"]),
            float(request.form["query3"]),
            float(request.form["query4"]),
            float(request.form["query5"]),
        ]
    except (KeyError, ValueError):
        abort(400, description="All five inputs must be provided as numeric values.")

    new_df = pd.DataFrame([vals], columns=FEATURES)

    # Predict
    single = model.predict(new_df)[0]                # -> 0 or 1
    proba_pos = model.predict_proba(new_df)[0, 1]    # -> float in [0,1]

    if single == 1:
        output1 = "The patient is diagnosed with Breast Cancer"
        output2 = f"Confidence: {proba_pos * 100:.1f}%"
    else:
        output1 = "The patient is not diagnosed with Breast Cancer"
        output2 = f"Confidence: {(1 - proba_pos) * 100:.1f}%"

    return render_template(
        "home.html",
        output1=output1,
        output2=output2,
        query1=request.form.get("query1", ""),
        query2=request.form.get("query2", ""),
        query3=request.form.get("query3", ""),
        query4=request.form.get("query4", ""),
        query5=request.form.get("query5", ""),
    )

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    app.run(debug=True)
