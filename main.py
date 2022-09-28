from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

model = pickle.load(open('model-iris.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    feature = [np.array(float_feature)]
    prediction = model.predict(feature)
    Acc = model.predict_proba(feature)
    print(Acc)
    return render_template("predict.html",data=prediction)


if __name__ == '__main__':
    app.run(debug=True)