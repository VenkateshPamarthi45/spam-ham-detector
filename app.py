from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
import pickle
import joblib

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

loaded_vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
loaded_model = joblib.load("ham_spam_model.sav")


@app.route('/predict')
def predict():
    message = request.args.get('msg')
    return predict_msg_from_loaded(message)


def predict_msg_from_loaded(msg):
    msg = loaded_vectorizer.transform([msg])
    prediction = loaded_model.predict(msg.toarray())
    return jsonify({'type':prediction[0]})


if __name__ == '__main__':
    app.run()
