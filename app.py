from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session to work

# Load TF-IDF vectorizer
with open("model/tfidf_vectorizer_model.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load Naive Bayes model
with open("model/best_model_nb.pkl", "rb") as model_file:
    model_pandas = pickle.load(model_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get input from the form
    input_user = request.form.get("input_user")
    
    # Transform input text into TF-IDF features
    input_features = tfidf_vectorizer.transform([input_user])
    
    # Predict sentiment
    result = model_pandas.predict(input_features)[0]
    
    # Store input and result in session
    session['input_user'] = input_user
    session['result'] = result
    
    if result == 'Positive':
        return redirect(url_for('positive'))
    else:
        return redirect(url_for('negative'))

@app.route("/positive")
def positive():
    input_user = session.get('input_user', '')
    return render_template("positive.html", input_user=input_user)

@app.route("/negative")
def negative():
    input_user = session.get('input_user', '')
    return render_template("negative.html", input_user=input_user)

if __name__ == "__main__":
    app.run(debug=True)
