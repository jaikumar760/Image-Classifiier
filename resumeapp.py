from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load the model and vectorizer
model = joblib.load('resume_analyzer_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Define the API route to handle resume analysis
@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Get the resume text from the request
        data = request.get_json()
        
        # Debug: log the incoming request data
        print("Received request:", data)

        if 'resume' not in data:
            return jsonify({"error": "Missing resume text in the request"}), 400

        resume_text = data['resume']

        # Preprocess the resume text
        cleaned_resume = preprocess(resume_text)

        # Transform the resume text into TF-IDF features
        resume_tfidf = vectorizer.transform([cleaned_resume])

        # Debug: log the TF-IDF features
        print("Transformed resume TF-IDF:", resume_tfidf)

        # Predict the category using the trained model
        predicted_category = model.predict(resume_tfidf)

        # Debug: log the prediction result
        print("Prediction result:", predicted_category)

        return jsonify({"predicted_category": predicted_category[0]})

    except Exception as e:
        # Debug: log the error
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
