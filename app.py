import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
MODEL_DIR = 'Models'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model_rf_best_params.pkl')
CV_PATH = os.path.join(MODEL_DIR, 'countVectorizer.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
try:
    loaded_rf_model = joblib.load(MODEL_PATH)
    loaded_cv = joblib.load(CV_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    print("* Models and preprocessors loaded successfully!")
except FileNotFoundError:
    print(f"! Error: One or more model files not found in '{MODEL_DIR}'. Please check paths.")
    loaded_rf_model, loaded_cv, loaded_scaler = None, None, None
except Exception as e:
    print(f"! Error loading models: {e}")
    loaded_rf_model, loaded_cv, loaded_scaler = None, None, None

def predict_sentiment(text_input):
    """
    Predicts the sentiment of a given text using the loaded
    CountVectorizer, scaler, and Random Forest model.
    """
    if not all([loaded_rf_model, loaded_cv, loaded_scaler]):
        raise RuntimeError("Models or preprocessors are not loaded. Cannot predict.")

    if isinstance(text_input, str):
        text_input = [text_input]  

    try:
        vectorized_text = loaded_cv.transform(text_input)
        scaled_text = loaded_scaler.transform(vectorized_text.toarray())
        predictions = loaded_rf_model.predict(scaled_text)
        sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'} 
        readable_predictions = [sentiment_mapping.get(p, "Unknown") for p in predictions]

        return readable_predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error in prediction"] 

# --- Define Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not all([loaded_rf_model, loaded_cv, loaded_scaler]):
            return render_template('index.html', error="Model not loaded. Please check server logs.")

        user_text = request.form.get('text')
        if not user_text:
            return render_template('index.html', error="Please enter some text.")

        try:
            prediction = predict_sentiment(user_text)
            return render_template('index.html', prediction=prediction[0], text=user_text)
        except RuntimeError as e:
             return render_template('index.html', error=str(e))
        except Exception as e:
            app.logger.error(f"Error processing request: {e}") 
            return render_template('index.html', error="An error occurred during prediction.")

    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict_api():
    if not all([loaded_rf_model, loaded_cv, loaded_scaler]):
        return jsonify({'error': 'Model not loaded properly. Check server logs.'}), 500

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in JSON payload.'}), 400

        text_input = data['text']
        if not isinstance(text_input, (str, list)):
            return jsonify({'error': '"text" field must be a string or a list of strings.'}), 400

        predictions = predict_sentiment(text_input)

        if "Error in prediction" in predictions:
             return jsonify({'error': 'Failed to make a prediction.'}), 500

        return jsonify({'predictions': predictions})

    except Exception as e:
        app.logger.error(f"API Error: {e}") 
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)