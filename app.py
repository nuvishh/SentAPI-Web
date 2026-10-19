import pickle
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Calculate absolute path for the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(os.path.join(BASE_DIR, 'sentapi.log'), maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK data - use a specific folder for PythonAnywhere
nltk_data_path = os.path.join(BASE_DIR, 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_path])
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Malayalam stop words
mal_stop_words = [
    "ഏത്", "എന്തിന്റെ", "ഇത്ര", "ആ", "അ", "ഏറെ", "ഇതേ", "എന്ന", "വേറെ",
    "മതി", ",", "എല്ലാ", "നിങ്ങളെ", "വളരെ", "എവിടെ", "എപ്പോൾ", "ഇന്ന്",
    "ഉള്ള", "ഇന്ന്", "മറ്റു", "ഈ", "ഇത്", "പോലെ", "തന്റെ", "വരെ",
    "മാത്രം", "എന്നാൽ", "മുമ്പ്", "തന്റെ", "ഇനി", "എങ്കിൽ", "അല്ലെങ്കിൽ",
    "കുറിച്ചു", "പിന്നെ", "എന്നത്", "അന്ന്", "എന്ത്", ",", ".", "അത്",
    "സിനിമ", "എല്ലാം"
]

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load model using absolute path
try:
    model_path = os.path.join(BASE_DIR, 'model8.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model at {model_path}: {e}")
    # Initialize a dummy model or handle the error gracefully to prevent app crash
    model = None

def preprocess_text(text):
    """Tokenize and filter stop words"""
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in mal_stop_words]
    return {token: True for token in filtered_tokens}


def classify_text(text):
    """Run the model on text. Returns (prediction, confidence) or raises."""
    if not model:
        raise RuntimeError("Model is not loaded")
    features = preprocess_text(text)
    prediction = model.classify(features)
    prob_dist = model.prob_classify(features)
    confidence = round(prob_dist.prob(prediction) * 100, 2)
    return prediction, confidence


@app.route('/')
def home():
    """Render the main page with initialized variables"""
    return render_template(
        "index.html",
        val='അവർ പോയി',
        prediction=None,
        confidence=None,
        error=None
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle text classification requests"""
    try:
        text = request.form.get('message', '').strip()
        if not text:
            return render_template(
                'index.html',
                error="Please enter some text",
                val='',
                prediction=None,
                confidence=None
            )

        logger.info(f"Processing text: {text[:50]}...")

        # Preprocess and classify
        try:
            prediction, confidence = classify_text(text)
        except RuntimeError:
            logger.error("Model is not loaded")
            return render_template(
                'index.html',
                error="Model loading failed. Please check the server logs.",
                val=text,
                prediction=None,
                confidence=None
            )

        logger.info(f"Result: {prediction} ({confidence}% confidence)")

        return render_template(
            'index.html',
            val=text,
            prediction=prediction,
            confidence=confidence,
            error=None
        )

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        return render_template(
            'index.html',
            error="Analysis failed. Please try again.",
            val=request.form.get('message', ''),
            prediction=None,
            confidence=None
        )

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """JSON API: POST {"text": "..."} -> {"prediction", "confidence"}"""
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400
    try:
        logger.info(f"API request: {text[:50]}...")
        prediction, confidence = classify_text(text)
        logger.info(f"API result: {prediction} ({confidence}%)")
        return jsonify({
            "text": text,
            "prediction": prediction,
            "confidence": confidence
        })
    except RuntimeError as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": "Analysis failed"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)