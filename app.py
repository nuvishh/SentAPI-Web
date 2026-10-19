import pickle
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template, jsonify
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_TRANSLITERATION = True
except ImportError:
    HAS_TRANSLITERATION = False
import os
import json as _json
import urllib.request
import urllib.parse
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




def google_transliterate(text):
    """Use Google Input Tools (free, no key) for smart Manglish->Malayalam."""
    url = ("https://inputtools.google.com/request?itc=ml-t-i0-und&num=5&cp=0&cs=1"
           "&ie=utf-8&oe=utf-8&app=test&text=" + urllib.parse.quote(text))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    # data: ["SUCCESS", [[input, [suggestions...]]]]
    if data and data[0] == "SUCCESS" and data[1]:
        suggestions = data[1][0][1]
        if suggestions:
            return suggestions[0]
    raise ValueError("No transliteration result")

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
        val='ഇത് നല്ല പടം ആണ്',
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


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """JSON API: POST {"texts": ["...", "..."]} -> {"results": [...]} (max 20)"""
    data = request.get_json(silent=True) or {}
    texts = data.get('texts')
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Provide 'texts' as a non-empty JSON array"}), 400
    texts = [str(t).strip() for t in texts[:20] if str(t).strip()]
    if not texts:
        return jsonify({"error": "No valid texts provided"}), 400
    results = []
    for text in texts:
        try:
            prediction, confidence = classify_text(text)
            results.append({"text": text, "prediction": prediction, "confidence": confidence})
        except Exception as e:
            logger.error(f"Batch item error: {e}")
            results.append({"text": text, "error": "Analysis failed"})
    return jsonify({"results": results, "count": len(results)})


@app.route('/api/transliterate', methods=['POST'])
def api_transliterate():
    """Manglish -> Malayalam: POST {"text": "ithu nalla padam"} -> {"transliterated": "..."}"""
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400
    # Try Google Input Tools first (smarter for casual Manglish)
    try:
        out = google_transliterate(text)
        return jsonify({"text": text, "transliterated": out, "engine": "google"})
    except Exception as e:
        logger.warning(f"Google transliteration failed, trying fallback: {e}")
    # Fallback: indic-transliteration (offline)
    if not HAS_TRANSLITERATION:
        return jsonify({"error": "Transliteration not available"}), 503
    try:
        out = transliterate(text, sanscript.ITRANS, sanscript.MALAYALAM)
        return jsonify({"text": text, "transliterated": out, "engine": "indic"})
    except Exception as e:
        logger.error(f"Transliteration error: {e}")
        return jsonify({"error": "Transliteration failed"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)