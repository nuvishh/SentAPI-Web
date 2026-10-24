# SentAPI — Malayalam Sentiment Analysis

A web app + JSON API that analyzes Malayalam text and classifies it as **positive**, **negative**, or **neutral**, with a confidence score. Built with Flask and a Naive Bayes classifier trained on Malayalam text.

🌐 **Live:** https://sentapii.pythonanywhere.com/

## Features

- **Sentiment analysis** — paste Malayalam text, get prediction + confidence meter
- **Batch analysis** — analyze up to 20 texts at once (UI + API)
- **Manglish → മലയാളം** — type in Roman script, convert to Malayalam (Google Input Tools, with offline fallback)
- **JSON API** — use SentAPI programmatically (`/api/analyze`, `/api/batch`, `/api/transliterate`)
- **Dark mode** — toggle with preference saved in the browser
- **Copy result** — one-click copy of prediction + confidence
- Character counter (2000 max)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| ML | NLTK + Naive Bayes classifier (`model8.pkl`) |
| Transliteration | Google Input Tools API → `indic-transliteration` fallback |
| Frontend | Single HTML template, vanilla JS, Font Awesome |
| Hosting | PythonAnywhere |

## API Reference

### `POST /api/analyze`
Analyze a single text.

```json
// Request
{ "text": "ഈ സിനിമ വളരെ നന്നായിരുന്നു" }

// Response (200)
{ "text": "ഈ സിനിമ വളരെ നന്നായിരുന്നു", "prediction": "positive", "confidence": 92.5 }
```

### `POST /api/batch`
Analyze up to 20 texts at once.

```json
// Request
{ "texts": ["ഈ സിനിമ നന്നായിരുന്നു", "ഞാൻ നിരാശനാണ്"] }

// Response (200)
{
  "count": 2,
  "results": [
    { "text": "ഈ സിനിമ നന്നായിരുന്നു", "prediction": "positive", "confidence": 70.86 },
    { "text": "ഞാൻ നിരാശനാണ്", "prediction": "negative", "confidence": 47.87 }
  ]
}
```

### `POST /api/transliterate`
Convert Manglish (Roman script) to Malayalam.

```json
// Request
{ "text": "ithu nalla padam" }

// Response (200)
{ "text": "ithu nalla padam", "transliterated": "ഇത് നല്ല പടം", "engine": "google" }
```

`engine` is `"google"` when Google Input Tools answered, `"indic"` when the offline fallback was used.

Error responses return `{"error": "..."}` with status `400` (bad input) or `500`/`503` (server-side failure).

## Run Locally

```bash
git clone https://github.com/nuvishh/SentAPI-Web.git
cd SentAPI-Web
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

The app expects `model4.pkl`–`model8.pkl` at the repo root (`model8.pkl` is loaded by default) and the `templates/` folder with its static assets.

## Project Structure

```
├── app.py                 # Flask app: routes, classifier, API endpoints
├── requirements.txt       # Flask, gunicorn, nltk, indic-transliteration
├── model4.pkl … model8.pkl# Trained Naive Bayes models (model8 active)
├── templates/
│   ├── index.html         # UI: analyzer, batch, transliteration, API docs
│   └── images/            # Static assets
├── procfile               # Process file for hosting
└── request.py             # Helper script
```

## Team

- **Vishnu N** — [@nuvishh](https://github.com/nuvishh)
- **Sreerag E** — [@cheer-g](https://github.com/cheer-g)
- **Firdousa K** — [@firdousa](https://github.com/firdousa)

## License

MIT — see [LICENSE](LICENSE).
