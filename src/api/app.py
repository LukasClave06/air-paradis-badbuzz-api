import os
from flask import Flask, request, jsonify, render_template_string

from src.api.predictor import PredictorConfig, SentimentPredictor

app = Flask(__name__)

RUN_ID = os.environ.get("MLFLOW_RUN_ID", "1ce1649c820d4e33ab0795e777b8cb4c")
MAX_LEN = int(os.environ.get("MAX_LEN", "40"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

predictor = SentimentPredictor(
    PredictorConfig(run_id=RUN_ID, max_len=MAX_LEN, threshold=THRESHOLD)
)

# --- Mini page HTML intégrée (ultra léger, pas de template à déployer) ---
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Air Paradis - Sentiment</title>
  </head>
  <body style="font-family: Arial; max-width: 800px; margin: 40px auto;">
    <h2>Air Paradis - Prédiction de sentiment</h2>
    <form method="post" action="/">
      <textarea name="text" rows="4" style="width: 100%;" placeholder="Écris un tweet...">{{ text }}</textarea>
      <br/><br/>
      <button type="submit">Prédire</button>
    </form>

    {% if result %}
      <hr/>
      <h3>Résultat</h3>
      <pre>{{ result }}</pre>
    {% endif %}
  </body>
</html>
"""

@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    result = None
    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            result = predictor.predict_one(text)
        else:
            result = {"error": "Texte vide."}
    return render_template_string(HTML, text=text, result=result)

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing field 'text' (non-empty string)."}), 400

    out = predictor.predict_one(text)
    return jsonify(out), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)