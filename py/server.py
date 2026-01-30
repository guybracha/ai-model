from __future__ import annotations

from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from binding import ModelLib

app = Flask(__name__)
CORS(app)

# --- Model setup ---
in_features = 4
out_features = 3

W = np.array([
    0.1, 0.2, 0.3, 0.4,
    -0.5, 0.1, 0.0, 0.2,
    0.3, -0.2, 0.8, -0.1
], dtype=np.float32)

b = np.array([0.0, 0.5, -0.2], dtype=np.float32)

lib = ModelLib()  # loads build/model.dll via your binding.py


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    x = data.get("x")

    if not isinstance(x, list):
        return jsonify({"error": "Expected JSON: {\"x\": [..]}"}), 400
    if len(x) != in_features:
        return jsonify({"error": f"x must have length {in_features}"}), 400

    try:
        x_np = np.array([float(v) for v in x], dtype=np.float32)
    except ValueError:
        return jsonify({"error": "x must contain numbers"}), 400

    y = lib.dense_relu(
        x=x_np,
        W=W,
        b=b,
        in_features=in_features,
        out_features=out_features,
    )

    return jsonify({"y": y.tolist()})


if __name__ == "__main__":
    # Note: run from py/ folder OR make sure python can import binding.py
    app.run(host="127.0.0.1", port=8000, debug=True)
