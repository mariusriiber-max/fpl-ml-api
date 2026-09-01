from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "service": "FPL ML API",
        "status": "ok"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predictions")
def predictions():
    return jsonify({
        "status": "ok",
        "model": "test-v1",
        "predictions": [
            {
                "player": "Erling Haaland",
                "next_gw": 7.8,
                "next_5": 35.4,
                "next_10": 68.2
            },
            {
                "player": "Bukayo Saka",
                "next_gw": 6.5,
                "next_5": 31.8,
                "next_10": 61.7
            },
            {
                "player": "Bruno Fernandes",
                "next_gw": 6.2,
                "next_5": 30.1,
                "next_10": 58.9
            }
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
