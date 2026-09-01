from dastan import predictor
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def get_fpl_players():
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=20)
    response.raise_for_status()

    data = response.json()

    teams = {
        team["id"]: {
            "name": team["name"],
            "short_name": team["short_name"]
        }
        for team in data["teams"]
    }

    players = []

    for p in data["elements"]:
        team = teams.get(p["team"], {})

        players.append({
            "fpl_id": p["id"],
            "fpl_code": p["code"],
            "player": p["web_name"],
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "team": team.get("short_name"),
            "team_name": team.get("name"),
            "position": p["element_type"],
            "price": p["now_cost"] / 10,
            "minutes": p["minutes"],
            "total_points": p["total_points"],
            "form": p["form"],
            "points_per_game": p["points_per_game"],
            "selected_by_percent": p["selected_by_percent"],
            "expected_goals": p.get("expected_goals"),
            "expected_assists": p.get("expected_assists"),
            "expected_goal_involvements": p.get(
                "expected_goal_involvements"
            ),
            "status": p["status"],
            "chance_of_playing_next_round": p.get(
                "chance_of_playing_next_round"
            )
        })

    return players


@app.route("/")
def home():
    return jsonify({
        "service": "FPL ML API",
        "status": "ok",
        "version": "0.2"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/players")
def players():
    try:
        player_data = get_fpl_players()

        return jsonify({
            "status": "ok",
            "count": len(player_data),
            "players": player_data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
@app.route("/ml-health")
def ml_health():
    try:
        model = predictor.Dastan()

        # Load one real model artifact to verify that
        # Dastan + model files + XGBoost work correctly.
        model._load("p60_MID")

        return jsonify({
            "status": "ok",
            "model": "Dastan",
            "model_loaded": True,
            "feature_count": len(model.features),
            "message": "Dastan model and weights loaded successfully."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "model": "Dastan",
            "model_loaded": False,
            "error": str(e)
        }), 500

@app.route("/predictions")
def predictions():

    # IMPORTANT:
    # The external ML model is NOT connected yet.
    #
    # This endpoint temporarily returns no production predictions
    # rather than fake/hard-coded values.

    return jsonify({
        "status": "ok",
        "model": "ml-not-connected",
        "predictions": [],
        "message": (
            "FPL player data service is live. "
            "External ML inference will be connected next."
        )
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
