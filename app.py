from pathlib import Path
from threading import Thread, Lock
from datetime import datetime, timezone

from dastan import predictor, data
from dastan.rebuild import fplcache
from flask import Flask, jsonify
from live import run_live_predictions
import requests
import pandas as pd

app = Flask(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

pipeline_lock = Lock()
pipeline_state = {
    "status": "idle",
    "stage": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


def get_fpl_players():
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=20)
    response.raise_for_status()

    data = response.json()

    teams = {
        team["id"]: {
            "name": team["name"],
            "short_name": team["short_name"],
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
            ),
        })

    return players


@app.route("/")
def home():
    return jsonify({
        "service": "FPL ML API",
        "status": "ok",
        "version": "0.3",
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
    })


@app.route("/players")
def players():
    try:
        player_data = get_fpl_players()

        return jsonify({
            "status": "ok",
            "count": len(player_data),
            "players": player_data,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
        }), 500


@app.route("/ml-health")
def ml_health():
    try:
        model = predictor.Dastan()

        model._load("p60_MID")

        return jsonify({
            "status": "ok",
            "model": "Dastan",
            "model_loaded": True,
            "feature_count": len(model.features),
            "message": "Dastan model and weights loaded successfully.",
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "model": "Dastan",
            "model_loaded": False,
            "error": str(e),
        }), 500


def run_pipeline():
    global pipeline_state

    try:
        from dastan.rebuild import sources, features
        with pipeline_lock:
            pipeline_state.update({
                "status": "running",
                "stage": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "result": None,
                "error": None,
            })

        root = Path(__file__).resolve().parent
        raw_dir = root / ".cache" / "dastan-raw"
        seasons = ["2025-26"]

        with pipeline_lock:
            pipeline_state["stage"] = "downloading_sources"

        sources.download_sources(
            raw_dir=raw_dir,
            seasons=seasons,
            workers=1,
            force=False,
            allow_missing_understat=True,
        )

        with pipeline_lock:
            pipeline_state["stage"] = "building_canonical_matches"

        player_matches, team_matches, player_lookup = (
            sources.build_canonical_matches(raw_dir, seasons)
        )

        with pipeline_lock:
            pipeline_state["stage"] = "building_features"

        frame = features.build_feature_frame(
            player_matches,
            team_matches,
        )
        player_match_rows = len(player_matches)
        team_match_rows = len(team_matches)

        import gc
        del player_matches
        del team_matches
        del player_lookup
        gc.collect()
        with pipeline_lock:
            pipeline_state["stage"] = "building_predeadline_artifacts"

        output_dir = root / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        frame.to_parquet(
            output_dir / "features.parquet",
            index=False,
        )
        del frame
        import gc
        gc.collect()
        fplcache.build_predeadline_artifacts(
            raw_dir,
            output_dir,
            seasons,
        )

        frame = data.load(
            data_dir=output_dir,
            check_rows=False,
       )
        
        with pipeline_lock:
            pipeline_state["stage"] = "running_predictions"

        model = predictor.Dastan()
        out = model.predict_frame(frame, with_parts=True)
        out.to_parquet(output_dir / "predictions.parquet", index=False)

        result = {
            "seasons": seasons,
            "player_match_rows": player_match_rows,
            "team_match_rows": team_match_rows,
            "feature_rows": len(frame),
            "feature_columns": len(frame.columns),
            "prediction_rows": len(out),
            "model_features": len(model.features),
        }

        with pipeline_lock:
            pipeline_state.update({
                "status": "completed",
                "stage": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "error": None,
            })

    except Exception as e:
        with pipeline_lock:
            pipeline_state.update({
                "status": "error",
                "stage": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            })


@app.route("/pipeline-test")
def pipeline_test():
    global pipeline_state

    with pipeline_lock:
        if pipeline_state["status"] == "running":
            return jsonify({
                "status": "running",
                "stage": pipeline_state["stage"],
                "message": "Pipeline is already running.",
                "status_url": "/pipeline-status",
            }), 202

        pipeline_state.update({
            "status": "starting",
            "stage": "queued",
            "result": None,
            "error": None,
        })

    worker = Thread(target=run_pipeline, daemon=True)
    worker.start()

    return jsonify({
        "status": "started",
        "message": "Dastan pipeline started in the background.",
        "status_url": "/pipeline-status",
    }), 202


@app.route("/pipeline-status")
def pipeline_status():
    with pipeline_lock:
        return jsonify(dict(pipeline_state))


@app.route("/predictions")
def predictions():
    try:
        output_dir = Path(__file__).resolve().parent / "data"
        predictions_path = output_dir / "predictions.parquet"

        if not predictions_path.exists():
            return jsonify({
                "status": "error",
                "error": "Predictions are not ready yet. Run /pipeline-test first.",
            }), 503

        out = pd.read_parquet(predictions_path)

        latest_gw = int(out["gameweek"].max())
        latest = out[out["gameweek"] == latest_gw].copy()
        latest = latest.sort_values("xpts", ascending=False)

        rows = []
        for _, row in latest.head(50).iterrows():
            rows.append({
                "player": row.get("player_name"),
                "team": row.get("team_name"),
                "position": row.get("position"),
                "gameweek": int(row["gameweek"]),
                "xpts": round(float(row["xpts"]), 2),
            })

        return jsonify({
            "status": "ok",
            "gameweek": latest_gw,
            "count": len(rows),
            "predictions": rows,
        })

    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc),
        }), 500

live_lock = Lock()
live_state = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


def run_live_pipeline():
    with live_lock:
        live_state.update({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
        })

    try:
        result = run_live_predictions()

        with live_lock:
            live_state.update({
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "error": None,
            })

    except Exception as exc:
        with live_lock:
            live_state.update({
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": None,
                "error": str(exc),
            })


@app.route("/live-test")
def live_test():
    with live_lock:
        if live_state["status"] == "running":
            return jsonify({
                "status": "running",
                "status_url": "/live-status",
            }), 202

        live_state.update({
            "status": "starting",
            "result": None,
            "error": None,
        })

    worker = Thread(target=run_live_pipeline, daemon=True)
    worker.start()

    return jsonify({
        "status": "started",
        "message": "Live Dastan pipeline started in the background.",
        "status_url": "/live-status",
    }), 202


@app.route("/live-status")
def live_status():
    with live_lock:
        return jsonify(dict(live_state))
# --- Base44 production endpoint: Dastan current-GW predictions ---
@app.get("/api/dastan/predictions")
def dastan_predictions():
    """Return the latest successfully generated live Dastan player-GW predictions."""
    try:
        live_file = Path(__file__).resolve().parent / "data" / "live" / "predictions_player_gw.parquet"
        if not live_file.exists():
            return jsonify({
                "status": "not_ready",
                "error": "No live Dastan prediction file exists yet."
            }), 503

        df = pd.read_parquet(live_file)

        required = {
            "fpl_code", "player", "team", "position", "price",
            "xpts", "expected_minutes", "p60", "fixtures"
        }
        missing = sorted(required - set(df.columns))
        if missing:
            return jsonify({
                "status": "error",
                "error": f"Prediction file is missing columns: {missing}"
            }), 500

        df = df.sort_values("xpts", ascending=False).copy()

        players = []
        for rank, row in enumerate(df.itertuples(index=False), start=1):
            players.append({
                "rank": rank,
                "fpl_code": int(row.fpl_code),
                "player": str(row.player),
                "team": str(row.team),
                "position": str(row.position),
                "price": round(float(row.price), 1),
                "xpts": round(float(row.xpts), 3),
                "expected_minutes": round(float(row.expected_minutes), 1),
                "p60": round(float(row.p60), 4),
                "fixtures": int(row.fixtures),
            })

        return jsonify({
            "status": "ok",
            "model": "Dastan",
            "season": "2026-27",
            "gameweek": 4,
            "count": len(players),
            "players": players,
        })

    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc),
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
