from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import requests

from dastan import data, predictor
from dastan.rebuild import features, sources, fplcache

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ACTIVE_SEASON = "2026-27"
HISTORY_SEASONS = ["2025-26", ACTIVE_SEASON]


def _get_json(url: str) -> dict | list:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "fpl-ml-api/1.0"},
    )
    response.raise_for_status()
    return response.json()


def _next_gameweek(bootstrap: dict) -> dict:
    next_events = [event for event in bootstrap["events"] if event.get("is_next")]
    if len(next_events) != 1:
        raise RuntimeError(
            f"Expected exactly one next FPL gameweek, found {len(next_events)}."
        )
    return next_events[0]


def _position_name(element_type: int) -> str:
    return {
        1: "GKP",
        2: "DEF",
        3: "MID",
        4: "FWD",
    }[int(element_type)]


def _future_player_rows(
    player_matches: pd.DataFrame,
    bootstrap: dict,
    fixtures: list[dict],
    gameweek: int,
) -> pd.DataFrame:
    teams = {int(team["id"]): team["name"] for team in bootstrap["teams"]}
    elements = bootstrap["elements"]

    gw_fixtures = [
        fixture
        for fixture in fixtures
        if fixture.get("event") == gameweek
        and fixture.get("kickoff_time")
        and not fixture.get("finished", False)
    ]
    if not gw_fixtures:
        raise RuntimeError(f"No future fixtures found for GW{gameweek}.")

    # The public feature builder is safe for a normal single-fixture gameweek.
    # Refuse a DGW here rather than silently allowing synthetic future team rows
    # to influence another future fixture's rolling team history.
    appearances: dict[int, int] = {}
    for fixture in gw_fixtures:
        appearances[int(fixture["team_h"])] = appearances.get(int(fixture["team_h"]), 0) + 1
        appearances[int(fixture["team_a"])] = appearances.get(int(fixture["team_a"]), 0) + 1
    doubled = [team_id for team_id, count in appearances.items() if count > 1]
    if doubled:
        names = [teams.get(team_id, str(team_id)) for team_id in doubled]
        raise RuntimeError(
            "Live builder detected a double gameweek. "
            "DGW needs the deadline-safe multi-fixture path before scoring: "
            + ", ".join(names)
        )

    assignment_map = (
        player_matches[player_matches["season"].eq(ACTIVE_SEASON)]
        [["fpl_code", "understat_id"]]
        .drop_duplicates("fpl_code")
        .set_index("fpl_code")["understat_id"]
        .to_dict()
    )

    by_team: dict[int, list[dict]] = {}
    for player in elements:
        by_team.setdefault(int(player["team"]), []).append(player)

    rows: list[dict] = []
    for fixture in gw_fixtures:
        fixture_id = int(fixture["id"])
        kickoff = pd.to_datetime(fixture["kickoff_time"], utc=True)
        home_id = int(fixture["team_h"])
        away_id = int(fixture["team_a"])

        for team_id, opponent_id, is_home in (
            (home_id, away_id, True),
            (away_id, home_id, False),
        ):
            team_name = teams[team_id]
            opponent_name = teams[opponent_id]

            for player in by_team.get(team_id, []):
                fpl_code = int(player["code"])
                row = {
                    "season": ACTIVE_SEASON,
                    "gameweek": int(gameweek),
                    "fixture": fixture_id,
                    "fpl_code": fpl_code,
                    "element": int(player["id"]),
                    "player_name": player.get("web_name") or player.get("second_name"),
                    "position": _position_name(player["element_type"]),
                    "team_name": team_name,
                    "opponent_team_name": opponent_name,
                    "us_opponent": features.fpl_to_understat(opponent_name),
                    "kickoff_time": kickoff,
                    "match_date": kickoff.date().isoformat(),
                    "is_home": bool(is_home),
                    "understat_id": assignment_map.get(fpl_code, np.nan),
                    # Outcome/current-fixture columns must contain no future information.
                    "minutes": 0.0,
                    "total_points": 0.0,
                    "expected_points_pre_deadline": 0.0,
                    "starts": np.nan,
                    "clearances_blocks_interceptions": np.nan,
                    "defensive_contribution": np.nan,
                    "recoveries": np.nan,
                    "tackles": np.nan,
                }

                # All ordinary player-match numeric sources are zero on the
                # synthetic target row. Every historical family in Dastan uses
                # shift(1), so these values are not used to predict this row.
                for column in features.PLAYER_NUMERIC:
                    row.setdefault(column, 0.0)

                rows.append(row)

    return pd.DataFrame(rows)


def _future_team_rows(
    team_matches: pd.DataFrame,
    bootstrap: dict,
    fixtures: list[dict],
    gameweek: int,
) -> pd.DataFrame:
    teams = {int(team["id"]): team["name"] for team in bootstrap["teams"]}
    rows: list[dict] = []

    gw_fixtures = [
        fixture
        for fixture in fixtures
        if fixture.get("event") == gameweek
        and fixture.get("kickoff_time")
        and not fixture.get("finished", False)
    ]

    for fixture in gw_fixtures:
        fixture_id = int(fixture["id"])
        kickoff = pd.to_datetime(fixture["kickoff_time"], utc=True)
        home_name = features.fpl_to_understat(teams[int(fixture["team_h"])])
        away_name = features.fpl_to_understat(teams[int(fixture["team_a"])])

        # Unique matching markers are used only so build_opponent_lookup can
        # pair the two synthetic team rows correctly when several EPL matches
        # share the same kickoff time. compute_team_rolling uses shift(1), so
        # the target fixture never sees its own marker values.
        marker_a = float(10000 + fixture_id)
        marker_b = float(20000 + fixture_id)
        marker_xg_a = float(30000 + fixture_id)
        marker_xg_b = float(40000 + fixture_id)

        home = {
            "season": ACTIVE_SEASON,
            "date": kickoff,
            "understat_team": home_name,
            "is_home": 1.0,
            "scored": marker_a,
            "missed": marker_b,
            "xG": marker_xg_a,
            "xGA": marker_xg_b,
            "pts": 0.0,
        }
        away = {
            "season": ACTIVE_SEASON,
            "date": kickoff,
            "understat_team": away_name,
            "is_home": 0.0,
            "scored": marker_b,
            "missed": marker_a,
            "xG": marker_xg_b,
            "xGA": marker_xg_a,
            "pts": 0.0,
        }

        for row in (home, away):
            for column in features.TEAM_NUMERIC:
                row.setdefault(column, 0.0)
            rows.append(row)

    future = pd.DataFrame(rows)

    # Match the existing table's columns without losing required future fields.
    for column in team_matches.columns:
        if column not in future:
            future[column] = np.nan
    for column in future.columns:
        if column not in team_matches:
            team_matches[column] = np.nan

    return future[team_matches.columns]


def _write_live_snapshot_artifacts(
    output_dir: Path,
    bootstrap: dict,
    gameweek: int,
) -> None:
    ep_rows = []
    signal_rows = []

    for player in bootstrap["elements"]:
        fpl_code = int(player["code"])
        raw_ep = player.get("ep_next")
        chance = player.get("chance_of_playing_next_round")

        ep_rows.append({
            "season": ACTIVE_SEASON,
            "gameweek": int(gameweek),
            "fpl_code": fpl_code,
            "ep_next": np.nan if raw_ep in (None, "") else float(raw_ep),
        })

        signal_rows.append({
            "season": ACTIVE_SEASON,
            "gameweek": int(gameweek),
            "fpl_code": fpl_code,
            "sig_status_risk": float(
                fplcache.STATUS_RISK.get(str(player.get("status") or "a"), 0)
            ),
            "sig_chance_playing": -1.0 if chance is None else float(chance),
            "sig_has_news": float(bool((player.get("news") or "").strip())),
        })

    pd.DataFrame(ep_rows).to_parquet(
        output_dir / "pre_deadline_ep_next.parquet",
        index=False,
    )
    pd.DataFrame(signal_rows).to_parquet(
        output_dir / "pre_deadline_signals.parquet",
        index=False,
    )


def run_live_predictions(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    raw_dir = root / ".cache" / "dastan-live-raw"
    output_dir = root / "data" / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = _get_json(FPL_BOOTSTRAP_URL)
    fixtures = _get_json(FPL_FIXTURES_URL)
    next_event = _next_gameweek(bootstrap)
    gameweek = int(next_event["id"])
    deadline = pd.to_datetime(next_event["deadline_time"], utc=True)
    now = pd.Timestamp.now(tz="UTC")

    if now >= deadline:
        raise RuntimeError(
            f"GW{gameweek} deadline has already passed ({deadline.isoformat()})."
        )

    print(
        f"Live Dastan: building {ACTIVE_SEASON} GW{gameweek} "
        f"(deadline {deadline.isoformat()})",
        flush=True,
    )

    sources.download_sources(
        raw_dir=raw_dir,
        seasons=HISTORY_SEASONS,
        workers=1,
        force=False,
        allow_missing_understat=True,
    )

    player_matches, team_matches, _ = sources.build_canonical_matches(
        raw_dir,
        HISTORY_SEASONS,
    )

    future_players = _future_player_rows(
        player_matches,
        bootstrap,
        fixtures,
        gameweek,
    )
    future_teams = _future_team_rows(
        team_matches,
        bootstrap,
        fixtures,
        gameweek,
    )

    player_matches = pd.concat(
        [player_matches, future_players],
        ignore_index=True,
        sort=False,
    )
    team_matches = pd.concat(
        [team_matches, future_teams],
        ignore_index=True,
        sort=False,
    )

    del future_teams
    gc.collect()

    frame = features.build_feature_frame(
        player_matches,
        team_matches,
    )

    del player_matches
    del team_matches
    gc.collect()

    live_frame = frame[
        frame["season"].eq(ACTIVE_SEASON)
        & frame["gameweek"].eq(gameweek)
    ].copy()

    if live_frame.empty:
        raise RuntimeError(f"Feature builder produced no rows for GW{gameweek}.")

    live_frame.to_parquet(
        output_dir / "features.parquet",
        index=False,
    )
    _write_live_snapshot_artifacts(
        output_dir,
        bootstrap,
        gameweek,
    )

    scored_frame = data.load(
        data_dir=output_dir,
        check_rows=False,
    )

    model = predictor.Dastan()
    predictions = model.predict_frame(
        scored_frame,
        with_parts=True,
    )

    names = pd.DataFrame([
        {
            "fpl_code": int(player["code"]),
            "player_name": player.get("web_name") or player.get("second_name"),
            "team_id": int(player["team"]),
            "price": float(player["now_cost"]) / 10.0,
        }
        for player in bootstrap["elements"]
    ])
    team_names = {
        int(team["id"]): team["name"]
        for team in bootstrap["teams"]
    }
    names["current_team_name"] = names["team_id"].map(team_names)

    predictions = predictions.drop(
        columns=["player_name"],
        errors="ignore",
    ).merge(
        names,
        on="fpl_code",
        how="left",
        validate="many_to_one",
    )

    predictions.to_parquet(
        output_dir / "predictions.parquet",
        index=False,
    )

    top = (
        predictions.groupby(
            ["fpl_code", "player_name", "current_team_name", "position", "price"],
            dropna=False,
            as_index=False,
        )
        .agg(
            xpts=("xpts", "sum"),
            expected_minutes=("expected_minutes", "sum"),
            p60=("p60", "max"),
            fixtures=("fixture", "nunique"),
        )
        .sort_values("xpts", ascending=False)
        .reset_index(drop=True)
    )

    top.to_parquet(
        output_dir / "predictions_player_gw.parquet",
        index=False,
    )

    return {
        "status": "ok",
        "season": ACTIVE_SEASON,
        "gameweek": gameweek,
        "deadline": deadline.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "players": int(len(top)),
        "fixture_rows": int(len(predictions)),
        "model_features": int(len(model.features)),
        "top_10": [
            {
                "player": row.player_name,
                "team": row.current_team_name,
                "position": row.position,
                "price": round(float(row.price), 1),
                "xpts": round(float(row.xpts), 2),
                "expected_minutes": round(float(row.expected_minutes), 1),
                "p60": round(float(row.p60), 3),
            }
            for row in top.head(10).itertuples()
        ],
    }
