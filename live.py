from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import requests

from dastan import data, predictor, mappings
from dastan.rebuild import features, sources, fplcache

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ACTIVE_SEASON = "2026-27"
HISTORY_SEASONS = ["2025-26", ACTIVE_SEASON]


@contextmanager
def _live_operational_mapping_mode():
    """Use current player identities for live inference without the retraining club gate.

    Dastan's public mapping guard blocks a new-season *retraining* build until every
    promoted club has a reviewed Understat club ID. Live scoring does not use those
    club IDs: current player Understat identities are joined by fpl_code, while team
    match data is loaded independently from the Understat league/team histories.

    Keep the upstream guard intact everywhere else and relax it only while this live
    inference job resolves current-season assignments.
    """
    original = mappings.assert_operational_clubs_ready

    def live_check(season: str) -> None:
        roster = mappings.load_roster()
        active = str(roster["season"].iat[0])
        if str(season) != active:
            original(season)

    mappings.assert_operational_clubs_ready = live_check
    try:
        yield
    finally:
        mappings.assert_operational_clubs_ready = original


def _get_json(url: str) -> dict | list:
    # Force a fresh FPL response. The public endpoints can sit behind CDN/proxy
    # caches, and stale player/team metadata is unacceptable for live scoring.
    cache_buster = int(datetime.now(timezone.utc).timestamp())
    separator = "&" if "?" in url else "?"
    fresh_url = f"{url}{separator}_={cache_buster}"
    response = requests.get(
        fresh_url,
        timeout=30,
        headers={
            "User-Agent": "fpl-ml-api/1.0",
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        },
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


def _current_roster(bootstrap: dict) -> pd.DataFrame:
    teams = {
        int(team["id"]): team["name"]
        for team in bootstrap["teams"]
    }
    if len(teams) != 20:
        raise RuntimeError(
            f"Official FPL bootstrap returned {len(teams)} teams, expected 20."
        )

    roster = pd.DataFrame([
        {
            "element": int(player["id"]),
            "fpl_code_current": int(player["code"]),
            "player_name_current": player.get("web_name") or player.get("second_name"),
            "first_name_current": player.get("first_name"),
            "second_name_current": player.get("second_name"),
            "web_name_current": player.get("web_name"),
            "team_id_current": int(player["team"]),
            "team_short_current": next(
                team["short_name"]
                for team in bootstrap["teams"]
                if int(team["id"]) == int(player["team"])
            ),
            "position_current": _position_name(player["element_type"]),
            "price_current": float(player["now_cost"]) / 10.0,
        }
        for player in bootstrap["elements"]
    ])

    if roster["element"].duplicated().any():
        raise RuntimeError("Official FPL bootstrap contains duplicate element IDs.")

    unknown_team_ids = sorted(
        set(roster["team_id_current"].dropna().astype(int)) - set(teams)
    )
    if unknown_team_ids:
        raise RuntimeError(
            f"Official FPL bootstrap players reference unknown teams: {unknown_team_ids}"
        )

    roster["current_team_name"] = roster["team_id_current"].map(teams)
    return roster


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
    official_next_gameweek: int,
) -> None:
    ep_rows = []
    signal_rows = []

    for player in bootstrap["elements"]:
        fpl_code = int(player["code"])
        # FPL ep_next is explicitly only for the official next GW. Never leak/reuse
        # it for later horizons. Dastan's missing-value contract uses -1.
        raw_ep = player.get("ep_next") if gameweek == official_next_gameweek else None
        chance = (
            player.get("chance_of_playing_next_round")
            if gameweek == official_next_gameweek
            else None
        )

        ep_rows.append({
            "season": ACTIVE_SEASON,
            "gameweek": int(gameweek),
            "fpl_code": fpl_code,
            "ep_next": -1.0 if raw_ep in (None, "") else float(raw_ep),
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
            "sig_pens_order": -1.0,
            "sig_fk_order": -1.0,
            "sig_corners_order": -1.0,
            "sig_age_years": -1.0,
            "sig_days_at_club": -1.0,
        })

    pd.DataFrame(ep_rows).to_parquet(
        output_dir / "pre_deadline_ep_next.parquet",
        index=False,
    )
    pd.DataFrame(signal_rows).to_parquet(
        output_dir / "pre_deadline_signals.parquet",
        index=False,
    )


def _carry_understat_identity_and_history_into_live_rows(
    frame: pd.DataFrame,
    active_season: str,
    gameweek: int,
) -> pd.DataFrame:
    """Repair synthetic live rows before feature engineering.

    For each stable player identity, copy the latest *raw Understat source fields*
    from a completed historical match into the synthetic current-GW row.
    This does NOT copy targets, FPL points, current-fixture outcomes or already
    rolled model features. The normal Dastan rolling code still performs its
    own shift/rolling/deadline anchoring afterwards.
    """
    out = frame.copy()

    if "fpl_code" not in out.columns:
        raise RuntimeError("Cannot carry Understat history: fpl_code missing.")

    target_mask = (
        out["season"].eq(active_season)
        & pd.to_numeric(out["gameweek"], errors="coerce").eq(gameweek)
    )

    # Only provider-level Understat columns. Never copy model/target/output fields.
    blocked_fragments = (
        "target",
        "points",
        "minutes",
        "fixture",
        "gameweek",
        "kickoff",
        "expected_minutes",
        "p60",
        "ep_next",
        "sig_",
        "player_fpl_",
    )
    us_cols = [
        c for c in out.columns
        if c.startswith("us_")
        and not any(fragment in c.lower() for fragment in blocked_fragments)
    ]

    if not us_cols:
        return out

    history = out.loc[~target_mask].copy()
    if history.empty:
        return out

    # Prefer chronological latest completed observation.
    sort_cols = [
        c for c in ["kickoff_time", "season", "gameweek", "fixture"]
        if c in history.columns
    ]
    if sort_cols:
        history = history.sort_values(sort_cols)

    # Stable FPL code is the repository's cross-season player identity.
    latest = (
        history.groupby("fpl_code", dropna=False)[us_cols]
        .last()
        .reset_index()
    )

    target = out.loc[target_mask, ["fpl_code"]].merge(
        latest,
        on="fpl_code",
        how="left",
        validate="many_to_one",
    )
    target.index = out.index[target_mask]

    # Fill only missing synthetic values. For raw US metrics, zeros generated
    # solely because the future row has no match observation are also repaired
    # when a non-null historical provider value exists.
    for col in us_cols:
        hist_values = target[col]
        current = out.loc[target_mask, col]
        replace = current.isna()
        if pd.api.types.is_numeric_dtype(out[col]):
            replace = replace | (
                pd.to_numeric(current, errors="coerce").eq(0)
                & pd.to_numeric(hist_values, errors="coerce").notna()
            )
        idx = current.index[replace]
        out.loc[idx, col] = hist_values.loc[idx]

    return out


def _model_team_feature_audit(
    scored_frame: pd.DataFrame,
    model,
    current_roster: pd.DataFrame,
) -> dict:
    """Audit only columns that are truly consumed by the released Dastan model."""
    model_cols = [c for c in model.features if c in scored_frame.columns]

    # Team/opponent/OpenFPL rolling families among the actual 286 model inputs.
    keywords = (
        "team", "opp", "opponent", "xg", "xga", "npxg", "npxga",
        "deep", "ppda", "scored", "missed", "pts"
    )
    team_model_cols = [
        c for c in model_cols
        if any(k in c.lower() for k in keywords)
    ]

    roster = current_roster[
        ["element", "player_name_current", "current_team_name"]
    ].copy()
    wanted = {"Rogers", "Palmer", "Gabriel", "João Pedro", "Joao Pedro"}
    roster = roster[
        roster["player_name_current"].fillna("").astype(str).isin(wanted)
    ]

    sample = scored_frame.merge(
        roster,
        on="element",
        how="inner",
        validate="many_to_one",
    )

    # Keep the output readable: report up to 80 genuine model columns.
    cols = team_model_cols[:80]

    def clean(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return round(float(v), 6)
        return v

    rows = []
    for _, row in sample.iterrows():
        values = {c: clean(row[c]) for c in cols}
        numeric = [
            float(row[c]) for c in cols
            if pd.notna(row[c]) and isinstance(row[c], (int, float, np.integer, np.floating))
        ]
        rows.append({
            "player": row["player_name_current"],
            "team": row["current_team_name"],
            "fixture": clean(row.get("fixture")),
            "model_team_feature_count": len(cols),
            "nonzero_numeric_count": sum(abs(v) > 1e-12 for v in numeric),
            "features": values,
        })

    return {
        "actual_model_feature_count": len(model_cols),
        "team_opponent_model_feature_count": len(team_model_cols),
        "audited_columns": cols,
        "players": rows,
    }


def _live_feature_audit(
    scored_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    current_roster: pd.DataFrame,
) -> list[dict]:
    """Expose the actual model inputs for a few high-value current players."""
    roster = current_roster[
        ["element", "player_name_current", "current_team_name"]
    ].copy()

    names = {"Rogers", "Palmer", "Gabriel", "João Pedro", "Joao Pedro"}
    roster = roster[
        roster["player_name_current"].fillna("").astype(str).isin(names)
    ].copy()

    if roster.empty:
        return []

    # Candidate columns: exact released features plus the most useful live inputs.
    wanted_patterns = (
        "ep_next",
        "chance",
        "status",
        "news",
        "minutes",
        "form",
        "points",
        "xg",
        "xa",
        "xgi",
        "expected",
        "team_",
        "opp",
        "fixture",
        "gameweek",
    )

    identity_cols = [
        c for c in ["element", "fpl_code", "fixture", "gameweek"]
        if c in scored_frame.columns
    ]
    candidate_cols = [
        c for c in scored_frame.columns
        if c not in identity_cols
        and any(p in c.lower() for p in wanted_patterns)
    ]

    # Keep audit readable while prioritising known Dastan live-signal columns.
    priority = [
        "ar_ep_next",
        "sig_status_risk",
        "sig_chance_playing",
        "sig_has_news",
    ]
    ordered = []
    for c in priority + candidate_cols:
        if c in scored_frame.columns and c not in ordered:
            ordered.append(c)
    ordered = ordered[:45]

    audit_frame = scored_frame[identity_cols + ordered].copy()
    audit_frame = audit_frame.merge(
        roster,
        on="element",
        how="inner",
        validate="many_to_one",
    )

    # Attach output-side p60 / expected minutes / xPts for the exact fixture row.
    pred_cols = [
        c for c in [
            "element", "fixture", "xpts", "p60", "expected_minutes"
        ] if c in predictions.columns
    ]
    if {"element", "fixture"}.issubset(pred_cols):
        audit_frame = audit_frame.merge(
            predictions[pred_cols],
            on=["element", "fixture"],
            how="left",
            validate="one_to_one",
        )

    def clean(value):
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return round(float(value), 5)
        if isinstance(value, float):
            return round(value, 5)
        return value

    rows = []
    for _, row in audit_frame.iterrows():
        item = {
            "player": row["player_name_current"],
            "team": row["current_team_name"],
        }
        for col in identity_cols + ordered + [
            "xpts", "p60", "expected_minutes"
        ]:
            if col in row.index:
                item[col] = clean(row[col])
        rows.append(item)

    return rows


def _validate_live_fixtures(
    frame: pd.DataFrame,
    bootstrap: dict,
    fixtures: list,
    gameweek: int,
) -> dict:
    """Hard-check that live target rows point at the official current GW fixtures."""
    team_names = {int(t["id"]): t["name"] for t in bootstrap["teams"]}
    official = {
        int(f["id"]): (int(f["team_h"]), int(f["team_a"]))
        for f in fixtures
        if f.get("event") == gameweek
    }
    if len(official) != 10:
        raise RuntimeError(
            f"Official FPL API returned {len(official)} fixtures for GW{gameweek}, expected 10."
        )

    if "fixture" not in frame.columns:
        raise RuntimeError("Live scoring frame has no fixture column.")

    frame_fixture_ids = set(
        pd.to_numeric(frame["fixture"], errors="coerce").dropna().astype(int)
    )
    unknown = sorted(frame_fixture_ids - set(official))
    if unknown:
        raise RuntimeError(
            f"Live scoring frame contains non-GW{gameweek} fixture IDs: {unknown[:10]}"
        )

    fixture_summary = []
    for fixture_id, (home_id, away_id) in official.items():
        fixture_summary.append({
            "fixture": fixture_id,
            "home": team_names[home_id],
            "away": team_names[away_id],
        })

    return {
        "count": len(official),
        "fixtures": fixture_summary,
    }


def _restrict_scoring_frame_to_current_roster(
    frame: pd.DataFrame,
    current_roster: pd.DataFrame,
) -> pd.DataFrame:
    """Allow only exact current FPL element+code identities into live scoring."""
    required = {"element", "fpl_code"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Live scoring frame lacks identity columns: {sorted(missing)}"
        )

    roster_keys = current_roster[
        ["element", "fpl_code_current"]
    ].copy()

    checked = frame.copy()
    checked["element"] = pd.to_numeric(
        checked["element"], errors="coerce"
    ).astype("Int64")
    checked["fpl_code"] = pd.to_numeric(
        checked["fpl_code"], errors="coerce"
    ).astype("Int64")

    roster_keys["element"] = pd.to_numeric(
        roster_keys["element"], errors="coerce"
    ).astype("Int64")
    roster_keys["fpl_code_current"] = pd.to_numeric(
        roster_keys["fpl_code_current"], errors="coerce"
    ).astype("Int64")

    checked = checked.merge(
        roster_keys,
        on="element",
        how="inner",
        validate="many_to_one",
    )

    checked = checked.loc[
        checked["fpl_code"].eq(checked["fpl_code_current"])
    ].drop(columns=["fpl_code_current"]).copy()

    if checked.empty:
        raise RuntimeError(
            "No rows survived exact current FPL roster validation before scoring."
        )

    return checked


def _roster_diagnostics(current_roster: pd.DataFrame) -> list[dict]:
    """Return exact current FPL rows for suspicious names before Dastan metadata is used."""
    search = current_roster[
        current_roster[
            ["first_name_current", "second_name_current", "web_name_current"]
        ]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.contains(r"rogers|salah", regex=True)
    ].copy()

    cols = [
        "element",
        "fpl_code_current",
        "first_name_current",
        "second_name_current",
        "web_name_current",
        "team_id_current",
        "current_team_name",
        "position_current",
        "price_current",
    ]
    return search[cols].to_dict("records")


def _score_target_gameweek(
    *,
    root: Path,
    raw_dir: Path,
    output_dir: Path,
    bootstrap: dict,
    fixtures: list[dict],
    current_roster: pd.DataFrame,
    base_player_matches: pd.DataFrame,
    base_team_matches: pd.DataFrame,
    gameweek: int,
    official_next_gameweek: int,
) -> tuple[pd.DataFrame, dict]:
    """Score one future GW from the same observed history.

    Critical rule: each horizon is built independently. Synthetic GW4 rows are
    never allowed to become 'history' for GW5, etc.
    """
    event = next((e for e in bootstrap["events"] if int(e["id"]) == int(gameweek)), None)
    if event is None:
        raise RuntimeError(f"Official FPL bootstrap has no GW{gameweek} event.")
    deadline = pd.to_datetime(event["deadline_time"], utc=True)

    future_players = _future_player_rows(
        base_player_matches.copy(), bootstrap, fixtures, gameweek
    )
    future_teams = _future_team_rows(
        base_team_matches.copy(), bootstrap, fixtures, gameweek
    )

    player_matches = pd.concat(
        [base_player_matches.copy(), future_players], ignore_index=True, sort=False
    )
    team_matches = pd.concat(
        [base_team_matches.copy(), future_teams], ignore_index=True, sort=False
    )
    del future_players, future_teams
    gc.collect()

    player_matches = _carry_understat_identity_and_history_into_live_rows(
        player_matches, ACTIVE_SEASON, gameweek
    )
    frame = features.build_feature_frame(player_matches, team_matches)
    del player_matches, team_matches
    gc.collect()

    live_frame = frame[
        frame["season"].eq(ACTIVE_SEASON)
        & frame["gameweek"].eq(gameweek)
    ].copy()
    del frame
    gc.collect()
    if live_frame.empty:
        raise RuntimeError(f"Feature builder produced no rows for GW{gameweek}.")

    gw_dir = output_dir / f"gw{gameweek}"
    gw_dir.mkdir(parents=True, exist_ok=True)
    live_frame.to_parquet(gw_dir / "features.parquet", index=False)
    _write_live_snapshot_artifacts(
        gw_dir, bootstrap, gameweek, official_next_gameweek
    )

    scored_frame = data.load(data_dir=gw_dir, check_rows=False)
    scored_frame = _restrict_scoring_frame_to_current_roster(
        scored_frame, current_roster
    )
    fixture_validation = _validate_live_fixtures(
        scored_frame, bootstrap, fixtures, gameweek
    )

    model = predictor.Dastan()
    predictions = model.predict_frame(scored_frame, with_parts=True)
    predictions["element"] = pd.to_numeric(
        predictions["element"], errors="coerce"
    ).astype("Int64")

    for col in [
        "player", "player_name", "web_name", "team", "team_name",
        "current_team_name", "position", "price",
    ]:
        if col in predictions.columns:
            predictions = predictions.drop(columns=[col])

    publish_roster = current_roster[[
        "element", "fpl_code_current", "player_name_current",
        "current_team_name", "position_current", "price_current",
    ]].copy()
    predictions = predictions.merge(
        publish_roster, on="element", how="inner", validate="many_to_one"
    )

    pred_codes = pd.to_numeric(predictions["fpl_code"], errors="coerce").astype("Int64")
    roster_codes = pd.to_numeric(
        predictions["fpl_code_current"], errors="coerce"
    ).astype("Int64")
    if pred_codes.ne(roster_codes).any():
        raise RuntimeError(f"Current-roster identity mismatch while scoring GW{gameweek}.")

    predictions = predictions.rename(columns={
        "player_name_current": "player",
        "current_team_name": "team",
        "position_current": "position",
        "price_current": "price",
    })
    predictions["forecast_gameweek"] = int(gameweek)
    predictions.to_parquet(gw_dir / "predictions.parquet", index=False)

    top = (
        predictions.groupby(
            ["fpl_code", "player", "team", "position", "price"],
            dropna=False, as_index=False,
        )
        .agg(
            xpts=("xpts", "sum"),
            expected_minutes=("expected_minutes", "sum"),
            p60=("p60", "max"),
            fixtures=("fixture", "nunique"),
            fixture_ids=("fixture", lambda x: sorted(set(int(v) for v in x.dropna()))),
        )
        .sort_values("xpts", ascending=False)
        .reset_index(drop=True)
    )
    top["gameweek"] = int(gameweek)
    top.to_parquet(gw_dir / "predictions_player_gw.parquet", index=False)

    audit = {
        "gameweek": int(gameweek),
        "deadline": deadline.isoformat(),
        "players": int(len(top)),
        "fixture_rows": int(len(predictions)),
        "fixture_validation": fixture_validation,
        "model_features": int(len(model.features)),
        "top_10": [
            {
                "player": r.player, "team": r.team, "position": r.position,
                "price": round(float(r.price), 1), "xpts": round(float(r.xpts), 2),
                "expected_minutes": round(float(r.expected_minutes), 1),
                "p60": round(float(r.p60), 3), "fixture_ids": r.fixture_ids,
            }
            for r in top.head(10).itertuples()
        ],
    }
    return top, audit


def run_live_predictions(root: Path | None = None) -> dict:
    """Phase 1 multi-GW validation: score official next GW and the following GW.

    We intentionally start with two horizons. If GW4 remains identical to the
    validated baseline and GW5 is sane, the same independent-horizon path can
    safely be extended to Next 5 / Next 10.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    raw_dir = root / ".cache" / "dastan-live-raw"
    output_dir = root / "data" / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = _get_json(FPL_BOOTSTRAP_URL)
    fixtures = _get_json(FPL_FIXTURES_URL)
    current_roster = _current_roster(bootstrap)
    next_event = _next_gameweek(bootstrap)
    next_gw = int(next_event["id"])
    now = pd.Timestamp.now(tz="UTC")
    next_deadline = pd.to_datetime(next_event["deadline_time"], utc=True)
    if now >= next_deadline:
        raise RuntimeError(
            f"GW{next_gw} deadline has already passed ({next_deadline.isoformat()})."
        )

    targets = [next_gw, next_gw + 1]
    print(f"Multi-GW phase 1: independently scoring {targets}", flush=True)

    with _live_operational_mapping_mode():
        sources.download_sources(
            raw_dir=raw_dir,
            seasons=HISTORY_SEASONS,
            workers=1,
            force=False,
            allow_missing_understat=True,
        )
        base_player_matches, base_team_matches, _ = sources.build_canonical_matches(
            raw_dir, HISTORY_SEASONS
        )

    all_tops = []
    audits = []
    for gw in targets:
        print(f"Multi-GW phase 1: scoring GW{gw}", flush=True)
        top, audit = _score_target_gameweek(
            root=root,
            raw_dir=raw_dir,
            output_dir=output_dir,
            bootstrap=bootstrap,
            fixtures=fixtures,
            current_roster=current_roster,
            base_player_matches=base_player_matches,
            base_team_matches=base_team_matches,
            gameweek=gw,
            official_next_gameweek=next_gw,
        )
        all_tops.append(top)
        audits.append(audit)
        gc.collect()

    combined = pd.concat(all_tops, ignore_index=True, sort=False)
    combined.to_parquet(output_dir / "predictions_multi_gw.parquet", index=False)

    # Preserve the existing API contract: current-GW endpoint still reads this file.
    current_top = all_tops[0].copy()
    current_top.to_parquet(
        output_dir / "predictions_player_gw.parquet", index=False
    )

    # Compact comparison for the players we have been using as integrity checks.
    watch = {"João Pedro", "Rogers", "Haaland", "Palmer", "Gabriel", "Saka"}
    comparison = []
    for name in watch:
        rows = combined[combined["player"].eq(name)].sort_values("gameweek")
        if rows.empty:
            continue
        comparison.append({
            "player": name,
            "team": str(rows.iloc[0]["team"]),
            "gameweeks": {
                str(int(r.gameweek)): {
                    "xpts": round(float(r.xpts), 3),
                    "expected_minutes": round(float(r.expected_minutes), 1),
                    "p60": round(float(r.p60), 3),
                    "fixture_ids": r.fixture_ids,
                }
                for r in rows.itertuples()
            },
        })

    return {
        "status": "ok",
        "mode": "multi_gw_phase_1",
        "season": ACTIVE_SEASON,
        "official_next_gameweek": next_gw,
        "forecast_gameweeks": targets,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_fpl_roster_players": int(len(current_roster)),
        "gameweek_audits": audits,
        "comparison": comparison,
        "safety": {
            "independent_horizons": True,
            "future_rows_used_as_history": False,
            "ep_next_used_only_for_official_next_gw": True,
            "later_gw_ep_next_missing_value": -1.0,
        },
    }
