"""Loading the frame, and the one correctness property it must satisfy.

The training frame is one row per player per fixture, six seasons, 163,072 rows. It
carries rolling historical features, the pre-deadline snapshot columns, and targets.

Everything here exists to protect a single invariant: **a feature vector may only
contain information that existed at the deadline the squad was picked for.** That is
harder than it sounds, and the subtle violation is documented in `anchor_to_deadline`
below.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import gc
from .seasons import load_registry

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FRAME = DATA / "features.parquet"
EP_NEXT = DATA / "pre_deadline_ep_next.parquet"
SIGNALS = DATA / "pre_deadline_signals.parquet"
FEATURE_COLS = ROOT / "models" / "feature_cols.json"

_REGISTRY = load_registry()
SEASONS = list(_REGISTRY["release_seasons"])
EXPECTED_ROWS = int(_REGISTRY["release_rows"])

# Rolling-history columns. `opp_` is deliberately excluded: opponent form genuinely
# differs between the two fixtures of a double gameweek and must not be flattened.
HISTORY_PREFIXES = ("player_", "team_", "ar_", "cx_", "p_any_", "p60_", "e_minutes_")


def anchor_to_deadline(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    if cols is None:
        cols = [c for c in df.columns if c.startswith(HISTORY_PREFIXES)]

    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df

    keys = ["season", "gameweek", "fpl_code"]
    sort_cols = ["season", "gameweek", "fpl_code", "kickoff_time"]

    chunk_size = 20

    for start in range(0, len(cols), chunk_size):
        chunk = cols[start:start + chunk_size]

        tmp = df[sort_cols + chunk].copy()
        tmp["_row_id"] = np.arange(len(tmp), dtype=np.int32)

        tmp = tmp.sort_values(sort_cols, kind="mergesort")

        tmp[chunk] = (
            tmp.groupby(keys, sort=False)[chunk]
            .transform("first")
        )

        tmp = tmp.sort_values("_row_id", kind="mergesort")

        df.loc[:, chunk] = tmp[chunk].to_numpy(copy=False)

        del tmp
        gc.collect()

    return df


def assert_deadline_anchored(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    """One player, one deadline, one history. Raises if that is violated."""
    if cols is None:
        cols = [c for c in df.columns if c.startswith(HISTORY_PREFIXES)]
    cols = [c for c in cols if c in df.columns]
    dup = df[df.duplicated(["season", "gameweek", "fpl_code"], keep=False)]
    if dup.empty:
        return
    worst = dup.groupby(["season", "gameweek", "fpl_code"])[cols].nunique().max()
    bad = worst[worst > 1]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} history columns differ within a player-gameweek -- the frame "
            f"is not deadline-anchored. Worst: {bad.sort_values(ascending=False).head(5).to_dict()}")


# Pre-deadline snapshot columns, joined from separate artefacts. Provenance for these
# is strict and is described in docs/DATA.md: a snapshot is used only if it names the
# gameweek as `is_next`, agrees on its deadline, and was captured before it.
CANDIDATES = {
    "availability": {
        "art": SIGNALS, "on": ["season", "gameweek", "fpl_code"],
        "cols": ["sig_status_risk", "sig_chance_playing", "sig_has_news"]},
    "set_pieces": {
        "art": SIGNALS, "on": ["season", "gameweek", "fpl_code"],
        "cols": ["sig_pens_order", "sig_fk_order", "sig_corners_order"]},
    "biographical": {
        "art": SIGNALS, "on": ["season", "gameweek", "fpl_code"],
        "cols": ["sig_age_years", "sig_days_at_club"]},
}
EP = {"art": EP_NEXT, "on": ["season", "gameweek", "fpl_code"], "cols": ["ep_next"]}

SHIPPED_CANDIDATES = ["availability"]


def load(
    candidates=tuple(CANDIDATES),
    check_rows: bool = True,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """The frame with every snapshot artefact joined.

    Missing values become **-1, never 0**. This matters more than it looks: 54,430
    rows carry a genuine `ep_next` of exactly zero (FPL expecting nothing from a
    player), and filling absent snapshots with 0 would teach the model that "we have
    no data" and "FPL expects nothing" are the same event. They are not.
    """
    base = Path(data_dir) if data_dir is not None else DATA
    frame_path = base / FRAME.name
    df = pd.read_parquet(frame_path)
    if check_rows and len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS:,} rows, got {len(df):,}")
    assert_deadline_anchored(df)
    n0 = len(df)

    wanted: dict = {}
    for name in list(candidates) + ["__ep__"]:
        spec = EP if name == "__ep__" else CANDIDATES[name]
        artifact = base / Path(spec["art"]).name
        wanted.setdefault((artifact, tuple(spec["on"])), []).extend(spec["cols"])

    for (art, on), cols in wanted.items():
        a = pd.read_parquet(art)[list(on) + cols].drop_duplicates(subset=list(on))
        df = df.merge(a, on=list(on), how="left", validate="many_to_one")
        if len(df) != n0:
            raise RuntimeError(f"{Path(art).name} join fanned out: {n0:,} -> {len(df):,}")
        for c in cols:
            df[c] = df[c].fillna(-1.0)
    df["ar_ep_next"] = df["ep_next"]
    return df


def feature_sets(df: pd.DataFrame, candidates=SHIPPED_CANDIDATES):
    """(baseline features, baseline + the named candidate families)."""
    base = [c for c in json.loads(FEATURE_COLS.read_text()) if c in df.columns]
    extra = [c for n in candidates for c in CANDIDATES[n]["cols"]]
    keep = [c for c in base if c not in extra]
    return keep, keep + [c for c in extra if c in df.columns]


def shipped_features(df: pd.DataFrame) -> list[str]:
    """Exactly the 286 columns the released weights were trained on, in order."""
    cols = json.loads(FEATURE_COLS.read_text())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"frame is missing {len(missing)} shipped features: {missing[:5]}")
    return cols
