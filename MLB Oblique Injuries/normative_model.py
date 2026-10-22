"""
normative_model.py — does a covariate model explain enough bat-speed noise to be worth it?

The idea is a risk-adjusted chart: predict each swing's bat speed from the player's
own level plus the conditions of that swing (count, pitch type, pitch speed,
location, platoon, park, home/away), then monitor the residuals instead of raw
bat speed. Before building that, this module answers one question:

    How much of the game-to-game variation in a player's bat speed do those
    conditions explain?

If the answer is a few percent, residual charts will look like the raw-bat-speed
charts and are not worth building. If it is substantial, they are.

Method
------
Within each player-season, bat speed and every covariate are centered on the
player-season mean (the "within" or fixed-effects estimator). Regressing centered
bat speed on centered covariates gives each condition's effect net of who the
hitter is, which is the same fixed-effect part a random-intercept mixed model
would estimate. Effects are fit on half the player-seasons and evaluated on the
other half (then swapped), so the variance explained is out of sample.

Data
----
Pitch-level Statcast via pybaseball, one player-season per request. Only swings
with bat-tracking data are kept, with only the columns used here, and each
player-season is cached under SWING_DIR. Swings during or within
HEALTHY_BUFFER_DAYS after any IL stint are dropped, so the norm describes
healthy performance.

Requires:  pip install pybaseball pandas numpy
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import oblique_study as study

SWING_DIR = Path("data/statcast_swings")
BAT_TRACKING_SEASONS = [2024, 2025, 2026]
HEALTHY_BUFFER_DAYS = 30    # drop games this many days after an IL activation
SAMPLE_PER_SEASON = 60      # league hitters sampled per season for the norm
STATCAST_PAUSE = 3.0        # seconds between Baseball Savant requests

KEEP_COLS = ["game_pk", "game_date", "game_type", "batter", "at_bat_number", "pitch_number",
             "description", "des", "events", "bat_speed", "swing_length", "launch_speed",
             "balls", "strikes", "pitch_type", "release_speed", "plate_x", "plate_z",
             "sz_top", "sz_bot", "stand", "p_throws", "inning_topbot", "home_team"]

SWING_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
                      "hit_into_play"}
PITCH_GROUPS = {
    "FF": "fastball", "SI": "fastball", "FC": "cutter",
    "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking",
    "KC": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
}


# =============================================================================
# 1. Pull swings, one player-season at a time
# =============================================================================
def fangraphs_to_mlbam(fg_ids):
    """Map FanGraphs IDs to MLBAM IDs through the Chadwick register."""
    from pybaseball import playerid_reverse_lookup
    ids = pd.to_numeric(pd.Series(list(fg_ids)), errors="coerce").dropna().astype(int).unique().tolist()
    lookup = playerid_reverse_lookup(ids, key_type="fangraphs")
    return {str(int(f)): int(m) for f, m in zip(lookup["key_fangraphs"], lookup["key_mlbam"])}


def get_swings(mlbam_id, season, pause=None):
    """
    Bat-tracked swings for one player-season (regular season, no bunts), cached.
    Returns an empty frame if the player has none.
    """
    path = SWING_DIR / str(season) / f"{mlbam_id}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["game_date"])

    from pybaseball import statcast_batter
    df = statcast_batter(f"{season}-03-01", f"{season}-11-30", player_id=mlbam_id)
    time.sleep(STATCAST_PAUSE if pause is None else pause)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df[(df["game_type"] == "R")
            & df["description"].isin(SWING_DESCRIPTIONS)
            & df["bat_speed"].notna()]
    is_bunt = df["des"].fillna("").str.contains("bunt", case=False) | \
        df["events"].fillna("").str.contains("bunt", case=False)
    df = df[~is_bunt].drop(columns=["des", "events", "game_type"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = season

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def drop_competitive_floor(swings, pct=0.10, ev_keep=90):
    """
    Approximate Baseball Savant's "competitive swings": drop each player-season's
    slowest `pct` of swings unless the ball was hit at least `ev_keep` mph.
    (Savant's exact rule may differ; check before publishing.)
    """
    floor = swings.groupby(["batter", "season"])["bat_speed"].transform(lambda s: s.quantile(pct))
    keep = (swings["bat_speed"] >= floor) | (swings["launch_speed"] >= ev_keep)
    return swings[keep]


def healthy_only(swings, injuries, mlbam_map, buffer_days=None):
    """
    Drop swings during an IL stint or within buffer_days after activation.

    injuries: the wide injury table from study.load_all_injuries() (all players,
              all stints), keyed by key_fangraphs and Season.
    mlbam_map: FanGraphs ID -> MLBAM ID.
    """
    buffer_days = HEALTHY_BUFFER_DAYS if buffer_days is None else buffer_days
    stints = []
    k = 1
    while f"IL_Retro_{k}" in injuries.columns:
        s = injuries[["key_fangraphs", f"IL_Retro_{k}", f"Activate_{k}"]].dropna(subset=[f"IL_Retro_{k}"])
        stints.append(s.set_axis(["key_fangraphs", "start", "end"], axis=1))
        k += 1
    stints = pd.concat(stints, ignore_index=True)
    stints["batter"] = stints["key_fangraphs"].map(mlbam_map)
    stints = stints.dropna(subset=["batter"])
    stints["batter"] = stints["batter"].astype(int)
    stints["end"] = stints["end"].fillna(pd.Timestamp.max.normalize()) + pd.Timedelta(days=buffer_days)

    merged = swings.reset_index().merge(stints[["batter", "start", "end"]], on="batter", how="inner")
    bad = merged.loc[(merged["game_date"] >= merged["start"]) & (merged["game_date"] <= merged["end"]), "index"]
    return swings.drop(index=bad.unique())


def sample_league_swings(cohort, injuries, seasons=None, n_per_season=None, seed=2026, pause=None):
    """
    Healthy, competitive swings for a random sample of league hitters (200+ PA),
    excluding the oblique cohort's own player-seasons.
    """
    seasons = seasons or BAT_TRACKING_SEASONS
    n_per_season = SAMPLE_PER_SEASON if n_per_season is None else n_per_season
    rng = np.random.default_rng(seed)

    picks = []
    for season in seasons:
        batters = study.fg_league_batters(season)
        cohort_ids = set(cohort.loc[cohort["Season"] == season, "key_fangraphs"])
        pool = batters[~batters["key_fangraphs"].isin(cohort_ids)]
        idx = rng.choice(len(pool), size=min(n_per_season, len(pool)), replace=False)
        picks.append(pool.iloc[idx][["key_fangraphs", "Season"]])
    picks = pd.concat(picks, ignore_index=True)

    mlbam_map = fangraphs_to_mlbam(picks["key_fangraphs"])
    picks["batter"] = picks["key_fangraphs"].map(mlbam_map)
    print(f"{picks['batter'].isna().sum()} sampled hitters had no MLBAM match and were skipped")
    picks = picks.dropna(subset=["batter"])

    frames = []
    for i, (mid, season) in enumerate(picks[["batter", "Season"]].itertuples(index=False), 1):
        sw = get_swings(int(mid), int(season), pause)
        if not sw.empty:
            frames.append(sw)
        if i % 25 == 0:
            print(f"  {i}/{len(picks)} player-seasons")
    swings = pd.concat(frames, ignore_index=True)
    swings = drop_competitive_floor(swings)
    return healthy_only(swings, injuries, mlbam_map).reset_index(drop=True)


# =============================================================================
# 2. Covariates
# =============================================================================
def swing_features(swings):
    """
    Numeric design matrix for the swing conditions:
      count (12 levels), pitch group, pitch speed (and by group), location relative
      to the batter and the zone (with squares and interaction), platoon, home/away, park.
    """
    s = swings
    f = pd.DataFrame(index=s.index)

    count = s["balls"].astype(int).astype(str) + "-" + s["strikes"].astype(int).astype(str)
    f = f.join(pd.get_dummies(count, prefix="count", dtype=float).drop(columns="count_0-0", errors="ignore"))

    group = s["pitch_type"].map(PITCH_GROUPS).fillna("other")
    f = f.join(pd.get_dummies(group, prefix="pitch", dtype=float).drop(columns="pitch_fastball", errors="ignore"))

    velo = s["release_speed"] - s["release_speed"].mean()
    f["velo"] = velo
    for g in ("breaking", "offspeed", "cutter"):
        if f"pitch_{g}" in f.columns:
            f[f"velo_x_{g}"] = velo * f[f"pitch_{g}"]

    # Horizontal location toward the batter: positive = inside
    x_in = np.where(s["stand"] == "R", -s["plate_x"], s["plate_x"])
    height = s["sz_top"] - s["sz_bot"]
    z_rel = (s["plate_z"] - (s["sz_bot"] + height / 2)) / height  # 0 = middle of the zone
    f["loc_in"], f["loc_in2"] = x_in, x_in ** 2
    f["loc_z"], f["loc_z2"] = z_rel, z_rel ** 2
    f["loc_in_z"] = x_in * z_rel

    f["same_hand"] = (s["stand"] == s["p_throws"]).astype(float)
    f["home"] = (s["inning_topbot"] == "Bot").astype(float)
    f = f.join(pd.get_dummies(s["home_team"], prefix="park", dtype=float).iloc[:, 1:])
    return f


# =============================================================================
# 3. How much do the covariates explain?
# =============================================================================
def _within(df, cols, groups):
    """Center columns within groups (player-season)."""
    return df[cols] - df.groupby(groups)[cols].transform("mean")


def variance_explained_check(swings, min_swings_per_game=3, seed=2026):
    """
    Out-of-sample share of within-player-season bat-speed variance explained by the
    swing conditions, at the swing level and at the game level (what the chart sees).

    Returns (summary dict, coefficient Series fit on all data, game-level frame).
    """
    s = swings.dropna(subset=["bat_speed", "release_speed", "plate_x", "plate_z",
                              "sz_top", "sz_bot", "balls", "strikes"]).copy()
    s["ps"] = s["batter"].astype(str) + "_" + s["season"].astype(str)
    X = swing_features(s)
    cols = list(X.columns)
    data = pd.concat([s[["ps", "game_pk", "bat_speed"]], X], axis=1)

    yw = data["bat_speed"] - data.groupby("ps")["bat_speed"].transform("mean")
    Xw = _within(data, cols, "ps")

    # Two folds of player-seasons: fit on one, evaluate on the other
    rng = np.random.default_rng(seed)
    ps_ids = data["ps"].unique()
    fold = dict(zip(ps_ids, rng.integers(0, 2, len(ps_ids))))
    data["fold"] = data["ps"].map(fold)
    data["resid"] = np.nan
    for k in (0, 1):
        train, test = data["fold"] != k, data["fold"] == k
        beta, *_ = np.linalg.lstsq(Xw[train].to_numpy(), yw[train].to_numpy(), rcond=None)
        data.loc[test, "resid"] = yw[test] - Xw[test].to_numpy() @ beta
    data["yw"] = yw

    swing_r2 = 1 - data["resid"].var() / data["yw"].var()

    # Game level: a player's game-average deviation, raw vs adjusted
    games = data.groupby(["ps", "game_pk"]).agg(
        n=("yw", "size"), raw=("yw", "mean"), adjusted=("resid", "mean")).reset_index()
    games = games[games["n"] >= min_swings_per_game]
    # re-center within player-season so only game-to-game variation remains
    for c in ("raw", "adjusted"):
        games[c] = games[c] - games.groupby("ps")[c].transform("mean")
    game_r2 = 1 - games["adjusted"].var() / games["raw"].var()

    beta_all, *_ = np.linalg.lstsq(Xw.to_numpy(), yw.to_numpy(), rcond=None)
    coefs = pd.Series(beta_all, index=cols)

    summary = {
        "player_seasons": len(ps_ids),
        "swings": len(data),
        "games": len(games),
        "swing_sd_within": data["yw"].std(),
        "swing_sd_adjusted": data["resid"].std(),
        "swing_variance_explained": swing_r2,
        "game_sd_raw": games["raw"].std(),
        "game_sd_adjusted": games["adjusted"].std(),
        "game_variance_explained": game_r2,
    }
    return summary, coefs, games
