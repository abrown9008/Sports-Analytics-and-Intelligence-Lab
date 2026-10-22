"""
oblique_study.py — data pipeline for the MLB batter oblique reinjury study.

The notebook is a sequence of short steps that call these functions:

    1. Injury data       load_all_injuries(), select_oblique_batters()
    2. Outcomes          add_injury_groups(), summarize_outcomes()
    3. Game logs         get_metric_logs()            (FanGraphs, cached to disk)
    4. League variation  get_league_metric_logs(), league_within_player_variation()
    5. Reference values  add_reference_values(), estimate_prior_weight()
    6. Monitoring data   build_monitoring_frame()
    7. Control chart     add_standardized_scores(), run_chart_grid(),
                         evaluate_grid(), leave_one_season_out()
    8. Everything above for one metric:  analyze_metric()

Game logs come from FanGraphs' game-log API, one small JSON request per
player-season. Each response is saved to disk with every column the study
might use (wOBA, exit velocity, hard-hit rate, bat speed, ...), so switching
metrics never requires downloading again, and an interrupted run picks up
where it stopped.

Monitored metrics are defined in METRICS. Each has a per-game value and a
per-game weight (the number of events the value averages over), so games
with more plate appearances or batted balls count for more.

Requires:  pip install requests pandas numpy openpyxl
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# Configuration
# =============================================================================
INJURY_URL = (
    "https://raw.githubusercontent.com/abrown9008/Sports-Analytics-and-Intelligence-Lab/"
    "main/MLB%20Oblique%20Injuries/MLB%20Injuries%20Raw%20Data.xlsx"
)
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PITCHER_POSITIONS = {"P", "SP", "RP"}

# League-average wOBA by season (FanGraphs league totals). Cite FanGraphs.
LEAGUE_WOBA = {2021: 0.314, 2022: 0.310, 2023: 0.318, 2024: 0.310, 2025: 0.313, 2026: 0.316}

PRIOR_WEIGHT = 7        # default games of league average blended into each baseline
LEAGUE_MIN_PA = 200     # batters below this season PA total are left out of league variation
LEAGUE_MIN_GAMES = 20   # ...as are batters with fewer games than this
REQUEST_PAUSE = 2.0     # seconds between FanGraphs requests

CACHE_DIR = Path("data/fg_gamelogs")    # league batter lists live here
GAMES_DIR = CACHE_DIR / "games"         # full per-game logs: games/<season>/<playerid>.csv

# Monitored metrics. Lower values must mean worse performance.
#   value:   FanGraphs game-log column holding the game's value
#   weight:  events the value averages over ("woba_denom" is computed from AB, BB, ...)
#   first_season: first season the metric exists
METRICS = {
    "woba":      {"value": "wOBA",        "weight": "woba_denom",        "first_season": 2021,
                  "label": "wOBA"},
    "ev":        {"value": "EV",          "weight": "Events",            "first_season": 2021,
                  "label": "Average exit velocity"},
    "hardhit":   {"value": "HardHit%",    "weight": "Events",            "first_season": 2021,
                  "label": "Hard-hit rate"},
    "bat_speed": {"value": "AvgBatSpeed", "weight": "CompetitiveSwings", "first_season": 2024,
                  "label": "Average bat speed"},
}

# Columns saved from each game-log response (whatever subset FanGraphs returns)
SAVE_COLS = [
    "Date", "dh", "Team", "Opp", "BatOrder", "Pos",
    "PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "IBB", "SO", "HBP", "SF", "SH",
    "wOBA", "wRC+", "ISO", "BABIP", "K%", "BB%",
    "SwStr%", "Contact%", "O-Swing%", "Z-Contact%", "Pull%", "Hard%", "Soft%",
    "Events", "EV", "LA", "maxEV", "Barrels", "Barrel%", "HardHit", "HardHit%",
    "AvgBatSpeed", "FastSwing%", "SwingLength", "CompetitiveSwings",
    "SquaredUpSwing%", "BlastSwing%",
]


# =============================================================================
# 1. Injury data -> one row per player-season, then the oblique batter cohort
# =============================================================================
STINT_COLS = {"IL Retro Date": "IL_Retro", "Return Date": "Activate", "Injury / Surgery": "Injury"}
KEEP_COLS = ["Name", "URL", "Team", "Pos", "IL Retro Date", "Return Date",
             "Injury / Surgery", "sub_url", "key_fangraphs"]


def load_injury_sheet(season, source=None):
    """Read one season's sheet, recovering missing FanGraphs IDs from the URL."""
    source = source or INJURY_URL
    raw = pd.read_excel(source, engine="openpyxl", sheet_name=str(season))[KEEP_COLS].copy()
    from_url = raw["URL"].str.extract(r"/players/[^/]+/([^/]+)/", expand=False)
    raw["key_fangraphs"] = raw["key_fangraphs"].fillna(from_url).astype(str)
    for col in ("IL Retro Date", "Return Date"):
        raw[col] = pd.to_datetime(raw[col], errors="coerce")
    return raw


def pivot_stints(raw):
    """One row per player: player attributes plus IL_Retro_k / Activate_k / Injury_k."""
    df = raw.sort_values(["key_fangraphs", "IL Retro Date"], na_position="last").copy()
    df["stint"] = df.groupby("key_fangraphs").cumcount() + 1

    # Player-level attributes come from the first stint (a player may change teams)
    players = df.drop_duplicates("key_fangraphs")[
        ["key_fangraphs", "Name", "Team", "Pos", "URL", "sub_url"]
    ]
    wide = (
        df.set_index(["key_fangraphs", "stint"])[list(STINT_COLS)]
        .rename(columns=STINT_COLS)
        .unstack("stint")
    )
    wide.columns = [f"{col}_{n}" for col, n in wide.columns]
    max_n = int(df["stint"].max())
    ordered = [f"{c}_{n}" for n in range(1, max_n + 1) for c in STINT_COLS.values()]
    wide = wide[ordered].reset_index()

    out = players.merge(wide, on="key_fangraphs", how="left")
    assert out["key_fangraphs"].is_unique
    assert out["key_fangraphs"].ne("nan").all(), "unresolved FanGraphs IDs"
    return out


def load_all_injuries(seasons=None, source=None):
    """Stack every season's wide table, with a Season column."""
    seasons = seasons or SEASONS
    return pd.concat(
        [pivot_stints(load_injury_sheet(s, source)).assign(Season=s) for s in seasons],
        ignore_index=True,
    )


def select_oblique_batters(all_wide):
    """Non-pitchers whose first IL stint of the season was an oblique and who returned."""
    # A position counts as pitcher-only if every part is a pitcher role ("SP/RP" yes,
    # two-way "SP/DH" no)
    pitcher_only = all_wide["Pos"].fillna("").str.split("/").apply(
        lambda parts: all(p in PITCHER_POSITIONS for p in parts))
    is_batter = ~pitcher_only
    first_oblique = all_wide["Injury_1"].str.contains("oblique", case=False, na=False)
    returned = all_wide["Activate_1"].notna()
    return all_wide[is_batter & first_oblique & returned].reset_index(drop=True)


def load_cohort(seasons=None, source=None):
    """Steps 1a-1c in one call: load, pivot, and filter to the study cohort."""
    return select_oblique_batters(load_all_injuries(seasons, source))


# =============================================================================
# 2. Classify the second injury into analysis groups
# =============================================================================
EXACT_MAP = {
    "Strained oblique": "oblique_recurrence",

    "Strained hamstring": "lower_body_strain",
    "Strained quad": "lower_body_strain",
    "Strained calf": "lower_body_strain",
    "Strained hip flexor": "lower_body_strain",

    "Strained lower back": "lower_back",
    "Lower back inflammation": "lower_back",
    "Back inflammation": "lower_back",
    "Back spasms": "lower_back",

    "Knee inflammation": "knee_nontraumatic",
    "Knee discomfort (patellar subluxation)": "knee_nontraumatic",
    "Sprained knee": "knee_nontraumatic",
    "Arthroscopic knee surgery": "knee_nontraumatic",

    "Plantar fascitis": "other_overuse",
    "Ankle/foot tendinitis": "other_overuse",
    "Hand inflammation": "other_overuse",

    "Sprained ankle": "ankle_sprain",

    "Fractured finger (left pinky)": "traumatic",
    "Fractured hand (right pinkie)": "traumatic",
    "Fractured thumb (left)": "traumatic",
    "Forearm surgery (fracture)": "traumatic",
    "Hamate surgery (left hand)": "traumatic",  # judgment call: possibly swing-related
    "Sprained thumb": "traumatic",
    "Sprained hand (left)": "traumatic",
    "Fractured knee": "traumatic",
    "Knee bone bruise": "traumatic",
    "Shoulder subluxation (left)": "traumatic",
}

# Fallback for labels not in EXACT_MAP. First matching rule wins; matches are flagged.
KEYWORD_RULES = [
    (["fracture", "hamate", "bruise", "contusion", "hit by pitch", "concussion",
      "subluxation (left)", "subluxation (right)", "dislocat"], "traumatic"),
    (["oblique"], "oblique_recurrence"),
    (["back", "lumbar", "spasm"], "lower_back"),
    (["hamstring", "quad", "calf", "hip flexor", "groin", "adductor"], "lower_body_strain"),
    (["knee", "patella", "meniscus"], "knee_nontraumatic"),
    (["ankle sprain", "sprained ankle"], "ankle_sprain"),
    (["sprained hand", "sprained thumb", "sprained finger", "sprained wrist"], "traumatic"),
    (["tendinitis", "tendinopathy", "fasci", "inflammation", "bursitis"], "other_overuse"),
]

PRIMARY_GROUPS = {"oblique_recurrence", "lower_body_strain", "lower_back"}
SENSITIVITY_GROUPS = PRIMARY_GROUPS | {"knee_nontraumatic", "other_overuse", "ankle_sprain"}
NEGATIVE_CONTROL_GROUPS = {"traumatic"}


def classify_injury(label):
    """Return (group, match_method) for one injury label."""
    if pd.isna(label) or str(label).strip() == "":
        return "none", "none"
    label = str(label).strip()
    if label in EXACT_MAP:
        return EXACT_MAP[label], "exact"
    lowered = label.lower()
    for keywords, group in KEYWORD_RULES:
        if any(k in lowered for k in keywords):
            return group, "keyword"
    return "unclassified", "unmatched"


def add_injury_groups(df, injury_col="Injury_2"):
    """Add group, outcome flags, and competing-risks codes (0 censored, 1 event, 2 competing)."""
    out = df.copy()
    results = out[injury_col].apply(classify_injury)
    out["injury_group"] = results.str[0]
    out["match_method"] = results.str[1]

    has_injury = out["injury_group"] != "none"
    out["primary_outcome"] = out["injury_group"].isin(PRIMARY_GROUPS).astype(int)
    out["sensitivity_outcome"] = out["injury_group"].isin(SENSITIVITY_GROUPS).astype(int)
    out["negative_control"] = out["injury_group"].isin(NEGATIVE_CONTROL_GROUPS).astype(int)
    out["event_primary"] = np.select([out["primary_outcome"] == 1, has_injury], [1, 2], default=0)
    out["event_sensitivity"] = np.select([out["sensitivity_outcome"] == 1, has_injury], [1, 2], default=0)

    review = out.loc[out["match_method"].isin(["keyword", "unmatched"]), [injury_col, "injury_group"]]
    if not review.empty:
        print("Review these labels and add them to EXACT_MAP:")
        for label, group in review.drop_duplicates().itertuples(index=False):
            print(f"  {label!r} -> {group}")
    return out


def summarize_outcomes(cohort):
    """Print group counts and outcome totals."""
    print(cohort["injury_group"].value_counts().to_string(), "\n")
    print(f"Players with a second injury: {(cohort['injury_group'] != 'none').sum()} of {len(cohort)}")
    print(f"Primary outcome events:       {cohort['primary_outcome'].sum()}")
    print(f"Sensitivity outcome events:   {cohort['sensitivity_outcome'].sum()}")
    print(f"Negative control events:      {cohort['negative_control'].sum()}")


# =============================================================================
# 3. FanGraphs game logs (port of baseballr::fg_batter_game_logs), cached
# =============================================================================
FG_GAMELOG_URL = "https://www.fangraphs.com/api/players/game-log"
FG_LEADERS_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
HEADERS = {"User-Agent": "okhttp/4.12.0"}  # same agent baseballr uses

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))


def fg_batter_game_logs(playerid, season, timeout=30):
    """One season of raw FanGraphs game logs (one row per game), or an empty frame."""
    params = {"playerid": playerid, "position": "", "type": 0, "gds": "", "gde": "",
              "z": 1703085978, "season": season}
    try:
        resp = _session.get(FG_GAMELOG_URL, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json().get("mlb", [])
    except (requests.RequestException, ValueError) as e:
        print(f"  FanGraphs request failed for {playerid} ({season}): {e}")
        return pd.DataFrame()

    if len(rows) <= 1:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:])  # first row is the season total
    df["Date"] = df["Date"].astype(str).str.extract(r">([^<]+)<", expand=False)
    return df


def _cache_path(playerid, season):
    return Path(GAMES_DIR) / str(season) / f"{playerid}.csv"


def fetch_player_season(playerid, season, pause=None):
    """
    Make sure one player-season is in the cache. Returns True if it is (or was
    just downloaded), False if FanGraphs returned nothing. Failures are not cached,
    so they are retried on the next run.
    """
    path = _cache_path(playerid, season)
    if path.exists():
        return True
    raw = fg_batter_game_logs(playerid, season)
    time.sleep(REQUEST_PAUSE if pause is None else pause)
    if raw.empty:
        return False
    keep = raw[[c for c in SAVE_COLS if c in raw.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(path, index=False)
    return True


def _to_number(series):
    """Numeric conversion that also handles strings like '45.2%'."""
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.rstrip("%").str.strip()
    return pd.to_numeric(series, errors="coerce")


def read_metric_log(playerid, season, metric):
    """
    One player-season from the cache, reduced to one metric:
    key_fangraphs, Season, game_date, dh, value, weight, num (= value x weight), game_num.
    Games with zero weight (no PA or no batted balls) are dropped.
    """
    spec = METRICS[metric]
    path = _cache_path(playerid, season)
    header = pd.read_csv(path, nrows=0).columns
    wanted = ["Date", "dh", spec["value"]] + (
        ["AB", "BB", "IBB", "SF", "HBP"] if spec["weight"] == "woba_denom" else [spec["weight"]])
    df = pd.read_csv(path, usecols=[c for c in wanted if c in header])

    if spec["value"] not in df.columns:
        return pd.DataFrame()
    for col in df.columns:
        if col != "Date":
            df[col] = _to_number(df[col])

    if spec["weight"] == "woba_denom":
        parts = [df.get(c, pd.Series(0, index=df.index)).fillna(0)
                 for c in ("AB", "BB", "IBB", "SF", "HBP")]
        weight = parts[0] + parts[1] - parts[2] + parts[3] + parts[4]
    else:
        weight = df[spec["weight"]].fillna(0) if spec["weight"] in df.columns else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "key_fangraphs": str(playerid),
        "Season": int(season),
        "game_date": pd.to_datetime(df["Date"], errors="coerce"),
        "dh": df["dh"].fillna(0).astype(int) if "dh" in df.columns else 0,
        "value": df[spec["value"]],
        "weight": weight,
    })
    out = out[(out["weight"] > 0) & out["value"].notna()].copy()
    out["num"] = out["value"] * out["weight"]
    out = out.sort_values(["game_date", "dh"]).reset_index(drop=True)
    out["game_num"] = np.arange(1, len(out) + 1)
    return out


def get_metric_logs(pairs, metric, id_col="key_fangraphs", season_col="Season", pause=None):
    """
    Game-by-game values of one metric for every (player, season) pair, downloading
    any player-season not yet in the cache. Seasons before the metric existed are skipped.
    """
    pairs = pairs[[id_col, season_col]].dropna().drop_duplicates()
    pairs = pairs[pairs[season_col] >= METRICS[metric]["first_season"]]
    logs, missing = [], []
    for i, (pid, season) in enumerate(pairs.itertuples(index=False), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(pairs)} player-seasons")
        log = (read_metric_log(pid, int(season), metric)
               if fetch_player_season(pid, int(season), pause) else pd.DataFrame())
        (logs.append(log) if not log.empty else missing.append((pid, int(season))))
    if missing:
        print(f"{metric}: {len(missing)} player-seasons have no games with this metric")
        if len(missing) == len(pairs):
            print(f"  None at all. Check that '{METRICS[metric]['value']}' is the right "
                  "FanGraphs column name (look at a cached CSV).")
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


# =============================================================================
# 4. League-level within-player variation
# =============================================================================
def fg_league_batters(season, min_pa=None):
    """FanGraphs IDs of non-pitchers with at least min_pa PA in a season (cached)."""
    min_pa = LEAGUE_MIN_PA if min_pa is None else min_pa
    path = Path(CACHE_DIR) / f"league_batters_{season}_{min_pa}PA.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"key_fangraphs": str})

    params = {"pos": "np", "stats": "bat", "lg": "all", "qual": min_pa, "type": 8,
              "season": season, "season1": season, "month": 0, "ind": 0, "team": 0,
              "rost": 0, "pageitems": 2000000000, "pagenum": 1}
    resp = _session.get(FG_LEADERS_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = pd.DataFrame(resp.json()["data"])

    out = pd.DataFrame({
        "key_fangraphs": data["playerid"].astype(str),
        "Season": season,
        "PlayerName": data.get("PlayerName"),
        "PA": pd.to_numeric(data["PA"], errors="coerce"),
    })
    out = out[out["PA"] >= min_pa].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def get_league_metric_logs(metric, seasons=None, min_pa=None, pause=None):
    """One metric's game logs for every qualifying batter (slow only the first time)."""
    seasons = [s for s in (seasons or SEASONS) if s >= METRICS[metric]["first_season"]]
    batters = pd.concat([fg_league_batters(s, min_pa) for s in seasons], ignore_index=True)
    return get_metric_logs(batters, metric, pause=pause)


def league_within_player_variation(game_logs, min_games=None):
    """
    Per season, how much a typical batter's game value varies around his own mean.

      n_players             batters with at least min_games games
      league_mean           weighted league mean among those batters
      mean_weight_per_game  average events (PA, batted balls, swings) per game
      pooled_within_sd      pooled game-level SD: sqrt(sum((n_i-1) s_i^2) / sum(n_i-1))
      sigma_unit            pooled per-event SD; a game with k events has
                            expected SD sigma_unit / sqrt(k)

    sigma_unit comes from weighted squared deviations from each batter's weighted
    mean: if single events have variance sigma^2, a k-event game average has
    variance sigma^2 / k, so k * (game value - mean)^2 estimates sigma^2.
    """
    min_games = LEAGUE_MIN_GAMES if min_games is None else min_games
    rows = []
    for season, g in game_logs.groupby("Season"):
        counts = g.groupby("key_fangraphs")["value"].count()
        g = g[g["key_fangraphs"].isin(counts[counts >= min_games].index)].copy()

        per_player = g.groupby("key_fangraphs").agg(
            n=("value", "count"), var=("value", "var"),
            num=("num", "sum"), den=("weight", "sum"))
        g["w_mean"] = g["key_fangraphs"].map(per_player["num"] / per_player["den"])
        g["wsq"] = g["weight"] * (g["value"] - g["w_mean"]) ** 2

        dof = (per_player["n"] - 1).sum()
        rows.append({
            "Season": season,
            "n_players": len(per_player),
            "league_mean": g["num"].sum() / g["weight"].sum(),
            "mean_weight_per_game": g["weight"].mean(),
            "mean_within_sd": np.sqrt(per_player["var"]).mean(),
            "pooled_within_sd": np.sqrt(((per_player["n"] - 1) * per_player["var"]).sum() / dof),
            "sigma_unit": np.sqrt(g["wsq"].sum() / dof),
        })
    return pd.DataFrame(rows)


def estimate_prior_weight(game_logs, league_var, min_games=None):
    """
    Data-driven shrinkage weight k, in games, per season.

    k = (game-level noise variance) / (variance of true ability across batters),
    where true-ability variance = observed variance of batter means - average
    sampling noise in those means (a method-of-moments estimate).
    """
    min_games = LEAGUE_MIN_GAMES if min_games is None else min_games
    rows = []
    for season, g in game_logs.groupby("Season"):
        lv = league_var.set_index("Season").loc[season]
        per = g.groupby("key_fangraphs").agg(n=("value", "count"),
                                             num=("num", "sum"), den=("weight", "sum"))
        per = per[per["n"] >= min_games]
        means = per["num"] / per["den"]
        noise = (lv["sigma_unit"] ** 2 / per["den"]).mean()
        tau2 = max(means.var() - noise, 1e-12)
        game_var = lv["sigma_unit"] ** 2 / lv["mean_weight_per_game"]
        rows.append({"Season": season, "ability_sd": np.sqrt(tau2),
                     "game_sd": np.sqrt(game_var), "prior_weight_games": game_var / tau2})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Each player's pre-injury reference value
# =============================================================================
def add_reference_values(cohort, cohort_logs, league_mean, prior_weight=None):
    """
    ref = league mean if the player had no games before his oblique IL stint;
    otherwise (n * player pre-injury mean + k * league mean) / (n + k).

    league_mean: dict or Series mapping Season -> league mean for this metric.
    """
    prior_weight = PRIOR_WEIGHT if prior_weight is None else prior_weight
    out = cohort.drop(columns=["n_prior_games", "prior_mean", "league_mean", "ref"], errors="ignore")
    logs = cohort_logs.merge(out[["key_fangraphs", "Season", "IL_Retro_1"]],
                             on=["key_fangraphs", "Season"])
    prior = (
        logs[logs["game_date"] < logs["IL_Retro_1"]]
        .groupby(["key_fangraphs", "Season"])["value"]
        .agg(n_prior_games="count", prior_mean="mean")
        .reset_index()
    )
    out = out.merge(prior, on=["key_fangraphs", "Season"], how="left")
    out["league_mean"] = out["Season"].map(pd.Series(league_mean))
    out["n_prior_games"] = out["n_prior_games"].fillna(0).astype(int)

    n = out["n_prior_games"]
    out["ref"] = np.where(
        n == 0,
        out["league_mean"],
        (n * out["prior_mean"].fillna(0) + prior_weight * out["league_mean"]) / (n + prior_weight),
    )
    return out


# =============================================================================
# 6. Post-return monitoring data
# =============================================================================
def build_monitoring_frame(cohort, cohort_logs, league_var):
    """
    One row per post-return game, from the return date until the second IL stint
    (or the end of the season), with the player's reference value and the expected
    SD for that game's number of events.

      sigma_game  = sigma_unit / sqrt(weight)
      z           = (value - ref) / sigma_game
      games_to_il = post-return games remaining before IL_Retro_2 (NaN if censored)
    """
    cols = ["key_fangraphs", "Season", "Name", "Activate_1", "IL_Retro_2", "ref",
            "injury_group", "primary_outcome", "sensitivity_outcome", "negative_control",
            "event_primary", "event_sensitivity"]
    m = cohort_logs.merge(cohort[cols], on=["key_fangraphs", "Season"])
    m = m.merge(league_var[["Season", "sigma_unit", "pooled_within_sd"]], on="Season", how="left")

    after_return = m["game_date"] >= m["Activate_1"]
    before_il2 = m["IL_Retro_2"].isna() | (m["game_date"] < m["IL_Retro_2"])
    m = m[after_return & before_il2].sort_values(["key_fangraphs", "Season", "game_date", "dh"])

    grp = m.groupby(["key_fangraphs", "Season"])
    m["post_game_num"] = grp.cumcount() + 1
    m["sigma_game"] = m["sigma_unit"] / np.sqrt(m["weight"])
    m["z"] = (m["value"] - m["ref"]) / m["sigma_game"]
    n_post = grp["post_game_num"].transform("max")
    m["games_to_il"] = np.where(m["IL_Retro_2"].notna(), n_post - m["post_game_num"] + 1, np.nan)
    return m.reset_index(drop=True)


# =============================================================================
# 7. One-sided EWMA-CUSUM chart and its evaluation
# =============================================================================
# Same chart as the NBA study:
#   LSL_i  = ref_i - 2 * sigma
#   W_ij   = (x_ij - LSL_i) / sigma          (W = 2 when a game equals the baseline)
#   c_ij   = min(0, lambda * (W_ij - 2) + (1 - lambda) * c_i(j-1)),  c_i0 = 0
#   signal when c_ij < h   (h < 0)
LAMBDA_GRID = np.round([0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.95], 2)
H_GRID = np.round(-np.arange(0.1, 2.01, 0.1), 2)


def add_standardized_scores(monitor, sigma="pooled"):
    """
    Add LSL and W to the monitoring frame.

    sigma="pooled": the season's pooled within-player game SD (as in the NBA study)
    sigma="event":  the SD expected for that game's number of events,
                    sigma_unit / sqrt(weight), so thin games count for less
    """
    m = monitor.drop(columns=["sigma_used", "LSL", "W"], errors="ignore").copy()
    if sigma == "pooled":
        m["sigma_used"] = m["pooled_within_sd"]
    elif sigma == "event":
        m["sigma_used"] = m["sigma_game"]
    else:
        raise ValueError("sigma must be 'pooled' or 'event'")
    m["LSL"] = m["ref"] - 2 * m["sigma_used"]
    m["W"] = (m["value"] - m["LSL"]) / m["sigma_used"]
    return m


def ewma_cusum(w, lam):
    """Lower-sided EWMA-CUSUM path for one player's W series."""
    c = np.empty(len(w))
    prev = 0.0
    for j, wj in enumerate(w):
        prev = min(0.0, lam * (wj - 2) + (1 - lam) * prev)
        c[j] = prev
    return c


def run_chart_grid(monitor, lambdas=None, hs=None):
    """
    Run every (lambda, h) combination for every player-season.

    Returns one row per player-season x lambda x h with:
      signaled     whether c ever fell below h
      signal_game  post-return game number of the first signal
      lead_games   for players with a second IL stint, games from the first signal
                   through the last game before the stint (1 = signaled in the
                   final game before going on the IL)
    """
    lambdas = LAMBDA_GRID if lambdas is None else lambdas
    hs = H_GRID if hs is None else hs
    keys = ["key_fangraphs", "Season"]
    info_cols = ["Name", "injury_group", "primary_outcome", "sensitivity_outcome",
                 "negative_control"]

    rows = []
    m = monitor.sort_values(keys + ["post_game_num"])
    for (pid, season), g in m.groupby(keys, sort=False):
        w = g["W"].to_numpy()
        to_il = g["games_to_il"].to_numpy()
        info = g.iloc[0][info_cols].to_dict()
        for lam in lambdas:
            running_min = np.minimum.accumulate(ewma_cusum(w, lam))
            for h in hs:
                below = np.flatnonzero(running_min < h)
                first = below[0] if len(below) else None
                rows.append({
                    "key_fangraphs": pid, "Season": season, **info,
                    "lam": lam, "h": h, "n_post_games": len(w),
                    "signaled": first is not None,
                    "signal_game": first + 1 if first is not None else np.nan,
                    "lead_games": to_il[first] if first is not None else np.nan,
                })
    return pd.DataFrame(rows)


def _metrics(df, outcome):
    """Classification metrics for one (lambda, h). Negatives = no second injury."""
    pos = df[df[outcome] == 1]
    neg = df[df["injury_group"] == "none"]
    tp, fn = pos["signaled"].sum(), (~pos["signaled"]).sum()
    fp, tn = neg["signaled"].sum(), (~neg["signaled"]).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    leads = pos.loc[pos["signaled"], "lead_games"]
    ctrl = df[df["negative_control"] == 1]
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "sensitivity": sens, "specificity": spec,
        "accuracy": (tp + tn) / (tp + fn + fp + tn),
        "balanced_accuracy": (sens + spec) / 2,
        "mean_lead": leads.mean(), "median_lead": leads.median(),
        "neg_control_signal_rate": ctrl["signaled"].mean() if len(ctrl) else np.nan,
    }


def evaluate_grid(grid, outcome="primary_outcome"):
    """
    Metrics for every (lambda, h).

    Positives are players with the chosen outcome; negatives are players with no
    second IL stint. Players with other second injuries (e.g., the traumatic
    negative controls) are left out of sensitivity/specificity; how often the chart
    signals for the negative controls is reported separately.
    """
    out = [{"lam": lam, "h": h, **_metrics(df, outcome)}
           for (lam, h), df in grid.groupby(["lam", "h"])]
    return pd.DataFrame(out)


def leave_one_season_out(grid, outcome="primary_outcome", objective="balanced_accuracy"):
    """
    For each season: choose (lambda, h) on the other seasons, then apply it to the
    held-out season. Returns the chosen parameters per fold and pooled out-of-sample
    metrics across all held-out seasons.
    """
    folds, held_out = [], []
    for season in sorted(grid["Season"].unique()):
        train = evaluate_grid(grid[grid["Season"] != season], outcome)
        best = train.sort_values([objective, "sensitivity"], ascending=False).iloc[0]
        test = grid[(grid["Season"] == season) & (grid["lam"] == best["lam"]) & (grid["h"] == best["h"])]
        held_out.append(test)
        folds.append({"held_out": season, "lam": best["lam"], "h": best["h"],
                      f"train_{objective}": best[objective]})
    pooled = pd.concat(held_out, ignore_index=True)
    return pd.DataFrame(folds), pd.Series(_metrics(pooled, outcome)), pooled


def separation(monitor, n_games=15):
    """
    Is there a difference to detect? Mean standardized shortfall (W - 2, or z when
    W is absent) per player over the first n post-return games, by outcome group,
    plus the AUC for "lower mean shortfall = reinjured" (0.5 = no separation).
    """
    d = monitor[monitor["post_game_num"] <= n_games].copy()
    d["shortfall"] = d["W"] - 2 if "W" in d.columns else d["z"]
    d["group"] = np.select(
        [d["primary_outcome"] == 1, d["injury_group"] == "none"],
        ["primary outcome", "no second injury"], default="other second injury")
    per = d.groupby(["key_fangraphs", "Season", "group"])["shortfall"].mean().reset_index()
    a = per.loc[per["group"] == "primary outcome", "shortfall"].to_numpy()
    b = per.loc[per["group"] == "no second injury", "shortfall"].to_numpy()
    auc = np.mean((a[:, None] < b[None, :]) + 0.5 * (a[:, None] == b[None, :])) if len(a) and len(b) else np.nan
    table = per.groupby("group")["shortfall"].agg(["mean", "std", "count"])
    return table, auc


# =============================================================================
# 8. The whole analysis for one metric
# =============================================================================
def analyze_metric(metric, cohort, prior_weight="estimate", sigma="pooled",
                   lambdas=None, hs=None, outcome="primary_outcome", pause=None):
    """
    Run steps 3-7 for one metric and return everything in a dict.

    prior_weight: a number of games, or "estimate" to use the data-driven k
                  (the across-season median from estimate_prior_weight).
    """
    label = METRICS[metric]["label"]
    print(f"--- {label} ---")
    cohort_m = cohort[cohort["Season"] >= METRICS[metric]["first_season"]]

    cohort_logs = get_metric_logs(cohort_m, metric, pause=pause)
    league_logs = get_league_metric_logs(metric, pause=pause)
    if cohort_logs.empty or league_logs.empty:
        print(f"Skipping {label}: no game data for this metric.")
        return None
    league_var = league_within_player_variation(league_logs)
    k_table = estimate_prior_weight(league_logs, league_var)
    k = float(k_table["prior_weight_games"].median()) if prior_weight == "estimate" else prior_weight

    league_mean = (pd.Series(LEAGUE_WOBA) if metric == "woba"
                   else league_var.set_index("Season")["league_mean"])
    with_ref = add_reference_values(cohort_m, cohort_logs, league_mean, prior_weight=k)
    monitor = add_standardized_scores(build_monitoring_frame(with_ref, cohort_logs, league_var), sigma)

    grid = run_chart_grid(monitor, lambdas, hs)
    results = evaluate_grid(grid, outcome)
    folds, oos, _ = leave_one_season_out(grid, outcome)
    sep_table, auc = separation(monitor)
    best = results.sort_values("balanced_accuracy", ascending=False).iloc[0]

    summary = {
        "metric": label,
        "player_seasons": monitor.groupby(["key_fangraphs", "Season"]).ngroups,
        "primary_events": int(with_ref[outcome].sum()),
        "prior_weight_k": round(k, 1),
        "first15_AUC": auc,
        "best_in_sample_BA": best["balanced_accuracy"],
        "best_lam": best["lam"], "best_h": best["h"],
        "LOSO_BA": oos["balanced_accuracy"],
        "LOSO_sensitivity": oos["sensitivity"],
        "LOSO_specificity": oos["specificity"],
        "LOSO_median_lead": oos["median_lead"],
    }
    return {"summary": summary, "cohort": with_ref, "cohort_logs": cohort_logs,
            "league_var": league_var, "k_table": k_table, "monitor": monitor,
            "grid": grid, "results": results, "folds": folds, "oos": oos,
            "separation": sep_table}
