"""
oblique_study.py — data pipeline for the MLB batter oblique reinjury study.

Everything the notebook needs lives here, so the notebook itself is just a
sequence of short, readable steps:

    1. Injury data      load_cohort()
    2. Outcomes         add_injury_groups(), summarize_outcomes()
    3. Game logs        get_game_logs()                (FanGraphs, cached to disk)
    4. League variation get_league_game_logs(), league_within_player_variation()
    5. Reference values add_reference_values(), estimate_prior_weight()
    6. Monitoring data  build_monitoring_frame()

Game logs come from FanGraphs' game-log API (one small JSON request per
player-season), so no pitch-level Statcast data is downloaded. Every
player-season is saved under CACHE_DIR, so an interrupted run picks up where
it stopped and later runs make no requests at all.

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

PRIOR_WEIGHT = 7        # games of league-average wOBA blended into each player's baseline
LEAGUE_MIN_PA = 200     # batters below this season PA total are left out of league variation
LEAGUE_MIN_GAMES = 20   # ...as are batters with fewer games than this
REQUEST_PAUSE = 2.0     # seconds between FanGraphs requests

CACHE_DIR = Path("data/fg_gamelogs")


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


def _tidy_game_log(raw, playerid, season):
    """Keep the columns the study needs and add the wOBA numerator/denominator."""
    df = raw.copy()
    for col in ("AB", "BB", "IBB", "SF", "HBP", "PA", "wOBA", "dh"):
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # FanGraphs' wOBA denominator; numerator recovers the weighted event total
    df["woba_denom"] = df["AB"] + df["BB"] - df["IBB"] + df["SF"] + df["HBP"]
    df["woba_num"] = df["wOBA"] * df["woba_denom"]

    out = pd.DataFrame({
        "key_fangraphs": str(playerid),
        "Season": int(season),
        "game_date": pd.to_datetime(df["Date"], errors="coerce"),
        "dh": df["dh"].astype(int),
        "Team": df.get("Team"),
        "Opp": df.get("Opp"),
        "PA": df["PA"],
        "woba_num": df["woba_num"],
        "woba_denom": df["woba_denom"],
    })
    out = out[out["woba_denom"] > 0].copy()  # no wOBA-eligible PA (e.g., a lone sac bunt)
    out["woba"] = out["woba_num"] / out["woba_denom"]
    out = out.sort_values(["game_date", "dh"]).reset_index(drop=True)
    out["game_num"] = np.arange(1, len(out) + 1)
    return out


def get_game_woba(playerid, season, cache_dir=None, pause=None):
    """
    Game-by-game wOBA for one player-season. Reads from the cache when possible;
    otherwise requests FanGraphs, waits `pause` seconds, and caches the result.
    """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    pause = REQUEST_PAUSE if pause is None else pause
    path = Path(cache_dir) / str(season) / f"{playerid}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["game_date"], dtype={"key_fangraphs": str})

    raw = fg_batter_game_logs(playerid, season)
    time.sleep(pause)
    if raw.empty:
        return pd.DataFrame()  # failures are not cached, so they are retried next run

    log = _tidy_game_log(raw, playerid, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def get_game_logs(pairs, id_col="key_fangraphs", season_col="Season", **kwargs):
    """Stack game logs for every (player, season) pair in a DataFrame."""
    pairs = pairs[[id_col, season_col]].dropna().drop_duplicates()
    logs, missing = [], []
    for i, (pid, season) in enumerate(pairs.itertuples(index=False), 1):
        log = get_game_woba(pid, int(season), **kwargs)
        (logs.append(log) if not log.empty else missing.append((pid, int(season))))
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} player-seasons done")
    if missing:
        print(f"{len(missing)} player-seasons returned no games: {missing}")
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


# =============================================================================
# 4. League-level within-player variation
# =============================================================================
def fg_league_batters(season, min_pa=None, cache_dir=None):
    """FanGraphs IDs of non-pitchers with at least min_pa PA in a season (cached)."""
    min_pa = LEAGUE_MIN_PA if min_pa is None else min_pa
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    path = Path(cache_dir) / f"league_batters_{season}_{min_pa}PA.csv"
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


def get_league_game_logs(seasons=None, min_pa=None, **kwargs):
    """Game logs for every qualifying batter in each season (slow the first time)."""
    seasons = seasons or SEASONS
    min_pa = LEAGUE_MIN_PA if min_pa is None else min_pa
    logs = []
    for season in seasons:
        batters = fg_league_batters(season, min_pa)
        print(f"{season}: {len(batters)} batters with {min_pa}+ PA")
        logs.append(get_game_logs(batters, **kwargs))
    return pd.concat(logs, ignore_index=True)


def league_within_player_variation(game_logs, min_games=None):
    """
    Per season, how much a typical batter's game wOBA varies around his own mean.

      n_players         batters with at least min_games games
      mean_pa_per_game  average wOBA-eligible PA per game
      mean_within_sd    simple mean of each batter's game-wOBA SD
      pooled_within_sd  pooled game-level SD: sqrt(sum((n_i-1) s_i^2) / sum(n_i-1))
      sigma_pa          pooled per-PA SD; the expected SD of a game with k PA
                        is sigma_pa / sqrt(k)

    sigma_pa comes from PA-weighted squared deviations from each batter's PA-weighted
    mean. If PA outcomes have variance sigma^2, a k-PA game has variance sigma^2/k,
    so k * (game wOBA - mean)^2 estimates sigma^2 for every game.
    """
    min_games = LEAGUE_MIN_GAMES if min_games is None else min_games
    rows = []
    for season, g in game_logs.groupby("Season"):
        counts = g.groupby("key_fangraphs")["woba"].count()
        g = g[g["key_fangraphs"].isin(counts[counts >= min_games].index)].copy()

        per_player = g.groupby("key_fangraphs").agg(
            n=("woba", "count"), var=("woba", "var"),
            num=("woba_num", "sum"), den=("woba_denom", "sum"))
        g["pa_mean"] = g["key_fangraphs"].map(per_player["num"] / per_player["den"])
        g["wsq"] = g["woba_denom"] * (g["woba"] - g["pa_mean"]) ** 2

        dof = (per_player["n"] - 1).sum()
        rows.append({
            "Season": season,
            "n_players": len(per_player),
            "mean_pa_per_game": g["woba_denom"].mean(),
            "mean_within_sd": np.sqrt(per_player["var"]).mean(),
            "pooled_within_sd": np.sqrt(((per_player["n"] - 1) * per_player["var"]).sum() / dof),
            "sigma_pa": np.sqrt(g["wsq"].sum() / dof),
        })
    return pd.DataFrame(rows)


def estimate_prior_weight(game_logs, league_var, min_games=None):
    """
    Data-driven check on PRIOR_WEIGHT, in games, per season.

    The weight that balances a batter's own games against the league average is
    (game-level noise variance) / (variance of true talent across batters).
    True-talent variance = observed variance of batter wOBA - average sampling noise.
    """
    min_games = LEAGUE_MIN_GAMES if min_games is None else min_games
    rows = []
    for season, g in game_logs.groupby("Season"):
        lv = league_var.set_index("Season").loc[season]
        per = g.groupby("key_fangraphs").agg(n=("woba", "count"),
                                             num=("woba_num", "sum"), den=("woba_denom", "sum"))
        per = per[per["n"] >= min_games]
        means = per["num"] / per["den"]
        noise = (lv["sigma_pa"] ** 2 / per["den"]).mean()
        tau2 = max(means.var() - noise, 1e-8)
        game_var = lv["sigma_pa"] ** 2 / lv["mean_pa_per_game"]
        rows.append({"Season": season, "talent_sd": np.sqrt(tau2),
                     "game_sd": np.sqrt(game_var), "prior_weight_games": game_var / tau2})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Each player's pre-injury reference wOBA
# =============================================================================
def add_reference_values(cohort, cohort_logs, league_woba=None, prior_weight=None):
    """
    wOBA_ref = league average if the player had no games before his oblique IL stint;
    otherwise (n * player pre-injury mean + w * league average) / (n + w).
    """
    league_woba = LEAGUE_WOBA if league_woba is None else league_woba
    prior_weight = PRIOR_WEIGHT if prior_weight is None else prior_weight
    out = cohort.drop(columns=["num_games_prior_to_injury", "player_mean_wOBA_prior_to_injury",
                               "league_avg_wOBA", "wOBA_ref"], errors="ignore")
    logs = cohort_logs.merge(out[["key_fangraphs", "Season", "IL_Retro_1"]],
                             on=["key_fangraphs", "Season"])
    prior = (
        logs[logs["game_date"] < logs["IL_Retro_1"]]
        .groupby(["key_fangraphs", "Season"])["woba"]
        .agg(num_games_prior_to_injury="count", player_mean_wOBA_prior_to_injury="mean")
        .reset_index()
    )
    out = out.merge(prior, on=["key_fangraphs", "Season"], how="left")
    out["league_avg_wOBA"] = out["Season"].map(league_woba)
    out["num_games_prior_to_injury"] = out["num_games_prior_to_injury"].fillna(0).astype(int)

    n = out["num_games_prior_to_injury"]
    out["wOBA_ref"] = np.where(
        n == 0,
        out["league_avg_wOBA"],
        (n * out["player_mean_wOBA_prior_to_injury"].fillna(0)
         + prior_weight * out["league_avg_wOBA"]) / (n + prior_weight),
    )
    return out


# =============================================================================
# 6. Post-return monitoring data, ready for the control chart
# =============================================================================
def build_monitoring_frame(cohort, cohort_logs, league_var):
    """
    One row per post-return game: from the return date until the second IL stint
    (or the end of the season), with the player's reference value and the expected
    game-level SD for that game's PA.

      sigma_game  = sigma_pa / sqrt(woba_denom)
      z           = (woba - wOBA_ref) / sigma_game
      games_to_il = post-return games remaining before IL_Retro_2 (NaN if censored)
    """
    cols = ["key_fangraphs", "Season", "Name", "Activate_1", "IL_Retro_2", "wOBA_ref",
            "injury_group", "primary_outcome", "sensitivity_outcome", "negative_control",
            "event_primary", "event_sensitivity"]
    m = cohort_logs.merge(cohort[cols], on=["key_fangraphs", "Season"])
    m = m.merge(league_var[["Season", "sigma_pa"]], on="Season", how="left")

    after_return = m["game_date"] >= m["Activate_1"]
    before_il2 = m["IL_Retro_2"].isna() | (m["game_date"] < m["IL_Retro_2"])
    m = m[after_return & before_il2].sort_values(["key_fangraphs", "Season", "game_date", "dh"])

    grp = m.groupby(["key_fangraphs", "Season"])
    m["post_game_num"] = grp.cumcount() + 1
    m["sigma_game"] = m["sigma_pa"] / np.sqrt(m["woba_denom"])
    m["z"] = (m["woba"] - m["wOBA_ref"]) / m["sigma_game"]
    n_post = grp["post_game_num"].transform("max")
    m["games_to_il"] = np.where(m["IL_Retro_2"].notna(), n_post - m["post_game_num"] + 1, np.nan)
    return m.reset_index(drop=True)
