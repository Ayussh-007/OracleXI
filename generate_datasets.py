"""
OracleXI v2.0 – Rich Synthetic Dataset Generator
====================================================
Generates comprehensive synthetic CSV datasets covering:

Football:
    - International matches (1990-2026): 5,000+ matches
    - Club matches (top leagues + UCL): 3,000+ matches
    - FIFA Rankings history
    - Goalscorers dataset
    - Tournament metadata

Cricket:
    - IPL (2008-2026): 1,200+ matches
    - ICC T20 World Cup: 400+ matches
    - ICC ODI World Cup: 400+ matches
    - ICC Champions Trophy: 200+ matches
    - Bilateral internationals: 800+ matches
    - Ball-by-ball data for all matches
    - Venue & toss metadata

Usage:
    python generate_datasets.py
"""

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FB_DIR = os.path.join(DATA_DIR, "football")
CR_DIR = os.path.join(DATA_DIR, "cricket")

for d in [DATA_DIR, FB_DIR, CR_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(42)

# ═════════════════════════════════════════════
# FOOTBALL CONFIGURATION
# ═════════════════════════════════════════════

INTL_TEAMS = {
    "Argentina": 0.90, "Brazil": 0.88, "France": 0.92, "England": 0.85,
    "Germany": 0.83, "Spain": 0.86, "Netherlands": 0.82, "Portugal": 0.87,
    "Belgium": 0.84, "Italy": 0.81, "Croatia": 0.80, "Uruguay": 0.78,
    "Colombia": 0.76, "Denmark": 0.77, "Mexico": 0.74, "United States": 0.75,
    "Switzerland": 0.76, "Japan": 0.73, "South Korea": 0.72, "Morocco": 0.77,
    "Senegal": 0.75, "Australia": 0.70, "Poland": 0.74, "Serbia": 0.73,
    "Nigeria": 0.72, "Ghana": 0.71, "Ecuador": 0.72, "Iran": 0.70,
    "Wales": 0.71, "Tunisia": 0.69, "Cameroon": 0.70, "Canada": 0.68,
    "Costa Rica": 0.65, "Saudi Arabia": 0.66, "Qatar": 0.63,
    "India": 0.55, "China": 0.50, "Egypt": 0.68, "Algeria": 0.72,
    "Chile": 0.76, "Paraguay": 0.67, "Peru": 0.70, "Sweden": 0.74,
    "Norway": 0.72, "Czech Republic": 0.73, "Turkey": 0.74,
    "Russia": 0.72, "Ukraine": 0.73, "Scotland": 0.68, "Ireland": 0.65,
}

CLUB_TEAMS = {
    "Manchester City": 0.93, "Arsenal": 0.88, "Liverpool": 0.90,
    "Manchester United": 0.82, "Chelsea": 0.84, "Tottenham": 0.79,
    "Newcastle United": 0.78, "Aston Villa": 0.76, "Brighton": 0.74,
    "West Ham": 0.73, "Real Madrid": 0.94, "Barcelona": 0.91,
    "Atletico Madrid": 0.84, "Real Sociedad": 0.76, "Villarreal": 0.75,
    "Athletic Bilbao": 0.74, "Sevilla": 0.77, "Real Betis": 0.73,
    "Bayern Munich": 0.92, "Borussia Dortmund": 0.85,
    "RB Leipzig": 0.80, "Bayer Leverkusen": 0.83,
    "Inter Milan": 0.87, "AC Milan": 0.83, "Juventus": 0.84,
    "Napoli": 0.85, "Roma": 0.78, "Lazio": 0.76,
    "Paris Saint-Germain": 0.91, "Marseille": 0.77,
    "Monaco": 0.76, "Lyon": 0.75,
    "Ajax": 0.78, "Benfica": 0.79, "Porto": 0.78,
    "Celtic": 0.72, "Rangers": 0.70,
    "Galatasaray": 0.73, "Fenerbahce": 0.72,
    "Flamengo": 0.77, "River Plate": 0.76, "Boca Juniors": 0.75,
}

INTL_TOURNAMENTS = [
    "FIFA World Cup", "FIFA World Cup Qualification",
    "UEFA Euro", "UEFA Euro Qualification",
    "Copa America", "African Cup of Nations",
    "AFC Asian Cup", "CONCACAF Gold Cup",
    "UEFA Nations League", "Friendly",
]

CLUB_TOURNAMENTS = [
    "UEFA Champions League", "UEFA Europa League",
    "Premier League", "La Liga", "Bundesliga",
    "Serie A", "Ligue 1",
]

CITIES = {
    "Argentina": ("Buenos Aires", "Argentina"), "Brazil": ("Rio de Janeiro", "Brazil"),
    "France": ("Paris", "France"), "England": ("London", "England"),
    "Germany": ("Berlin", "Germany"), "Spain": ("Madrid", "Spain"),
    "Netherlands": ("Amsterdam", "Netherlands"), "Italy": ("Rome", "Italy"),
    "Portugal": ("Lisbon", "Portugal"), "Belgium": ("Brussels", "Belgium"),
    "Croatia": ("Zagreb", "Croatia"), "Uruguay": ("Montevideo", "Uruguay"),
    "Japan": ("Tokyo", "Japan"), "South Korea": ("Seoul", "South Korea"),
    "United States": ("New York", "United States"),
    "Mexico": ("Mexico City", "Mexico"), "Morocco": ("Casablanca", "Morocco"),
    "Denmark": ("Copenhagen", "Denmark"), "Switzerland": ("Zurich", "Switzerland"),
    "Manchester City": ("Manchester", "England"), "Arsenal": ("London", "England"),
    "Liverpool": ("Liverpool", "England"), "Real Madrid": ("Madrid", "Spain"),
    "Barcelona": ("Barcelona", "Spain"), "Bayern Munich": ("Munich", "Germany"),
    "Paris Saint-Germain": ("Paris", "France"), "Juventus": ("Turin", "Italy"),
    "Inter Milan": ("Milan", "Italy"), "Borussia Dortmund": ("Dortmund", "Germany"),
}


def _gen_football_matches(teams: dict, tournaments: list,
                          start_year: int, end_year: int,
                          n_matches: int, is_club: bool) -> pd.DataFrame:
    """Generate football matches for a set of teams."""
    team_list = list(teams.keys())
    records = []
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    total_days = (end - start).days

    for _ in range(n_matches):
        day = np.random.randint(0, total_days)
        date = start + pd.Timedelta(days=int(day))
        t = np.random.choice(team_list, size=2, replace=False)
        home, away = t[0], t[1]
        tournament = np.random.choice(tournaments)
        neutral = np.random.random() < 0.25
        city_info = CITIES.get(home, ("Unknown", "Unknown"))
        city, country = city_info if not neutral else CITIES.get(
            np.random.choice(team_list), ("Neutral", "Neutral"))

        hs = teams.get(home, 0.70)
        aws = teams.get(away, 0.70)
        ha = 0.0 if neutral else 0.25
        lam_h = max(0.4, 1.3 * (hs + ha))
        lam_a = max(0.4, 1.3 * aws)
        home_score = min(int(np.random.poisson(lam_h)), 8)
        away_score = min(int(np.random.poisson(lam_a)), 8)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "tournament": tournament,
            "city": city, "country": country,
            "neutral": neutral,
            "match_type": "club" if is_club else "international",
        })

    return pd.DataFrame(records)


def _gen_goalscorers(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate goalscorer records from match results."""
    records = []
    scorers_pool = [
        "Messi", "Ronaldo", "Mbappe", "Haaland", "Neymar",
        "Lewandowski", "Kane", "Salah", "Vinicius Jr", "Bellingham",
        "De Bruyne", "Modric", "Suarez", "Muller", "Son",
        "Firmino", "Griezmann", "Benzema", "Lukaku", "Sterling",
    ]
    for _, row in results_df.iterrows():
        for g in range(row["home_score"]):
            records.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "team": row["home_team"],
                "scorer": np.random.choice(scorers_pool),
                "minute": int(np.random.randint(1, 91)),
                "own_goal": False,
                "penalty": np.random.random() < 0.12,
            })
        for g in range(row["away_score"]):
            records.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "team": row["away_team"],
                "scorer": np.random.choice(scorers_pool),
                "minute": int(np.random.randint(1, 91)),
                "own_goal": False,
                "penalty": np.random.random() < 0.12,
            })
    return pd.DataFrame(records) if records else pd.DataFrame()


def _gen_rankings(teams: dict) -> pd.DataFrame:
    """Generate FIFA-style ranking history."""
    records = []
    for year in range(2010, 2027):
        for month in [1, 4, 7, 10]:
            sorted_teams = sorted(
                teams.items(),
                key=lambda x: x[1] + np.random.normal(0, 0.03),
                reverse=True,
            )
            for rank, (team, strength) in enumerate(sorted_teams, 1):
                points = max(0, int(strength * 2000 + np.random.normal(0, 50)))
                records.append({
                    "rank_date": f"{year}-{month:02d}-01",
                    "rank": rank,
                    "team": team,
                    "total_points": points,
                    "previous_points": max(0, points + int(np.random.normal(0, 30))),
                    "rank_change": int(np.random.normal(0, 2)),
                })
    return pd.DataFrame(records)


# ═════════════════════════════════════════════
# CRICKET CONFIGURATION
# ═════════════════════════════════════════════

IPL_TEAMS = {
    "Mumbai Indians": 0.85, "Chennai Super Kings": 0.83,
    "Royal Challengers Bangalore": 0.78, "Kolkata Knight Riders": 0.80,
    "Delhi Capitals": 0.76, "Punjab Kings": 0.72,
    "Rajasthan Royals": 0.79, "Sunrisers Hyderabad": 0.75,
    "Gujarat Titans": 0.82, "Lucknow Super Giants": 0.74,
}

INTL_CRICKET = {
    "India": 0.90, "Australia": 0.88, "England": 0.86,
    "South Africa": 0.82, "New Zealand": 0.84, "Pakistan": 0.80,
    "Sri Lanka": 0.72, "West Indies": 0.70, "Bangladesh": 0.65,
    "Afghanistan": 0.68, "Zimbabwe": 0.55, "Ireland": 0.58,
    "Netherlands": 0.50, "Scotland": 0.48, "Nepal": 0.45,
    "Namibia": 0.44, "Oman": 0.42, "UAE": 0.40,
    "USA": 0.52, "Canada": 0.46,
}

CRICKET_VENUES = [
    ("Wankhede Stadium", "Mumbai"), ("M.A. Chidambaram Stadium", "Chennai"),
    ("M. Chinnaswamy Stadium", "Bangalore"), ("Eden Gardens", "Kolkata"),
    ("Arun Jaitley Stadium", "Delhi"), ("IS Bindra Stadium", "Mohali"),
    ("Sawai Mansingh Stadium", "Jaipur"),
    ("Rajiv Gandhi Intl Stadium", "Hyderabad"),
    ("Narendra Modi Stadium", "Ahmedabad"),
    ("BRSABV Ekana Stadium", "Lucknow"),
    ("MCG", "Melbourne"), ("SCG", "Sydney"),
    ("Lord's", "London"), ("The Oval", "London"),
    ("Headingley", "Leeds"), ("Old Trafford", "Manchester"),
    ("Kensington Oval", "Bridgetown"), ("Sabina Park", "Kingston"),
    ("SuperSport Park", "Centurion"), ("Newlands", "Cape Town"),
    ("Basin Reserve", "Wellington"), ("Seddon Park", "Hamilton"),
    ("Gaddafi Stadium", "Lahore"), ("National Stadium", "Karachi"),
    ("R. Premadasa Stadium", "Colombo"), ("Shere Bangla Stadium", "Dhaka"),
    ("Dubai International Stadium", "Dubai"),
    ("Nassau County Stadium", "New York"),
]

BATTERS = {
    "Mumbai Indians": ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma", "Tim David"],
    "Chennai Super Kings": ["Ruturaj Gaikwad", "Devon Conway", "Shivam Dube", "Moeen Ali", "Ravindra Jadeja"],
    "Royal Challengers Bangalore": ["Virat Kohli", "Faf du Plessis", "Glenn Maxwell", "Rajat Patidar", "Dinesh Karthik"],
    "Kolkata Knight Riders": ["Shreyas Iyer", "Venkatesh Iyer", "Nitish Rana", "Andre Russell", "Rinku Singh"],
    "Delhi Capitals": ["David Warner", "Prithvi Shaw", "Rishabh Pant", "Mitchell Marsh", "Axar Patel"],
    "Punjab Kings": ["Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran", "Jitesh Sharma"],
    "Rajasthan Royals": ["Yashasvi Jaiswal", "Jos Buttler", "Sanju Samson", "Shimron Hetmyer", "Devdutt Padikkal"],
    "Sunrisers Hyderabad": ["Travis Head", "Abhishek Sharma", "Aiden Markram", "Heinrich Klaasen", "Rahul Tripathi"],
    "Gujarat Titans": ["Shubman Gill", "Wriddhiman Saha", "Hardik Pandya", "David Miller", "Rashid Khan"],
    "Lucknow Super Giants": ["KL Rahul", "Quinton de Kock", "Ayush Badoni", "Marcus Stoinis", "Deepak Hooda"],
    "India": ["Virat Kohli", "Rohit Sharma", "Shubman Gill", "Suryakumar Yadav", "KL Rahul"],
    "Australia": ["David Warner", "Travis Head", "Steve Smith", "Glenn Maxwell", "Marcus Stoinis"],
    "England": ["Jos Buttler", "Phil Salt", "Ben Stokes", "Harry Brook", "Liam Livingstone"],
    "South Africa": ["Quinton de Kock", "Aiden Markram", "David Miller", "Heinrich Klaasen", "Temba Bavuma"],
    "New Zealand": ["Kane Williamson", "Devon Conway", "Glenn Phillips", "Daryl Mitchell", "Mark Chapman"],
    "Pakistan": ["Babar Azam", "Mohammad Rizwan", "Fakhar Zaman", "Iftikhar Ahmed", "Shadab Khan"],
}

BOWLERS = {
    "Mumbai Indians": ["Jasprit Bumrah", "Piyush Chawla", "Daniel Sams", "Hrithik Shokeen"],
    "Chennai Super Kings": ["Deepak Chahar", "Tushar Deshpande", "Maheesh Theekshana", "Matheesha Pathirana"],
    "Royal Challengers Bangalore": ["Mohammed Siraj", "Josh Hazlewood", "Wanindu Hasaranga", "Harshal Patel"],
    "Kolkata Knight Riders": ["Mitchell Starc", "Sunil Narine", "Varun Chakravarthy", "Harshit Rana"],
    "Delhi Capitals": ["Anrich Nortje", "Khaleel Ahmed", "Kuldeep Yadav", "Mustafizur Rahman"],
    "Punjab Kings": ["Arshdeep Singh", "Kagiso Rabada", "Rahul Chahar", "Harpreet Brar"],
    "Rajasthan Royals": ["Trent Boult", "Yuzvendra Chahal", "Prasidh Krishna", "Sandeep Sharma"],
    "Sunrisers Hyderabad": ["Bhuvneshwar Kumar", "T Natarajan", "Umran Malik", "Washington Sundar"],
    "Gujarat Titans": ["Mohammed Shami", "Alzarri Joseph", "Noor Ahmad", "R Sai Kishore"],
    "Lucknow Super Giants": ["Avesh Khan", "Mark Wood", "Ravi Bishnoi", "Krunal Pandya"],
    "India": ["Jasprit Bumrah", "Mohammed Siraj", "Kuldeep Yadav", "Arshdeep Singh"],
    "Australia": ["Mitchell Starc", "Pat Cummins", "Adam Zampa", "Josh Hazlewood"],
    "England": ["Jofra Archer", "Mark Wood", "Adil Rashid", "Chris Woakes"],
    "South Africa": ["Kagiso Rabada", "Anrich Nortje", "Lungi Ngidi", "Keshav Maharaj"],
    "New Zealand": ["Trent Boult", "Tim Southee", "Ish Sodhi", "Lockie Ferguson"],
    "Pakistan": ["Shaheen Afridi", "Haris Rauf", "Naseem Shah", "Shadab Khan"],
}


def _gen_cricket_matches(teams: dict, tournament: str, fmt: str,
                         start_year: int, end_year: int,
                         n_matches: int, match_id_start: int) -> pd.DataFrame:
    """Generate cricket match info records."""
    team_list = list(teams.keys())
    records = []
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    total_days = max(1, (end - start).days)

    for i in range(n_matches):
        day = np.random.randint(0, total_days)
        date = start + pd.Timedelta(days=int(day))
        t = np.random.choice(team_list, size=2, replace=False)
        team1, team2 = t[0], t[1]
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(["bat", "field"], p=[0.35, 0.65])
        venue_idx = np.random.randint(0, len(CRICKET_VENUES))
        venue, city = CRICKET_VENUES[venue_idx]

        s1 = teams.get(team1, 0.60)
        s2 = teams.get(team2, 0.60)
        # Toss advantage small bump
        toss_bump = 0.02 if toss_winner == team1 else -0.02
        win_prob_1 = (s1 + toss_bump) / (s1 + s2 + 0.001)
        winner = team1 if np.random.random() < win_prob_1 else team2

        result_type = np.random.choice(["runs", "wickets"], p=[0.45, 0.55])
        if result_type == "runs":
            result_margin = min(int(np.random.exponential(25)) + 1, 200)
        else:
            result_margin = np.random.randint(1, 11)

        records.append({
            "match_id": match_id_start + i,
            "date": date.strftime("%Y-%m-%d"),
            "team1": team1, "team2": team2,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "venue": venue, "city": city,
            "winner": winner,
            "result_margin": result_margin, "result_type": result_type,
            "format": fmt, "tournament": tournament,
        })

    return pd.DataFrame(records)


def _gen_ball_by_ball(match_df: pd.DataFrame, is_odi: bool = False) -> pd.DataFrame:
    """Generate ball-by-ball data for matches."""
    all_records = []
    max_overs = 50 if is_odi else 20

    for _, match in match_df.iterrows():
        mid = match["match_id"]
        team1, team2 = match["team1"], match["team2"]

        for inning in [1, 2]:
            bat = team1 if inning == 1 else team2
            bowl = team2 if inning == 1 else team1
            batters = BATTERS.get(bat, ["Batter1", "Batter2", "Batter3", "Batter4", "Batter5"])
            bowlers_list = BOWLERS.get(bowl, ["Bowler1", "Bowler2", "Bowler3", "Bowler4"])

            wickets = 0
            bi = 0
            for over in range(max_overs):
                bowler = np.random.choice(bowlers_list)
                for ball in range(1, 7):
                    if wickets >= 10:
                        break
                    batter = batters[bi % len(batters)]
                    non_striker = batters[(bi + 1) % len(batters)]

                    run_probs = [0.33, 0.30, 0.12, 0.03, 0.15, 0.01, 0.06] if not is_odi else [0.38, 0.28, 0.12, 0.03, 0.13, 0.01, 0.05]
                    batsman_run = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=run_probs)

                    extra_run = 0
                    extra_type = ""
                    if np.random.random() < 0.07:
                        extra_run = np.random.choice([1, 2, 4], p=[0.7, 0.2, 0.1])
                        extra_type = np.random.choice(["wides", "noballs", "legbyes", "byes"], p=[0.45, 0.25, 0.20, 0.10])

                    total_run = batsman_run + extra_run

                    wp = 0.045 if not is_odi else 0.035
                    if over >= (max_overs - 4):
                        wp += 0.02
                    elif over <= 5:
                        wp -= 0.01

                    is_wicket = 1 if np.random.random() < wp else 0
                    dismissal_kind = ""
                    player_out = ""
                    if is_wicket:
                        wickets += 1
                        dismissal_kind = np.random.choice(
                            ["bowled", "caught", "lbw", "run out", "stumped", "caught and bowled"],
                            p=[0.18, 0.40, 0.15, 0.12, 0.08, 0.07])
                        player_out = batter
                        bi += 1

                    all_records.append({
                        "match_id": mid, "inning": inning,
                        "over": over, "ball": ball,
                        "batting_team": bat, "bowling_team": bowl,
                        "batter": batter, "bowler": bowler,
                        "non_striker": non_striker,
                        "batsman_run": batsman_run,
                        "extra_run": extra_run, "total_run": total_run,
                        "extra_type": extra_type,
                        "is_wicket": is_wicket,
                        "dismissal_kind": dismissal_kind,
                        "player_out": player_out,
                    })
                if wickets >= 10:
                    break

    return pd.DataFrame(all_records)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  OracleXI v2.0 – Rich Dataset Generator")
    print("=" * 60)
    print()

    # ── FOOTBALL ──
    print("[1/8] International Football (1990-2026)")
    fb_intl = _gen_football_matches(INTL_TEAMS, INTL_TOURNAMENTS, 1990, 2026, 5000, False)
    fb_intl.to_csv(os.path.join(FB_DIR, "international_results.csv"), index=False)
    print(f"  ✓ {len(fb_intl)} international matches")

    print("[2/8] Club Football (2005-2026)")
    fb_club = _gen_football_matches(CLUB_TEAMS, CLUB_TOURNAMENTS, 2005, 2026, 3000, True)
    fb_club.to_csv(os.path.join(FB_DIR, "club_results.csv"), index=False)
    print(f"  ✓ {len(fb_club)} club matches")

    # Combined football
    fb_all = pd.concat([fb_intl, fb_club], ignore_index=True).sort_values("date").reset_index(drop=True)
    fb_all.to_csv(os.path.join(DATA_DIR, "football_results.csv"), index=False)
    print(f"  ✓ {len(fb_all)} total football matches (combined)")

    print("[3/8] Football Goalscorers")
    goalscorers = _gen_goalscorers(fb_all)
    goalscorers.to_csv(os.path.join(FB_DIR, "goalscorers.csv"), index=False)
    print(f"  ✓ {len(goalscorers)} goal records")

    print("[4/8] FIFA Rankings")
    rankings = _gen_rankings(INTL_TEAMS)
    rankings.to_csv(os.path.join(FB_DIR, "rankings.csv"), index=False)
    print(f"  ✓ {len(rankings)} ranking records")

    # ── CRICKET ──
    mid = 1

    print("[5/8] IPL (2008-2026)")
    ipl = _gen_cricket_matches(IPL_TEAMS, "IPL", "T20", 2008, 2026, 1200, mid)
    ipl.to_csv(os.path.join(CR_DIR, "ipl_matches.csv"), index=False)
    mid += len(ipl)
    print(f"  ✓ {len(ipl)} IPL matches")

    print("[6/8] ICC T20 World Cup (2007-2026)")
    t20wc = _gen_cricket_matches(INTL_CRICKET, "ICC T20 World Cup", "T20", 2007, 2026, 400, mid)
    t20wc.to_csv(os.path.join(CR_DIR, "t20_worldcup.csv"), index=False)
    mid += len(t20wc)
    print(f"  ✓ {len(t20wc)} T20 WC matches")

    print("[7/8] ICC ODI World Cup + Champions Trophy + Bilateral")
    odi_wc = _gen_cricket_matches(INTL_CRICKET, "ICC ODI World Cup", "ODI", 1992, 2026, 400, mid)
    mid += len(odi_wc)
    ct = _gen_cricket_matches(INTL_CRICKET, "ICC Champions Trophy", "ODI", 1998, 2025, 200, mid)
    mid += len(ct)
    bilateral = _gen_cricket_matches(INTL_CRICKET, "ODI Bilateral", "ODI", 2000, 2026, 600, mid)
    mid += len(bilateral)
    t20_bilateral = _gen_cricket_matches(INTL_CRICKET, "T20 Bilateral", "T20", 2005, 2026, 500, mid)
    mid += len(t20_bilateral)

    odi_all = pd.concat([odi_wc, ct, bilateral, t20_bilateral], ignore_index=True)
    odi_all.to_csv(os.path.join(CR_DIR, "international_matches.csv"), index=False)
    print(f"  ✓ {len(odi_all)} international cricket matches")

    # Combine all cricket
    cr_all = pd.concat([ipl, t20wc, odi_all], ignore_index=True).sort_values("date").reset_index(drop=True)
    cr_all["match_id"] = range(1, len(cr_all) + 1)
    cr_all.to_csv(os.path.join(DATA_DIR, "cricket_match_info.csv"), index=False)
    print(f"  ✓ {len(cr_all)} total cricket matches (combined)")

    print("[8/8] Cricket Ball-by-Ball (all matches)")
    # Split T20 and ODI for different over counts
    t20_matches = cr_all[cr_all["format"] == "T20"]
    odi_matches = cr_all[cr_all["format"] == "ODI"]

    bbb_t20 = _gen_ball_by_ball(t20_matches, is_odi=False)
    bbb_odi = _gen_ball_by_ball(odi_matches, is_odi=True)
    bbb_all = pd.concat([bbb_t20, bbb_odi], ignore_index=True)
    bbb_all.to_csv(os.path.join(DATA_DIR, "cricket_ball_by_ball.csv"), index=False)
    print(f"  ✓ {len(bbb_all)} ball-by-ball deliveries")

    print()
    print("=" * 60)
    print("  Dataset Generation Complete!")
    print(f"  Football: {len(fb_all):,} matches + {len(goalscorers):,} goals + {len(rankings):,} rankings")
    print(f"  Cricket:  {len(cr_all):,} matches + {len(bbb_all):,} deliveries")
    print(f"  Formats:  International + Club + T20 + ODI + IPL + World Cup")
    print("=" * 60)


if __name__ == "__main__":
    main()
