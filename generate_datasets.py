import os
import random
import datetime
import numpy as np
import pandas as pd

from utils.constants import FOOTBALL_TEAMS, CRICKET_TEAMS, FOOTBALL_TOURNAMENTS, CRICKET_TOURNAMENTS

def generate_football_data(num_matches=2500):
    print("Generating synthetic football dataset...")
    data = []
    start_date = datetime.date(2015, 1, 1)
    
    for _ in range(num_matches):
        team1, team2 = random.sample(FOOTBALL_TEAMS, 2)
        # Poisson distribution for goals to avoid anomalies
        lam1, lam2 = random.uniform(0.8, 2.5), random.uniform(0.5, 2.0)
        home_score = np.random.poisson(lam1)
        away_score = np.random.poisson(lam2)
        
        days_offset = random.randint(0, 3000)
        match_date = start_date + datetime.timedelta(days=days_offset)
        
        tournament = random.choice(FOOTBALL_TOURNAMENTS)
        data.append({
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": team1,
            "away_team": team2,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "city": "Unknown",
            "country": "Unknown",
            "neutral": random.choice([True, False])
        })
        
    df = pd.DataFrame(data).sort_values("date")
    os.makedirs(os.path.join("data", "football"), exist_ok=True)
    df.to_csv(os.path.join("data", "football", "international_results.csv"), index=False)
    print(f"Generated {num_matches} football matches.")

def generate_cricket_data(num_matches=1000):
    print("Generating synthetic cricket datasets...")
    match_data = []
    ball_data = []
    start_date = datetime.date(2015, 1, 1)
    
    for match_id in range(1, num_matches + 1):
        team1, team2 = random.sample(CRICKET_TEAMS, 2)
        days_offset = random.randint(0, 3000)
        match_date = start_date + datetime.timedelta(days=days_offset)
        
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(["bat", "field"])
        
        score1 = max(np.random.normal(160, 20), 50)
        score2 = max(np.random.normal(150, 20), 50)
        
        winner = team1 if score1 > score2 else team2
        if int(score1) == int(score2):
            winner = "No Result"
            
        margin = int(abs(score1 - score2))
        
        match_data.append({
            "match_id": f"m_{match_id}",
            "date": match_date.strftime("%Y-%m-%d"),
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner,
            "result_margin": margin,
            "result_type": "runs",
            "format": "T20",
            "tournament": random.choice(CRICKET_TOURNAMENTS),
            "city": "Unknown"
        })
        
        # Ball by ball
        for inning, (bat, bowl, target) in enumerate([(team1, team2, 200), (team2, team1, int(score1))], 1):
            runs_scored = 0
            wickets = 0
            for over in range(20):
                for ball in range(1, 7):
                    run = np.random.choice([0, 1, 2, 3, 4, 6], p=[0.35, 0.35, 0.1, 0.05, 0.1, 0.05])
                    is_w = np.random.choice([0, 1], p=[0.95, 0.05])
                    extra = np.random.choice([0, 1], p=[0.96, 0.04])
                    
                    if is_w: run = 0
                    
                    ball_data.append({
                        "match_id": f"m_{match_id}",
                        "inning": inning,
                        "over": over,
                        "ball": ball,
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "total_run": run + extra,
                        "is_wicket": is_w,
                        "extra_run": extra
                    })
                    
                    runs_scored += (run + extra)
                    wickets += is_w
                    if wickets >= 10: break
                if wickets >= 10: break
                
    df_match = pd.DataFrame(match_data).sort_values("date")
    df_ball = pd.DataFrame(ball_data)
    
    os.makedirs(os.path.join("data", "cricket"), exist_ok=True)
    df_match.to_csv(os.path.join("data", "cricket", "cricket_match_info.csv"), index=False)
    df_ball.to_csv(os.path.join("data", "cricket", "cricket_ball_by_ball.csv"), index=False)
    print(f"Generated {num_matches} cricket matches and ball-by-ball data.")

def main():
    os.makedirs("data", exist_ok=True)
    generate_football_data()
    generate_cricket_data()
    print("Done! Synthetic datasets are ready.")

if __name__ == "__main__":
    main()
