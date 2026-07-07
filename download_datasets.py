import os
import urllib.request
import zipfile
import io
import csv
import pandas as pd

def download_football():
    print("Downloading real football data (this might take a few moments)...")
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        df = pd.read_csv(url)
        os.makedirs("data/football", exist_ok=True)
        
        print("Downloading club football data (European Top 5 Leagues)...")
        seasons = ["2324", "2223", "2122", "2021", "1920"]
        leagues = {"E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga", "I1": "Serie A", "F1": "Ligue 1"}
        club_frames = []
        for season in seasons:
            for league_code, league_name in leagues.items():
                url_club = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
                try:
                    df_club = pd.read_csv(url_club, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
                    df_club = df_club.rename(columns={
                        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                        "FTHG": "home_score", "FTAG": "away_score"
                    })
                    df_club["tournament"] = league_name
                    df_club["city"] = "Unknown"
                    df_club["country"] = "Europe"
                    df_club["neutral"] = False
                    # Convert dates properly
                    df_club["date"] = pd.to_datetime(df_club["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
                    club_frames.append(df_club)
                except Exception as e:
                    pass
        
        if club_frames:
            df_clubs_all = pd.concat(club_frames, ignore_index=True)
            df = pd.concat([df, df_clubs_all], ignore_index=True)
            print(f"Appended {len(df_clubs_all)} real club football matches.")

        df.to_csv("data/football/international_results.csv", index=False)
        print(f"Downloaded {len(df)} total football matches successfully.")
        
        print("Downloading football goalscorers data...")
        url_goals = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
        df_goals = pd.read_csv(url_goals)
        df_goals.to_csv("data/football/goalscorers.csv", index=False)
        print(f"Downloaded {len(df_goals)} football goalscorers records.")
    except Exception as e:
        print(f"Failed to download football data: {e}")

def download_cricket():
    print("Downloading real IPL cricket data...")
    try:
        out_df = pd.DataFrame()
        
        # Now download International formats and IPL from Cricsheet
        
        # Now download International formats and IPL from Cricsheet
        formats = [
            ("https://cricsheet.org/downloads/t20s_csv2.zip", "T20I"),
            ("https://cricsheet.org/downloads/odis_csv2.zip", "ODI"),
            ("https://cricsheet.org/downloads/tests_csv2.zip", "Test"),
            ("https://cricsheet.org/downloads/ipl_csv2.zip", "T20 Franchise")
        ]
        
        intl_matches = []
        intl_player_stats_list = []
        for zip_url, fmt_name in formats:
            print(f"Downloading {fmt_name} from Cricsheet...")
            req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                with urllib.request.urlopen(req) as response:
                    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                        csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.endswith('_info.csv')]
                        info_files = [f for f in z.namelist() if f.endswith('_info.csv')]
                        male_match_ids = set()
                        # Parse match info
                        for info_file in info_files:
                            match = {
                                "match_id": info_file.split('_')[0], "format": fmt_name,
                                "tournament": "International", "team1": None, "team2": None,
                                "date": None, "city": "Unknown", "toss_winner": None,
                                "toss_decision": None, "winner": "No Result",
                                "result_margin": 0, "result_type": "N/A", "gender": "unknown"
                            }
                            with z.open(info_file) as f:
                                content = f.read().decode('utf-8')
                                reader = csv.reader(io.StringIO(content))
                                teams = []
                                for row in reader:
                                    if len(row) < 3 or row[0] != 'info': continue
                                    key = row[1]
                                    if key == 'city': match['city'] = row[2]
                                    elif key in ['dates', 'date']: match['date'] = row[2].replace('/', '-')
                                    elif key == 'team': teams.append(row[2])
                                    elif key == 'toss' and row[2] == 'winner': match['toss_winner'] = row[3]
                                    elif key == 'toss' and row[2] == 'decision': match['toss_decision'] = row[3]
                                    elif key == 'winner': match['winner'] = row[2]
                                    elif key == 'winner_runs':
                                        match['result_type'] = 'runs'
                                        match['result_margin'] = int(row[2])
                                    elif key == 'gender': match['gender'] = row[2]
                                    elif key == 'winner_wickets':
                                        match['result_type'] = 'wickets'
                                        match['result_margin'] = int(row[2])
                            if match.get('gender') != 'male':
                                continue
                            male_match_ids.add(match['match_id'])
                            if len(teams) >= 2:
                                match['team1'], match['team2'] = teams[0], teams[1]
                            intl_matches.append(match)
                            
                        # Parse ball-by-ball for players
                        for match_file in csv_files:
                            match_id = match_file.split('.')[0]
                            if match_id not in male_match_ids: continue
                            match_runs = {}
                            match_wkts = {}
                            teams_map = {}
                            with z.open(match_file) as f:
                                content = f.read().decode('utf-8')
                                reader = csv.reader(io.StringIO(content))
                                # skip header
                                next(reader, None)
                                for row in reader:
                                    if len(row) < 22: continue
                                    # match_id,season,start_date,venue,innings,ball,actual_delivery,batting_team,bowling_team,striker,non_striker,bowler,runs_off_bat,extras,wides,noballs,byes,legbyes,penalty,non_boundary,wicket_type,player_dismissed
                                    bat_team = row[7]
                                    bowl_team = row[8]
                                    striker = row[9]
                                    bowler = row[11]
                                    runs = int(row[12]) if row[12].isdigit() else 0
                                    wicket_type = row[20] if len(row) > 20 else ""
                                    
                                    teams_map[striker] = bat_team
                                    teams_map[bowler] = bowl_team
                                    
                                    match_runs[striker] = match_runs.get(striker, 0) + runs
                                    if wicket_type and wicket_type not in ["run out", "retired hurt", "retired out", "obstructing the field", ""]:
                                        match_wkts[bowler] = match_wkts.get(bowler, 0) + 1
                                        
                            for player, runs in match_runs.items():
                                intl_player_stats_list.append({
                                    "match_id": match_id, "player": player, "team": teams_map.get(player, ""), 
                                    "runs": runs, "wickets": match_wkts.get(player, 0), "tournament": "International"
                                })
                            for player, wkts in match_wkts.items():
                                if player not in match_runs:
                                    intl_player_stats_list.append({
                                        "match_id": match_id, "player": player, "team": teams_map.get(player, ""),
                                        "runs": 0, "wickets": wkts, "tournament": "International"
                                    })
            except Exception as e:
                print(f"Failed to fetch {fmt_name}: {e}")
        
        if intl_matches:
            intl_df = pd.DataFrame(intl_matches)
            out_df = pd.concat([out_df, intl_df], ignore_index=True)
            
        os.makedirs("data/cricket", exist_ok=True)
        out_df.to_csv("data/cricket/cricket_match_info.csv", index=False)
        print(f"Downloaded {len(out_df)} total real cricket matches successfully (IPL + Internationals).")
        
        intl_stats_df = pd.DataFrame(intl_player_stats_list) if intl_player_stats_list else pd.DataFrame(columns=['match_id', 'player', 'team', 'runs', 'wickets', 'tournament'])
        
        # Save International Cricket Player Stats
        if not intl_stats_df.empty:
            intl_stats_df.to_csv("data/cricket/cricket_player_stats.csv", index=False)
            print(f"Extracted {len(intl_stats_df)} player records from International matches.")
            
    except Exception as e:
        print(f"Failed to download cricket data: {e}")

def main():
    os.makedirs("data", exist_ok=True)
    download_football()
    download_cricket()
    print("All downloads complete. Real data is now populated!")

if __name__ == "__main__":
    main()
