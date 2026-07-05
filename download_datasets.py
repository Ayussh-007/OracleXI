import os
import requests
import zipfile
import io
import pandas as pd
from typing import Optional

def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    downloaded = 0
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                print(f"  Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB", end='\r')
    print(f"\nSaved to {dest_path}")

def process_football_data():
    """Download and format football data."""
    print("--- Processing Football Data ---")
    fb_dir = os.path.join("data", "football")
    os.makedirs(fb_dir, exist_ok=True)
    
    # URLs from Mart Jürisoo's repository
    results_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    shootouts_url = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
    goalscorers_url = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
    
    download_file(results_url, os.path.join(fb_dir, "international_results.csv"))
    download_file(shootouts_url, os.path.join(fb_dir, "shootouts.csv"))
    download_file(goalscorers_url, os.path.join(fb_dir, "goalscorers.csv"))
    
    print(f"Formatted football results")

def _parse_cricsheet_info(lines, fmt):
    """Parse a cricsheet info CSV into a dictionary."""
    data = {"format": fmt.upper()}
    teams = []
    for line in lines:
        parts = line.decode('utf-8').strip().split(',')
        if len(parts) >= 3 and parts[0] == 'info':
            key = parts[1]
            val = parts[2]
            if key == 'team':
                teams.append(val)
            elif key == 'toss_winner':
                data['toss_winner'] = val
            elif key == 'toss_decision':
                data['toss_decision'] = val
            elif key == 'winner':
                data['winner'] = val
            elif key == 'date':
                data['date'] = val
            elif key == 'venue':
                data['tournament'] = val # map venue to tournament for now
            elif key == 'city':
                data['city'] = val
            
    if len(teams) >= 2:
        data['team1'] = teams[0]
        data['team2'] = teams[1]
    
    return data

def process_cricket_data():
    """Download and format cricket data."""
    print("--- Processing Cricket Data ---")
    cr_dir = os.path.join("data", "cricket")
    os.makedirs(cr_dir, exist_ok=True)
    
    # URLs from Cricsheet
    urls = {
        "t20": "https://cricsheet.org/downloads/t20s_csv2.zip",
    }
    
    match_infos = []
    
    for fmt, url in urls.items():
        zip_path = os.path.join(cr_dir, f"{fmt}.zip")
        download_file(url, zip_path)
        
        print(f"Extracting and parsing {fmt}.zip (this may take a minute) ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            info_files = [f for f in zip_ref.namelist() if f.endswith('_info.csv')]
            
            for info_file in info_files:
                with zip_ref.open(info_file) as f:
                    info_dict = _parse_cricsheet_info(f.readlines(), fmt)
                    info_dict['match_id'] = info_file.replace('_info.csv', '')
                    match_infos.append(info_dict)
                    
    if match_infos:
        master_df = pd.DataFrame(match_infos)
        # Ensure we don't have missing columns that DatasetManager expects
        if "winner" not in master_df.columns: master_df["winner"] = "No Result"
        if "toss_winner" not in master_df.columns: master_df["toss_winner"] = "Unknown"
        if "toss_decision" not in master_df.columns: master_df["toss_decision"] = "Unknown"
        if "result_margin" not in master_df.columns: master_df["result_margin"] = 0
        if "result_type" not in master_df.columns: master_df["result_type"] = "N/A"
        
        # Save to csv
        master_df.to_csv(os.path.join(cr_dir, "cricket_match_info.csv"), index=False)
        print(f"Formatted cricket match info: {len(master_df)} rows")
    else:
        print("Warning: No match info found.")

def main():
    print("Starting OracleXI Dataset Downloader...")
    os.makedirs("data", exist_ok=True)
    
    # Clear out synthetic datasets if they exist
    import shutil
    for path in ["data/cricket_match_info.csv", "data/cricket_ball_by_ball.csv", "data/football_results.csv"]:
        if os.path.exists(path):
            os.remove(path)
            
    process_football_data()
    process_cricket_data()
    print("Done! Data is ready for DatasetManager.")

if __name__ == "__main__":
    main()
