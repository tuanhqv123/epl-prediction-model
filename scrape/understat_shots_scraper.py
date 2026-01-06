import sys
sys.path.insert(0, '.')
import json
import time
import os
from understatapi import UnderstatClient
import pandas as pd

OUTPUT_DIR = "understat_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASONS = list(range(2014, 2025))  # 2014 = 2014/15 season

def get_league_matches(season):
    """Get all match IDs for EPL season"""
    with UnderstatClient() as client:
        matches = client.league(league="EPL").get_match_data(season=str(season))
    return matches

def get_match_shots(match_id):
    """Get shot data for a single match"""
    with UnderstatClient() as client:
        shots = client.match(match=str(match_id)).get_shot_data()
    return shots

def scrape_season(season):
    """Scrape all shots for a season"""
    print(f"\nScraping season {season}/{season+1}...")
    
    # Get all matches
    matches = get_league_matches(season)
    print(f"  Found {len(matches)} matches")
    
    all_shots = []
    failed_matches = []
    
    for i, match in enumerate(matches):
        match_id = match.get('id')
        if not match_id:
            continue
            
        try:
            shots_data = get_match_shots(match_id)
            
            # Combine home and away shots
            home_shots = shots_data.get('h', [])
            away_shots = shots_data.get('a', [])
            
            for shot in home_shots + away_shots:
                shot['season_year'] = season
                all_shots.append(shot)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(matches)} matches, {len(all_shots)} shots so far")
            
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            failed_matches.append({'match_id': match_id, 'error': str(e)})
            print(f"  Error match {match_id}: {e}")
            time.sleep(1)
    
    print(f"  Season {season}/{season+1}: {len(all_shots)} shots from {len(matches)} matches")
    if failed_matches:
        print(f"  Failed: {len(failed_matches)} matches")
    
    return all_shots, matches, failed_matches

def main():
    print("UNDERSTAT EPL SHOTS SCRAPER")
    print("-" * 50)
    
    all_shots = []
    all_matches = []
    all_failed = []
    
    for season in SEASONS:
        shots, matches, failed = scrape_season(season)
        all_shots.extend(shots)
        all_matches.extend(matches)
        all_failed.extend(failed)
        
        # Save intermediate results
        df_shots = pd.DataFrame(all_shots)
        df_shots.to_csv(f"{OUTPUT_DIR}/epl_shots_partial.csv", index=False)
        print(f"  Saved {len(all_shots)} shots so far")
        
        time.sleep(2)  # Pause between seasons
    
    # Save final results
    df_shots = pd.DataFrame(all_shots)
    df_shots.to_csv(f"{OUTPUT_DIR}/epl_shots_all.csv", index=False)
    
    df_matches = pd.DataFrame(all_matches)
    df_matches.to_csv(f"{OUTPUT_DIR}/epl_matches_all.csv", index=False)
    
    if all_failed:
        df_failed = pd.DataFrame(all_failed)
        df_failed.to_csv(f"{OUTPUT_DIR}/failed_matches.csv", index=False)
    
    print("\n" + "=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Total shots: {len(all_shots)}")
    print(f"Total matches: {len(all_matches)}")
    print(f"Failed matches: {len(all_failed)}")
    print(f"\nFiles saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
