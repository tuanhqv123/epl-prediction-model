import time
import os
from understatapi import UnderstatClient
import pandas as pd

OUTPUT_DIR = "understat_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load existing match IDs
shots = pd.read_csv(f"{OUTPUT_DIR}/epl_shots_all.csv")
match_ids = shots['match_id'].unique()
print(f"Total matches to scrape: {len(match_ids)}")

all_player_stats = []
failed = []

for i, match_id in enumerate(match_ids):
    try:
        with UnderstatClient() as client:
            roster = client.match(match=str(match_id)).get_roster_data()
        
        # Get match info from shots
        match_shots = shots[shots['match_id'] == match_id].iloc[0]
        
        for side in ['h', 'a']:
            for player_id, player_data in roster[side].items():
                player_data['match_id'] = match_id
                player_data['side'] = side
                player_data['date'] = match_shots['date']
                player_data['season'] = match_shots['season']
                player_data['h_team'] = match_shots['h_team']
                player_data['a_team'] = match_shots['a_team']
                all_player_stats.append(player_data)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(match_ids)} matches")
            # Save intermediate
            df = pd.DataFrame(all_player_stats)
            df.to_csv(f"{OUTPUT_DIR}/player_stats_partial.csv", index=False)
        
        time.sleep(0.3)
        
    except Exception as e:
        failed.append({'match_id': match_id, 'error': str(e)})
        print(f"Error match {match_id}: {e}")
        time.sleep(1)

# Save final
df = pd.DataFrame(all_player_stats)
df.to_csv(f"{OUTPUT_DIR}/player_stats_all.csv", index=False)

print(f"\nDone! {len(all_player_stats)} player records from {len(match_ids)} matches")
print(f"Failed: {len(failed)}")
print(f"Saved to {OUTPUT_DIR}/player_stats_all.csv")
