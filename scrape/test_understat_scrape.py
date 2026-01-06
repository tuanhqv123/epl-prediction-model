import sys
sys.path.insert(0, '.')
import time
from understatapi import UnderstatClient
import pandas as pd


# Test 1: Get matches for one season
print("\n1. Getting matches for 2023/24 season...")
with UnderstatClient() as client:
    matches = client.league(league="EPL").get_match_data(season="2023")

print(f"   Found {len(matches)} matches")
print(f"   Sample match: {matches[0] if matches else 'None'}")

# Test 2: Get shots for first 5 matches
print("\n2. Getting shots for first 5 matches...")
all_shots = []
for i, match in enumerate(matches[:5]):
    match_id = match.get('id')
    print(f"   Match {i+1}: ID={match_id}, {match.get('h', {}).get('title', 'N/A')} vs {match.get('a', {}).get('title', 'N/A')}")
    
    with UnderstatClient() as client:
        shots = client.match(match=str(match_id)).get_shot_data()
    
    home_shots = shots.get('h', [])
    away_shots = shots.get('a', [])
    print(f"      Home shots: {len(home_shots)}, Away shots: {len(away_shots)}")
    
    all_shots.extend(home_shots)
    all_shots.extend(away_shots)
    time.sleep(0.5)

print(f"\n3. Total shots from 5 matches: {len(all_shots)}")

# Test 3: Check shot data structure
print("\n4. Shot data structure:")
if all_shots:
    shot = all_shots[0]
    for key, value in shot.items():
        print(f"   {key}: {value} ({type(value).__name__})")

# Test 4: Check key fields
print("\n5. Key fields check:")
required_fields = ['X', 'Y', 'xG', 'result', 'shotType', 'situation', 'player', 'match_id', 'date']
for field in required_fields:
    present = all(field in shot for shot in all_shots)
    print(f"   {field}: {'OK' if present else 'MISSING'}")

# Test 5: Check result values
print("\n6. Result values distribution:")
results = [shot.get('result') for shot in all_shots]
from collections import Counter
for result, count in Counter(results).items():
    print(f"   {result}: {count}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
