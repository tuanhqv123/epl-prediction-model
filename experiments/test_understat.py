from understatapi import UnderstatClient

with UnderstatClient() as client:
    # Get shots data for a match
    match_data = client.match(match="14711").get_shot_data()
    print(f"Type: {type(match_data)}")
    print(f"Keys: {match_data.keys() if hasattr(match_data, 'keys') else 'N/A'}")
    print(f"\nData: {match_data}")
