import requests
import ssl
import re
import time
import os
from datetime import datetime
from bs4 import BeautifulSoup
import csv
import gzip

# SSL certificate fix for macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def decode_response(response):
    """Handle compressed responses"""
    content = response.content

    # Check if content is gzip compressed (starts with gzip magic number)
    if content.startswith(b'\x1f\x8b'):
        try:
            return gzip.decompress(content).decode('utf-8', errors='ignore')
        except:
            return response.text
    else:
        return response.text

class EPLScraper:
    def __init__(self):
        """Initialize scraper with enhanced headers"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.worldfootball.net/',
        }

        # Your Cloudflare clearance cookie
        self.cookies = {
            'cf_clearance': 'S0GFCtOdNuU_5JEsGpbkkSY_0rNmFVIOg_6egtyy3KI-1763488243-1.2.1.1-HzbJVuT_ZFuZYCRN0LtQXHZ1EKxlJjm9bkBMJd_xPkMOi89JXc1lBBBMlO5DGiZiGkd00YfWNz3.5Iq8VOABx0fyJ49MVsGRMsaoXdrbODsuPPc5rDmtROJSU5dSLj8r43pNKFUXCLyrCPLDlRbfho2mBDEjS3dUqTUZL1kTjThEcrw.g0_.D1XRV0h60xdg8TaIOlIu6YYTD.VIbWamDdhpmWm6h9vLl434fae1MG12D8miDcfShxgTCOo.Ab_eQ'
        }

    def get_teams_for_season(self, season):
        """Get all teams from Premier League season page"""
        season_patterns = {
            '2023-2024': 'se52517',
            '2022-2023': 'se45794',
            '2021-2022': 'se52515',
            '2020-2021': 'se52514',
            '2019-2020': 'se52513',
            '2018-2019': 'se52512',
            '2017-2018': 'se52511',
            '2016-2017': 'se52510',
            '2015-2016': 'se52509'
        }

        season_id = season_patterns.get(season)
        if not season_id:
            print(f"❌ No season ID found for {season}")
            return []

        url = f"https://www.worldfootball.net/competition/co91/england-premier-league/{season_id}/{season}/results-and-standings/"
        print(f"🌐 Fetching teams from: {url}")

        try:
            response = requests.get(url, headers=self.headers, cookies=self.cookies, timeout=30)
            print(f"📊 Response status: {response.status_code}")

            if response.status_code == 200:
                # Try to decode the response
                content = decode_response(response)

                # Check if it's still gibberish
                if not content or len(content) < 100:
                    print("❌ Response content appears to be empty or corrupted")
                    print(f"Raw content preview: {response.content[:100]}")
                    return []

                soup = BeautifulSoup(content, 'html.parser')
                team_links = []

                team_elements = soup.find_all('td', class_='team-name')

                for team_element in team_elements:
                    link = team_element.find('a')
                    if link and link.get('href'):
                        team_name = link.get_text(strip=True)
                        team_url = f"https://www.worldfootball.net{link['href']}"
                        team_id_match = re.search(r'/teams/te(\d+)/', team_url)

                        if team_id_match and team_name and team_name not in [team['name'] for team in team_links]:
                            team_links.append({
                                'name': team_name,
                                'url': team_url,
                                'id': team_id_match.group(1)
                            })

                print(f"✅ Found {len(team_links)} teams for {season}")
                return team_links
            else:
                print(f"❌ HTTP {response.status_code} for {url}")

                # Try to decode the error response too
                try:
                    error_content = decode_response(response)
                    if "Just a moment" in error_content or "Cloudflare" in error_content:
                        print("🛡️ Cloudflare protection detected - cookies may be expired")
                        print("🔄 Try updating your cf_clearance cookie from your browser")
                    print(f"Decoded content preview: {error_content[:200]}")
                except:
                    print(f"Raw content preview: {response.content[:200]}")

                return []

        except Exception as e:
            print(f"❌ Error fetching teams for {season}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_team_matches_for_season(self, team, season):
        """Get all matches for a team in a specific season"""
        base_team_name = team['name'].lower().replace(' ', '-').replace("'", "")
        matches_url = f"https://www.worldfootball.net/teams/te{team['id']}/{base_team_name}/vs{season}/all-matches/"

        print(f"⚽ Fetching matches for {team['name']} from: {matches_url}")

        try:
            time.sleep(2)  # Be respectful
            response = requests.get(matches_url, headers=self.headers, cookies=self.cookies, timeout=30)
            print(f"📈 Matches response status: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                matches = []

                finished_matches = soup.find_all('div', class_='finished')

                for match_div in finished_matches:
                    datetime_attr = match_div.get('data-datetime')
                    if not datetime_attr:
                        continue

                    try:
                        dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                        date_formatted = dt.strftime('%d/%m/%Y')
                    except:
                        continue

                    winner_element = match_div.find('div', class_='team-name hs-winner')

                    if winner_element:
                        winner = winner_element.get_text(strip=True)
                        matches.append({
                            'date': date_formatted,
                            'season': season,
                            'team': team['name'],
                            'team_win': winner
                        })

                print(f"🎯 Found {len(matches)} matches for {team['name']} in {season}")
                return matches
            else:
                print(f"❌ HTTP {response.status_code} for {team['name']} matches")
                return []

        except Exception as e:
            print(f"❌ Error fetching matches for {team['name']}: {e}")
            return []

    def save_to_csv(self, matches, filename):
        """Save matches to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['date', 'season', 'team', 'team_win']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for match in matches:
                writer.writerow(match)

    def scrape_season(self, season, output_file):
        """Scrape all matches for a given season and save to CSV"""
        print(f"\n🏆 === Scraping season {season} ===")

        teams = self.get_teams_for_season(season)

        if not teams:
            print(f"❌ No teams found for {season}")
            return

        print(f"📋 Teams found: {[team['name'] for team in teams[:5]]}{'...' if len(teams) > 5 else ''}")

        all_matches = []

        for i, team in enumerate(teams, 1):
            print(f"\n⚽ Processing team {i}/{len(teams)}: {team['name']}")
            matches = self.get_team_matches_for_season(team, season)
            all_matches.extend(matches)

            # Add delay between requests
            if i < len(teams):
                delay = 3
                print(f"⏰ Waiting {delay} seconds...")
                time.sleep(delay)

        if all_matches:
            self.save_to_csv(all_matches, output_file)
            print(f"\n✅ Saved {len(all_matches)} matches to {output_file}")
        else:
            print(f"❌ No matches found for {season}")

def parse_teams_from_html_file(html_file):
    """Parse teams from saved HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        team_links = []

        team_elements = soup.find_all('td', class_='team-name')

        for team_element in team_elements:
            link = team_element.find('a')
            if link and link.get('href'):
                team_name = link.get_text(strip=True)
                team_url = f"https://www.worldfootball.net{link['href']}"
                team_id_match = re.search(r'/teams/te(\d+)/', team_url)

                if team_id_match and team_name and team_name not in [team['name'] for team in team_links]:
                    team_links.append({
                        'name': team_name,
                        'url': team_url,
                        'id': team_id_match.group(1)
                    })

        print(f"✅ Found {len(team_links)} teams in HTML file")
        return team_links

    except Exception as e:
        print(f"❌ Error parsing teams from {html_file}: {e}")
        return []

def parse_matches_from_html_file(html_file, team_name, season):
    """Parse matches from saved HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        matches = []

        finished_matches = soup.find_all('div', class_='finished')

        for match_div in finished_matches:
            datetime_attr = match_div.get('data-datetime')
            if not datetime_attr:
                continue

            try:
                dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                date_formatted = dt.strftime('%d/%m/%Y')
            except:
                continue

            winner_element = match_div.find('div', class_='team-name hs-winner')

            if winner_element:
                winner = winner_element.get_text(strip=True)
                matches.append({
                    'date': date_formatted,
                    'season': season,
                    'team': team_name,
                    'team_win': winner
                })

        print(f"✅ Found {len(matches)} matches for {team_name}")
        return matches

    except Exception as e:
        print(f"❌ Error parsing matches from {html_file}: {e}")
        return []

def print_instructions():
    """Print clear instructions for manual HTML collection"""
    print("""
🔧 MANUAL HTML COLLECTION INSTRUCTIONS:

Since the site has strong Cloudflare protection, here's how to get the data:

1️⃣ SAVE TEAMS HTML:
   • Open: https://www.worldfootball.net/competition/co91/england-premier-league/se52517/2023-2024/results-and-standings/
   • Wait for page to load completely
   • Right-click → "Save Page As" → "Webpage, HTML only"
   • Save as: "teams_2023_2024.html"

2️⃣ SAVE TEAM MATCHES HTML:
   For each team from step 1:
   • Click on the team name (e.g., "Manchester City")
   • Navigate to: vs2023-2024/all-matches/
   • Save page as: "matches_[team_name].html"
   • Example: "matches_manchester_city.html"

3️⃣ RUN PARSER:
   Update the file paths in the main() function below and run!

This approach bypasses all Cloudflare issues and works reliably! 🚀
""")

def main():
    """Main function - automated or manual approach"""
    print("🚀 Starting EPL scraper...")

    # First try automated approach
    scraper = EPLScraper()
    test_season = "2023-2024"
    output_filename = f"premier_league_matches_{test_season.replace('-', '_')}.csv"

    try:
        scraper.scrape_season(test_season, output_filename)
        if os.path.exists(output_filename):
            print("🎉 Automated scraping completed successfully!")
            return
        else:
            print("⚠️ Automated scraping found no data - switching to manual approach...")
    except Exception as e:
        print(f"⚠️ Automated approach failed: {e}")

    print("\n🔄 Manual HTML Collection Required:")

    # Manual HTML approach
    print_instructions()

    # Example usage - update these paths after you save the HTML files
    teams_html_file = "teams_2023_2024.html"
    matches_files = [
        "matches_manchester_city.html",
        "matches_arsenal.html",
        "matches_liverpool.html",
        # Add more team match files as you save them
    ]

    if os.path.exists(teams_html_file):
        print(f"\n📂 Processing teams from {teams_html_file}")
        teams = parse_teams_from_html_file(teams_html_file)

        all_matches = []
        season = "2023-2024"

        for match_file in matches_files:
            if os.path.exists(match_file):
                team_name = match_file.replace("matches_", "").replace(".html", "").replace("_", " ").title()
                matches = parse_matches_from_html_file(match_file, team_name, season)
                all_matches.extend(matches)

        if all_matches:
            scraper = EPLScraper()
            scraper.save_to_csv(all_matches, output_filename)
            print(f"✅ Saved {len(all_matches)} matches to {output_filename}")
        else:
            print("❌ No matches found")
    else:
        print(f"❌ Teams HTML file not found: {teams_html_file}")
        print("Please follow the instructions above to save HTML files first.")

if __name__ == "__main__":
    main()