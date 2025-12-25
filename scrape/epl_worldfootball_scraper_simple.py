#!/usr/bin/env python3
"""
Simple EPL Non-League Matches Scraper
Streamlined version - single window, no ad blocking, just get elements and move on
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
import os
from datetime import datetime

class EPLScraper:
    def __init__(self):
        self.base_url = "https://www.worldfootball.net"
        self.csv_file = "epl_non_league_matches.csv"
        self.driver = None
        self.all_matches = []

    def _create_simple_driver(self):
        """Create simple driver without extensions"""
        options = uc.ChromeOptions()
        options.version_main = 120
        options.page_load_strategy = 'eager'  # Wait for DOM but not all resources

        # Minimal options for speed but keep JavaScript
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-images")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-notifications")

        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(10)  # Reasonable timeout
        driver.implicitly_wait(2)  # Short implicit wait
        return driver

    def scrape_all(self):
        """Scrape all seasons using single window, step by step"""
        seasons = [
            ('2015-2016', f"{self.base_url}/competition/co91/england-premier-league/se18350/2015-2016/results-and-standings/"),
            ('2016-2017', f"{self.base_url}/competition/co91/england-premier-league/se20827/2016-2017/results-and-standings/"),
            ('2017-2018', f"{self.base_url}/competition/co91/england-premier-league/se23911/2017-2018/results-and-standings/"),
            ('2018-2019', f"{self.base_url}/competition/co91/england-premier-league/se28514/2018-2019/results-and-standings/"),
            ('2019-2020', f"{self.base_url}/competition/co91/england-premier-league/se31730/2019-2020/results-and-standings/"),
            ('2020-2021', f"{self.base_url}/competition/co91/england-premier-league/se36131/2020-2021/results-and-standings/"),
            ('2021-2022', f"{self.base_url}/competition/co91/england-premier-league/se39343/2021-2022/results-and-standings/"),
            ('2022-2023', f"{self.base_url}/competition/co91/england-premier-league/se45794/2022-2023/results-and-standings/"),
            ('2023-2024', f"{self.base_url}/competition/co91/england-premier-league/se52517/2023-2024/results-and-standings/"),
            ('2024-2025', f"{self.base_url}/competition/co91/england-premier-league/se74714/2024-2025/results-and-standings/")
        ]

        print(f"Starting scraper with {len(seasons)} seasons")

        # Create single driver
        self.driver = self._create_simple_driver()

        try:
            for i, (season_name, season_url) in enumerate(seasons):
                print(f"\n{'='*60}")
                print(f"PROCESSING SEASON {i+1}/{len(seasons)}: {season_name}")
                print(f"{'='*60}")

                try:
                    # Navigate to season page
                    print(f"Navigating to: {season_url}")
                    try:
                        self.driver.get(season_url)
                    except:
                        # If page load times out, continue anyway
                        pass

                    # Get teams quickly - don't wait for full page load
                    teams = self._get_teams_fast(season_name)

                    if not teams:
                        print(f"No teams found for {season_name}, skipping...")
                        continue

                    print(f"Found {len(teams)} teams, processing...")

                    # Process each team
                    for j, (team_name, team_href) in enumerate(teams, 1):
                        try:
                            matches = self._get_team_matches_fast(team_name, team_href, season_name)
                            self.all_matches.extend(matches)
                            print(f"    [{j}/{len(teams)}] {team_name}: {len(matches)} matches")

                            # Save every 10 matches to avoid memory issues
                            if len(self.all_matches) >= 10:
                                self.save_matches()

                            # Minimal delay
                            time.sleep(random.uniform(0.1, 0.3))

                        except Exception as e:
                            print(f"    Error with {team_name}: {str(e)[:30]}")
                            continue

                    # Save remaining matches for this season
                    self.save_matches()

                except Exception as e:
                    print(f"Error processing {season_name}: {e}")
                    continue

        finally:
            # Clean up driver
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass

        print(f"\nCompleted! All matches saved to {self.csv_file}")

    def _get_teams_fast(self, season_name):
        """Get teams quickly - don't wait for full page load"""
        print(f"Fetching teams for {season_name}")

        try:
            teams = []

            # Wait a bit longer for the page to load some content
            time.sleep(1.5)

            # Try multiple selectors and approaches quickly
            selectors = [
                'td.team-name.team-name- a',
                'td.team-name a',
                '.team-name a',
                'a[href*="/team/"]',
                'table.standard_tabelle a[href*="/team/"]',
                'tbody a[href*="/team/"]',
                '.standard_tabelle a[href*="/team/"]'
            ]

            print(f"Trying {len(selectors)} different selectors...")

            # Try each selector with minimal wait
            for i, selector in enumerate(selectors):
                try:
                    print(f"  Trying selector {i+1}: {selector}")
                    team_links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    print(f"    Found {len(team_links)} links")

                    if team_links:
                        for link in team_links:
                            try:
                                team_name = link.text.strip()
                                team_href = link.get_attribute('href')
                                print(f"    Team candidate: '{team_name}' -> {team_href}")

                                if team_href and team_name and len(team_name) > 2:
                                    # Filter out obvious non-team links
                                    if any(word in team_name.lower() for word in ['table', 'fixture', 'result', 'match', 'goal']):
                                        continue

                                    team_href = team_href.replace(self.base_url, '')
                                    if (team_name, team_href) not in teams:  # Avoid duplicates
                                        teams.append((team_name, team_href))
                                        print(f"    ✓ Added: {team_name}")
                            except Exception as e:
                                print(f"    Error processing link: {e}")
                                continue

                        if teams:
                            print(f"  Success with selector {i+1}! Found {len(teams)} teams")
                            break  # Got teams, stop trying other selectors

                except Exception as e:
                    print(f"  Selector {i+1} failed: {e}")
                    continue

            # If still no teams, try a more aggressive approach
            if not teams:
                print("No teams found with standard selectors, trying aggressive approach...")
                time.sleep(1)
                try:
                    # Get all links and filter for team links
                    all_links = self.driver.find_elements(By.TAG_NAME, 'a')
                    print(f"Found {len(all_links)} total links on page")

                    for link in all_links:
                        try:
                            href = link.get_attribute('href')
                            text = link.text.strip()

                            if href and '/team/' in href and text and len(text) > 2:
                                # Filter out obvious non-team text
                                if not any(word in text.lower() for word in ['table', 'fixture', 'result', 'match', 'goal', 'more']):
                                    team_href = href.replace(self.base_url, '')
                                    if (text, team_href) not in teams:
                                        teams.append((text, team_href))
                                        print(f"    ✓ Found team: {text}")
                        except:
                            continue

                except Exception as e:
                    print(f"Aggressive approach failed: {e}")

            print(f"Final result: Found {len(teams)} teams for {season_name}")
            if teams:
                print("Teams found:")
                for i, (name, href) in enumerate(teams[:5], 1):  # Show first 5
                    print(f"  {i}. {name}")
                if len(teams) > 5:
                    print(f"  ... and {len(teams) - 5} more")

            return teams

        except Exception as e:
            print(f"Error getting teams: {e}")
            return []

    def _get_team_matches_fast(self, team_name, team_href, season):
        """Get matches quickly - just grab what's available"""
        url = f"{self.base_url}{team_href}vs{season}/all-matches/"

        try:
            # Navigate without waiting for full load
            try:
                self.driver.get(url)
            except:
                # Continue even if page load times out
                pass

            matches = []

            # Try multiple approaches to get match data quickly
            selectors = [
                'tbody tr[data-match_id]',
                'tr[data-match_id]',
                'table tr[data-match_id]'
            ]

            for selector in selectors:
                try:
                    rows = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    if rows:
                        in_epl = False
                        for row in rows:
                            try:
                                comp_id = row.get_attribute('data-competition_id')

                                # Skip EPL matches (competition_id = 91)
                                if comp_id == '91':
                                    in_epl = True
                                    continue

                                if in_epl and comp_id != '91':
                                    in_epl = False

                                # Only process non-EPL matches
                                if comp_id != '91' and not in_epl:
                                    cells = row.find_elements(By.TAG_NAME, 'td')
                                    if len(cells) >= 8:
                                        try:
                                            # Try multiple ways to get score
                                            score_text = ""
                                            try:
                                                score_element = cells[7].find_element(By.CSS_SELECTOR, 'span a')
                                                score_text = score_element.text.strip()
                                            except:
                                                try:
                                                    score_text = cells[7].text.strip()
                                                except:
                                                    score_text = "N/A"

                                            match = {
                                                'team': team_name,
                                                'season': season,
                                                'date': cells[0].text.strip() if cells[0].text else "N/A",
                                                'home_away': cells[3].text.strip() if len(cells) > 3 and cells[3].text else "N/A",
                                                'opponent': cells[5].text.strip() if len(cells) > 5 and cells[5].text else "N/A",
                                                'result_wl': cells[6].text.strip() if len(cells) > 6 and cells[6].text else "N/A",
                                                'result_score': score_text,
                                                'competition_id': comp_id or "N/A",
                                                'match_id': row.get_attribute('data-match_id') or "N/A"
                                            }
                                            matches.append(match)
                                        except Exception as cell_error:
                                            continue
                            except Exception as row_error:
                                continue

                        # If we got matches, return them
                        if matches:
                            return matches

                except Exception as selector_error:
                    continue

            # If no matches found with any selector, try one more time with brief wait
            if not matches:
                time.sleep(0.2)
                try:
                    rows = self.driver.find_elements(By.CSS_SELECTOR, 'tr')
                    for row in rows:
                        try:
                            if row.get_attribute('data-match_id'):
                                comp_id = row.get_attribute('data-competition_id')
                                if comp_id and comp_id != '91':
                                    cells = row.find_elements(By.TAG_NAME, 'td')
                                    if len(cells) >= 6:
                                        match = {
                                            'team': team_name,
                                            'season': season,
                                            'date': cells[0].text.strip() if cells[0].text else "N/A",
                                            'home_away': "N/A",
                                            'opponent': "N/A",
                                            'result_wl': "N/A",
                                            'result_score': "N/A",
                                            'competition_id': comp_id,
                                            'match_id': row.get_attribute('data-match_id')
                                        }
                                        matches.append(match)
                        except:
                            continue
                except:
                    pass

            return matches

        except Exception as e:
            print(f"    Error scraping {team_name}: {str(e)[:50]}")
            return []

    def save_matches(self):
        """Save matches to CSV"""
        if not self.all_matches:
            return

        df = pd.DataFrame(self.all_matches)

        if not os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, index=False)
            print(f"\nCreated {self.csv_file}")
        else:
            df.to_csv(self.csv_file, mode='a', header=False, index=False)

        self.all_matches = []


if __name__ == "__main__":
    print("Starting EPL Non-League Matches Scraper - Fast Version")
    print("="*60)

    scraper = EPLScraper()

    try:
        scraper.scrape_all()
        print("\n" + "="*60)
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {scraper.csv_file}")
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nScraping failed with error: {e}")
    finally:
        # Cleanup
        try:
            if scraper.driver:
                scraper.driver.quit()
        except:
            pass
