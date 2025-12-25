#!/usr/bin/env python3
"""
EPL Lineup Scraper - Robust Version
Step 1: Collect all match URLs
Step 2: Scrape each URL
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json, os, csv, re

BASE = "https://www.worldfootball.net"
OUT = "epl_dynamic_data"
os.makedirs(OUT, exist_ok=True)

SEASONS = [
    ('2015-2016', '/competition/co91/england-premier-league/se18350/2015-2016/results-and-standings/'),
    ('2016-2017', '/competition/co91/england-premier-league/se20827/2016-2017/results-and-standings/'),
    ('2017-2018', '/competition/co91/england-premier-league/se23911/2017-2018/results-and-standings/'),
    ('2018-2019', '/competition/co91/england-premier-league/se28514/2018-2019/results-and-standings/'),
    ('2019-2020', '/competition/co91/england-premier-league/se31730/2019-2020/results-and-standings/'),
    ('2020-2021', '/competition/co91/england-premier-league/se36131/2020-2021/results-and-standings/'),
    ('2021-2022', '/competition/co91/england-premier-league/se39343/2021-2022/results-and-standings/'),
    ('2022-2023', '/competition/co91/england-premier-league/se45794/2022-2023/results-and-standings/'),
    ('2023-2024', '/competition/co91/england-premier-league/se52517/2023-2024/results-and-standings/'),
    ('2024-2025', '/competition/co91/england-premier-league/se74714/2024-2025/results-and-standings/'),
]

URLS_FILE = f"{OUT}/all_match_urls.json"
DATA_FILE = f"{OUT}/all_matches.json"
FAILED_FILE = f"{OUT}/failed_urls.json"

# Global driver
driver = None


def create_driver():
    """Create new driver"""
    global driver
    # Kill old driver
    try:
        if driver:
            driver.quit()
    except:
        pass
    
    options = uc.ChromeOptions()
    options.page_load_strategy = 'eager'
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-images')
    options.add_argument('--disable-extensions')
    
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(15)
    driver.implicitly_wait(2)
    return driver


def check_driver_alive():
    """Check if driver is still responsive"""
    global driver
    try:
        _ = driver.current_url
        return True
    except:
        return False


def scrape_match(url, season, round_num):
    """Scrape single match - returns data dict or None"""
    global driver
    
    # Navigate
    try:
        driver.get(url)
    except:
        pass  # Timeout OK
    
    time.sleep(1.5)
    
    # Check driver alive
    if not check_driver_alive():
        return None
    
    data = {'season': season, 'round': round_num, 'url': url}
    
    # Wait for lineup (max 3 sec)
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.hs-lineup--starter'))
        )
    except:
        pass
    
    # Date
    try:
        el = driver.find_element(By.CSS_SELECTOR, 'div[data-datetime]')
        data['date'] = el.get_attribute('data-datetime')
    except:
        pass
    
    # Score
    try:
        data['score'] = driver.find_element(By.CSS_SELECTOR, 'div.match-result').text
    except:
        pass
    
    # Home team
    try:
        h = driver.find_element(By.CSS_SELECTOR, 'div.hs-lineup--starter.home h3').text
        m = re.match(r'(.+?)\s*\(([^)]+)\)', h)
        data['home'] = m.group(1).strip() if m else h
        data['home_f'] = m.group(2).strip() if m else ''
    except:
        pass
    
    # Away team
    try:
        a = driver.find_element(By.CSS_SELECTOR, 'div.hs-lineup--starter.away h3').text
        m = re.match(r'(.+?)\s*\(([^)]+)\)', a)
        data['away'] = m.group(1).strip() if m else a
        data['away_f'] = m.group(2).strip() if m else ''
    except:
        pass
    
    # Players
    try:
        data['home_p'] = [e.text for e in driver.find_elements(By.CSS_SELECTOR, 'div.hs-lineup--starter.home div.person-name a')[:11] if e.text]
        data['away_p'] = [e.text for e in driver.find_elements(By.CSS_SELECTOR, 'div.hs-lineup--starter.away div.person-name a')[:11] if e.text]
    except:
        data['home_p'] = []
        data['away_p'] = []
    
    # Check if we got the essential data
    if data.get('home') and data.get('away'):
        return data
    return None


def step2_scrape_data():
    """Step 2: Scrape each URL"""
    global driver
    
    print("=" * 60)
    print("STEP 2: Scraping match data")
    print("=" * 60)
    
    if not os.path.exists(URLS_FILE):
        print(f"No URLs file! Run step1 first.")
        return
    
    with open(URLS_FILE) as f:
        all_urls = json.load(f)
    
    # Load existing data
    all_data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            all_data = json.load(f)
    
    scraped_urls = {d['url'] for d in all_data}
    to_scrape = [u for u in all_urls if u['url'] not in scraped_urls]
    
    print(f"Total URLs: {len(all_urls)}")
    print(f"Already scraped: {len(scraped_urls)}")
    print(f"Remaining: {len(to_scrape)}")
    
    if not to_scrape:
        print("Nothing to scrape!")
        return
    
    driver = create_driver()
    failed = []
    consecutive_fails = 0
    
    try:
        for i, item in enumerate(to_scrape, 1):
            url = item['url']
            season = item['season']
            round_num = item['round']
            
            print(f"[{i}/{len(to_scrape)}] {season} R{round_num}: ", end="", flush=True)
            
            # Try up to 3 times
            data = None
            for attempt in range(3):
                # Check if driver is alive, restart if not
                if not check_driver_alive():
                    print(f"[restart]", end="", flush=True)
                    driver = create_driver()
                    time.sleep(1)
                
                data = scrape_match(url, season, round_num)
                
                if data:
                    break
                elif attempt < 2:
                    print(f"[retry{attempt+1}]", end="", flush=True)
                    time.sleep(1)
            
            if data:
                all_data.append(data)
                print(f"✓ {data['home']} vs {data['away']}")
                consecutive_fails = 0
            else:
                failed.append(item)
                print("✗")
                consecutive_fails += 1
                
                # If 5 consecutive fails, restart driver
                if consecutive_fails >= 5:
                    print("  [5 fails, restarting driver...]")
                    driver = create_driver()
                    consecutive_fails = 0
                    time.sleep(2)
            
            # Save every 20
            if i % 20 == 0:
                with open(DATA_FILE, 'w') as f:
                    json.dump(all_data, f, indent=2)
                with open(FAILED_FILE, 'w') as f:
                    json.dump(failed, f, indent=2)
                print(f"  --- Saved: {len(all_data)} OK, {len(failed)} failed ---")
                    
    except KeyboardInterrupt:
        print("\n\nStopped by user!")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Save final
        with open(DATA_FILE, 'w') as f:
            json.dump(all_data, f, indent=2)
        with open(FAILED_FILE, 'w') as f:
            json.dump(failed, f, indent=2)
        
        # CSV export
        with open(f"{OUT}/all_matches.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Season','Round','Date','Home','Away','Score','HomeF','AwayF','HomeP','AwayP','URL'])
            for d in all_data:
                w.writerow([
                    d['season'], d.get('round',''), d.get('date',''), 
                    d.get('home',''), d.get('away',''), d.get('score',''),
                    d.get('home_f',''), d.get('away_f',''),
                    ';'.join(d.get('home_p',[])), ';'.join(d.get('away_p',[])), 
                    d['url']
                ])
        
        print(f"\n{'='*60}")
        print(f"Final: {len(all_data)} matches scraped, {len(failed)} failed")
        try:
            driver.quit()
        except:
            pass


def step1_collect_urls(start_season=0):
    """Step 1: Collect all match URLs"""
    global driver
    
    print("=" * 60)
    print("STEP 1: Collecting all match URLs")
    print("=" * 60)
    
    driver = create_driver()
    
    # Load existing URLs
    all_urls = []
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE) as f:
            all_urls = json.load(f)
    
    seen_urls = {u['url'] for u in all_urls}
    
    try:
        for s_idx, (season, s_url) in enumerate(SEASONS):
            if s_idx < start_season:
                continue
                
            print(f"\n{'='*60}")
            print(f"[{s_idx+1}/10] SEASON: {season}")
            print(f"{'='*60}")
            
            full_url = BASE + s_url
            
            rounds = []
            for attempt in range(5):
                if not check_driver_alive():
                    driver = create_driver()
                
                try:
                    driver.get(full_url)
                except:
                    pass
                time.sleep(2)
                
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'select[aria-label="Select round"]'))
                    )
                except:
                    print(f"  Select not found, retry {attempt+1}...")
                    continue
                
                opts = driver.find_elements(By.CSS_SELECTOR, 'select[aria-label="Select round"] option')
                for opt in opts:
                    try:
                        v = opt.get_attribute('value')
                        if v:
                            full = BASE + v if v.startswith('/') else v
                            if full not in rounds:
                                rounds.append(full)
                    except:
                        pass
                
                if len(rounds) >= 30:
                    break
                print(f"  Got {len(rounds)} rounds, retry {attempt+1}...")
                time.sleep(2)
            
            print(f"Found {len(rounds)} rounds")
            
            if not rounds:
                continue
            
            season_urls = 0
            for r_idx, r_url in enumerate(rounds, 1):
                match_urls = []
                
                for attempt in range(5):
                    if not check_driver_alive():
                        driver = create_driver()
                    
                    try:
                        driver.get(r_url)
                    except:
                        pass
                    time.sleep(1.5)
                    
                    try:
                        WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/match-report/"]'))
                        )
                    except:
                        pass
                    
                    links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/match-report/"][href*="/lineup/"]')
                    for a in links:
                        try:
                            href = a.get_attribute('href')
                            if href and href not in seen_urls:
                                seen_urls.add(href)
                                match_urls.append(href)
                        except:
                            pass
                    
                    if match_urls:
                        break
                    if attempt < 4:
                        print(f"  R{r_idx}: 0 matches, retry {attempt+1}...", end="", flush=True)
                        time.sleep(1)
                
                for href in match_urls:
                    all_urls.append({'season': season, 'round': r_idx, 'url': href})
                
                season_urls += len(match_urls)
                print(f"  Round {r_idx:2d}: {len(match_urls)} matches | Season total: {season_urls}")
                
                with open(URLS_FILE, 'w') as f:
                    json.dump(all_urls, f, indent=2)
            
            print(f"\n  Season {season} complete: {season_urls} matches")
                
    except KeyboardInterrupt:
        print("\n\nStopped by user!")
    finally:
        with open(URLS_FILE, 'w') as f:
            json.dump(all_urls, f, indent=2)
        print(f"\nSaved {len(all_urls)} URLs to {URLS_FILE}")
        try:
            driver.quit()
        except:
            pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python epl_scraper_final.py 1       # Step 1: Collect URLs")
        print("  python epl_scraper_final.py 1 N     # Start from season N (0-9)")
        print("  python epl_scraper_final.py 2       # Step 2: Scrape data")
    elif sys.argv[1] == '1':
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        step1_collect_urls(start_season=start)
    elif sys.argv[1] == '2':
        step2_scrape_data()
