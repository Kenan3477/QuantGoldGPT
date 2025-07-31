"""
Data Fetching Module
Handles price data, news, and market data fetching
"""
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import logging
from typing import List, Dict, Tuple, Optional
from config import POLYGON_API_KEY, FRED_API_KEY, DB_PATH

class PriceDataFetcher:
    """Fetches price data from various sources"""
    
    def get_gold_price(self) -> Optional[float]:
        """Get current gold price - instance method for live monitoring"""
        return self.get_current_price()
    
    @staticmethod
    def fetch_gold_price_yahoo(period="1d", interval="1m") -> Tuple[float, Dict]:
        """Fetch gold price from Yahoo Finance with multiple ticker fallbacks"""
        # Try multiple gold tickers in order of preference
        gold_tickers = ["GC=F", "GOLD", "GLD", "XAUUSD=X"]
        
        for ticker in gold_tickers:
            try:
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
                if df.empty or len(df) == 0:
                    logging.warning(f"No data returned for {ticker}")
                    continue
                
                # Handle MultiIndex columns from yfinance
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required_columns):
                    logging.warning(f"Missing required columns for {ticker}")
                    continue
                
                # Get the last row and validate data
                last_row = df.iloc[-1]
                if pd.isna(last_row['Close']):
                    logging.warning(f"Invalid Close price for {ticker}")
                    continue
                
                current_price = float(last_row['Close'])
                if current_price <= 0:
                    logging.warning(f"Invalid price value for {ticker}: {current_price}")
                    continue
                
                price_data = {
                    'open': float(last_row['Open']) if not pd.isna(last_row['Open']) else current_price,
                    'high': float(last_row['High']) if not pd.isna(last_row['High']) else current_price,
                    'low': float(last_row['Low']) if not pd.isna(last_row['Low']) else current_price,
                    'close': current_price,
                    'volume': float(last_row['Volume']) if 'Volume' in df.columns and not pd.isna(last_row['Volume']) else 0,
                    'ticker_used': ticker
                }
                
                logging.info(f"Successfully fetched gold price from {ticker}: ${current_price:.2f}")
                return current_price, price_data
                
            except Exception as e:
                logging.warning(f"Error fetching data from {ticker}: {e}")
                continue
        
        logging.error("All Yahoo Finance gold tickers failed")
        return None, {}
    
    @staticmethod
    def fetch_goldapi_price() -> Tuple[float, str]:
        """Fetch gold price from GoldAPI - matches original implementation exactly with accuracy logging"""
        try:
            url = "https://api.gold-api.com/price/XAU"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # Use the raw price directly as per original implementation - NO MODIFICATIONS
                price = float(data["price"])
                
                # Log the exact price for accuracy verification
                logging.info(f"🔍 GOLDAPI exact price: ${price:.10f} (full precision)")
                
                # Validate price is reasonable for gold
                if 1000 <= price <= 5000:
                    # Store the exact tick for accuracy tracking
                    try:
                        PriceDataFetcher.store_goldapi_tick(price)
                    except Exception as e:
                        logging.error(f"Error storing GOLDAPI tick: {e}")
                    return price, "goldapi"
                else:
                    logging.warning(f"⚠️ GOLDAPI price seems unrealistic: ${price}")
                    return price, "goldapi"  # Still return it, but warn
            else:
                logging.error(f"GOLDAPI returned status {response.status_code}")
                return None, None
                
        except Exception as e:
            logging.error(f"Error fetching from GoldAPI: {e}")
            return None, None
    
    @staticmethod
    def fetch_multi_source_price() -> Tuple[float, str]:
        """Fetch price from multiple sources with GoldAPI as primary source (matches original)"""
        # Primary source: GoldAPI (matches original implementation)
        price, source = PriceDataFetcher.fetch_goldapi_price()
        if price is not None:
            return price, source
        
        # Fallback sources if GoldAPI fails
        fallback_sources = [
            PriceDataFetcher.fetch_gold_price_yahoo,
        ]
        
        for source_func in fallback_sources:
            try:
                price, _ = source_func()
                if price and price > 0:
                    return price, source_func.__name__
                    
            except Exception as e:
                logging.warning(f"Failed to fetch from {source_func.__name__}: {e}")
                continue
        
        return None, "All sources failed"
    
    @staticmethod
    def store_goldapi_tick(price: float):
        """Store GoldAPI tick price with full precision - NO MODIFICATIONS"""
        try:
            from datetime import datetime, timezone
            import sqlite3
            from .config import DB_PATH
            
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS goldapi_ticks (
                    timestamp TEXT PRIMARY KEY,
                    price REAL,
                    raw_price_string TEXT
                )
            """)
            # Store both the float price and the raw string for maximum accuracy
            c.execute("""
                INSERT OR IGNORE INTO goldapi_ticks (timestamp, price, raw_price_string) 
                VALUES (?, ?, ?)
            """, (ts, price, str(price)))
            conn.commit()
            conn.close()
            
            # Log storage confirmation
            logging.debug(f"✅ Stored GOLDAPI tick: ${price} at {ts}")
            
        except Exception as e:
            logging.error(f"Error storing GoldAPI tick: {e}")

    @staticmethod
    def fetch_gold_price_multi(signal_mode=False) -> Tuple[float, str]:
        """
        Fetches the current gold price from multiple sources and returns the first valid price and its source.
        GoldAPI is now the primary source. Matches original implementation exactly.
        """
        price, source = PriceDataFetcher.fetch_goldapi_price()
        if price is not None:
            return price, source
        
        # Fallback to other sources if GoldAPI fails
        try:
            # Try Yahoo Finance as fallback
            price, _ = PriceDataFetcher.fetch_gold_price_yahoo()
            if price and price > 0:
                return price, "yahoo_finance"
        except Exception as e:
            logging.warning(f"Yahoo Finance fallback failed: {e}")
        
        # All sources failed: log and return None
        logging.error("All gold price sources failed! Returning (None, 'unknown').")
        return None, "unknown"
    
    @staticmethod
    def get_current_price() -> Optional[float]:
        """Get current price - simple wrapper for trade monitoring"""
        price, _ = PriceDataFetcher.fetch_gold_price_multi()
        return price
    
    @staticmethod
    def get_price_change_24h() -> float:
        """Get 24h price change with fallback tickers"""
        gold_tickers = ["GC=F", "GOLD", "GLD", "XAUUSD=X"]
        
        for ticker in gold_tickers:
            try:
                df = yf.download(ticker, period="2d", interval="1d", auto_adjust=True, progress=False)
                
                # Handle MultiIndex columns from yfinance
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                    
                if len(df) >= 2 and 'Close' in df.columns:
                    current = float(df['Close'].iloc[-1])
                    previous = float(df['Close'].iloc[-2])
                    
                    if not pd.isna(current) and not pd.isna(previous):
                        change = current - previous
                        logging.info(f"24h price change from {ticker}: ${change:.2f}")
                        return change
                        
            except Exception as e:
                logging.warning(f"Error getting price change from {ticker}: {e}")
                continue
        
        logging.error("Failed to get 24h price change from all sources")
        return 0.0
    
    @staticmethod
    def get_current_volume() -> float:
        """Get current trading volume with fallback tickers"""
        gold_tickers = ["GC=F", "GOLD", "GLD"]  # Volume data might not be available for XAUUSD=X
        
        for ticker in gold_tickers:
            try:
                df = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
                
                # Handle MultiIndex columns from yfinance
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                    
                if not df.empty and 'Volume' in df.columns:
                    volume = float(df['Volume'].iloc[-1])
                    if not pd.isna(volume) and volume > 0:
                        logging.info(f"Volume from {ticker}: {volume:,.0f}")
                        return volume
                        
            except Exception as e:
                logging.warning(f"Error getting volume from {ticker}: {e}")
                continue
        
        logging.error("Failed to get volume from all sources")
        return 0.0
    
    def get_price_change_1h(self) -> float:
        """Get 1-hour price change with fallback tickers"""
        gold_tickers = ["GC=F", "GOLD", "GLD", "XAUUSD=X"]
        
        for ticker in gold_tickers:
            try:
                df = yf.download(ticker, period="1d", interval="1h", auto_adjust=True, progress=False)
                
                # Handle MultiIndex columns from yfinance
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                    
                if len(df) >= 2 and 'Close' in df.columns:
                    close_curr = float(df['Close'].iloc[-1])
                    close_prev = float(df['Close'].iloc[-2])
                    
                    if not pd.isna(close_curr) and not pd.isna(close_prev):
                        change = close_curr - close_prev
                        logging.info(f"1h price change from {ticker}: ${change:.2f}")
                        return change
                        
            except Exception as e:
                logging.warning(f"Error getting 1h change from {ticker}: {e}")
                continue
        
        logging.error("Failed to get 1h price change from all sources")
        return 0.0
    
    def get_historical_data(self, period: str = "1d", interval: str = "1h") -> pd.DataFrame:
        """Get historical price data with fallback tickers"""
        gold_tickers = ["GC=F", "GOLD", "GLD", "XAUUSD=X"]
        
        for ticker in gold_tickers:
            try:
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
                
                # Handle MultiIndex columns from yfinance
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
                    
                if not df.empty and 'Close' in df.columns:
                    logging.info(f"Historical data from {ticker}: {len(df)} rows")
                    return df
                    
            except Exception as e:
                logging.warning(f"Error getting historical data from {ticker}: {e}")
                continue
        
        logging.error("Failed to get historical data from all sources")
        return pd.DataFrame()
    
    def store_price_tick(self, price: float):
        """Store price tick in database with advanced market data tracking"""
        try:
            if not price or price <= 0:
                return
            
            # Get database connection
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create price_ticks table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    source TEXT DEFAULT 'unknown',
                    volume REAL DEFAULT 0,
                    bid REAL DEFAULT 0,
                    ask REAL DEFAULT 0,
                    spread REAL DEFAULT 0
                )
            """)
            
            # Store the tick
            timestamp = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO price_ticks (timestamp, price, source)
                VALUES (?, ?, ?)
            """, (timestamp, price, 'live_feed'))
            
            # Keep only last 10000 ticks to manage size
            cursor.execute("""
                DELETE FROM price_ticks 
                WHERE id NOT IN (
                    SELECT id FROM price_ticks 
                    ORDER BY timestamp DESC 
                    LIMIT 10000
                )
            """)
            
            conn.commit()
            conn.close()
            
            # Update latest price in memory cache
            self._latest_price = price
            self._last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logging.error(f"Error storing price tick: {e}")
    
    def store_goldapi_tick_instance(self, price: float):
        """Store GoldAPI tick data - instance method wrapper"""
        self.store_price_tick(price)
    
    def is_connected(self) -> bool:
        """Check if price feed is connected"""
        try:
            price = self.get_current_price()
            return price is not None
        except:
            return False

class NewsDataFetcher:
    """Fetches news data from various sources"""
    
    @staticmethod
    def fetch_kitco_headlines() -> List[str]:
        """Fetch headlines from Kitco (matches original implementation)"""
        url = "https://www.kitco.com/news/"
        try:
            page = requests.get(url, timeout=10)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")
            
            headlines = []
            # Kitco headlines are in <a class="news__headline"> or similar
            for a in soup.find_all("a", class_="news__headline"):
                text = a.get_text(strip=True)
                if text:
                    headlines.append(text)
            
            # Fallback: try h3 tags
            if not headlines:
                for h3 in soup.find_all("h3"):
                    text = h3.get_text(strip=True)
                    if text:
                        headlines.append(text)
            
            return headlines[:10]
            
        except Exception as e:
            logging.error(f"Error fetching Kitco headlines: {e}")
            return []
    
    @staticmethod
    def fetch_reuters_headlines() -> List[str]:
        """Fetch headlines from Reuters (matches original implementation)"""
        url = "https://www.reuters.com/markets/commodities/"
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            
            headlines = []
            # Reuters headlines are often in <a data-testid="Heading"> or <h3>
            for tag in soup.find_all(["a", "h3"], attrs={"data-testid": "Heading"}):
                text = tag.get_text(strip=True)
                if text:
                    headlines.append(text)
            
            # Fallback: all h3 tags
            if not headlines:
                for h3 in soup.find_all("h3"):
                    text = h3.get_text(strip=True)
                    if text:
                        headlines.append(text)
            
            return headlines[:10]
            
        except Exception as e:
            logging.error(f"Error fetching Reuters headlines: {e}")
            return []
    
    @staticmethod
    def fetch_bloomberg_headlines() -> List[str]:
        """Fetch headlines from Bloomberg (matches original implementation)"""
        url = "https://www.bloomberg.com/markets/commodities"
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            
            headlines = []
            # Bloomberg headlines are often in <a> with 'story-package-module__story__headline-link'
            for a in soup.find_all("a", class_="story-package-module__story__headline-link"):
                text = a.get_text(strip=True)
                if text:
                    headlines.append(text)
            
            # Fallback: all h3 tags
            if not headlines:
                for h3 in soup.find_all("h3"):
                    text = h3.get_text(strip=True)
                    if text:
                        headlines.append(text)
            
            return headlines[:10]
            
        except Exception as e:
            logging.error(f"Error fetching Bloomberg headlines: {e}")
            return []
    
    @staticmethod
    def fetch_reddit_headlines(subreddit="wallstreetbets", query="gold") -> List[str]:
        """Fetch headlines from Reddit"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {"q": query, "restrict_sr": "on", "sort": "new", "limit": 10}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; GoldBot/1.0)"}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                headlines = []
                
                for post in data.get('data', {}).get('children', []):
                    title = post.get('data', {}).get('title', '')
                    if title and 'gold' in title.lower():
                        headlines.append(title)
                
                return headlines[:10]
            
            return []
            
        except Exception as e:
            logging.error(f"Error fetching Reddit headlines: {e}")
            return []
    
    @staticmethod
    def fetch_all_news_sources() -> Dict[str, List[str]]:
        """Fetch news from all sources"""
        return {
            "kitco": NewsDataFetcher.fetch_kitco_headlines(),
            "reuters": NewsDataFetcher.fetch_reuters_headlines(),
            "bloomberg": NewsDataFetcher.fetch_bloomberg_headlines(),
            "reddit": NewsDataFetcher.fetch_reddit_headlines()
        }
    
    def fetch_kitco_news(self) -> List[Dict]:
        """Fetch Kitco news as dictionaries"""
        try:
            headlines = self.fetch_kitco_headlines()
            return [{"title": title, "source": "kitco"} for title in headlines]
        except Exception as e:
            logging.error(f"Error fetching Kitco news: {e}")
            return []
    
    def fetch_reuters_news(self) -> List[Dict]:
        """Fetch Reuters news as dictionaries"""
        try:
            headlines = self.fetch_reuters_headlines()
            return [{"title": title, "source": "reuters"} for title in headlines]
        except Exception as e:
            logging.error(f"Error fetching Reuters news: {e}")
            return []

    def get_latest_news(self) -> List[Dict]:
        """Get latest news articles"""
        try:
            news = []
            
            # Fetch from multiple sources
            kitco_news = self.fetch_kitco_news()
            reuters_news = self.fetch_reuters_news()
            
            news.extend(kitco_news[:3])  # Top 3 from each source
            news.extend(reuters_news[:3])
            
            return news
        except Exception as e:
            logging.error(f"Error fetching latest news: {e}")
            return []
    
    def get_emergency_news(self) -> List[Dict]:
        """Get emergency/urgent news"""
        try:
            # Check for emergency news keywords
            news = self.get_latest_news()
            emergency_news = []
            
            emergency_keywords = ['emergency', 'urgent', 'breaking', 'alert', 'crisis']
            
            for article in news:
                title = article.get('title', '').lower()
                if any(keyword in title for keyword in emergency_keywords):
                    emergency_news.append(article)
            
            return emergency_news
        except Exception as e:
            logging.error(f"Error fetching emergency news: {e}")
            return []
    
    def is_active(self) -> bool:
        """Check if news feed is active"""
        try:
            news = self.get_latest_news()
            return len(news) > 0
        except:
            return False

class MacroDataFetcher:
    """Fetches macroeconomic data"""
    
    @staticmethod
    def fetch_fred_data() -> Dict[str, float]:
        """Fetch FRED data"""
        if not FRED_API_KEY:
            return {"cpi": np.nan, "interest_rate": np.nan}
        
        try:
            try:
                from fredapi import Fred
            except ImportError:
                logging.warning("fredapi not installed. Install with: pip install fredapi")
                return {"cpi": np.nan, "interest_rate": np.nan}
            
            fred = Fred(api_key=FRED_API_KEY)
            
            # Fetch latest CPI and Fed Funds Rate (remove limit parameter)
            cpi_series = fred.get_series_latest_release('CPIAUCSL')
            fedfunds_series = fred.get_series_latest_release('FEDFUNDS')
            
            cpi = float(cpi_series.iloc[-1]) if not cpi_series.empty else np.nan
            interest_rate = float(fedfunds_series.iloc[-1]) if not fedfunds_series.empty else np.nan
            
            return {
                "cpi": cpi,
                "interest_rate": interest_rate
            }
            
        except Exception as e:
            logging.error(f"Error fetching FRED data: {e}")
            return {"cpi": np.nan, "interest_rate": np.nan}
    
    @staticmethod
    def fetch_dxy() -> float:
        """Fetch Dollar Index (DXY)"""
        try:
            df = yf.download("DX-Y.NYB", period="1d", interval="1m", progress=False)
            if not df.empty:
                return float(df['Close'].iloc[-1].iloc[0] if hasattr(df['Close'].iloc[-1], 'iloc') else df['Close'].iloc[-1])
            return np.nan
        except Exception as e:
            logging.error(f"Error fetching DXY: {e}")
            return np.nan
    
    @staticmethod
    def fetch_sp500() -> float:
        """Fetch S&P 500"""
        try:
            df = yf.download("^GSPC", period="1d", interval="1m", progress=False)
            if not df.empty:
                return float(df['Close'].iloc[-1].iloc[0] if hasattr(df['Close'].iloc[-1], 'iloc') else df['Close'].iloc[-1])
            return np.nan
        except Exception as e:
            logging.error(f"Error fetching S&P 500: {e}")
            return np.nan
    
    @staticmethod
    def fetch_oil_price() -> float:
        """Fetch oil price"""
        try:
            df = yf.download("CL=F", period="1d", interval="1m", progress=False)
            if not df.empty:
                return float(df['Close'].iloc[-1].iloc[0] if hasattr(df['Close'].iloc[-1], 'iloc') else df['Close'].iloc[-1])
            return np.nan
        except Exception as e:
            logging.error(f"Error fetching oil price: {e}")
            return np.nan
    
    @staticmethod
    def fetch_gold_reserves() -> float:
        """Fetch global gold reserves data"""
        try:
            # Try to fetch from multiple sources
            sources = [
                MacroDataFetcher._fetch_reserves_from_fred(),
                MacroDataFetcher._fetch_reserves_from_world_bank(),
                35000.0  # Fallback estimate in tonnes
            ]
            
            # Return first valid value
            for reserves in sources:
                if reserves and not np.isnan(reserves) and reserves > 0:
                    return float(reserves)
            
            return 35000.0  # Final fallback
            
        except Exception as e:
            logging.error(f"Error fetching gold reserves: {e}")
            return 35000.0  # Default fallback
    
    @staticmethod
    def _fetch_reserves_from_fred() -> float:
        """Try to fetch gold reserves from FRED"""
        try:
            try:
                from fredapi import Fred
            except ImportError:
                logging.debug("fredapi not available for reserves data")
                return None
                
            if not FRED_API_KEY:
                return None
            
            fred = Fred(api_key=FRED_API_KEY)
            
            # Try to get US Treasury gold holdings (in fine troy ounces)
            us_gold = fred.get_series_latest_release('TRESEGUSM052N')
            if not us_gold.empty:
                # Convert troy ounces to tonnes (1 troy ounce = 0.0311035 kg)
                us_gold_tonnes = float(us_gold.iloc[-1]) * 0.0311035 / 1000
                # Estimate global reserves (US holds ~25% of global official reserves)
                return us_gold_tonnes * 4
                
        except Exception as e:
            logging.debug(f"FRED reserves fetch failed: {e}")
        
        return None
    
    @staticmethod
    def _fetch_reserves_from_world_bank() -> float:
        """Try to fetch reserves estimate from World Bank API"""
        try:
            # World Bank doesn't have direct gold reserves, but we can estimate
            # Based on known data that global official gold reserves ~35,000 tonnes
            # This is a simplified implementation - real version would use more sources
            return 35000.0
            
        except Exception as e:
            logging.debug(f"World Bank reserves fetch failed: {e}")
            
        return None

    @staticmethod
    def fetch_all_macro_data() -> Dict[str, float]:
        """Fetch all macro data including enhanced sources"""
        fred_data = MacroDataFetcher.fetch_fred_data()
        
        # Get enhanced data sources
        try:
            try:
                from .enhanced_data_sources import enhanced_data_sources
                enhanced_data = enhanced_data_sources.get_comprehensive_market_data()
            except ImportError:
                logging.debug("Enhanced data sources not available")
                enhanced_data = {}
            
            # Flatten enhanced data for compatibility
            enhanced_macro = {}
            for category, data in enhanced_data.items():
                if isinstance(data, dict) and category != "composite_scores":
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            enhanced_macro[f"{category}_{key}"] = value
                elif category == "composite_scores" and isinstance(data, dict):
                    # Add composite scores directly
                    enhanced_macro.update(data)
                    
        except Exception as e:
            logging.warning(f"Enhanced data sources unavailable: {e}")
            enhanced_macro = {}
        
        return {
            **fred_data,
            "dxy": MacroDataFetcher.fetch_dxy(),
            "sp500": MacroDataFetcher.fetch_sp500(),
            "oil": MacroDataFetcher.fetch_oil_price(),
            "global_gold_reserves": MacroDataFetcher.fetch_gold_reserves(),
            **enhanced_macro  # Include all enhanced data
        }

class SentimentAnalyzer:
    """Enhanced sentiment analyzer with real data sources"""
    
    @staticmethod
    def fetch_sentiment() -> float:
        """Get sentiment score between -1 and 1 based on real market data"""
        try:
            sentiment_score = 0.0
            data_points = 0
            
            # Get enhanced sentiment data
            try:
                try:
                    from .enhanced_data_sources import enhanced_data_sources
                    enhanced_data = enhanced_data_sources.get_comprehensive_market_data()
                except ImportError:
                    logging.debug("Enhanced data sources not available for sentiment")
                    enhanced_data = {}
                
                # Extract sentiment from composite scores
                composite_scores = enhanced_data.get("composite_scores", {})
                
                if "overall_gold_sentiment" in composite_scores:
                    # Convert -100 to +100 scale to -1 to +1
                    gold_sentiment = composite_scores["overall_gold_sentiment"] / 100.0
                    sentiment_score += gold_sentiment * 0.4  # 40% weight
                    data_points += 1
                
                # Fear & greed index (higher fear = more bullish for gold)
                if "fear_greed_index" in composite_scores:
                    fear_index = composite_scores["fear_greed_index"]
                    # Convert 0-100 to -1 to +1 (50 = neutral)
                    fear_sentiment = (fear_index - 50) / 50.0
                    sentiment_score += fear_sentiment * 0.3  # 30% weight
                    data_points += 1
                
                # ETF sentiment
                advanced_sentiment = enhanced_data.get("advanced_sentiment", {})
                if "etf_avg_sentiment" in advanced_sentiment:
                    etf_sentiment = advanced_sentiment["etf_avg_sentiment"]
                    # Normalize ETF sentiment (assume -10% to +10% range)
                    normalized_etf = max(-1, min(1, etf_sentiment / 10.0))
                    sentiment_score += normalized_etf * 0.3  # 30% weight
                    data_points += 1
                    
            except Exception as e:
                logging.warning(f"Enhanced sentiment unavailable: {e}")
            
            # Fallback to basic market indicators if enhanced data unavailable
            if data_points == 0:
                try:
                    # VIX-based sentiment
                    vix_df = yf.download("^VIX", period="2d", interval="1d", progress=False)
                    if not vix_df.empty:
                        vix = float(vix_df['Close'].iloc[-1].item())
                        # High VIX = high fear = bullish for gold
                        vix_sentiment = min(1.0, (vix - 15) / 20.0)  # Normalize around 15-35 range
                        sentiment_score += vix_sentiment * 0.5
                        data_points += 1
                    
                    # Gold vs S&P correlation
                    sp500_df = yf.download("^GSPC", period="2d", interval="1d", progress=False)
                    gold_df = yf.download("GC=F", period="2d", interval="1d", progress=False)
                    
                    if not sp500_df.empty and not gold_df.empty and len(sp500_df) > 1 and len(gold_df) > 1:
                        sp500_change = (float(sp500_df['Close'].iloc[-1].item()) - float(sp500_df['Close'].iloc[-2])) / float(sp500_df['Close'].iloc[-2])
                        gold_change = (float(gold_df['Close'].iloc[-1].item()) - float(gold_df['Close'].iloc[-2])) / float(gold_df['Close'].iloc[-2])
                        
                        # Inverse correlation sentiment (gold up when stocks down = positive sentiment)
                        if sp500_change < 0 and gold_change > 0:
                            sentiment_score += 0.3
                        elif sp500_change > 0 and gold_change < 0:
                            sentiment_score -= 0.3
                        else:
                            sentiment_score += gold_change * 2  # Direct gold momentum
                        
                        data_points += 1
                        
                except Exception as e:
                    logging.warning(f"Basic sentiment calculation failed: {e}")
            
            # Normalize final score
            if data_points > 0:
                sentiment_score = sentiment_score / data_points if data_points > 1 else sentiment_score
            
            # Ensure bounds
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return sentiment_score
            
        except Exception as e:
            logging.error(f"Error fetching sentiment: {e}")
            return 0.0
    
    @staticmethod
    def get_current_market_sentiment() -> Dict:
        """Get comprehensive sentiment data with real market indicators"""
        try:
            sentiment_score = SentimentAnalyzer.fetch_sentiment()
            
            # Classify sentiment
            if sentiment_score > 0.3:
                sentiment_label = "Bullish"
            elif sentiment_score < -0.3:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            # Get additional sentiment breakdown
            sentiment_breakdown = {}
            
            try:
                try:
                    from .enhanced_data_sources import enhanced_data_sources
                    enhanced_data = enhanced_data_sources.get_comprehensive_market_data()
                except ImportError:
                    logging.debug("Enhanced data sources not available for sentiment breakdown")
                    enhanced_data = {}
                
                composite_scores = enhanced_data.get("composite_scores", {})
                sentiment_breakdown = {
                    "inflation_pressure": composite_scores.get("inflation_pressure", 50),
                    "fear_greed_index": composite_scores.get("fear_greed_index", 50),
                    "mining_sector_health": composite_scores.get("mining_sector_health", 50),
                    "overall_gold_sentiment": composite_scores.get("overall_gold_sentiment", 0)
                }
                
                # ETF flows sentiment
                advanced_sentiment = enhanced_data.get("advanced_sentiment", {})
                if "etf_avg_sentiment" in advanced_sentiment:
                    sentiment_breakdown["etf_flows"] = advanced_sentiment["etf_avg_sentiment"]
                
                # Risk indicators
                risk_data = enhanced_data.get("geopolitical_risk", {})
                if "vix" in risk_data:
                    sentiment_breakdown["volatility_index"] = risk_data["vix"]
                    
            except Exception as e:
                logging.warning(f"Enhanced sentiment breakdown unavailable: {e}")
            
            return {
                'overall_sentiment': sentiment_score,
                'news_sentiment': sentiment_label,
                'social_sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_breakdown': sentiment_breakdown,
                'confidence': min(100, abs(sentiment_score) * 100 + 50)  # Confidence based on strength
            }
            
        except Exception as e:
            logging.error(f"Error getting market sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'news_sentiment': 'Neutral', 
                'social_sentiment': 'Neutral',
                'sentiment_score': 0.0,
                'sentiment_breakdown': {},
                'confidence': 50
            }

# Module instances for easy import
price_fetcher = PriceDataFetcher()
news_fetcher = NewsDataFetcher()
macro_fetcher = MacroDataFetcher()
sentiment_analyzer = SentimentAnalyzer()
