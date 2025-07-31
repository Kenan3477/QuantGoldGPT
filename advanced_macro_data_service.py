#!/usr/bin/env python3
"""
GoldGPT Macro Data Service
Economic indicators service with inflation, interest rates, and macro data
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import time
import numpy as np
from data_pipeline_core import DataPipelineCore, DataType, DataSourceTier

logger = logging.getLogger(__name__)

@dataclass
class MacroIndicator:
    """Macro economic indicator data structure"""
    indicator_name: str
    value: float
    previous_value: Optional[float]
    change_percentage: float
    timestamp: datetime
    source: str
    confidence: float
    impact_level: str  # 'low', 'medium', 'high'
    market_impact: float  # -1.0 to 1.0 for gold

@dataclass
class EconomicEvent:
    """Economic event/release data structure"""
    event_name: str
    country: str
    importance: str  # 'low', 'medium', 'high'
    actual_value: Optional[float]
    forecast_value: Optional[float]
    previous_value: Optional[float]
    release_time: datetime
    currency: str
    impact_on_gold: float  # -1.0 to 1.0

@dataclass
class MacroAnalysisResult:
    """Complete macro analysis result"""
    timestamp: datetime
    overall_sentiment: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    key_indicators: List[MacroIndicator]
    upcoming_events: List[EconomicEvent]
    inflation_outlook: str
    interest_rate_outlook: str
    dollar_strength_index: float
    risk_sentiment: str

class AdvancedMacroDataService:
    """Advanced macro economic data service"""
    
    def __init__(self, pipeline: DataPipelineCore, db_path: str = "goldgpt_macro_data.db"):
        self.pipeline = pipeline
        self.db_path = db_path
        
        # Macro indicators configuration with gold impact weights
        self.indicator_configs = {
            'inflation': {
                'cpi': {'weight': 2.5, 'gold_impact': 'positive'},  # Higher inflation = higher gold
                'core_cpi': {'weight': 2.2, 'gold_impact': 'positive'},
                'pce': {'weight': 2.0, 'gold_impact': 'positive'},
                'ppi': {'weight': 1.8, 'gold_impact': 'positive'}
            },
            'interest_rates': {
                'fed_funds_rate': {'weight': 3.0, 'gold_impact': 'negative'},  # Higher rates = lower gold
                '10y_treasury': {'weight': 2.8, 'gold_impact': 'negative'},
                '2y_treasury': {'weight': 2.5, 'gold_impact': 'negative'},
                'real_rates': {'weight': 3.2, 'gold_impact': 'negative'}
            },
            'economic_growth': {
                'gdp': {'weight': 2.0, 'gold_impact': 'negative'},  # Strong growth = less gold demand
                'employment': {'weight': 2.2, 'gold_impact': 'negative'},
                'retail_sales': {'weight': 1.5, 'gold_impact': 'negative'},
                'industrial_production': {'weight': 1.8, 'gold_impact': 'negative'}
            },
            'monetary_policy': {
                'money_supply': {'weight': 2.5, 'gold_impact': 'positive'},  # More money printing = higher gold
                'qe_programs': {'weight': 3.0, 'gold_impact': 'positive'},
                'fed_balance_sheet': {'weight': 2.8, 'gold_impact': 'positive'}
            },
            'currency': {
                'dxy': {'weight': 3.5, 'gold_impact': 'negative'},  # Stronger dollar = lower gold
                'eur_usd': {'weight': 2.0, 'gold_impact': 'positive'},
                'jpy_usd': {'weight': 1.8, 'gold_impact': 'mixed'}
            },
            'geopolitical': {
                'vix': {'weight': 2.2, 'gold_impact': 'positive'},  # Higher volatility = higher gold
                'geopolitical_risk': {'weight': 2.8, 'gold_impact': 'positive'},
                'safe_haven_demand': {'weight': 3.0, 'gold_impact': 'positive'}
            }
        }
        
        # Countries and their impact on gold
        self.country_impact_weights = {
            'US': 3.0,    # Highest impact
            'EU': 2.2,
            'China': 2.5,
            'Japan': 1.8,
            'UK': 1.5,
            'Switzerland': 1.3,
            'Germany': 1.7,
            'India': 2.0  # Large gold consumer
        }
        
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize macro data database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Macro indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                category TEXT NOT NULL,
                value REAL NOT NULL,
                previous_value REAL,
                change_percentage REAL,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                impact_level TEXT NOT NULL,
                market_impact REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(indicator_name, timestamp)
            )
        ''')
        
        # Economic events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL,
                country TEXT NOT NULL,
                importance TEXT NOT NULL,
                actual_value REAL,
                forecast_value REAL,
                previous_value REAL,
                release_time DATETIME NOT NULL,
                currency TEXT NOT NULL,
                impact_on_gold REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Central bank communications
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS central_bank_communications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                central_bank TEXT NOT NULL,
                communication_type TEXT NOT NULL,
                content TEXT NOT NULL,
                hawkish_dovish_score REAL NOT NULL,
                gold_impact_score REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Macro analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                overall_sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                inflation_outlook TEXT NOT NULL,
                interest_rate_outlook TEXT NOT NULL,
                dollar_strength_index REAL NOT NULL,
                risk_sentiment TEXT NOT NULL,
                key_factors TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Macro data database initialized")
    
    async def fetch_inflation_data(self) -> List[MacroIndicator]:
        """Fetch inflation indicators"""
        indicators = []
        
        try:
            # Simulate fetching from FRED API, BLS, etc.
            # In production, integrate with actual economic data APIs
            
            inflation_data = await self.fetch_simulated_inflation_data()
            
            for data in inflation_data:
                indicator = MacroIndicator(
                    indicator_name=data['name'],
                    value=data['value'],
                    previous_value=data.get('previous_value'),
                    change_percentage=data.get('change_pct', 0.0),
                    timestamp=datetime.now(),
                    source=data['source'],
                    confidence=0.95,  # High confidence for official data
                    impact_level=self.determine_impact_level(data['name'], data['value']),
                    market_impact=self.calculate_gold_impact(data['name'], data['value'], 'inflation')
                )
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching inflation data: {e}")
            return []
    
    async def fetch_simulated_inflation_data(self) -> List[Dict]:
        """Simulate fetching inflation data from APIs"""
        # This would integrate with real APIs like FRED, Alpha Vantage, etc.
        return [
            {
                'name': 'cpi_yoy',
                'value': 3.2,  # 3.2% YoY inflation
                'previous_value': 3.1,
                'change_pct': 3.2,
                'source': 'bls'
            },
            {
                'name': 'core_cpi_yoy',
                'value': 2.8,
                'previous_value': 2.9,
                'change_pct': -3.4,
                'source': 'bls'
            },
            {
                'name': 'pce_yoy',
                'value': 2.5,
                'previous_value': 2.4,
                'change_pct': 4.2,
                'source': 'bea'
            }
        ]
    
    async def fetch_interest_rate_data(self) -> List[MacroIndicator]:
        """Fetch interest rate indicators"""
        indicators = []
        
        try:
            rate_data = await self.fetch_simulated_rate_data()
            
            for data in rate_data:
                indicator = MacroIndicator(
                    indicator_name=data['name'],
                    value=data['value'],
                    previous_value=data.get('previous_value'),
                    change_percentage=data.get('change_pct', 0.0),
                    timestamp=datetime.now(),
                    source=data['source'],
                    confidence=0.98,  # Very high confidence for rates
                    impact_level=self.determine_impact_level(data['name'], data['value']),
                    market_impact=self.calculate_gold_impact(data['name'], data['value'], 'interest_rates')
                )
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching interest rate data: {e}")
            return []
    
    async def fetch_simulated_rate_data(self) -> List[Dict]:
        """Simulate fetching interest rate data"""
        return [
            {
                'name': 'fed_funds_rate',
                'value': 5.25,  # 5.25% Fed Funds Rate
                'previous_value': 5.0,
                'change_pct': 5.0,
                'source': 'fred'
            },
            {
                'name': '10y_treasury',
                'value': 4.45,
                'previous_value': 4.35,
                'change_pct': 2.3,
                'source': 'treasury'
            },
            {
                'name': '2y_treasury',
                'value': 4.85,
                'previous_value': 4.80,
                'change_pct': 1.0,
                'source': 'treasury'
            }
        ]
    
    async def fetch_currency_data(self) -> List[MacroIndicator]:
        """Fetch currency indicators"""
        indicators = []
        
        try:
            currency_data = await self.fetch_simulated_currency_data()
            
            for data in currency_data:
                indicator = MacroIndicator(
                    indicator_name=data['name'],
                    value=data['value'],
                    previous_value=data.get('previous_value'),
                    change_percentage=data.get('change_pct', 0.0),
                    timestamp=datetime.now(),
                    source=data['source'],
                    confidence=0.92,
                    impact_level=self.determine_impact_level(data['name'], data['value']),
                    market_impact=self.calculate_gold_impact(data['name'], data['value'], 'currency')
                )
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching currency data: {e}")
            return []
    
    async def fetch_simulated_currency_data(self) -> List[Dict]:
        """Simulate fetching currency data"""
        return [
            {
                'name': 'dxy',
                'value': 104.2,  # Dollar Index
                'previous_value': 103.8,
                'change_pct': 0.4,
                'source': 'ice'
            },
            {
                'name': 'eur_usd',
                'value': 1.0875,
                'previous_value': 1.0890,
                'change_pct': -0.14,
                'source': 'market_data'
            }
        ]
    
    async def fetch_economic_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        events = []
        
        try:
            # Simulate fetching from economic calendar APIs
            event_data = await self.fetch_simulated_economic_events(days_ahead)
            
            for data in event_data:
                event = EconomicEvent(
                    event_name=data['name'],
                    country=data['country'],
                    importance=data['importance'],
                    actual_value=data.get('actual'),
                    forecast_value=data.get('forecast'),
                    previous_value=data.get('previous'),
                    release_time=data['release_time'],
                    currency=data['currency'],
                    impact_on_gold=self.calculate_event_gold_impact(data)
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching economic events: {e}")
            return []
    
    async def fetch_simulated_economic_events(self, days_ahead: int) -> List[Dict]:
        """Simulate fetching economic calendar events"""
        base_time = datetime.now()
        return [
            {
                'name': 'CPI Release',
                'country': 'US',
                'importance': 'high',
                'forecast': 3.1,
                'previous': 3.2,
                'release_time': base_time + timedelta(days=2),
                'currency': 'USD'
            },
            {
                'name': 'FOMC Meeting',
                'country': 'US',
                'importance': 'high',
                'forecast': None,
                'previous': None,
                'release_time': base_time + timedelta(days=5),
                'currency': 'USD'
            },
            {
                'name': 'ECB Rate Decision',
                'country': 'EU',
                'importance': 'high',
                'forecast': 4.0,
                'previous': 4.0,
                'release_time': base_time + timedelta(days=3),
                'currency': 'EUR'
            },
            {
                'name': 'Non-Farm Payrolls',
                'country': 'US',
                'importance': 'high',
                'forecast': 200000,
                'previous': 185000,
                'release_time': base_time + timedelta(days=4),
                'currency': 'USD'
            }
        ]
    
    def determine_impact_level(self, indicator_name: str, value: float) -> str:
        """Determine impact level of indicator based on value"""
        # Simplified impact determination
        for category, indicators in self.indicator_configs.items():
            if indicator_name in indicators or any(ind in indicator_name for ind in indicators.keys()):
                weight = indicators.get(indicator_name, {}).get('weight', 1.0)
                if weight >= 2.5:
                    return 'high'
                elif weight >= 1.5:
                    return 'medium'
                else:
                    return 'low'
        return 'medium'
    
    def calculate_gold_impact(self, indicator_name: str, value: float, category: str) -> float:
        """Calculate impact of indicator on gold prices"""
        try:
            if category not in self.indicator_configs:
                return 0.0
            
            category_config = self.indicator_configs[category]
            
            # Find matching indicator
            for ind_name, config in category_config.items():
                if ind_name in indicator_name or indicator_name in ind_name:
                    impact_direction = config.get('gold_impact', 'neutral')
                    weight = config.get('weight', 1.0)
                    
                    # Calculate normalized impact
                    if category == 'inflation':
                        # Higher inflation typically bullish for gold
                        if value > 3.0:  # Above 3% inflation
                            impact = 0.3 * (weight / 3.0)
                        elif value > 2.0:
                            impact = 0.1 * (weight / 3.0)
                        else:
                            impact = -0.1 * (weight / 3.0)
                    
                    elif category == 'interest_rates':
                        # Higher rates typically bearish for gold
                        if value > 5.0:  # Above 5% rates
                            impact = -0.4 * (weight / 3.0)
                        elif value > 3.0:
                            impact = -0.2 * (weight / 3.0)
                        else:
                            impact = 0.1 * (weight / 3.0)
                    
                    elif category == 'currency':
                        if 'dxy' in indicator_name:
                            # Higher DXY bearish for gold
                            if value > 105:
                                impact = -0.3 * (weight / 3.0)
                            elif value > 100:
                                impact = -0.1 * (weight / 3.0)
                            else:
                                impact = 0.2 * (weight / 3.0)
                        else:
                            impact = 0.0
                    
                    else:
                        # Default calculation
                        impact = 0.1 * (weight / 3.0)
                    
                    # Ensure impact is within bounds
                    return max(-1.0, min(1.0, impact))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating gold impact: {e}")
            return 0.0
    
    def calculate_event_gold_impact(self, event_data: Dict) -> float:
        """Calculate gold impact of economic event"""
        try:
            importance_weights = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
            country_weight = self.country_impact_weights.get(event_data['country'], 1.0)
            importance_weight = importance_weights.get(event_data['importance'], 0.6)
            
            base_impact = importance_weight * (country_weight / 3.0)
            
            # Event-specific impacts
            event_name = event_data['name'].lower()
            
            if 'cpi' in event_name or 'inflation' in event_name:
                # Inflation events - typically bullish for gold if higher than expected
                base_impact *= 0.8  # Positive bias for gold
            elif 'rate' in event_name or 'fomc' in event_name:
                # Rate events - typically bearish for gold if hawkish
                base_impact *= -0.6  # Negative bias for gold
            elif 'employment' in event_name or 'payroll' in event_name:
                # Employment events - strong employment can be bearish for gold
                base_impact *= -0.4
            elif 'gdp' in event_name:
                # GDP events - strong growth can be bearish for gold
                base_impact *= -0.3
            
            return max(-1.0, min(1.0, base_impact))
            
        except Exception as e:
            logger.error(f"Error calculating event impact: {e}")
            return 0.0
    
    async def store_macro_indicators(self, indicators: List[MacroIndicator]):
        """Store macro indicators in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for indicator in indicators:
            try:
                # Determine category
                category = self.get_indicator_category(indicator.indicator_name)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO macro_indicators
                    (indicator_name, category, value, previous_value, change_percentage,
                     timestamp, source, confidence, impact_level, market_impact)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    indicator.indicator_name,
                    category,
                    indicator.value,
                    indicator.previous_value,
                    indicator.change_percentage,
                    indicator.timestamp.isoformat(),
                    indicator.source,
                    indicator.confidence,
                    indicator.impact_level,
                    indicator.market_impact
                ))
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing indicator {indicator.indicator_name}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Stored {stored_count} macro indicators")
    
    def get_indicator_category(self, indicator_name: str) -> str:
        """Get category for indicator"""
        for category, indicators in self.indicator_configs.items():
            if any(ind in indicator_name for ind in indicators.keys()):
                return category
        return 'other'
    
    async def store_economic_events(self, events: List[EconomicEvent]):
        """Store economic events in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for event in events:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO economic_events
                    (event_name, country, importance, actual_value, forecast_value,
                     previous_value, release_time, currency, impact_on_gold)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_name,
                    event.country,
                    event.importance,
                    event.actual_value,
                    event.forecast_value,
                    event.previous_value,
                    event.release_time.isoformat(),
                    event.currency,
                    event.impact_on_gold
                ))
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing event {event.event_name}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Stored {stored_count} economic events")
    
    async def generate_macro_analysis(self) -> MacroAnalysisResult:
        """Generate comprehensive macro analysis"""
        try:
            # Fetch all current indicators
            inflation_indicators = await self.fetch_inflation_data()
            rate_indicators = await self.fetch_interest_rate_data()
            currency_indicators = await self.fetch_currency_data()
            
            all_indicators = inflation_indicators + rate_indicators + currency_indicators
            
            # Store indicators
            if all_indicators:
                await self.store_macro_indicators(all_indicators)
            
            # Fetch upcoming events
            upcoming_events = await self.fetch_economic_events(days_ahead=7)
            if upcoming_events:
                await self.store_economic_events(upcoming_events)
            
            # Calculate overall sentiment
            overall_sentiment, confidence = self.calculate_overall_sentiment(all_indicators)
            
            # Determine outlooks
            inflation_outlook = self.determine_inflation_outlook(inflation_indicators)
            rate_outlook = self.determine_rate_outlook(rate_indicators)
            
            # Calculate dollar strength index
            dollar_strength = self.calculate_dollar_strength_index(currency_indicators)
            
            # Determine risk sentiment
            risk_sentiment = self.determine_risk_sentiment(all_indicators, upcoming_events)
            
            # Create analysis result
            result = MacroAnalysisResult(
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                key_indicators=all_indicators,
                upcoming_events=upcoming_events,
                inflation_outlook=inflation_outlook,
                interest_rate_outlook=rate_outlook,
                dollar_strength_index=dollar_strength,
                risk_sentiment=risk_sentiment
            )
            
            # Store analysis result
            await self.store_macro_analysis(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating macro analysis: {e}")
            return self.generate_neutral_analysis()
    
    def calculate_overall_sentiment(self, indicators: List[MacroIndicator]) -> Tuple[str, float]:
        """Calculate overall macro sentiment for gold"""
        if not indicators:
            return 'neutral', 0.5
        
        total_impact = 0.0
        total_weight = 0.0
        
        for indicator in indicators:
            weight = self.get_indicator_weight(indicator.indicator_name)
            total_impact += indicator.market_impact * weight
            total_weight += weight
        
        if total_weight == 0:
            return 'neutral', 0.5
        
        avg_impact = total_impact / total_weight
        confidence = min(1.0, total_weight / len(indicators))
        
        if avg_impact > 0.2:
            sentiment = 'bullish'
        elif avg_impact < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return sentiment, confidence
    
    def get_indicator_weight(self, indicator_name: str) -> float:
        """Get weight for indicator"""
        for category, indicators in self.indicator_configs.items():
            for ind_name, config in indicators.items():
                if ind_name in indicator_name:
                    return config.get('weight', 1.0)
        return 1.0
    
    def determine_inflation_outlook(self, inflation_indicators: List[MacroIndicator]) -> str:
        """Determine inflation outlook"""
        if not inflation_indicators:
            return 'uncertain'
        
        avg_inflation = np.mean([ind.value for ind in inflation_indicators])
        avg_change = np.mean([ind.change_percentage for ind in inflation_indicators if ind.change_percentage])
        
        if avg_inflation > 3.5 or avg_change > 5:
            return 'rising'
        elif avg_inflation < 2.0 or avg_change < -5:
            return 'falling'
        else:
            return 'stable'
    
    def determine_rate_outlook(self, rate_indicators: List[MacroIndicator]) -> str:
        """Determine interest rate outlook"""
        if not rate_indicators:
            return 'uncertain'
        
        fed_funds = next((ind for ind in rate_indicators if 'fed_funds' in ind.indicator_name), None)
        treasury_10y = next((ind for ind in rate_indicators if '10y' in ind.indicator_name), None)
        
        if fed_funds and fed_funds.value > 5.0:
            if fed_funds.change_percentage > 0:
                return 'rising'
            else:
                return 'peak'
        elif treasury_10y and treasury_10y.change_percentage > 5:
            return 'rising'
        elif treasury_10y and treasury_10y.change_percentage < -5:
            return 'falling'
        else:
            return 'stable'
    
    def calculate_dollar_strength_index(self, currency_indicators: List[MacroIndicator]) -> float:
        """Calculate normalized dollar strength index"""
        dxy_indicator = next((ind for ind in currency_indicators if 'dxy' in ind.indicator_name), None)
        
        if dxy_indicator:
            # Normalize DXY to 0-1 scale (assuming range 90-120)
            normalized = (dxy_indicator.value - 90) / 30
            return max(0.0, min(1.0, normalized))
        
        return 0.5  # Neutral if no data
    
    def determine_risk_sentiment(self, indicators: List[MacroIndicator], events: List[EconomicEvent]) -> str:
        """Determine overall risk sentiment"""
        # Count high-impact events in next 7 days
        high_impact_events = len([e for e in events if e.importance == 'high'])
        
        # Check for high uncertainty indicators
        high_uncertainty = any(ind.impact_level == 'high' and abs(ind.market_impact) > 0.3 for ind in indicators)
        
        if high_impact_events >= 3 or high_uncertainty:
            return 'risk_off'
        elif high_impact_events <= 1:
            return 'risk_on'
        else:
            return 'mixed'
    
    def generate_neutral_analysis(self) -> MacroAnalysisResult:
        """Generate neutral analysis when data is unavailable"""
        return MacroAnalysisResult(
            timestamp=datetime.now(),
            overall_sentiment='neutral',
            confidence=0.3,
            key_indicators=[],
            upcoming_events=[],
            inflation_outlook='uncertain',
            interest_rate_outlook='uncertain',
            dollar_strength_index=0.5,
            risk_sentiment='mixed'
        )
    
    async def store_macro_analysis(self, result: MacroAnalysisResult):
        """Store macro analysis result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        key_factors = [ind.indicator_name for ind in result.key_indicators[:5]]
        
        cursor.execute('''
            INSERT OR REPLACE INTO macro_analysis_results
            (timestamp, overall_sentiment, confidence, inflation_outlook,
             interest_rate_outlook, dollar_strength_index, risk_sentiment, key_factors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp.isoformat(),
            result.overall_sentiment,
            result.confidence,
            result.inflation_outlook,
            result.interest_rate_outlook,
            result.dollar_strength_index,
            result.risk_sentiment,
            json.dumps(key_factors)
        ))
        
        conn.commit()
        conn.close()
        logger.info("✅ Stored macro analysis result")
    
    async def get_macro_summary(self) -> Dict[str, Any]:
        """Get comprehensive macro summary for dashboard"""
        try:
            analysis = await self.generate_macro_analysis()
            
            summary = {
                'overall_sentiment': analysis.overall_sentiment,
                'confidence': round(analysis.confidence, 3),
                'inflation_outlook': analysis.inflation_outlook,
                'interest_rate_outlook': analysis.interest_rate_outlook,
                'dollar_strength_index': round(analysis.dollar_strength_index, 3),
                'risk_sentiment': analysis.risk_sentiment,
                'key_indicators': [
                    {
                        'name': ind.indicator_name,
                        'value': ind.value,
                        'impact': round(ind.market_impact, 3),
                        'level': ind.impact_level
                    } for ind in analysis.key_indicators[:5]
                ],
                'upcoming_events': [
                    {
                        'name': event.event_name,
                        'country': event.country,
                        'importance': event.importance,
                        'release_time': event.release_time.isoformat(),
                        'gold_impact': round(event.impact_on_gold, 3)
                    } for event in analysis.upcoming_events[:5]
                ],
                'last_updated': analysis.timestamp.isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating macro summary: {e}")
            return {'error': str(e)}

# Global instance
macro_service = AdvancedMacroDataService(DataPipelineCore())

if __name__ == "__main__":
    async def test_macro_service():
        print("🧪 Testing Advanced Macro Data Service...")
        
        # Test macro analysis
        analysis = await macro_service.generate_macro_analysis()
        print(f"📊 Macro Analysis:")
        print(f"   Overall Sentiment: {analysis.overall_sentiment}")
        print(f"   Confidence: {analysis.confidence:.3f}")
        print(f"   Inflation Outlook: {analysis.inflation_outlook}")
        print(f"   Rate Outlook: {analysis.interest_rate_outlook}")
        print(f"   Dollar Strength: {analysis.dollar_strength_index:.3f}")
        print(f"   Risk Sentiment: {analysis.risk_sentiment}")
        print(f"   Key Indicators: {len(analysis.key_indicators)}")
        print(f"   Upcoming Events: {len(analysis.upcoming_events)}")
        
        # Test macro summary
        summary = await macro_service.get_macro_summary()
        print(f"\n📈 Macro Summary: {summary}")
    
    asyncio.run(test_macro_service())
