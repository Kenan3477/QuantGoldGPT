#!/usr/bin/env python3
"""
GoldGPT Macro Data Service
Economic indicators service with inflation, interest rates, and economic data
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import numpy as np
from data_pipeline_core import data_pipeline, DataType

logger = logging.getLogger(__name__)

@dataclass
class EconomicIndicator:
    """Economic indicator data structure"""
    name: str
    value: float
    previous_value: float
    change: float
    change_percent: float
    unit: str
    release_date: datetime
    frequency: str
    source: str
    impact_level: str  # HIGH, MEDIUM, LOW

@dataclass
class MacroAnalysis:
    """Macro economic analysis result"""
    overall_sentiment: str
    risk_level: str
    gold_impact: str
    key_indicators: List[EconomicIndicator]
    analysis_summary: str
    confidence: float

class MacroDataService:
    """Advanced macro economic data service"""
    
    def __init__(self, db_path: str = "goldgpt_macro.db"):
        self.db_path = db_path
        self.macro_cache = {}
        self.indicator_definitions = self.setup_indicator_definitions()
        self.data_sources = self.setup_data_sources()
        self.initialize_database()
    
    def setup_indicator_definitions(self) -> Dict:
        """Define economic indicators and their gold market impact"""
        return {
            'inflation_rate': {
                'name': 'US Inflation Rate (CPI)',
                'unit': '%',
                'frequency': 'monthly',
                'impact_level': 'HIGH',
                'gold_correlation': 'positive',  # Higher inflation = bullish for gold
                'description': 'Consumer Price Index year-over-year change'
            },
            'fed_funds_rate': {
                'name': 'Federal Funds Rate',
                'unit': '%',
                'frequency': 'meeting',
                'impact_level': 'HIGH',
                'gold_correlation': 'negative',  # Higher rates = bearish for gold
                'description': 'Federal Reserve overnight lending rate'
            },
            'gdp_growth': {
                'name': 'GDP Growth Rate',
                'unit': '%',
                'frequency': 'quarterly',
                'impact_level': 'MEDIUM',
                'gold_correlation': 'negative',  # Strong growth = bearish for gold
                'description': 'Gross Domestic Product quarterly change'
            },
            'unemployment_rate': {
                'name': 'Unemployment Rate',
                'unit': '%',
                'frequency': 'monthly',
                'impact_level': 'MEDIUM',
                'gold_correlation': 'positive',  # High unemployment = bullish for gold
                'description': 'Percentage of unemployed labor force'
            },
            'dxy_index': {
                'name': 'US Dollar Index (DXY)',
                'unit': 'index',
                'frequency': 'continuous',
                'impact_level': 'HIGH',
                'gold_correlation': 'negative',  # Strong dollar = bearish for gold
                'description': 'Trade-weighted USD against major currencies'
            },
            'bond_yields_10y': {
                'name': '10-Year Treasury Yield',
                'unit': '%',
                'frequency': 'continuous',
                'impact_level': 'HIGH',
                'gold_correlation': 'negative',  # Higher yields = bearish for gold
                'description': '10-year US Treasury bond yield'
            },
            'vix_index': {
                'name': 'VIX Volatility Index',
                'unit': 'index',
                'frequency': 'continuous',
                'impact_level': 'MEDIUM',
                'gold_correlation': 'positive',  # High volatility = bullish for gold
                'description': 'Market volatility and fear gauge'
            },
            'ppi': {
                'name': 'Producer Price Index',
                'unit': '%',
                'frequency': 'monthly',
                'impact_level': 'MEDIUM',
                'gold_correlation': 'positive',  # Producer inflation = bullish for gold
                'description': 'Producer-level inflation measurement'
            },
            'retail_sales': {
                'name': 'Retail Sales',
                'unit': '%',
                'frequency': 'monthly',
                'impact_level': 'MEDIUM',
                'gold_correlation': 'negative',  # Strong consumer = bearish for gold
                'description': 'Consumer spending on retail goods'
            },
            'nonfarm_payrolls': {
                'name': 'Non-Farm Payrolls',
                'unit': 'thousands',
                'frequency': 'monthly',
                'impact_level': 'HIGH',
                'gold_correlation': 'negative',  # Job growth = bearish for gold
                'description': 'Monthly job creation in non-agricultural sectors'
            }
        }
    
    def setup_data_sources(self) -> List[Dict]:
        """Configure macro data sources"""
        return [
            {
                'name': 'fred_api',
                'url': 'https://api.stlouisfed.org/fred/series/observations',
                'api_key': 'YOUR_FRED_API_KEY',
                'rate_limit': 120,
                'tier': 'primary'
            },
            {
                'name': 'alpha_vantage_macro',
                'url': 'https://www.alphavantage.co/query',
                'api_key': 'YOUR_ALPHA_VANTAGE_KEY',
                'rate_limit': 25,
                'tier': 'primary'
            },
            {
                'name': 'yahoo_finance_macro',
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
                'rate_limit': 60,
                'tier': 'secondary'
            }
        ]
    
    def initialize_database(self):
        """Initialize macro economic database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Economic indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                previous_value REAL,
                change_value REAL,
                change_percent REAL,
                unit TEXT,
                release_date DATETIME,
                data_date DATETIME,
                frequency TEXT,
                source TEXT,
                impact_level TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX(indicator_name, data_date)
            )
        ''')
        
        # Macro analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date DATETIME NOT NULL,
                overall_sentiment TEXT,
                risk_level TEXT,
                gold_impact TEXT,
                analysis_summary TEXT,
                confidence REAL,
                indicators_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Central bank policy table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS central_bank_policy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bank_name TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                announcement_date DATETIME,
                effective_date DATETIME,
                rate_change REAL,
                new_rate REAL,
                policy_stance TEXT,
                statement_summary TEXT,
                gold_impact_assessment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Economic calendar table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL,
                country TEXT,
                release_date DATETIME,
                importance TEXT,
                actual_value REAL,
                forecast_value REAL,
                previous_value REAL,
                currency_impact TEXT,
                gold_impact_expected TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Macro economic database initialized")
    
    async def fetch_fred_data(self, series_id: str, limit: int = 100) -> List[Dict]:
        """Fetch data from FRED API"""
        try:
            params = {
                'series_id': series_id,
                'api_key': 'demo',  # Replace with actual FRED API key
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.stlouisfed.org/fred/series/observations',
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get('observations', [])
                        
                        return [{
                            'date': obs['date'],
                            'value': float(obs['value']) if obs['value'] != '.' else None,
                            'source': 'fred'
                        } for obs in observations if obs['value'] != '.']
        
        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
        
        return []
    
    async def fetch_alpha_vantage_macro(self, function: str) -> List[Dict]:
        """Fetch macro data from Alpha Vantage"""
        try:
            params = {
                'function': function,
                'apikey': 'demo'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://www.alphavantage.co/query',
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle different response formats
                        if 'data' in data:
                            return [{
                                'date': item.get('date'),
                                'value': float(item.get('value', 0)),
                                'source': 'alpha_vantage'
                            } for item in data['data']]
        
        except Exception as e:
            logger.error(f"Alpha Vantage macro error for {function}: {e}")
        
        return []
    
    async def fetch_simulated_macro_data(self) -> Dict[str, List[Dict]]:
        """Generate simulated macro economic data for testing"""
        current_date = datetime.now()
        
        # Simulate realistic economic data
        simulated_data = {
            'inflation_rate': [{
                'date': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'value': 3.2,
                'previous_value': 3.1,
                'source': 'simulated'
            }],
            'fed_funds_rate': [{
                'date': (current_date - timedelta(days=45)).strftime('%Y-%m-%d'),
                'value': 5.25,
                'previous_value': 5.00,
                'source': 'simulated'
            }],
            'gdp_growth': [{
                'date': (current_date - timedelta(days=90)).strftime('%Y-%m-%d'),
                'value': 2.1,
                'previous_value': 1.8,
                'source': 'simulated'
            }],
            'unemployment_rate': [{
                'date': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'value': 3.7,
                'previous_value': 3.8,
                'source': 'simulated'
            }],
            'dxy_index': [{
                'date': current_date.strftime('%Y-%m-%d'),
                'value': 104.2,
                'previous_value': 103.8,
                'source': 'simulated'
            }],
            'bond_yields_10y': [{
                'date': current_date.strftime('%Y-%m-%d'),
                'value': 4.35,
                'previous_value': 4.28,
                'source': 'simulated'
            }],
            'vix_index': [{
                'date': current_date.strftime('%Y-%m-%d'),
                'value': 18.5,
                'previous_value': 16.2,
                'source': 'simulated'
            }],
            'ppi': [{
                'date': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'value': 2.8,
                'previous_value': 2.9,
                'source': 'simulated'
            }],
            'retail_sales': [{
                'date': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'value': 0.3,
                'previous_value': 0.1,
                'source': 'simulated'
            }],
            'nonfarm_payrolls': [{
                'date': (current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'value': 187000,
                'previous_value': 209000,
                'source': 'simulated'
            }]
        }
        
        return simulated_data
    
    def create_economic_indicator(self, name: str, data_point: Dict, indicator_def: Dict) -> EconomicIndicator:
        """Create EconomicIndicator object from data"""
        current_value = data_point.get('value', 0)
        previous_value = data_point.get('previous_value', current_value)
        
        change = current_value - previous_value
        change_percent = (change / previous_value * 100) if previous_value != 0 else 0
        
        try:
            release_date = datetime.strptime(data_point.get('date'), '%Y-%m-%d')
        except:
            release_date = datetime.now()
        
        return EconomicIndicator(
            name=indicator_def['name'],
            value=current_value,
            previous_value=previous_value,
            change=change,
            change_percent=change_percent,
            unit=indicator_def['unit'],
            release_date=release_date,
            frequency=indicator_def['frequency'],
            source=data_point.get('source', 'unknown'),
            impact_level=indicator_def['impact_level']
        )
    
    def calculate_gold_impact_score(self, indicators: List[EconomicIndicator]) -> float:
        """Calculate overall gold impact score from economic indicators"""
        total_score = 0
        total_weight = 0
        
        for indicator in indicators:
            indicator_name = next(
                (key for key, value in self.indicator_definitions.items() 
                 if value['name'] == indicator.name), 
                None
            )
            
            if not indicator_name:
                continue
            
            definition = self.indicator_definitions[indicator_name]
            
            # Weight by impact level
            weight = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[definition['impact_level']]
            
            # Calculate directional impact
            correlation = definition['gold_correlation']
            change_magnitude = abs(indicator.change_percent)
            
            if correlation == 'positive':
                # Positive correlation: increase = bullish for gold
                impact = indicator.change_percent * weight
            else:
                # Negative correlation: increase = bearish for gold
                impact = -indicator.change_percent * weight
            
            total_score += impact
            total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
            return max(-100, min(100, normalized_score))  # Clamp between -100 and 100
        
        return 0
    
    def determine_risk_level(self, indicators: List[EconomicIndicator]) -> str:
        """Determine overall economic risk level"""
        risk_factors = 0
        total_indicators = len(indicators)
        
        for indicator in indicators:
            # High volatility or extreme changes indicate risk
            if abs(indicator.change_percent) > 10:  # Large changes
                risk_factors += 2
            elif abs(indicator.change_percent) > 5:  # Moderate changes
                risk_factors += 1
            
            # Specific risk indicators
            if 'VIX' in indicator.name and indicator.value > 25:
                risk_factors += 2
            elif 'Unemployment' in indicator.name and indicator.value > 5:
                risk_factors += 1
            elif 'Inflation' in indicator.name and indicator.value > 4:
                risk_factors += 1
        
        risk_ratio = risk_factors / (total_indicators * 2) if total_indicators > 0 else 0
        
        if risk_ratio > 0.6:
            return "HIGH"
        elif risk_ratio > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_analysis_summary(self, indicators: List[EconomicIndicator], 
                                 gold_impact_score: float, risk_level: str) -> str:
        """Generate human-readable analysis summary"""
        summary_parts = []
        
        # Overall economic sentiment
        if gold_impact_score > 20:
            summary_parts.append("Strongly bullish economic environment for gold")
        elif gold_impact_score > 5:
            summary_parts.append("Moderately bullish economic conditions for gold")
        elif gold_impact_score < -20:
            summary_parts.append("Strongly bearish economic environment for gold")
        elif gold_impact_score < -5:
            summary_parts.append("Moderately bearish economic conditions for gold")
        else:
            summary_parts.append("Neutral economic environment for gold")
        
        # Key driver analysis
        high_impact_indicators = [ind for ind in indicators if ind.impact_level == 'HIGH']
        
        if high_impact_indicators:
            significant_changes = [
                ind for ind in high_impact_indicators 
                if abs(ind.change_percent) > 2
            ]
            
            if significant_changes:
                summary_parts.append(
                    f"Key drivers: {', '.join([ind.name for ind in significant_changes[:3]])}"
                )
        
        # Risk assessment
        summary_parts.append(f"Economic risk level: {risk_level}")
        
        # Specific concerns or opportunities
        inflation_indicators = [ind for ind in indicators if 'Inflation' in ind.name or 'CPI' in ind.name]
        if inflation_indicators and inflation_indicators[0].value > 3:
            summary_parts.append("Elevated inflation supports gold as hedge")
        
        rate_indicators = [ind for ind in indicators if 'Rate' in ind.name or 'Fed' in ind.name]
        if rate_indicators and rate_indicators[0].change > 0:
            summary_parts.append("Rising interest rates create headwind for gold")
        
        return ". ".join(summary_parts) + "."
    
    async def get_macro_analysis(self) -> MacroAnalysis:
        """Get comprehensive macro economic analysis"""
        logger.info("🔍 Fetching macro economic analysis...")
        
        try:
            # Try to fetch real data (would use actual APIs in production)
            # For now, use simulated data
            macro_data = await self.fetch_simulated_macro_data()
            
            indicators = []
            
            # Process each indicator
            for indicator_name, data_points in macro_data.items():
                if indicator_name in self.indicator_definitions and data_points:
                    indicator_def = self.indicator_definitions[indicator_name]
                    data_point = data_points[0]  # Most recent
                    
                    indicator = self.create_economic_indicator(
                        indicator_name, data_point, indicator_def
                    )
                    indicators.append(indicator)
            
            # Store indicators in database
            self.store_indicators(indicators)
            
            # Calculate analysis
            gold_impact_score = self.calculate_gold_impact_score(indicators)
            risk_level = self.determine_risk_level(indicators)
            
            # Determine overall sentiment and gold impact
            if gold_impact_score > 15:
                overall_sentiment = "BULLISH"
                gold_impact = "POSITIVE"
            elif gold_impact_score < -15:
                overall_sentiment = "BEARISH"
                gold_impact = "NEGATIVE"
            else:
                overall_sentiment = "NEUTRAL"
                gold_impact = "MIXED"
            
            # Generate summary
            analysis_summary = self.generate_analysis_summary(
                indicators, gold_impact_score, risk_level
            )
            
            # Calculate confidence based on data quality and recency
            confidence = self.calculate_analysis_confidence(indicators)
            
            analysis = MacroAnalysis(
                overall_sentiment=overall_sentiment,
                risk_level=risk_level,
                gold_impact=gold_impact,
                key_indicators=indicators,
                analysis_summary=analysis_summary,
                confidence=confidence
            )
            
            # Store analysis
            self.store_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            
            # Return default analysis on error
            return MacroAnalysis(
                overall_sentiment="NEUTRAL",
                risk_level="MEDIUM",
                gold_impact="MIXED",
                key_indicators=[],
                analysis_summary="Unable to complete macro analysis due to data limitations.",
                confidence=0.1
            )
    
    def calculate_analysis_confidence(self, indicators: List[EconomicIndicator]) -> float:
        """Calculate confidence level of the analysis"""
        if not indicators:
            return 0.1
        
        # Base confidence on number and quality of indicators
        indicator_count_factor = min(1.0, len(indicators) / 8)  # Full confidence at 8+ indicators
        
        # Check data recency
        now = datetime.now()
        recency_scores = []
        
        for indicator in indicators:
            days_old = (now - indicator.release_date).days
            
            if indicator.frequency == 'continuous':
                max_age = 1  # Should be very recent
            elif indicator.frequency == 'daily':
                max_age = 7
            elif indicator.frequency == 'monthly':
                max_age = 45
            else:
                max_age = 120
            
            recency_score = max(0, 1 - (days_old / max_age))
            recency_scores.append(recency_score)
        
        recency_factor = np.mean(recency_scores) if recency_scores else 0.5
        
        # Check for high-impact indicators
        high_impact_count = sum(1 for ind in indicators if ind.impact_level == 'HIGH')
        impact_factor = min(1.0, high_impact_count / 4)  # Full confidence at 4+ high-impact indicators
        
        # Combine factors
        overall_confidence = (
            indicator_count_factor * 0.4 +
            recency_factor * 0.4 +
            impact_factor * 0.2
        )
        
        return round(overall_confidence, 3)
    
    def store_indicators(self, indicators: List[EconomicIndicator]):
        """Store economic indicators in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for indicator in indicators:
            cursor.execute('''
                INSERT OR REPLACE INTO economic_indicators 
                (indicator_name, value, previous_value, change_value, change_percent,
                 unit, release_date, data_date, frequency, source, impact_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                indicator.name,
                indicator.value,
                indicator.previous_value,
                indicator.change,
                indicator.change_percent,
                indicator.unit,
                indicator.release_date.isoformat(),
                indicator.release_date.isoformat(),
                indicator.frequency,
                indicator.source,
                indicator.impact_level
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"📝 Stored {len(indicators)} economic indicators")
    
    def store_analysis(self, analysis: MacroAnalysis):
        """Store macro analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO macro_analysis 
            (analysis_date, overall_sentiment, risk_level, gold_impact,
             analysis_summary, confidence, indicators_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            analysis.overall_sentiment,
            analysis.risk_level,
            analysis.gold_impact,
            analysis.analysis_summary,
            analysis.confidence,
            len(analysis.key_indicators)
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_indicators(self, limit: int = 10) -> List[Dict]:
        """Get latest economic indicators from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT indicator_name, value, previous_value, change_percent,
                   release_date, impact_level, unit
            FROM economic_indicators 
            ORDER BY release_date DESC, created_at DESC
            LIMIT ?
        ''', (limit,))
        
        indicators = []
        for row in cursor.fetchall():
            indicators.append({
                'name': row[0],
                'value': row[1],
                'previous_value': row[2],
                'change_percent': row[3],
                'release_date': row[4],
                'impact_level': row[5],
                'unit': row[6]
            })
        
        conn.close()
        return indicators
    
    async def get_economic_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming economic events (simulated for demo)"""
        upcoming_events = []
        
        base_date = datetime.now()
        
        # Simulate upcoming economic releases
        events = [
            {'name': 'CPI Release', 'importance': 'HIGH', 'days': 2},
            {'name': 'Fed Meeting Minutes', 'importance': 'HIGH', 'days': 5},
            {'name': 'Non-Farm Payrolls', 'importance': 'HIGH', 'days': 3},
            {'name': 'GDP Preliminary', 'importance': 'MEDIUM', 'days': 7},
            {'name': 'Retail Sales', 'importance': 'MEDIUM', 'days': 4},
            {'name': 'PPI Release', 'importance': 'MEDIUM', 'days': 6},
            {'name': 'FOMC Statement', 'importance': 'HIGH', 'days': 1}
        ]
        
        for event in events:
            if event['days'] <= days_ahead:
                event_date = base_date + timedelta(days=event['days'])
                upcoming_events.append({
                    'event_name': event['name'],
                    'release_date': event_date.isoformat(),
                    'importance': event['importance'],
                    'gold_impact_expected': 'HIGH' if event['importance'] == 'HIGH' else 'MEDIUM',
                    'country': 'US'
                })
        
        return upcoming_events

# Global instance
macro_service = MacroDataService()

if __name__ == "__main__":
    # Test the macro service
    async def test_macro_service():
        print("🧪 Testing Macro Data Service...")
        
        # Test macro analysis
        analysis = await macro_service.get_macro_analysis()
        
        print(f"📊 Overall Sentiment: {analysis.overall_sentiment}")
        print(f"⚠️ Risk Level: {analysis.risk_level}")
        print(f"🥇 Gold Impact: {analysis.gold_impact}")
        print(f"🎯 Confidence: {analysis.confidence:.2f}")
        print(f"📈 Indicators: {len(analysis.key_indicators)}")
        print(f"💡 Summary: {analysis.analysis_summary}")
        
        if analysis.key_indicators:
            print("\n🔝 Key Economic Indicators:")
            for indicator in analysis.key_indicators[:5]:
                print(f"  • {indicator.name}: {indicator.value}{indicator.unit} "
                      f"({indicator.change_percent:+.1f}%) [{indicator.impact_level}]")
        
        # Test economic calendar
        calendar = await macro_service.get_economic_calendar(7)
        print(f"\n📅 Upcoming Events: {len(calendar)}")
        for event in calendar[:3]:
            print(f"  • {event['event_name']} - {event['importance']} impact")
    
    asyncio.run(test_macro_service())
