#!/usr/bin/env python3
"""
Enhanced ML Data Provider for GoldGPT
Provides high-quality, multi-source data to ML engines using the data pipeline core
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import data pipeline
from data_pipeline_core import data_pipeline, DataType
from data_pipeline_web_integration import web_data_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLDataPackage:
    """Comprehensive data package for ML training and prediction"""
    symbol: str
    timestamp: datetime
    price_data: Dict[str, Any]
    technical_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    macro_data: Dict[str, Any]
    news_data: Dict[str, Any]
    quality_metrics: Dict[str, float]
    confidence_score: float
    feature_vector: np.ndarray
    metadata: Dict[str, Any]

class EnhancedMLDataProvider:
    """Enhanced data provider for ML engines using multi-source pipeline"""
    
    def __init__(self):
        self.pipeline = data_pipeline
        self.cache_duration = 300  # 5 minutes
        self.data_cache = {}
        
        # Feature extraction configuration
        self.feature_config = {
            'price_features': ['current_price', 'price_change', 'volatility'],
            'technical_features': ['rsi', 'macd', 'bollinger_bands', 'moving_averages'],
            'sentiment_features': ['news_sentiment', 'market_sentiment', 'social_sentiment'],
            'macro_features': ['inflation', 'interest_rates', 'usd_index', 'vix']
        }
    
    async def get_comprehensive_data(self, symbol: str, lookback_hours: int = 24) -> Optional[MLDataPackage]:
        """Get comprehensive data package for ML processing"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_hours}_{int(datetime.now().timestamp() / 300)}"
            if cache_key in self.data_cache:
                logger.info(f"📋 Cache hit for ML data: {symbol}")
                return self.data_cache[cache_key]
            
            # Fetch data from all sources concurrently
            price_task = self.pipeline.get_unified_data(symbol, DataType.PRICE)
            technical_task = self.pipeline.get_unified_data(symbol, DataType.TECHNICAL)
            sentiment_task = self.pipeline.get_unified_data(symbol, DataType.SENTIMENT)
            macro_task = self.pipeline.get_unified_data(symbol, DataType.MACRO)
            news_task = self.pipeline.get_unified_data(symbol, DataType.NEWS)
            
            # Execute all tasks concurrently
            price_data, technical_data, sentiment_data, macro_data, news_data = await asyncio.gather(
                price_task, technical_task, sentiment_task, macro_task, news_task,
                return_exceptions=True
            )
            
            # Handle exceptions and missing data
            price_data = price_data if not isinstance(price_data, Exception) else None
            technical_data = technical_data if not isinstance(technical_data, Exception) else None
            sentiment_data = sentiment_data if not isinstance(sentiment_data, Exception) else None
            macro_data = macro_data if not isinstance(macro_data, Exception) else None
            news_data = news_data if not isinstance(news_data, Exception) else None
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(
                price_data, technical_data, sentiment_data, macro_data, news_data
            )
            
            # Generate feature vector
            feature_vector = self.generate_feature_vector(
                price_data, technical_data, sentiment_data, macro_data, news_data
            )
            
            # Calculate overall confidence score
            confidence_score = self.calculate_confidence_score(quality_metrics)
            
            # Create comprehensive data package
            ml_package = MLDataPackage(
                symbol=symbol,
                timestamp=datetime.now(),
                price_data=price_data or {},
                technical_data=technical_data or {},
                sentiment_data=sentiment_data or {},
                macro_data=macro_data or {},
                news_data=news_data or {},
                quality_metrics=quality_metrics,
                confidence_score=confidence_score,
                feature_vector=feature_vector,
                metadata={
                    'lookback_hours': lookback_hours,
                    'data_sources': self.get_active_sources(),
                    'generation_time': datetime.now().isoformat(),
                    'pipeline_status': await self.get_pipeline_health()
                }
            )
            
            # Cache the result
            self.data_cache[cache_key] = ml_package
            
            logger.info(f"✅ Generated ML data package for {symbol} (confidence: {confidence_score:.2f})")
            return ml_package
            
        except Exception as e:
            logger.error(f"❌ Error generating ML data package: {e}")
            return None
    
    def calculate_quality_metrics(self, price_data, technical_data, sentiment_data, macro_data, news_data) -> Dict[str, float]:
        """Calculate data quality metrics for each data source"""
        metrics = {}
        
        # Price data quality
        if price_data:
            metrics['price_quality'] = price_data.get('confidence', 0)
            metrics['price_freshness'] = self.calculate_freshness(price_data.get('timestamp'))
        else:
            metrics['price_quality'] = 0
            metrics['price_freshness'] = 0
        
        # Technical data quality
        if technical_data:
            metrics['technical_quality'] = technical_data.get('confidence', 0)
            metrics['technical_completeness'] = self.calculate_completeness(technical_data, 'technical')
        else:
            metrics['technical_quality'] = 0
            metrics['technical_completeness'] = 0
        
        # Sentiment data quality
        if sentiment_data:
            metrics['sentiment_quality'] = sentiment_data.get('confidence', 0)
            metrics['sentiment_reliability'] = self.calculate_sentiment_reliability(sentiment_data)
        else:
            metrics['sentiment_quality'] = 0
            metrics['sentiment_reliability'] = 0
        
        # Macro data quality
        if macro_data:
            metrics['macro_quality'] = macro_data.get('confidence', 0)
            metrics['macro_relevance'] = self.calculate_macro_relevance(macro_data)
        else:
            metrics['macro_quality'] = 0
            metrics['macro_relevance'] = 0
        
        # News data quality
        if news_data:
            metrics['news_quality'] = news_data.get('confidence', 0)
            metrics['news_recency'] = self.calculate_freshness(news_data.get('timestamp'))
        else:
            metrics['news_quality'] = 0
            metrics['news_recency'] = 0
        
        return metrics
    
    def calculate_freshness(self, timestamp_str: Optional[str]) -> float:
        """Calculate data freshness score (1.0 = very fresh, 0.0 = stale)"""
        if not timestamp_str:
            return 0.0
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age = datetime.now() - timestamp.replace(tzinfo=None)
            
            # Exponential decay: fresh data gets high scores
            max_age_hours = 24
            freshness = np.exp(-age.total_seconds() / (3600 * max_age_hours))
            return min(1.0, freshness)
            
        except Exception:
            return 0.0
    
    def calculate_completeness(self, data: Dict[str, Any], data_type: str) -> float:
        """Calculate data completeness score"""
        if not data:
            return 0.0
        
        expected_fields = self.feature_config.get(f"{data_type}_features", [])
        if not expected_fields:
            return 0.5  # Unknown data type
        
        present_fields = sum(1 for field in expected_fields if field in data)
        return present_fields / len(expected_fields)
    
    def calculate_sentiment_reliability(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate sentiment data reliability based on multiple factors"""
        if not sentiment_data:
            return 0.0
        
        reliability = 0.0
        
        # Check for multiple sentiment sources
        sentiment_sources = sentiment_data.get('sources', [])
        if len(sentiment_sources) > 1:
            reliability += 0.3
        
        # Check sentiment confidence
        confidence = sentiment_data.get('confidence', 0)
        reliability += confidence * 0.4
        
        # Check for sentiment consistency
        sentiment_scores = sentiment_data.get('scores', {})
        if sentiment_scores and len(sentiment_scores) > 1:
            variance = np.var(list(sentiment_scores.values()))
            consistency = 1.0 - min(1.0, variance)  # Lower variance = higher consistency
            reliability += consistency * 0.3
        
        return min(1.0, reliability)
    
    def calculate_macro_relevance(self, macro_data: Dict[str, Any]) -> float:
        """Calculate macro data relevance to gold trading"""
        if not macro_data:
            return 0.0
        
        # Gold-relevant macro indicators and their weights
        relevance_weights = {
            'inflation': 0.25,
            'interest_rates': 0.25,
            'usd_index': 0.20,
            'vix': 0.15,
            'gdp_growth': 0.10,
            'employment': 0.05
        }
        
        relevance_score = 0.0
        for indicator, weight in relevance_weights.items():
            if indicator in macro_data:
                relevance_score += weight
        
        return relevance_score
    
    def generate_feature_vector(self, price_data, technical_data, sentiment_data, macro_data, news_data) -> np.ndarray:
        """Generate comprehensive feature vector for ML models"""
        features = []
        
        # Price features
        if price_data:
            features.extend([
                price_data.get('price', 0),
                price_data.get('price_change', 0),
                price_data.get('volatility', 0),
                price_data.get('volume', 0),
                price_data.get('confidence', 0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Technical features
        if technical_data:
            features.extend([
                technical_data.get('rsi', 50),
                technical_data.get('macd', 0),
                technical_data.get('macd_signal', 0),
                technical_data.get('bollinger_upper', 0),
                technical_data.get('bollinger_lower', 0),
                technical_data.get('ma_20', 0),
                technical_data.get('ma_50', 0),
                technical_data.get('ma_200', 0)
            ])
        else:
            features.extend([50, 0, 0, 0, 0, 0, 0, 0])
        
        # Sentiment features
        if sentiment_data:
            features.extend([
                sentiment_data.get('overall_sentiment', 0),
                sentiment_data.get('news_sentiment', 0),
                sentiment_data.get('social_sentiment', 0),
                sentiment_data.get('sentiment_strength', 0),
                sentiment_data.get('confidence', 0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Macro features
        if macro_data:
            features.extend([
                macro_data.get('inflation_rate', 0),
                macro_data.get('interest_rate', 0),
                macro_data.get('usd_index', 100),
                macro_data.get('vix', 20),
                macro_data.get('gdp_growth', 0)
            ])
        else:
            features.extend([0, 0, 100, 20, 0])
        
        # News features
        if news_data:
            features.extend([
                news_data.get('news_count', 0),
                news_data.get('positive_news', 0),
                news_data.get('negative_news', 0),
                news_data.get('news_sentiment_score', 0),
                news_data.get('market_impact_score', 0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24.0,  # Hour of day normalized
            now.weekday() / 6.0,  # Day of week normalized
            (now.day - 1) / 30.0,  # Day of month normalized
            (now.month - 1) / 11.0  # Month normalized
        ])
        
        return np.array(features, dtype=np.float32)
    
    def calculate_confidence_score(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall confidence score for the data package"""
        if not quality_metrics:
            return 0.0
        
        # Weighted confidence calculation
        weights = {
            'price_quality': 0.30,
            'price_freshness': 0.15,
            'technical_quality': 0.20,
            'technical_completeness': 0.10,
            'sentiment_quality': 0.10,
            'sentiment_reliability': 0.05,
            'macro_quality': 0.05,
            'macro_relevance': 0.05
        }
        
        confidence = 0.0
        for metric, weight in weights.items():
            confidence += quality_metrics.get(metric, 0) * weight
        
        return min(1.0, confidence)
    
    def get_active_sources(self) -> List[str]:
        """Get list of currently active data sources"""
        try:
            status = self.pipeline.get_source_status()
            active_sources = [
                source for source, info in status.items() 
                if info.get('reliability_score', 0) > 0.5
            ]
            return active_sources
        except Exception:
            return []
    
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status"""
        try:
            health = await self.pipeline.health_check()
            healthy_count = sum(1 for h in health.values() if h.get('status') == 'healthy')
            total_count = len(health)
            
            return {
                'healthy_sources': healthy_count,
                'total_sources': total_count,
                'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0,
                'last_check': datetime.now().isoformat()
            }
        except Exception:
            return {'healthy_sources': 0, 'total_sources': 0, 'health_percentage': 0}
    
    async def get_training_dataset(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Generate training dataset for ML models"""
        try:
            # This would typically fetch historical data
            # For now, we'll generate recent data points
            data_points = []
            
            for i in range(days * 24):  # Hourly data points
                timestamp = datetime.now() - timedelta(hours=i)
                
                # Simulate getting historical data
                # In production, this would fetch actual historical data
                data_package = await self.get_comprehensive_data(symbol)
                
                if data_package:
                    data_points.append({
                        'timestamp': timestamp.isoformat(),
                        'features': data_package.feature_vector.tolist(),
                        'confidence': data_package.confidence_score,
                        'price': data_package.price_data.get('price', 0)
                    })
                
                # Don't overwhelm the system
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
            
            # Convert to DataFrame
            if data_points:
                df = pd.DataFrame(data_points)
                logger.info(f"✅ Generated training dataset with {len(df)} data points")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating training dataset: {e}")
            return None

# Global enhanced ML data provider
enhanced_ml_provider = EnhancedMLDataProvider()

# Integration functions for existing ML systems
async def get_enhanced_ml_data(symbol: str = 'XAU', timeframe: str = '1h') -> Optional[Dict[str, Any]]:
    """Get enhanced ML data for existing ML engines"""
    try:
        # Map timeframe to lookback hours
        timeframe_mapping = {
            '1m': 1,
            '5m': 1,
            '15m': 2,
            '30m': 4,
            '1h': 8,
            '4h': 24,
            '1d': 168,  # 1 week
            '1w': 720   # 1 month
        }
        
        lookback_hours = timeframe_mapping.get(timeframe, 8)
        data_package = await enhanced_ml_provider.get_comprehensive_data(symbol, lookback_hours)
        
        if data_package:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': data_package.price_data.get('price', 0),
                'features': data_package.feature_vector.tolist(),
                'confidence': data_package.confidence_score,
                'quality_metrics': data_package.quality_metrics,
                'price_data': data_package.price_data,
                'technical_data': data_package.technical_data,
                'sentiment_data': data_package.sentiment_data,
                'macro_data': data_package.macro_data,
                'news_data': data_package.news_data,
                'metadata': data_package.metadata,
                'timestamp': data_package.timestamp.isoformat()
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML data retrieval failed: {e}")
        return None

async def get_signal_generation_data(symbol: str = 'XAU') -> Optional[Dict[str, Any]]:
    """Get enhanced data specifically for signal generation"""
    try:
        data_package = await enhanced_ml_provider.get_comprehensive_data(symbol, 4)  # 4 hours lookback
        
        if data_package:
            # Format data for signal generation
            return {
                'symbol': symbol,
                'current_price': data_package.price_data.get('price', 0),
                'price_change': data_package.price_data.get('price_change', 0),
                'technical_indicators': data_package.technical_data,
                'sentiment_analysis': data_package.sentiment_data,
                'macro_indicators': data_package.macro_data,
                'news_impact': data_package.news_data,
                'confidence_score': data_package.confidence_score,
                'quality_score': sum(data_package.quality_metrics.values()) / len(data_package.quality_metrics) if data_package.quality_metrics else 0,
                'data_sources': data_package.metadata.get('data_sources', []),
                'timestamp': data_package.timestamp.isoformat(),
                'feature_vector': data_package.feature_vector.tolist()
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Signal generation data retrieval failed: {e}")
        return None

if __name__ == "__main__":
    # Test the enhanced ML data provider
    async def test_provider():
        print("🧪 Testing Enhanced ML Data Provider...")
        
        # Test comprehensive data retrieval
        data_package = await enhanced_ml_provider.get_comprehensive_data('XAU')
        if data_package:
            print(f"✅ Data package generated with confidence: {data_package.confidence_score:.2f}")
            print(f"📊 Feature vector shape: {data_package.feature_vector.shape}")
            print(f"🎯 Quality metrics: {data_package.quality_metrics}")
        
        # Test integration functions
        ml_data = await get_enhanced_ml_data('XAU', '1h')
        if ml_data:
            print(f"✅ Enhanced ML data retrieved: {ml_data['symbol']}")
        
        signal_data = await get_signal_generation_data('XAU')
        if signal_data:
            print(f"✅ Signal generation data retrieved: {signal_data['symbol']}")
    
    asyncio.run(test_provider())
