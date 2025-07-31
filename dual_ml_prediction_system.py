#!/usr/bin/env python3
"""
Dual ML Engine Prediction System
Displays predictions from both engines and tracks their accuracy
"""
import logging
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class DualMLPredictionSystem:
    """Manages predictions from multiple ML engines with accuracy tracking"""
    
    def __init__(self, db_path='goldgpt_ml_tracking.db'):
        self.engines = [
            'enhanced_ml_prediction_engine',
            'intelligent_ml_predictor'
        ]
        self.db_path = db_path
        
    async def get_dual_predictions(self, symbol: str = 'XAUUSD') -> Dict:
        """Get predictions from both ML engines using GoldAPI for live prices"""
        try:
            from ml_engine_tracker import ml_tracker, track_prediction_from_engine
            
            # Get current price from GoldAPI via data pipeline (primary) with fallbacks
            current_price = await self._get_live_gold_price()
            
            # First, validate any ready predictions
            if ml_tracker:
                try:
                    ml_tracker.validate_predictions(current_price, symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Could not validate predictions: {e}")
            
            predictions = {
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'engines': [],
                'comparison': {},
                'current_price': current_price,
                'price_source': 'goldapi_data_pipeline',
                'tracking_enabled': True
            }
            
            logger.info(f"🎯 ML System using live GoldAPI price: ${current_price:.2f}")
            
            # Test Enhanced ML Engine
            enhanced_result = await self._get_enhanced_predictions()
            if enhanced_result and enhanced_result.get('success'):
                engine_data = {
                    'name': 'enhanced_ml_prediction_engine',
                    'display_name': 'Enhanced ML Engine',
                    'status': 'active',
                    'predictions': enhanced_result.get('predictions', []),
                    'current_price': current_price,  # Use GoldAPI price
                    'confidence_avg': sum(p.get('confidence', 0) for p in enhanced_result.get('predictions', [])) / max(len(enhanced_result.get('predictions', [])), 1),
                    'data_quality': enhanced_result.get('data_quality', 'unknown')
                }
                predictions['engines'].append(engine_data)
                
                # Track predictions using live GoldAPI price
                for pred in enhanced_result.get('predictions', []):
                    await track_prediction_from_engine(
                        'enhanced_ml_prediction_engine',
                        {
                            'symbol': symbol,
                            'current_price': current_price,  # Use GoldAPI price
                            'timeframe': pred['timeframe'],
                            'predicted_price': pred['predicted_price'],
                            'change_percent': pred['change_percent'],
                            'direction': pred['direction'],
                            'confidence': pred.get('confidence', 0.5)
                        }
                    )
            else:
                predictions['engines'].append({
                    'name': 'enhanced_ml_prediction_engine',
                    'display_name': 'Enhanced ML Engine',
                    'status': 'error',
                    'error': 'Failed to generate predictions',
                    'predictions': []
                })
            
            # Test Intelligent ML Predictor
            intelligent_result = await self._get_intelligent_predictions(symbol)
            if intelligent_result and intelligent_result.get('predictions'):
                engine_data = {
                    'name': 'intelligent_ml_predictor',
                    'display_name': 'Intelligent ML Predictor',
                    'status': 'active',
                    'predictions': intelligent_result.get('predictions', []),
                    'current_price': intelligent_result.get('current_price', predictions['current_price']),
                    'confidence_avg': sum(p.get('confidence', 0) for p in intelligent_result.get('predictions', [])) / max(len(intelligent_result.get('predictions', [])), 1),
                    'data_quality': intelligent_result.get('data_quality', 'unknown')
                }
                predictions['engines'].append(engine_data)
                
                # Track predictions
                for pred in intelligent_result.get('predictions', []):
                    await track_prediction_from_engine(
                        'intelligent_ml_predictor',
                        {
                            'symbol': symbol,
                            'current_price': predictions['current_price'],
                            'timeframe': pred['timeframe'],
                            'predicted_price': pred['predicted_price'],
                            'change_percent': pred['change_percent'],
                            'direction': pred['direction'],
                            'confidence': pred.get('confidence', 0.5)
                        }
                    )
            else:
                predictions['engines'].append({
                    'name': 'intelligent_ml_predictor',
                    'display_name': 'Intelligent ML Predictor',
                    'status': 'error',
                    'error': 'Failed to generate predictions',
                    'predictions': []
                })
            
            # Generate comparison analysis
            predictions['comparison'] = self._compare_predictions(predictions['engines'])
            
            # Add accuracy tracking stats
            predictions['accuracy_stats'] = self._get_accuracy_stats()
            
            # Convert numpy types to native Python types for JSON serialization
            predictions = convert_numpy_types(predictions)
            
            logger.info(f"✅ Generated dual predictions with {len(predictions['engines'])} engines")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Dual prediction system failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_enhanced_predictions(self) -> Dict:
        """Get predictions from Enhanced ML Engine"""
        try:
            from enhanced_ml_prediction_engine import get_enhanced_ml_predictions
            return get_enhanced_ml_predictions()
        except Exception as e:
            logger.error(f"❌ Enhanced ML Engine failed: {e}")
            return None
    
    async def _get_intelligent_predictions(self, symbol: str) -> Dict:
        """Get predictions from Intelligent ML Predictor"""
        try:
            from intelligent_ml_predictor import get_intelligent_ml_predictions
            return get_intelligent_ml_predictions(symbol)
        except Exception as e:
            logger.error(f"❌ Intelligent ML Predictor failed: {e}")
            return None
    
    def _compare_predictions(self, engines: List[Dict]) -> Dict:
        """Compare predictions between engines"""
        comparison = {
            'agreement': {},
            'divergence': {},
            'consensus': {},
            'conflict_areas': []
        }
        
        if len(engines) < 2:
            return comparison
        
        # Compare by timeframe
        timeframes = ['1H', '4H', '1D']
        for tf in timeframes:
            engine_predictions = []
            for engine in engines:
                if engine['status'] == 'active':
                    for pred in engine['predictions']:
                        if pred['timeframe'] == tf:
                            engine_predictions.append({
                                'engine': engine['name'],
                                'direction': pred['direction'],
                                'change_percent': pred['change_percent'],
                                'confidence': pred.get('confidence', 0.5)
                            })
            
            if len(engine_predictions) >= 2:
                # Check direction agreement
                directions = [p['direction'] for p in engine_predictions]
                changes = [p['change_percent'] for p in engine_predictions]
                
                direction_agreement = len(set(directions)) == 1
                change_similarity = abs(max(changes) - min(changes)) < 0.5  # Within 0.5%
                
                comparison['agreement'][tf] = {
                    'direction_agreement': direction_agreement,
                    'change_similarity': change_similarity,
                    'consensus_direction': directions[0] if direction_agreement else 'mixed',
                    'avg_change': sum(changes) / len(changes),
                    'confidence_level': 'high' if direction_agreement and change_similarity else 'low'
                }
                
                if not direction_agreement:
                    comparison['conflict_areas'].append({
                        'timeframe': tf,
                        'issue': 'Direction disagreement',
                        'details': f"Engines predict: {', '.join(directions)}"
                    })
        
        return comparison
    
    def _get_accuracy_stats(self) -> Dict:
        """Get accuracy statistics for dashboard display"""
        try:
            from ml_engine_tracker import get_ml_accuracy_dashboard_data
            return get_ml_accuracy_dashboard_data()
        except Exception as e:
            logger.error(f"❌ Failed to get accuracy stats: {e}")
            return {'engines': [], 'best_performer': None}

# Global instance
dual_ml_system = DualMLPredictionSystem()

async def get_dual_ml_predictions(symbol: str = 'XAUUSD') -> Dict:
    """Main function to get dual ML predictions"""
    return await dual_ml_system.get_dual_predictions(symbol)

if __name__ == "__main__":
    # Test the dual prediction system
    async def test_dual_system():
        print("🔄 Testing Dual ML Prediction System")
        print("=" * 50)
        
        result = await get_dual_ml_predictions()
        
        print(f"Success: {result.get('success')}")
        print(f"Engines tested: {len(result.get('engines', []))}")
        
        for engine in result.get('engines', []):
            status = engine['status']
            name = engine['display_name']
            pred_count = len(engine.get('predictions', []))
            print(f"  {name}: {status} ({pred_count} predictions)")
            
            if status == 'active':
                for pred in engine.get('predictions', []):
                    print(f"    {pred['timeframe']}: ${pred['predicted_price']:.2f} ({pred['change_percent']:+.2f}%) - {pred['direction']}")
        
        print(f"\nComparison: {len(result.get('comparison', {}).get('agreement', {}))} timeframes analyzed")
        print("✅ Dual ML system test complete")
    
    async def _get_live_gold_price(self) -> float:
        """Get live gold price from GoldAPI via data pipeline with fallbacks"""
        try:
            # Primary: Use data pipeline GoldAPI
            try:
                from data_pipeline_core import data_pipeline, DataType
                price_data = await data_pipeline.get_unified_data('XAU', DataType.PRICE)
                if price_data and 'price' in price_data:
                    live_price = float(price_data['price'])
                    logger.info(f"📡 Live GoldAPI price via data pipeline: ${live_price:.2f}")
                    return live_price
            except Exception as e:
                logger.warning(f"⚠️ Data pipeline GoldAPI failed: {e}")
            
            # Secondary: Direct GoldAPI call
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.gold-api.com/price/XAU",
                        headers={"X-API-KEY": "goldapi-YOUR_KEY"},
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            live_price = float(data.get('price', 0))
                            if live_price > 1000:  # Sanity check
                                logger.info(f"📡 Live GoldAPI price (direct): ${live_price:.2f}")
                                return live_price
            except Exception as e:
                logger.warning(f"⚠️ Direct GoldAPI call failed: {e}")
            
            # Tertiary: Price storage manager fallback
            try:
                from price_storage_manager import get_current_gold_price
                fallback_price = get_current_gold_price()
                if fallback_price and fallback_price > 1000:
                    logger.info(f"📋 Using price storage fallback: ${fallback_price:.2f}")
                    return fallback_price
            except Exception as e:
                logger.warning(f"⚠️ Price storage fallback failed: {e}")
            
            # Final fallback: Use most recent known Gold API price
            logger.warning("⚠️ All price sources failed, using current Gold API fallback")
            return 3430.0
            
        except Exception as e:
            logger.error(f"❌ Error getting live gold price: {e}")
            return 3430.0
    
    asyncio.run(test_dual_system())
