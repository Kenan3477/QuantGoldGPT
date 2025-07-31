#!/usr/bin/env python3
"""
GoldGPT Data Quality Validation System
Advanced validation and anomaly detection for all data sources
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statistics import mean, stdev
from data_pipeline_core import data_pipeline, DataType

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Data validation rule configuration"""
    name: str
    data_type: DataType
    rule_type: str  # range, trend, outlier, consistency, freshness
    parameters: Dict[str, Any]
    severity: str  # critical, warning, info
    enabled: bool = True

@dataclass
class ValidationResult:
    """Result of data validation"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    actual_value: Any
    expected_range: Optional[Tuple[float, float]]
    confidence: float
    timestamp: datetime

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    data_point: Any
    anomaly_score: float
    anomaly_type: str
    description: str
    severity: str
    recommendations: List[str]

class DataQualityValidator:
    """Advanced data quality validation and anomaly detection system"""
    
    def __init__(self, db_path: str = "goldgpt_data_quality.db"):
        self.db_path = db_path
        self.validation_rules = []
        self.anomaly_thresholds = self.setup_anomaly_thresholds()
        self.historical_baselines = {}
        self.initialize_database()
        self.setup_validation_rules()
    
    def setup_anomaly_thresholds(self) -> Dict[str, Dict]:
        """Configure anomaly detection thresholds"""
        return {
            'price': {
                'sudden_change_threshold': 0.05,  # 5% sudden change
                'volatility_threshold': 3.0,  # 3 standard deviations
                'gap_threshold': 0.10,  # 10% price gap
                'volume_spike_threshold': 5.0  # 5x normal volume
            },
            'sentiment': {
                'extreme_sentiment_threshold': 0.8,  # Very positive/negative
                'sentiment_volatility_threshold': 2.0,
                'sample_size_minimum': 5  # Minimum articles for reliable sentiment
            },
            'technical': {
                'indicator_divergence_threshold': 0.7,  # High divergence between indicators
                'signal_flip_frequency_max': 3,  # Max signal changes per hour
                'confidence_minimum': 0.3  # Minimum confidence for valid signals
            },
            'macro': {
                'data_age_threshold_days': 90,  # Data older than 90 days is stale
                'change_threshold': 0.5,  # Significant economic indicator change
                'correlation_break_threshold': 0.3  # Correlation breakdown threshold
            }
        }
    
    def initialize_database(self):
        """Initialize data quality database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Validation rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT UNIQUE NOT NULL,
                data_type TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                severity TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Validation results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                actual_value TEXT,
                expected_range TEXT,
                confidence REAL,
                validation_timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Anomaly detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                description TEXT,
                severity TEXT,
                data_point TEXT,
                recommendations TEXT,
                detected_at DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                baseline_value REAL,
                deviation_score REAL,
                quality_score REAL,
                measurement_date DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX(data_type, metric_name, measurement_date)
            )
        ''')
        
        # Historical baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                baseline_type TEXT NOT NULL,
                baseline_data TEXT NOT NULL,
                calculation_date DATETIME NOT NULL,
                valid_until DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(data_type, baseline_type)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Data quality validation database initialized")
    
    def setup_validation_rules(self):
        """Setup comprehensive validation rules"""
        
        # Price validation rules
        self.add_validation_rule(ValidationRule(
            name="price_range_check",
            data_type=DataType.PRICE,
            rule_type="range",
            parameters={"min_value": 1000, "max_value": 10000},
            severity="critical"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="price_freshness_check",
            data_type=DataType.PRICE,
            rule_type="freshness",
            parameters={"max_age_minutes": 5},
            severity="warning"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="price_change_check",
            data_type=DataType.PRICE,
            rule_type="trend",
            parameters={"max_change_percent": 10},
            severity="warning"
        ))
        
        # Sentiment validation rules
        self.add_validation_rule(ValidationRule(
            name="sentiment_range_check",
            data_type=DataType.SENTIMENT,
            rule_type="range",
            parameters={"min_value": -1.0, "max_value": 1.0},
            severity="critical"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="sentiment_sample_size_check",
            data_type=DataType.SENTIMENT,
            rule_type="consistency",
            parameters={"min_sample_size": 3},
            severity="warning"
        ))
        
        # Technical indicator validation rules
        self.add_validation_rule(ValidationRule(
            name="technical_confidence_check",
            data_type=DataType.TECHNICAL,
            rule_type="consistency",
            parameters={"min_confidence": 0.2},
            severity="warning"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="technical_signal_consistency",
            data_type=DataType.TECHNICAL,
            rule_type="consistency",
            parameters={"max_signal_changes_per_hour": 5},
            severity="info"
        ))
        
        # Macro data validation rules
        self.add_validation_rule(ValidationRule(
            name="macro_data_freshness",
            data_type=DataType.MACRO,
            rule_type="freshness",
            parameters={"max_age_days": 30},
            severity="warning"
        ))
        
        logger.info(f"✅ Setup {len(self.validation_rules)} validation rules")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.validation_rules.append(rule)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO validation_rules 
            (rule_name, data_type, rule_type, parameters, severity, enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            rule.name,
            rule.data_type.value,
            rule.rule_type,
            json.dumps(rule.parameters),
            rule.severity,
            rule.enabled
        ))
        
        conn.commit()
        conn.close()
    
    def validate_price_data(self, price_data: Dict) -> List[ValidationResult]:
        """Validate price data quality"""
        results = []
        
        if not price_data:
            results.append(ValidationResult(
                rule_name="price_data_availability",
                passed=False,
                severity="critical",
                message="No price data available",
                actual_value=None,
                expected_range=None,
                confidence=1.0,
                timestamp=datetime.now()
            ))
            return results
        
        price = price_data.get('price', 0)
        timestamp_str = price_data.get('timestamp', '')
        
        # Range check
        range_rule = next((r for r in self.validation_rules if r.name == "price_range_check"), None)
        if range_rule and range_rule.enabled:
            min_val = range_rule.parameters['min_value']
            max_val = range_rule.parameters['max_value']
            
            passed = min_val <= price <= max_val
            results.append(ValidationResult(
                rule_name=range_rule.name,
                passed=passed,
                severity=range_rule.severity,
                message=f"Price {price} {'within' if passed else 'outside'} expected range [{min_val}, {max_val}]",
                actual_value=price,
                expected_range=(min_val, max_val),
                confidence=1.0,
                timestamp=datetime.now()
            ))
        
        # Freshness check
        freshness_rule = next((r for r in self.validation_rules if r.name == "price_freshness_check"), None)
        if freshness_rule and freshness_rule.enabled and timestamp_str:
            try:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                data_time = data_time.replace(tzinfo=None)
                age_minutes = (datetime.now() - data_time).total_seconds() / 60
                max_age = freshness_rule.parameters['max_age_minutes']
                
                passed = age_minutes <= max_age
                results.append(ValidationResult(
                    rule_name=freshness_rule.name,
                    passed=passed,
                    severity=freshness_rule.severity,
                    message=f"Price data age {age_minutes:.1f} minutes {'within' if passed else 'exceeds'} limit {max_age} minutes",
                    actual_value=age_minutes,
                    expected_range=(0, max_age),
                    confidence=0.9,
                    timestamp=datetime.now()
                ))
            except:
                results.append(ValidationResult(
                    rule_name=freshness_rule.name,
                    passed=False,
                    severity="warning",
                    message="Unable to parse price data timestamp",
                    actual_value=timestamp_str,
                    expected_range=None,
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_sentiment_data(self, sentiment_data: Dict) -> List[ValidationResult]:
        """Validate sentiment analysis data quality"""
        results = []
        
        if not sentiment_data or 'metrics' not in sentiment_data:
            results.append(ValidationResult(
                rule_name="sentiment_data_availability",
                passed=False,
                severity="critical",
                message="No sentiment data available",
                actual_value=None,
                expected_range=None,
                confidence=1.0,
                timestamp=datetime.now()
            ))
            return results
        
        metrics = sentiment_data['metrics']
        overall_sentiment = metrics.get('overall_sentiment', 0)
        sample_size = metrics.get('sample_size', 0)
        
        # Range check
        range_rule = next((r for r in self.validation_rules if r.name == "sentiment_range_check"), None)
        if range_rule and range_rule.enabled:
            min_val = range_rule.parameters['min_value']
            max_val = range_rule.parameters['max_value']
            
            passed = min_val <= overall_sentiment <= max_val
            results.append(ValidationResult(
                rule_name=range_rule.name,
                passed=passed,
                severity=range_rule.severity,
                message=f"Sentiment {overall_sentiment} {'within' if passed else 'outside'} valid range [{min_val}, {max_val}]",
                actual_value=overall_sentiment,
                expected_range=(min_val, max_val),
                confidence=1.0,
                timestamp=datetime.now()
            ))
        
        # Sample size check
        sample_rule = next((r for r in self.validation_rules if r.name == "sentiment_sample_size_check"), None)
        if sample_rule and sample_rule.enabled:
            min_size = sample_rule.parameters['min_sample_size']
            
            passed = sample_size >= min_size
            results.append(ValidationResult(
                rule_name=sample_rule.name,
                passed=passed,
                severity=sample_rule.severity,
                message=f"Sentiment sample size {sample_size} {'adequate' if passed else 'insufficient'} (minimum {min_size})",
                actual_value=sample_size,
                expected_range=(min_size, float('inf')),
                confidence=0.8,
                timestamp=datetime.now()
            ))
        
        return results
    
    def validate_technical_data(self, technical_data: Dict) -> List[ValidationResult]:
        """Validate technical analysis data quality"""
        results = []
        
        if not technical_data or 'overall_assessment' not in technical_data:
            results.append(ValidationResult(
                rule_name="technical_data_availability",
                passed=False,
                severity="critical",
                message="No technical analysis data available",
                actual_value=None,
                expected_range=None,
                confidence=1.0,
                timestamp=datetime.now()
            ))
            return results
        
        assessment = technical_data['overall_assessment']
        confidence = assessment.get('confidence', 0)
        
        # Confidence check
        conf_rule = next((r for r in self.validation_rules if r.name == "technical_confidence_check"), None)
        if conf_rule and conf_rule.enabled:
            min_conf = conf_rule.parameters['min_confidence']
            
            passed = confidence >= min_conf
            results.append(ValidationResult(
                rule_name=conf_rule.name,
                passed=passed,
                severity=conf_rule.severity,
                message=f"Technical analysis confidence {confidence:.2f} {'adequate' if passed else 'low'} (minimum {min_conf})",
                actual_value=confidence,
                expected_range=(min_conf, 1.0),
                confidence=0.8,
                timestamp=datetime.now()
            ))
        
        return results
    
    def detect_price_anomalies(self, price_data: Dict, historical_context: List[Dict] = None) -> List[AnomalyDetection]:
        """Detect price anomalies"""
        anomalies = []
        
        if not price_data:
            return anomalies
        
        current_price = price_data.get('price', 0)
        
        # If we have historical context, look for anomalies
        if historical_context and len(historical_context) > 10:
            prices = [p.get('close', p.get('price', 0)) for p in historical_context if p.get('close') or p.get('price')]
            
            if len(prices) > 5:
                price_mean = mean(prices)
                price_std = stdev(prices) if len(prices) > 1 else 0
                
                # Z-score anomaly detection
                if price_std > 0:
                    z_score = abs(current_price - price_mean) / price_std
                    threshold = self.anomaly_thresholds['price']['volatility_threshold']
                    
                    if z_score > threshold:
                        anomalies.append(AnomalyDetection(
                            data_point=price_data,
                            anomaly_score=z_score,
                            anomaly_type="price_outlier",
                            description=f"Price {current_price} is {z_score:.2f} standard deviations from mean {price_mean:.2f}",
                            severity="warning" if z_score < threshold * 1.5 else "critical",
                            recommendations=[
                                "Verify price data source accuracy",
                                "Check for market news or events",
                                "Consider excluding from analysis if confirmed error"
                            ]
                        ))
                
                # Sudden change detection
                if len(prices) >= 2:
                    recent_change = abs(current_price - prices[-1]) / prices[-1]
                    change_threshold = self.anomaly_thresholds['price']['sudden_change_threshold']
                    
                    if recent_change > change_threshold:
                        anomalies.append(AnomalyDetection(
                            data_point=price_data,
                            anomaly_score=recent_change,
                            anomaly_type="sudden_price_change",
                            description=f"Sudden price change of {recent_change:.2%} detected",
                            severity="warning",
                            recommendations=[
                                "Investigate cause of sudden price movement",
                                "Check for market volatility events",
                                "Validate price feed accuracy"
                            ]
                        ))
        
        return anomalies
    
    def detect_sentiment_anomalies(self, sentiment_data: Dict) -> List[AnomalyDetection]:
        """Detect sentiment analysis anomalies"""
        anomalies = []
        
        if not sentiment_data or 'metrics' not in sentiment_data:
            return anomalies
        
        metrics = sentiment_data['metrics']
        overall_sentiment = metrics.get('overall_sentiment', 0)
        sample_size = metrics.get('sample_size', 0)
        confidence = metrics.get('confidence', 0)
        
        # Extreme sentiment detection
        extreme_threshold = self.anomaly_thresholds['sentiment']['extreme_sentiment_threshold']
        if abs(overall_sentiment) > extreme_threshold:
            anomalies.append(AnomalyDetection(
                data_point=sentiment_data,
                anomaly_score=abs(overall_sentiment),
                anomaly_type="extreme_sentiment",
                description=f"Extreme sentiment detected: {overall_sentiment:.2f}",
                severity="warning",
                recommendations=[
                    "Verify news sources for accuracy",
                    "Check for sentiment manipulation or bias",
                    "Consider market context and events"
                ]
            ))
        
        # Low sample size anomaly
        min_sample = self.anomaly_thresholds['sentiment']['sample_size_minimum']
        if sample_size < min_sample:
            anomalies.append(AnomalyDetection(
                data_point=sentiment_data,
                anomaly_score=min_sample - sample_size,
                anomaly_type="insufficient_sample_size",
                description=f"Low sentiment sample size: {sample_size} articles",
                severity="info",
                recommendations=[
                    "Increase news data collection timeframe",
                    "Add more news sources",
                    "Use with caution in analysis"
                ]
            ))
        
        return anomalies
    
    def calculate_data_quality_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall data quality score"""
        if not validation_results:
            return 0.0
        
        # Weight by severity
        severity_weights = {'critical': 1.0, 'warning': 0.7, 'info': 0.3}
        
        total_weight = 0
        passed_weight = 0
        
        for result in validation_results:
            weight = severity_weights.get(result.severity, 0.5) * result.confidence
            total_weight += weight
            
            if result.passed:
                passed_weight += weight
        
        if total_weight > 0:
            quality_score = passed_weight / total_weight
            return round(quality_score, 3)
        
        return 0.0
    
    async def validate_all_data(self, price_data: Dict = None, sentiment_data: Dict = None,
                               technical_data: Dict = None, macro_data: Dict = None) -> Dict:
        """Comprehensive data quality validation"""
        
        all_results = []
        all_anomalies = []
        quality_scores = {}
        
        # Validate price data
        if price_data:
            price_results = self.validate_price_data(price_data)
            all_results.extend(price_results)
            
            price_anomalies = self.detect_price_anomalies(price_data)
            all_anomalies.extend(price_anomalies)
            
            quality_scores['price'] = self.calculate_data_quality_score(price_results)
        
        # Validate sentiment data
        if sentiment_data:
            sentiment_results = self.validate_sentiment_data(sentiment_data)
            all_results.extend(sentiment_results)
            
            sentiment_anomalies = self.detect_sentiment_anomalies(sentiment_data)
            all_anomalies.extend(sentiment_anomalies)
            
            quality_scores['sentiment'] = self.calculate_data_quality_score(sentiment_results)
        
        # Validate technical data
        if technical_data:
            technical_results = self.validate_technical_data(technical_data)
            all_results.extend(technical_results)
            
            quality_scores['technical'] = self.calculate_data_quality_score(technical_results)
        
        # Store results
        self.store_validation_results(all_results)
        self.store_anomalies(all_anomalies)
        
        # Calculate overall quality score
        if quality_scores:
            overall_quality = sum(quality_scores.values()) / len(quality_scores)
        else:
            overall_quality = 0.0
        
        # Categorize overall quality
        if overall_quality >= 0.8:
            quality_level = "EXCELLENT"
        elif overall_quality >= 0.6:
            quality_level = "GOOD"
        elif overall_quality >= 0.4:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        return {
            'overall_quality_score': round(overall_quality, 3),
            'quality_level': quality_level,
            'component_scores': quality_scores,
            'validation_results': [{
                'rule_name': r.rule_name,
                'passed': r.passed,
                'severity': r.severity,
                'message': r.message,
                'confidence': r.confidence
            } for r in all_results],
            'anomalies': [{
                'type': a.anomaly_type,
                'score': a.anomaly_score,
                'description': a.description,
                'severity': a.severity,
                'recommendations': a.recommendations
            } for a in all_anomalies],
            'summary': {
                'total_checks': len(all_results),
                'passed_checks': sum(1 for r in all_results if r.passed),
                'failed_checks': sum(1 for r in all_results if not r.passed),
                'anomalies_found': len(all_anomalies),
                'critical_issues': len([r for r in all_results if not r.passed and r.severity == 'critical'])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def store_validation_results(self, results: List[ValidationResult]):
        """Store validation results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute('''
                INSERT INTO validation_results 
                (rule_name, data_type, passed, severity, message, actual_value, 
                 expected_range, confidence, validation_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.rule_name,
                'unknown',  # We'd need to map rule to data type
                result.passed,
                result.severity,
                result.message,
                json.dumps(result.actual_value) if result.actual_value is not None else None,
                json.dumps(result.expected_range) if result.expected_range else None,
                result.confidence,
                result.timestamp.isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def store_anomalies(self, anomalies: List[AnomalyDetection]):
        """Store detected anomalies in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for anomaly in anomalies:
            cursor.execute('''
                INSERT INTO anomalies 
                (data_type, anomaly_type, anomaly_score, description, severity, 
                 data_point, recommendations, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'unknown',  # Would be mapped from anomaly type
                anomaly.anomaly_type,
                anomaly.anomaly_score,
                anomaly.description,
                anomaly.severity,
                json.dumps(anomaly.data_point) if hasattr(anomaly.data_point, '__dict__') else str(anomaly.data_point),
                json.dumps(anomaly.recommendations),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def get_quality_report(self, days_back: int = 7) -> Dict:
        """Generate data quality report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Get recent validation results
        cursor.execute('''
            SELECT rule_name, passed, severity, COUNT(*) as count
            FROM validation_results 
            WHERE validation_timestamp >= ?
            GROUP BY rule_name, passed, severity
            ORDER BY rule_name
        ''', (start_date,))
        
        validation_summary = {}
        for row in cursor.fetchall():
            rule_name, passed, severity, count = row
            if rule_name not in validation_summary:
                validation_summary[rule_name] = {'passed': 0, 'failed': 0, 'severity': severity}
            
            if passed:
                validation_summary[rule_name]['passed'] += count
            else:
                validation_summary[rule_name]['failed'] += count
        
        # Get recent anomalies
        cursor.execute('''
            SELECT anomaly_type, severity, COUNT(*) as count
            FROM anomalies 
            WHERE detected_at >= ?
            GROUP BY anomaly_type, severity
        ''', (start_date,))
        
        anomaly_summary = {}
        for row in cursor.fetchall():
            anomaly_type, severity, count = row
            anomaly_summary[anomaly_type] = {
                'count': count,
                'severity': severity
            }
        
        conn.close()
        
        return {
            'report_period': f"{days_back} days",
            'generated_at': datetime.now().isoformat(),
            'validation_summary': validation_summary,
            'anomaly_summary': anomaly_summary,
            'recommendations': self.generate_quality_recommendations(validation_summary, anomaly_summary)
        }
    
    def generate_quality_recommendations(self, validation_summary: Dict, anomaly_summary: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Check for frequent failures
        for rule_name, stats in validation_summary.items():
            total = stats['passed'] + stats['failed']
            if total > 0:
                failure_rate = stats['failed'] / total
                if failure_rate > 0.2:  # More than 20% failure rate
                    recommendations.append(
                        f"High failure rate ({failure_rate:.1%}) for {rule_name} - review data source quality"
                    )
        
        # Check for frequent anomalies
        for anomaly_type, stats in anomaly_summary.items():
            if stats['count'] > 5:  # More than 5 anomalies
                recommendations.append(
                    f"Frequent {anomaly_type} anomalies detected - investigate data source stability"
                )
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations

# Global instance
quality_validator = DataQualityValidator()

if __name__ == "__main__":
    # Test the quality validator
    async def test_quality_validator():
        print("🧪 Testing Data Quality Validation System...")
        
        # Test with sample data
        sample_price_data = {
            'price': 3400.50,
            'timestamp': datetime.now().isoformat(),
            'source': 'test'
        }
        
        sample_sentiment_data = {
            'metrics': {
                'overall_sentiment': 0.3,
                'sample_size': 8,
                'confidence': 0.7
            }
        }
        
        sample_technical_data = {
            'overall_assessment': {
                'signal': 'BUY',
                'confidence': 0.6
            }
        }
        
        # Run comprehensive validation
        validation_report = await quality_validator.validate_all_data(
            price_data=sample_price_data,
            sentiment_data=sample_sentiment_data,
            technical_data=sample_technical_data
        )
        
        print(f"📊 Overall Quality Score: {validation_report['overall_quality_score']}")
        print(f"🏆 Quality Level: {validation_report['quality_level']}")
        print(f"✅ Passed Checks: {validation_report['summary']['passed_checks']}")
        print(f"❌ Failed Checks: {validation_report['summary']['failed_checks']}")
        print(f"🚨 Anomalies Found: {validation_report['summary']['anomalies_found']}")
        
        if validation_report['anomalies']:
            print("\n🔍 Detected Anomalies:")
            for anomaly in validation_report['anomalies']:
                print(f"  • {anomaly['type']}: {anomaly['description']}")
        
        # Test quality report
        quality_report = quality_validator.get_quality_report(7)
        print(f"\n📋 Quality Report Recommendations:")
        for rec in quality_report['recommendations']:
            print(f"  • {rec}")
    
    asyncio.run(test_quality_validator())
