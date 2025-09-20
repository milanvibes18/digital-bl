"""Test AI modules functionality"""

import pytest
import pandas as pd
import numpy as np

def test_health_score_calculator():
    """Test health score calculator"""
    try:
        from AI_MODULES.health_score import HealthScoreCalculator
        
        calculator = HealthScoreCalculator()
        sample_data = pd.DataFrame({
            'temperature': [20, 22, 21, 25, 23],
            'pressure': [1013, 1015, 1012, 1018, 1014],
            'vibration': [0.1, 0.12, 0.09, 0.15, 0.11],
            'efficiency': [85, 87, 83, 90, 86]
        })
        
        result = calculator.calculate_overall_health_score(sample_data, device_id="TEST_001")
        
        assert 'overall_score' in result
        assert 0 <= result['overall_score'] <= 1
        assert 'health_status' in result
        assert result['health_status'] in ['excellent', 'good', 'warning', 'critical', 'failure']
    except ImportError:
        pytest.skip("HealthScoreCalculator not available")

def test_predictive_analytics_engine():
    """Test predictive analytics engine"""
    try:
        from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
        
        analytics = PredictiveAnalyticsEngine()
        sample_data = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 100),
            'pressure': np.random.normal(1013, 10, 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
        })
        
        features, target = analytics.prepare_data(sample_data)
        
        assert features is not None
        assert len(features.columns) > 0
    except ImportError:
        pytest.skip("PredictiveAnalyticsEngine not available")

def test_alert_manager():
    """Test alert manager"""
    try:
        from AI_MODULES.alert_manager import AlertManager
        
        alert_manager = AlertManager()
        test_data = {'temperature': 95, 'pressure': 1050, 'vibration': 0.8}
        
        alerts = alert_manager.evaluate_conditions(test_data, device_id="TEST_001")
        
        assert isinstance(alerts, list)
        assert len(alerts) > 0
    except ImportError:
        pytest.skip("AlertManager not available")

def test_pattern_analyzer():
    """Test pattern analyzer"""
    try:
        from AI_MODULES.pattern_analyzer import PatternAnalyzer
        
        analyzer = PatternAnalyzer()
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'value': 20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 1, 100)
        })
        
        result = analyzer.analyze_temporal_patterns(sample_data, 'timestamp', ['value'])
        
        assert 'patterns_found' in result
        assert isinstance(result['patterns_found'], dict)
    except ImportError:
        pytest.skip("PatternAnalyzer not available")
