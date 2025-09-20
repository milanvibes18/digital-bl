"""Test Flask API endpoints"""

import pytest

def test_health_endpoint(flask_test_client):
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'timestamp' in data

def test_dashboard_data_endpoint(flask_test_client):
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/dashboard_data')
    assert response.status_code == 200
    assert isinstance(response.get_json(), dict)

def test_devices_endpoint(flask_test_client):
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/devices')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_alerts_endpoint(flask_test_client):
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/alerts')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_system_metrics_endpoint(flask_test_client):
    if flask_test_client is None:
        pytest.skip("Flask app not available")
    
    response = flask_test_client.get('/api/system_metrics')
    assert response.status_code == 200
    assert isinstance(response.get_json(), dict)
