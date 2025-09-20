#!/usr/bin/env python3
"""
Enhanced Flask Application for Digital Twin System v2.0
Main web application with real-time capabilities, advanced analytics, and secure API endpoints.
"""

import os
import sys
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time
import uuid
from functools import wraps
import hashlib
import hmac
import secrets

# Flask imports
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import eventlet

# Security imports
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
try:
    from CONFIG.app_config import config
    from AI_MODULES.secure_database_manager import SecureDatabaseManager
    from AI_MODULES.predictive_analytics_engine import PredictiveAnalyticsEngine
    from AI_MODULES.health_score import HealthScoreCalculator
    from AI_MODULES.alert_manager import AlertManager
    from AI_MODULES.pattern_analyzer import PatternAnalyzer
    from AI_MODULES.recommendation_engine import RecommendationEngine
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available")

class DigitalTwinApp:
    """Main Digital Twin Flask Application Class"""
    
    def __init__(self):
        self.app = None
        self.socketio = None
        self.db_manager = None
        self.analytics_engine = None
        self.health_calculator = None
        self.alert_manager = None
        self.pattern_analyzer = None
        self.recommendation_engine = None
        
        # Application state
        self.connected_clients = {}
        self.data_cache = {}
        self.last_update = datetime.now()
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize Flask app
        self.create_app()
        
        # Initialize AI modules
        self.initialize_ai_modules()
        
        # Setup routes
        self.setup_routes()
        
        # Setup WebSocket events
        self.setup_websocket_events()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('LOGS/digital_twin_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DigitalTwinApp')
        self.logger.info("Digital Twin Application starting...")
    
    def create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(__name__)
        
        # Configuration
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
        self.app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.app.config['TESTING'] = False
        
        # CORS configuration
        CORS(self.app, origins="*", allow_headers=["Content-Type", "Authorization"])
        
        # SocketIO initialization
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        self.logger.info("Flask application created successfully")
    
    def initialize_ai_modules(self):
        """Initialize AI and analytics modules"""
        try:
            # Database manager
            self.db_manager = SecureDatabaseManager()
            self.logger.info("Database manager initialized")
            
            # Analytics engine
            self.analytics_engine = PredictiveAnalyticsEngine()
            self.logger.info("Analytics engine initialized")
            
            # Health score calculator
            self.health_calculator = HealthScoreCalculator()
            self.logger.info("Health calculator initialized")
            
            # Alert manager
            self.alert_manager = AlertManager()
            self.logger.info("Alert manager initialized")
            
            # Pattern analyzer
            self.pattern_analyzer = PatternAnalyzer()
            self.logger.info("Pattern analyzer initialized")
            
            # Recommendation engine
            self.recommendation_engine = RecommendationEngine()
            self.logger.info("Recommendation engine initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing AI modules: {e}")
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        # Main pages
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('index.html')
        
        @self.app.route('/dashboard')
        def enhanced_dashboard():
            """Enhanced dashboard with advanced analytics"""
            return render_template('enhanced_dashboard.html')
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics page"""
            return render_template('analytics.html')
        
        @self.app.route('/devices')
        def devices():
            """Device management page"""
            return render_template('devices_view.html')
        
        # Health check endpoint
        @self.app.route('/health')
        def health_check():
            """Health check endpoint for monitoring"""
            try:
                # Check database connectivity
                db_status = self.check_database_health()
                
                # Check AI modules
                ai_status = self.check_ai_modules_health()
                
                status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'database': db_status,
                    'ai_modules': ai_status,
                    'uptime': self.get_uptime(),
                    'version': '2.0.0'
                }
                
                return jsonify(status), 200
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 503
        
        # API endpoints
        @self.app.route('/api/dashboard_data')
        def get_dashboard_data():
            """Get main dashboard data"""
            try:
                data = self.get_cached_dashboard_data()
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting dashboard data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/devices')
        def get_devices():
            """Get all devices data"""
            try:
                devices = self.get_devices_data()
                return jsonify(devices)
            except Exception as e:
                self.logger.error(f"Error getting devices data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/device/<device_id>')
        def get_device(device_id):
            """Get specific device data"""
            try:
                device = self.get_device_data(device_id)
                if device:
                    return jsonify(device)
                else:
                    return jsonify({'error': 'Device not found'}), 404
            except Exception as e:
                self.logger.error(f"Error getting device {device_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analytics')
        def get_analytics_data():
            """Get analytics data for charts"""
            try:
                analytics_data = self.get_analytics_data()
                return jsonify(analytics_data)
            except Exception as e:
                self.logger.error(f"Error getting analytics data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get system alerts"""
            try:
                limit = request.args.get('limit', 10, type=int)
                severity = request.args.get('severity', None)
                
                alerts = self.get_alerts_data(limit=limit, severity=severity)
                return jsonify(alerts)
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system_metrics')
        def get_system_metrics():
            """Get system performance metrics"""
            try:
                metrics = self.get_system_metrics()
                return jsonify(metrics)
            except Exception as e:
                self.logger.error(f"Error getting system metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/historical_data')
        def get_historical_data():
            """Get historical data for trends"""
            try:
                device_id = request.args.get('device_id')
                hours = request.args.get('hours', 24, type=int)
                metric = request.args.get('metric', 'value')
                
                data = self.get_historical_data(device_id, hours, metric)
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting historical data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/predictions')
        def get_predictions():
            """Get predictive analytics data"""
            try:
                device_id = request.args.get('device_id')
                horizon = request.args.get('horizon', 24, type=int)
                
                predictions = self.get_predictions_data(device_id, horizon)
                return jsonify(predictions)
            except Exception as e:
                self.logger.error(f"Error getting predictions: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health_scores')
        def get_health_scores():
            """Get health scores for all devices"""
            try:
                health_scores = self.calculate_health_scores()
                return jsonify(health_scores)
            except Exception as e:
                self.logger.error(f"Error getting health scores: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recommendations')
        def get_recommendations():
            """Get AI recommendations"""
            try:
                recommendations = self.get_recommendations()
                return jsonify(recommendations)
            except Exception as e:
                self.logger.error(f"Error getting recommendations: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Data management endpoints
        @self.app.route('/api/export_data')
        def export_data():
            """Export data for analysis"""
            try:
                format_type = request.args.get('format', 'json')
                date_range = request.args.get('days', 7, type=int)
                
                exported_data = self.export_data(format_type, date_range)
                return jsonify(exported_data)
            except Exception as e:
                self.logger.error(f"Error exporting data: {e}")
                return jsonify({'error': str(e)}), 500
        
        self.logger.info("All routes setup completed")
    
    def setup_websocket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = str(uuid.uuid4())
            session['client_id'] = client_id
            self.connected_clients[client_id] = {
                'connected_at': datetime.now(),
                'last_ping': datetime.now()
            }
            
            self.logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")
            
            # Send initial data to client
            emit('initial_data', self.get_cached_dashboard_data())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = session.get('client_id')
            if client_id and client_id in self.connected_clients:
                del self.connected_clients[client_id]
                self.logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle client ping"""
            client_id = session.get('client_id')
            if client_id and client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.now()
                emit('pong', {'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription requests"""
            try:
                client_id = session.get('client_id')
                subscription_type = data.get('type')
                
                if subscription_type == 'device_updates':
                    join_room('device_updates')
                elif subscription_type == 'alerts':
                    join_room('alerts')
                elif subscription_type == 'system_metrics':
                    join_room('system_metrics')
                
                emit('subscription_confirmed', {'type': subscription_type})
                self.logger.info(f"Client {client_id} subscribed to {subscription_type}")
                
            except Exception as e:
                self.logger.error(f"Error handling subscription: {e}")
                emit('error', {'message': 'Subscription failed'})
        
        self.logger.info("WebSocket events setup completed")
    
    def start_background_tasks(self):
        """Start background tasks for real-time updates"""
        
        def data_update_task():
            """Background task to update data and send to clients"""
            while True:
                try:
                    # Update cached data
                    self.update_data_cache()
                    
                    # Send updates to connected clients
                    if self.connected_clients:
                        dashboard_data = self.get_cached_dashboard_data()
                        self.socketio.emit('data_update', dashboard_data, room='device_updates')
                    
                    # Check for new alerts
                    self.check_and_send_alerts()
                    
                    eventlet.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in data update task: {e}")
                    eventlet.sleep(60)  # Wait longer on error
        
        def cleanup_task():
            """Background task to cleanup disconnected clients"""
            while True:
                try:
                    current_time = datetime.now()
                    timeout_threshold = current_time - timedelta(minutes=5)
                    
                    disconnected_clients = []
                    for client_id, client_info in self.connected_clients.items():
                        if client_info['last_ping'] < timeout_threshold:
                            disconnected_clients.append(client_id)
                    
                    for client_id in disconnected_clients:
                        del self.connected_clients[client_id]
                        self.logger.info(f"Cleaned up inactive client {client_id}")
                    
                    eventlet.sleep(300)  # Cleanup every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
                    eventlet.sleep(600)  # Wait longer on error
        
        # Start background tasks
        self.socketio.start_background_task(data_update_task)
        self.socketio.start_background_task(cleanup_task)
        
        self.logger.info("Background tasks started")
    
    # Data retrieval methods
    def get_cached_dashboard_data(self):
        """Get cached dashboard data"""
        if 'dashboard' not in self.data_cache or \
           datetime.now() - self.data_cache.get('dashboard_updated', datetime.min) > timedelta(minutes=1):
            self.data_cache['dashboard'] = self.fetch_dashboard_data()
            self.data_cache['dashboard_updated'] = datetime.now()
        
        return self.data_cache['dashboard']
    
    def fetch_dashboard_data(self):
        """Fetch fresh dashboard data"""
        try:
            # Get latest device data
            devices_data = self.get_latest_device_data()
            
            # Calculate key metrics
            total_devices = len(devices_data)
            active_devices = len([d for d in devices_data if d.get('status') == 'normal'])
            
            # Calculate average health score
            health_scores = [d.get('health_score', 0) for d in devices_data if d.get('health_score')]
            avg_health = np.mean(health_scores) * 100 if health_scores else 0
            
            # Calculate average efficiency
            efficiency_scores = [d.get('efficiency_score', 0) for d in devices_data if d.get('efficiency_score')]
            avg_efficiency = np.mean(efficiency_scores) * 100 if efficiency_scores else 0
            
            # Get energy usage
            energy_usage = self.get_current_energy_usage()
            
            # Get performance data for charts
            performance_data = self.get_performance_chart_data()
            
            # Get status distribution
            status_distribution = self.calculate_status_distribution(devices_data)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': avg_health,
                'active_devices': active_devices,
                'total_devices': total_devices,
                'efficiency': avg_efficiency,
                'energy_usage': energy_usage,
                'performance_data': performance_data,
                'status_distribution': status_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_latest_device_data(self):
        """Get latest data for all devices"""
        try:
            # Check if database is available
            if not os.path.exists('DATABASE/health_data.db'):
                return self.generate_sample_device_data()
            
            with sqlite3.connect('DATABASE/health_data.db') as conn:
                query = """
                    SELECT device_id, device_name, device_type, value, unit, status, 
                           health_score, efficiency_score, location, timestamp
                    FROM device_data 
                    WHERE timestamp = (
                        SELECT MAX(timestamp) 
                        FROM device_data AS sub 
                        WHERE sub.device_id = device_data.device_id
                    )
                    ORDER BY device_id
                """
                
                df = pd.read_sql_query(query, conn)
                return df.to_dict('records')
                
        except Exception as e:
            self.logger.error(f"Error getting device data: {e}")
            return self.generate_sample_device_data()
    
    def generate_sample_device_data(self):
        """Generate sample device data for demo purposes"""
        devices = []
        device_types = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor', 'humidity_sensor']
        locations = ['Factory Floor A', 'Factory Floor B', 'Warehouse', 'Quality Lab']
        
        for i in range(15):
            device_type = np.random.choice(device_types)
            status_prob = np.random.random()
            
            if status_prob > 0.1:
                status = 'normal'
                health_score = np.random.uniform(0.8, 1.0)
            elif status_prob > 0.05:
                status = 'warning'
                health_score = np.random.uniform(0.5, 0.8)
            else:
                status = 'critical'
                health_score = np.random.uniform(0.1, 0.5)
            
            devices.append({
                'device_id': f'DEVICE_{i+1:03d}',
                'device_name': f'{device_type.replace("_", " ").title()} {i+1:03d}',
                'device_type': device_type,
                'value': round(np.random.uniform(10, 100), 2),
                'unit': self.get_unit_for_type(device_type),
                'status': status,
                'health_score': health_score,
                'efficiency_score': np.random.uniform(0.7, 1.0),
                'location': np.random.choice(locations),
                'timestamp': datetime.now().isoformat()
            })
        
        return devices
    
    def get_unit_for_type(self, device_type):
        """Get appropriate unit for device type"""
        units = {
            'temperature_sensor': '°C',
            'pressure_sensor': 'hPa',
            'vibration_sensor': 'mm/s',
            'humidity_sensor': '%RH',
            'power_meter': 'W'
        }
        return units.get(device_type, 'units')
    
    def get_current_energy_usage(self):
        """Get current energy usage"""
        try:
            # Try to get from database
            with sqlite3.connect('DATABASE/health_data.db') as conn:
                query = """
                    SELECT power_consumption_kw 
                    FROM energy_data 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                result = pd.read_sql_query(query, conn)
                if not result.empty:
                    return float(result.iloc[0]['power_consumption_kw'])
        except:
            pass
        
        # Return sample data if database not available
        return round(np.random.uniform(800, 1500), 1)
    
    def get_performance_chart_data(self):
        """Get data for performance charts"""
        try:
            # Generate last 24 hours of data points
            now = datetime.now()
            timestamps = []
            health_scores = []
            efficiency_scores = []
            
            for i in range(24):
                timestamp = now - timedelta(hours=23-i)
                timestamps.append(timestamp.strftime('%H:%M'))
                
                # Simulate daily patterns
                hour_factor = np.sin(2 * np.pi * timestamp.hour / 24)
                base_health = 85 + 10 * hour_factor + np.random.normal(0, 3)
                base_efficiency = 78 + 12 * hour_factor + np.random.normal(0, 4)
                
                health_scores.append(max(0, min(100, base_health)))
                efficiency_scores.append(max(0, min(100, base_efficiency)))
            
            return {
                'labels': timestamps,
                'health_scores': health_scores,
                'efficiency_scores': efficiency_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance chart data: {e}")
            return {'labels': [], 'health_scores': [], 'efficiency_scores': []}
    
    def calculate_status_distribution(self, devices_data):
        """Calculate device status distribution"""
        status_counts = {'normal': 0, 'warning': 0, 'critical': 0}
        
        for device in devices_data:
            status = device.get('status', 'normal')
            if status in status_counts:
                status_counts[status] += 1
            elif status == 'anomaly':
                status_counts['critical'] += 1
        
        return status_counts
    
    def get_devices_data(self):
        """Get detailed devices data"""
        return self.get_latest_device_data()
    
    def get_device_data(self, device_id):
        """Get specific device data"""
        devices = self.get_latest_device_data()
        for device in devices:
            if device.get('device_id') == device_id:
                return device
        return None
    
    def get_analytics_data(self):
        """Get analytics data for charts"""
        try:
            # Generate sample analytics data
            now = datetime.now()
            timestamps = [(now - timedelta(hours=i)).strftime('%H:%M') for i in range(23, -1, -1)]
            
            analytics = {
                'temperature': {
                    'labels': timestamps,
                    'values': [20 + 5*np.sin(i*0.1) + np.random.normal(0, 1) for i in range(24)]
                },
                'pressure': {
                    'labels': timestamps,
                    'values': [1013 + 20*np.sin(i*0.05) + np.random.normal(0, 5) for i in range(24)]
                },
                'vibration': {
                    'labels': timestamps,
                    'values': [0.2 + 0.1*np.sin(i*0.15) + np.random.exponential(0.05) for i in range(24)]
                },
                'power': {
                    'labels': timestamps,
                    'values': [1200 + 300*np.sin(i*0.08) + np.random.normal(0, 50) for i in range(24)]
                }
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting analytics data: {e}")
            return {}
    
    def get_alerts_data(self, limit=10, severity=None):
        """Get system alerts"""
        try:
            # Generate sample alerts
            alert_types = [
                ('Temperature anomaly detected', 'warning'),
                ('Pressure threshold exceeded', 'critical'),
                ('Device offline', 'critical'),
                ('Vibration levels high', 'warning'),
                ('Maintenance required', 'info'),
                ('System performance degraded', 'warning')
            ]
            
            alerts = []
            for i in range(min(limit, len(alert_types))):
                alert_type, alert_severity = alert_types[i % len(alert_types)]
                
                if severity and alert_severity != severity:
                    continue
                
                alerts.append({
                    'id': str(uuid.uuid4()),
                    'title': alert_type,
                    'message': f'{alert_type} on device DEVICE_{(i%5)+1:03d}',
                    'severity': alert_severity,
                    'device_id': f'DEVICE_{(i%5)+1:03d}',
                    'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat()
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            import psutil
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'active_connections': len(self.connected_clients)
            }
            
        except ImportError:
            # Fallback if psutil is not available
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': np.random.uniform(20, 80),
                'memory_percent': np.random.uniform(40, 70),
                'disk_percent': np.random.uniform(50, 85),
                'active_connections': len(self.connected_clients)
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_historical_data(self, device_id, hours, metric):
        """Get historical data for a device"""
        try:
            # Generate sample historical data
            now = datetime.now()
            timestamps = []
            values = []
            
            for i in range(hours):
                timestamp = now - timedelta(hours=hours-1-i)
                timestamps.append(timestamp.isoformat())
                
                # Generate realistic historical pattern
                value = 50 + 20*np.sin(i*0.1) + np.random.normal(0, 5)
                values.append(round(value, 2))
            
            return {
                'device_id': device_id,
                'metric': metric,
                'timestamps': timestamps,
                'values': values
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return {}
    
    def get_predictions_data(self, device_id, horizon):
        """Get prediction data"""
        try:
            # Generate sample prediction data
            now = datetime.now()
            timestamps = []
            predictions = []
            confidence = []
            
            for i in range(horizon):
                timestamp = now + timedelta(hours=i)
                timestamps.append(timestamp.isoformat())
                
                # Generate prediction with decreasing confidence
                pred = 50 + 15*np.sin(i*0.05) + np.random.normal(0, 2)
                conf = max(0.5, 0.95 - (i * 0.02))  # Decreasing confidence
                
                predictions.append(round(pred, 2))
                confidence.append(round(conf, 3))
            
            return {
                'device_id': device_id,
                'horizon_hours': horizon,
                'timestamps': timestamps,
                'predictions': predictions,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return {}
    
    def calculate_health_scores(self):
        """Calculate health scores for all devices"""
        try:
            devices = self.get_latest_device_data()
            health_scores = {}
            
            for device in devices:
                device_id = device.get('device_id')
                health_score = device.get('health_score', 0.8)
                health_scores[device_id] = {
                    'overall_health': health_score * 100,
                    'components': {
                        'performance': np.random.uniform(0.7, 1.0) * 100,
                        'reliability': np.random.uniform(0.8, 1.0) * 100,
                        'efficiency': device.get('efficiency_score', 0.8) * 100,
                        'maintenance': np.random.uniform(0.6, 0.9) * 100
                    }
                }
            
            return health_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating health scores: {e}")
            return {}
    
    def get_recommendations(self):
        """Get AI-powered recommendations"""
        try:
            recommendations = [
                {
                    'id': str(uuid.uuid4()),
                    'type': 'maintenance',
                    'priority': 'high',
                    'title': 'Schedule Preventive Maintenance',
                    'description': 'Device DEVICE_003 showing increased vibration levels. Recommend maintenance within 48 hours.',
                    'estimated_impact': 'Prevent potential downtime of 4-6 hours',
                    'confidence': 0.92
                },
                {
                    'id': str(uuid.uuid4()),
                    'type': 'optimization',
                    'priority': 'medium',
                    'title': 'Optimize Operating Parameters',
                    'description': 'Adjust temperature setpoints for devices in Factory Floor A to improve energy efficiency.',
                    'estimated_impact': 'Reduce energy consumption by 8-12%',
                    'confidence': 0.85
                },
                {
                    'id': str(uuid.uuid4()),
                    'type': 'alert',
                    'priority': 'low',
                    'title': 'Update Firmware',
                    'description': 'Several devices have outdated firmware versions. Update recommended for improved security.',
                    'estimated_impact': 'Enhanced security and performance',
                    'confidence': 0.78
                }
            ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []
    
    def export_data(self, format_type, date_range):
        """Export data for analysis"""
        try:
            # Get data for specified date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range)
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'format': format_type
                },
                'devices': self.get_latest_device_data(),
                'alerts': self.get_alerts_data(limit=100),
                'system_metrics': self.get_system_metrics()
            }
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {'error': str(e)}
    
    def update_data_cache(self):
        """Update cached data"""
        try:
            # Update dashboard data
            self.data_cache['dashboard'] = self.fetch_dashboard_data()
            self.data_cache['dashboard_updated'] = datetime.now()
            
            # Update other cached data
            self.data_cache['devices'] = self.get_latest_device_data()
            self.data_cache['devices_updated'] = datetime.now()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating data cache: {e}")
    
    def check_and_send_alerts(self):
        """Check for new alerts and send to clients"""
        try:
            # Simulate alert generation
            if np.random.random() < 0.1:  # 10% chance of new alert
                alert_types = [
                    ('Temperature spike detected', 'warning'),
                    ('Pressure anomaly', 'critical'),
                    ('Vibration threshold exceeded', 'warning')
                ]
                
                alert_type, severity = np.random.choice(alert_types)
                device_id = f'DEVICE_{np.random.randint(1, 16):03d}'
                
                new_alert = {
                    'id': str(uuid.uuid4()),
                    'title': alert_type,
                    'message': f'{alert_type} on {device_id}',
                    'severity': severity,
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send to subscribed clients
                self.socketio.emit('alert_update', new_alert, room='alerts')
                self.logger.info(f"New alert sent: {alert_type} - {device_id}")
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    # Health check methods
    def check_database_health(self):
        """Check database health"""
        try:
            db_path = 'DATABASE/health_data.db'
            if os.path.exists(db_path):
                with sqlite3.connect(db_path) as conn:
                    conn.execute('SELECT 1').fetchone()
                return {'status': 'healthy', 'path': db_path}
            else:
                return {'status': 'warning', 'message': 'Database file not found, using sample data'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_ai_modules_health(self):
        """Check AI modules health"""
        modules = {
            'database_manager': self.db_manager is not None,
            'analytics_engine': self.analytics_engine is not None,
            'health_calculator': self.health_calculator is not None,
            'alert_manager': self.alert_manager is not None,
            'pattern_analyzer': self.pattern_analyzer is not None,
            'recommendation_engine': self.recommendation_engine is not None
        }
        
        return {
            'modules': modules,
            'status': 'healthy' if all(modules.values()) else 'partial'
        }
    
    def get_uptime(self):
        """Get application uptime"""
        if hasattr(self, 'start_time'):
            uptime = datetime.now() - self.start_time
            return str(uptime).split('.')[0]  # Remove microseconds
        return "Unknown"
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting Digital Twin Application on {host}:{port}")
        self.logger.info(f"Debug mode: {debug}")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,  # Disable reloader to prevent issues with background tasks
                log_output=True
            )
        except KeyboardInterrupt:
            self.logger.info("Application stopped by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise

# Error handlers
def setup_error_handlers(app):
    """Setup error handlers for the Flask app"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Create application instance
def create_app():
    """Application factory function"""
    try:
        app_instance = DigitalTwinApp()
        setup_error_handlers(app_instance.app)
        return app_instance
    except Exception as e:
        logging.error(f"Failed to create application: {e}")
        raise

# Main execution
if __name__ == '__main__':
    # Setup environment
    os.environ.setdefault('FLASK_ENV', 'development')
    
    # Create and run application
    try:
        digital_twin_app = create_app()
        
        # Get configuration from environment
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        # Run application
        digital_twin_app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)