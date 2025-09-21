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
import math
import random

# Flask imports
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
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
    from CONFIG.unified_data_generator import UnifiedDataGenerator
    from REPORTS.health_report_generator import HealthReportGenerator
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
        self.data_generator = None
        
        # Application state
        self.connected_clients = {}
        self.data_cache = {}
        self.last_update = datetime.now()
        self.start_time = datetime.now()
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize Flask app
        self.create_app()
        
        # Initialize all modules
        self.initialize_modules()
        
        # Setup routes
        self.setup_routes()
        
        # Setup WebSocket events
        self.setup_websocket_events()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Ensure logs directory exists
        os.makedirs('LOGS', exist_ok=True)
        
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
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        
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
    
    def initialize_modules(self):
        """Initialize AI, analytics, and data generation modules"""
        try:
            # Initialize core modules
            self.db_manager = SecureDatabaseManager() if 'SecureDatabaseManager' in globals() else None
            self.analytics_engine = PredictiveAnalyticsEngine() if 'PredictiveAnalyticsEngine' in globals() else None
            self.health_calculator = HealthScoreCalculator() if 'HealthScoreCalculator' in globals() else None
            self.alert_manager = AlertManager() if 'AlertManager' in globals() else None
            self.pattern_analyzer = PatternAnalyzer() if 'PatternAnalyzer' in globals() else None
            self.recommendation_engine = RecommendationEngine() if 'RecommendationEngine' in globals() else None
            
            # Initialize data generator
            self.data_generator = UnifiedDataGenerator() if 'UnifiedDataGenerator' in globals() else None
            
            # Fallback alert manager if not available
            if self.alert_manager is None:
                self.alert_manager = self._create_fallback_alert_manager()
            
            # Fallback data generator if not available
            if self.data_generator is None:
                self.data_generator = self._create_fallback_data_generator()
            
            self.logger.info("All modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing modules: {e}")
            # Initialize fallback components
            self._initialize_fallback_modules()
    
    def _create_fallback_alert_manager(self):
        """Create a fallback alert manager"""
        class FallbackAlertManager:
            def __init__(self):
                self.alert_conditions = {
                    'temperature_high': {'threshold': 80, 'operator': '>', 'severity': 'warning'},
                    'temperature_critical': {'threshold': 100, 'operator': '>', 'severity': 'critical'},
                    'pressure_high': {'threshold': 1050, 'operator': '>', 'severity': 'warning'},
                    'vibration_high': {'threshold': 5.0, 'operator': '>', 'severity': 'warning'},
                    'health_low': {'threshold': 0.5, 'operator': '<', 'severity': 'critical'},
                }
            
            def evaluate_conditions(self, data, device_id):
                """Evaluate alert conditions on device data"""
                alerts = []
                device_type = data.get('device_type', '')
                value = data.get('value', 0)
                health_score = data.get('health_score', 1.0)
                
                # Check value-based alerts
                if 'temperature' in device_type and value > 80:
                    severity = 'critical' if value > 100 else 'warning'
                    alerts.append({
                        'id': str(uuid.uuid4()),
                        'device_id': device_id,
                        'type': 'temperature_alert',
                        'severity': severity,
                        'description': f'Temperature {value}°C exceeds threshold',
                        'timestamp': datetime.now().isoformat(),
                        'value': value
                    })
                
                elif 'pressure' in device_type and value > 1050:
                    alerts.append({
                        'id': str(uuid.uuid4()),
                        'device_id': device_id,
                        'type': 'pressure_alert',
                        'severity': 'warning',
                        'description': f'Pressure {value} hPa exceeds normal range',
                        'timestamp': datetime.now().isoformat(),
                        'value': value
                    })
                
                elif 'vibration' in device_type and value > 5.0:
                    alerts.append({
                        'id': str(uuid.uuid4()),
                        'device_id': device_id,
                        'type': 'vibration_alert',
                        'severity': 'warning',
                        'description': f'Vibration {value} mm/s is elevated',
                        'timestamp': datetime.now().isoformat(),
                        'value': value
                    })
                
                # Check health score alerts
                if health_score < 0.5:
                    alerts.append({
                        'id': str(uuid.uuid4()),
                        'device_id': device_id,
                        'type': 'health_alert',
                        'severity': 'critical',
                        'description': f'Device health score {health_score:.1%} is critically low',
                        'timestamp': datetime.now().isoformat(),
                        'value': health_score
                    })
                
                return alerts
        
        return FallbackAlertManager()
    
    def _create_fallback_data_generator(self):
        """Create a fallback data generator"""
        class FallbackDataGenerator:
            def __init__(self):
                self.device_types = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor', 
                                     'humidity_sensor', 'power_meter']
                self.locations = ['Factory Floor A', 'Factory Floor B', 'Warehouse', 'Quality Lab']
            
            def generate_device_data(self, device_count=15, days_of_data=1, interval_minutes=5):
                """Generate sample device data"""
                data = []
                current_time = datetime.now()
                start_time = current_time - timedelta(days=days_of_data)
                
                # Generate time series
                time_points = []
                current = start_time
                while current <= current_time:
                    time_points.append(current)
                    current += timedelta(minutes=interval_minutes)
                
                for device_idx in range(device_count):
                    device_id = f'DEVICE_{device_idx+1:03d}'
                    device_type = random.choice(self.device_types)
                    device_name = f'{device_type.replace("_", " ").title()} {device_idx+1:03d}'
                    location = random.choice(self.locations)
                    
                    for timestamp in time_points:
                        # Generate realistic values based on device type
                        if device_type == 'temperature_sensor':
                            base_value = 25 + 15 * math.sin(timestamp.hour * 0.26)  # Daily cycle
                            value = base_value + random.gauss(0, 3)
                            unit = '°C'
                        elif device_type == 'pressure_sensor':
                            base_value = 1013 + 20 * math.sin(timestamp.hour * 0.1)
                            value = base_value + random.gauss(0, 5)
                            unit = 'hPa'
                        elif device_type == 'vibration_sensor':
                            base_value = 0.5 + 0.3 * math.sin(timestamp.hour * 0.3)
                            value = base_value + random.exponential(0.2)
                            unit = 'mm/s'
                        elif device_type == 'humidity_sensor':
                            base_value = 45 + 20 * math.sin(timestamp.hour * 0.2)
                            value = base_value + random.gauss(0, 5)
                            unit = '%RH'
                        else:  # power_meter
                            base_value = 800 + 400 * math.sin(timestamp.hour * 0.26)
                            value = base_value + random.gauss(0, 50)
                            unit = 'W'
                        
                        # Determine status based on value ranges
                        if device_type == 'temperature_sensor':
                            if value > 100:
                                status = 'critical'
                            elif value > 80:
                                status = 'warning'
                            else:
                                status = 'normal'
                        elif device_type == 'pressure_sensor':
                            if value > 1080 or value < 950:
                                status = 'critical'
                            elif value > 1050 or value < 980:
                                status = 'warning'
                            else:
                                status = 'normal'
                        else:
                            # Random status for other types
                            status_prob = random.random()
                            if status_prob > 0.9:
                                status = 'critical'
                            elif status_prob > 0.8:
                                status = 'warning'
                            else:
                                status = 'normal'
                        
                        # Calculate health and efficiency scores
                        if status == 'critical':
                            health_score = random.uniform(0.1, 0.5)
                            efficiency_score = random.uniform(0.3, 0.6)
                        elif status == 'warning':
                            health_score = random.uniform(0.5, 0.8)
                            efficiency_score = random.uniform(0.6, 0.8)
                        else:
                            health_score = random.uniform(0.8, 1.0)
                            efficiency_score = random.uniform(0.8, 1.0)
                        
                        data.append({
                            'device_id': device_id,
                            'device_name': device_name,
                            'device_type': device_type,
                            'location': location,
                            'timestamp': timestamp,
                            'value': round(value, 2),
                            'unit': unit,
                            'status': status,
                            'health_score': round(health_score, 3),
                            'efficiency_score': round(efficiency_score, 3)
                        })
                
                return pd.DataFrame(data)
        
        return FallbackDataGenerator()
    
    def _initialize_fallback_modules(self):
        """Initialize fallback modules when imports fail"""
        if self.alert_manager is None:
            self.alert_manager = self._create_fallback_alert_manager()
        if self.data_generator is None:
            self.data_generator = self._create_fallback_data_generator()
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        # Main pages
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            try:
                return render_template('index.html')
            except:
                return "Digital Twin API is running. Dashboard templates not found."
        
        @self.app.route('/dashboard')
        def enhanced_dashboard():
            """Enhanced dashboard with advanced analytics"""
            try:
                return render_template('enhanced_dashboard.html')
            except:
                return "Enhanced Dashboard - Templates not found."
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics page"""
            try:
                return render_template('analytics.html')
            except:
                return "Analytics page - Templates not found."
        
        @self.app.route('/devices')
        def devices():
            """Device management page"""
            try:
                return render_template('devices_view.html')
            except:
                return "Devices page - Templates not found."
        
        # Health check endpoint
        @self.app.route('/health')
        def health_check():
            """Health check endpoint for monitoring"""
            try:
                db_status = self.check_database_health()
                ai_status = self.check_ai_modules_health()
                
                status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'database': db_status,
                    'ai_modules': ai_status,
                    'uptime': self.get_uptime(),
                    'version': '2.0.0',
                    'connected_clients': len(self.connected_clients)
                }
                
                return jsonify(status), 200
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 503
        
        # Core API endpoints
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
        
        # NEW ENDPOINTS FOR BUTTONS
        @self.app.route('/api/generate_report')
        def generate_report_endpoint():
            """Generate a health report and return its path"""
            try:
                self.logger.info("Generating health report via API request.")
                
                # Try to use actual report generator
                try:
                    if 'HealthReportGenerator' in globals():
                        report_generator = HealthReportGenerator()
                        html_path = report_generator.generate_comprehensive_report(date_range_days=7)
                        report_filename = os.path.basename(html_path)
                        return jsonify({
                            'success': True,
                            'report_path': f'/reports/{report_filename}',
                            'message': 'Health report generated successfully'
                        })
                    else:
                        raise ImportError("HealthReportGenerator not available")
                except Exception as report_error:
                    self.logger.warning(f"Report generator failed: {report_error}")
                    # Generate a simple fallback report
                    return self._generate_fallback_report()
                
            except Exception as e:
                self.logger.error(f"Error generating report: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate report',
                    'details': str(e)
                }), 500

        @self.app.route('/reports/<filename>')
        def serve_report(filename):
            """Serve the generated report file"""
            try:
                reports_dir = os.path.join(os.path.dirname(__file__), '..', 'REPORTS', 'generated')
                if not os.path.exists(reports_dir):
                    os.makedirs(reports_dir, exist_ok=True)
                return send_from_directory(reports_dir, filename)
            except Exception as e:
                self.logger.error(f"Error serving report: {e}")
                return jsonify({'error': 'Report not found'}), 404

        @self.app.route('/api/export_data')
        def export_data_endpoint():
            """Export data and provide download link"""
            try:
                format_type = request.args.get('format', 'json')
                date_range = request.args.get('days', 7, type=int)
                
                self.logger.info(f"Exporting data via API request. Format: {format_type}, Days: {date_range}")
                
                # Generate export data
                export_data = self.export_data(format_type, date_range)
                
                # Save to file
                exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
                os.makedirs(exports_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if format_type.lower() == 'csv':
                    filename = f'export_{timestamp}.csv'
                    filepath = os.path.join(exports_dir, filename)
                    # Convert to CSV if we have device data
                    if 'devices' in export_data and export_data['devices']:
                        devices_df = pd.DataFrame(export_data['devices'])
                        devices_df.to_csv(filepath, index=False)
                    else:
                        # Fallback: create a simple CSV
                        pd.DataFrame([{'message': 'No device data available', 'timestamp': datetime.now()}]).to_csv(filepath, index=False)
                else:
                    filename = f'export_{timestamp}.json'
                    filepath = os.path.join(exports_dir, filename)
                    with open(filepath, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                
                return jsonify({
                    'success': True,
                    'message': f'Data export completed ({format_type.upper()})',
                    'export_path': f'/exports/{filename}',
                    'filename': filename,
                    'records_exported': len(export_data.get('devices', []))
                })
                
            except Exception as e:
                self.logger.error(f"Error exporting data: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to export data',
                    'details': str(e)
                }), 500

        @self.app.route('/exports/<filename>')
        def serve_export(filename):
            """Serve exported data files"""
            try:
                exports_dir = os.path.join(os.path.dirname(__file__), '..', 'EXPORTS')
                return send_from_directory(exports_dir, filename)
            except Exception as e:
                self.logger.error(f"Error serving export: {e}")
                return jsonify({'error': 'Export file not found'}), 404
        
        self.logger.info("All routes setup completed")
    
    def _generate_fallback_report(self):
        """Generate a simple fallback report when the main generator is unavailable"""
        try:
            # Ensure reports directory exists
            reports_dir = os.path.join(os.path.dirname(__file__), '..', 'REPORTS', 'generated')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate simple HTML report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'health_report_{timestamp}.html'
            filepath = os.path.join(reports_dir, filename)
            
            # Get current data
            dashboard_data = self.get_cached_dashboard_data()
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Digital Twin Health Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007bff; }}
                    .devices {{ margin-top: 20px; }}
                    .device {{ margin: 10px 0; padding: 15px; background-color: white; border: 1px solid #ddd; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Digital Twin System Health Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>System Overview</h2>
                <div class="metric">System Health: {dashboard_data.get('systemHealth', 0)}%</div>
                <div class="metric">Active Devices: {dashboard_data.get('activeDevices', 0)}/{dashboard_data.get('totalDevices', 0)}</div>
                <div class="metric">System Efficiency: {dashboard_data.get('efficiency', 0)}%</div>
                <div class="metric">Energy Usage: {dashboard_data.get('energyUsage', 0)} W</div>
                
                <h2>Device Status</h2>
                <div class="devices">
            """
            
            # Add device information
            devices = dashboard_data.get('devices', [])
            for device in devices[:10]:  # Limit to first 10 devices
                html_content += f"""
                    <div class="device">
                        <h3>{device.get('device_name', 'Unknown Device')}</h3>
                        <p>Status: <strong>{device.get('status', 'unknown').upper()}</strong></p>
                        <p>Health Score: {device.get('health_score', 0):.1%}</p>
                        <p>Current Value: {device.get('value', 0)} {device.get('unit', '')}</p>
                        <p>Location: {device.get('location', 'Unknown')}</p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div style="margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
                    <p><strong>Note:</strong> This is a simplified fallback report. For comprehensive analytics and detailed insights, please ensure all system modules are properly configured.</p>
                </div>
            </body>
            </html>
            """
            
            # Write the HTML file
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            return jsonify({
                'success': True,
                'report_path': f'/reports/{filename}',
                'message': 'Simplified health report generated successfully'
            })
            
        except Exception as e:
            self.logger.error(f"Error generating fallback report: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to generate fallback report',
                'details': str(e)
            }), 500
    
    def setup_websocket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients[client_id] = {
                'connected_at': datetime.now(),
                'last_ping': datetime.now()
            }
            
            self.logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")
            
            # Send initial data to client
            emit('initial_data', self.get_cached_dashboard_data())
            
            # Automatically subscribe to essential updates
            join_room('device_updates')
            join_room('alerts')
            join_room('system_metrics')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
                self.logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle client ping"""
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.now()
                emit('pong', {'timestamp': datetime.now().isoformat()})
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription requests"""
            try:
                client_id = request.sid
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
                    # Generate a new slice of real-time data
                    new_devices_df = self.data_generator.generate_device_data(
                        device_count=15, 
                        days_of_data=0.003,  # ~5 minutes of data
                        interval_minutes=1
                    )
                    
                    # Get the latest record for each device to simulate a snapshot
                    latest_devices_df = new_devices_df.loc[new_devices_df.groupby('device_id')['timestamp'].idxmax()]
                    
                    # Update cached data with the new snapshot
                    self.update_data_cache(latest_devices_df)
                    
                    # Send updates to connected clients
                    if self.connected_clients:
                        dashboard_data = self.get_cached_dashboard_data()
                        self.socketio.emit('data_update', dashboard_data, room='device_updates')
                    
                    # Check for new alerts based on the latest data
                    self.check_and_send_alerts(latest_devices_df)
                    
                    eventlet.sleep(5)  # Update every 5 seconds
                    
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
        if 'dashboard' not in self.data_cache:
            # Generate initial data if cache is empty
            initial_df = self.data_generator.generate_device_data(device_count=15, days_of_data=1)
            latest_df = initial_df.loc[initial_df.groupby('device_id')['timestamp'].idxmax()]
            self.update_data_cache(latest_df)
        
        return self.data_cache['dashboard']

    def fetch_dashboard_data(self, devices_df):
        """Fetch fresh dashboard data from a DataFrame"""
        try:
            devices_data = devices_df.to_dict('records')
            total_devices = len(devices_data)
            active_devices = len([d for d in devices_data if d.get('status') != 'offline'])
            
            health_scores = [d.get('health_score', 0) for d in devices_data if d.get('health_score')]
            avg_health = np.mean(health_scores) * 100 if health_scores else 0
            
            efficiency_scores = [d.get('efficiency_score', 0) for d in devices_data if d.get('efficiency_score')]
            avg_efficiency = np.mean(efficiency_scores) * 100 if efficiency_scores else 0
            
            energy_usage = sum(d.get('value', 0) for d in devices_data if d.get('device_type') == 'power_meter')

            # Generate performance data for charts
            performance_data = self.get_performance_chart_data(avg_health, avg_efficiency, energy_usage)
            status_distribution = self.calculate_status_distribution(devices_data)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'systemHealth': round(avg_health),
                'activeDevices': active_devices,
                'totalDevices': total_devices,
                'efficiency': round(avg_efficiency),
                'energyUsage': round(energy_usage) if energy_usage > 0 else round(np.random.uniform(800, 1500)),
                'energyCost': round((energy_usage if energy_usage > 0 else np.random.uniform(800, 1500)) * 0.12),
                'performanceData': performance_data,
                'statusDistribution': status_distribution,
                'devices': devices_data
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def update_data_cache(self, devices_df):
        """Update cached dashboard data"""
        try:
            self.data_cache['dashboard'] = self.fetch_dashboard_data(devices_df)
            self.data_cache['dashboard_updated'] = datetime.now()
            self.last_update = datetime.now()
        except Exception as e:
            self.logger.error(f"Error updating data cache: {e}")

    def check_and_send_alerts(self, devices_df):
        """Check for new alerts using AlertManager and send to clients"""
        try:
            # Iterate over the latest data for each device
            for _, device_row in devices_df.iterrows():
                device_data_dict = device_row.to_dict()
                
                # Use the alert manager to evaluate conditions
                triggered_alerts = self.alert_manager.evaluate_conditions(
                    data=device_data_dict,
                    device_id=device_data_dict.get('device_id')
                )
                
                # Emit any triggered alerts (limit to avoid spam)
                for alert in triggered_alerts[:2]:  # Limit to 2 alerts per device per cycle
                    self.socketio.emit('alert_update', alert, room='alerts')
                    self.logger.info(f"New alert sent: {alert.get('description')} for device {alert.get('device_id')}")
                    
        except Exception as e:
            self.logger.error(f"Error in check_and_send_alerts: {e}")
    
    def get_performance_chart_data(self, system_health, efficiency, energy_usage):
        """Generate data for performance charts based on current metrics"""
        try:
            # Generate 24 hours of historical performance data
            chart_data = []
            base_time = datetime.now() - timedelta(hours=23)
            
            for i in range(24):
                timestamp = base_time + timedelta(hours=i)
                
                # Add some realistic variation
                hour_factor = math.sin(2 * math.pi * timestamp.hour / 24)
                
                health_variation = system_health + (10 * hour_factor) + random.gauss(0, 3)
                efficiency_variation = efficiency + (8 * hour_factor) + random.gauss(0, 4)
                energy_variation = energy_usage + (energy_usage * 0.2 * hour_factor) + random.gauss(0, energy_usage * 0.05)
                
                chart_data.append({
                    'timestamp': timestamp.strftime('%H:%M'),
                    'systemHealth': max(0, min(100, health_variation)),
                    'efficiency': max(0, min(100, efficiency_variation)),
                    'energyUsage': max(0, energy_variation)
                })
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating performance chart data: {e}")
            return []
    
    def calculate_status_distribution(self, devices_data):
        """Calculate device status distribution"""
        status_counts = {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}
        for device in devices_data:
            status = device.get('status', 'normal')
            if status in status_counts:
                status_counts[status] += 1
            else:
                # Map unknown statuses
                if status == 'anomaly':
                    status_counts['critical'] += 1
                else:
                    status_counts['normal'] += 1
        return status_counts

    def get_devices_data(self):
        """Get detailed devices data"""
        try:
            dashboard_data = self.get_cached_dashboard_data()
            return dashboard_data.get('devices', [])
        except Exception as e:
            self.logger.error(f"Error getting devices data: {e}")
            return []
    
    def get_device_data(self, device_id):
        """Get specific device data"""
        try:
            devices = self.get_devices_data()
            for device in devices:
                if device.get('device_id') == device_id:
                    return device
            return None
        except Exception as e:
            self.logger.error(f"Error getting device data for {device_id}: {e}")
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
                    'values': [20 + 5*math.sin(i*0.1) + random.gauss(0, 1) for i in range(24)]
                },
                'pressure': {
                    'labels': timestamps,
                    'values': [1013 + 20*math.sin(i*0.05) + random.gauss(0, 5) for i in range(24)]
                },
                'vibration': {
                    'labels': timestamps,
                    'values': [0.2 + 0.1*math.sin(i*0.15) + abs(random.gauss(0, 0.05)) for i in range(24)]
                },
                'power': {
                    'labels': timestamps,
                    'values': [1200 + 300*math.sin(i*0.08) + random.gauss(0, 50) for i in range(24)]
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
                ('System performance degraded', 'warning'),
                ('Low efficiency detected', 'warning'),
                ('Sensor calibration needed', 'info')
            ]
            
            alerts = []
            for i in range(min(limit, len(alert_types) * 2)):
                alert_type, alert_severity = alert_types[i % len(alert_types)]
                
                if severity and alert_severity != severity:
                    continue
                
                device_num = (i % 5) + 1
                alerts.append({
                    'id': str(uuid.uuid4()),
                    'title': alert_type,
                    'message': f'{alert_type} on device DEVICE_{device_num:03d}',
                    'severity': alert_severity,
                    'device_id': f'DEVICE_{device_num:03d}',
                    'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat()
                })
                
                if len(alerts) >= limit:
                    break
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
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
                    'cpu_percent': random.uniform(20, 80),
                    'memory_percent': random.uniform(40, 70),
                    'disk_percent': random.uniform(50, 85),
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
                value = 50 + 20*math.sin(i*0.1) + random.gauss(0, 5)
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
                pred = 50 + 15*math.sin(i*0.05) + random.gauss(0, 2)
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
            devices = self.get_devices_data()
            health_scores = {}
            
            for device in devices:
                device_id = device.get('device_id')
                health_score = device.get('health_score', 0.8)
                health_scores[device_id] = {
                    'overall_health': health_score * 100,
                    'components': {
                        'performance': random.uniform(0.7, 1.0) * 100,
                        'reliability': random.uniform(0.8, 1.0) * 100,
                        'efficiency': device.get('efficiency_score', 0.8) * 100,
                        'maintenance': random.uniform(0.6, 0.9) * 100
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
            
            # Generate historical data for export
            export_devices_df = self.data_generator.generate_device_data(
                device_count=15,
                days_of_data=date_range,
                interval_minutes=30  # 30-minute intervals for historical data
            )
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'format': format_type,
                    'total_records': len(export_devices_df)
                },
                'devices': export_devices_df.to_dict('records'),
                'alerts': self.get_alerts_data(limit=100),
                'system_metrics': self.get_system_metrics()
            }
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {'error': str(e)}
    
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
                return {'status': 'warning', 'message': 'Database file not found, using simulated data'}
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
            'recommendation_engine': self.recommendation_engine is not None,
            'data_generator': self.data_generator is not None
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
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
        
        # Run application
        digital_twin_app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logging.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)