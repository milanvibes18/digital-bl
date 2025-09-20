import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append('.')
# Corrected AppConfig to point to the correct database path
class AppConfig:
    class Database:
        primary_path = 'DATABASE/health_data.db' # <--- CORRECTED PATH
    database = Database()
config = AppConfig()

class UnifiedDataGenerator:
    """
    Comprehensive data generator for Digital Twin applications.
    Generates realistic industrial IoT data with patterns, anomalies, and relationships.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 seed: int = 42):
        
        self.db_path = db_path or config.database.primary_path
        self.seed = seed
        self.logger = self._setup_logging()
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Device configurations
        self.device_types = {
            'temperature_sensor': {
                'normal_range': (15, 35),
                'critical_range': (0, 60),
                'noise_factor': 0.5,
                'seasonal_amplitude': 5,
                'daily_amplitude': 3
            },
            'pressure_sensor': {
                'normal_range': (900, 1100),
                'critical_range': (800, 1200),
                'noise_factor': 2.0,
                'seasonal_amplitude': 10,
                'daily_amplitude': 5
            },
            'vibration_sensor': {
                'normal_range': (0.1, 0.3),
                'critical_range': (0, 1.0),
                'noise_factor': 0.02,
                'seasonal_amplitude': 0.05,
                'daily_amplitude': 0.1
            },
            'humidity_sensor': {
                'normal_range': (40, 70),
                'critical_range': (10, 90),
                'noise_factor': 1.0,
                'seasonal_amplitude': 15,
                'daily_amplitude': 8
            },
            'power_meter': {
                'normal_range': (1000, 5000),
                'critical_range': (0, 10000),
                'noise_factor': 50,
                'seasonal_amplitude': 500,
                'daily_amplitude': 800
            }
        }
        
        # Location data for spatial patterns
        self.locations = [
            {'name': 'Factory Floor A', 'x': 10, 'y': 20, 'zone': 'production'},
            {'name': 'Factory Floor B', 'x': 50, 'y': 20, 'zone': 'production'},
            {'name': 'Warehouse Section 1', 'x': 10, 'y': 80, 'zone': 'storage'},
            {'name': 'Warehouse Section 2', 'x': 50, 'y': 80, 'zone': 'storage'},
            {'name': 'Quality Control Lab', 'x': 80, 'y': 50, 'zone': 'quality'},
            {'name': 'Maintenance Workshop', 'x': 90, 'y': 10, 'zone': 'maintenance'},
            {'name': 'Office Building', 'x': 20, 'y': 90, 'zone': 'administrative'},
            {'name': 'Server Room', 'x': 85, 'y': 85, 'zone': 'it_infrastructure'}
        ]
        
        self.anomaly_probability = 0.05  # 5% chance of anomalies
        self.maintenance_probability = 0.02  # 2% chance of maintenance events
        
    def _setup_logging(self):
        """Setup logging for data generator."""
        logger = logging.getLogger('UnifiedDataGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            Path('LOGS').mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler('LOGS/digital_twin_app.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_device_data(self, 
                            device_count: int = 20,
                            days_of_data: int = 30,
                            interval_minutes: int = 5) -> pd.DataFrame:
        """
        Generate comprehensive device sensor data.
        
        Args:
            device_count: Number of devices to simulate
            days_of_data: Number of days to generate data for
            interval_minutes: Data collection interval in minutes
            
        Returns:
            DataFrame with generated device data
        """
        try:
            self.logger.info(f"Generating device data for {device_count} devices over {days_of_data} days")
            
            # Calculate time points
            start_time = datetime.now() - timedelta(days=days_of_data)
            end_time = datetime.now()
            time_points = pd.date_range(start_time, end_time, freq=f'{interval_minutes}min')
            
            # Generate devices
            devices = self._generate_device_metadata(device_count, start_time)
            
            # Generate data for each device and time point
            all_data = []
            
            for device in devices:
                device_data = self._generate_device_time_series(device, time_points)
                all_data.extend(device_data)
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Add calculated fields
            df = self._add_calculated_fields(df)
            
            # Add anomalies and maintenance events
            df = self._add_anomalies(df)
            df = self._add_maintenance_events(df)
            
            self.logger.info(f"Generated {len(df):,} data records")
            return df
            
        except Exception as e:
            self.logger.error(f"Device data generation error: {e}")
            raise
    
    def _generate_device_metadata(self, count: int, start_time: datetime) -> List[Dict]:
        """Generate metadata for devices."""
        devices = []
        
        for i in range(count):
            device_type = np.random.choice(list(self.device_types.keys()))
            location = random.choice(self.locations)
            
            device = {
                'device_id': f"DEVICE_{i+1:03d}",
                'device_name': f"{device_type.replace('_', ' ').title()} {i+1:03d}",
                'device_type': device_type,
                'location': location['name'],
                'location_x': location['x'],
                'location_y': location['y'],
                'zone': location['zone'],
                'installation_date': start_time + timedelta(days=random.randint(0, 30)),
                'manufacturer': random.choice(['Siemens', 'ABB', 'Schneider', 'Rockwell', 'Honeywell']),
                'model': f"Model-{random.randint(1000, 9999)}",
                'firmware_version': f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                'status': 'active',
                'last_maintenance': start_time + timedelta(days=random.randint(0, 60)),
                'config': self.device_types[device_type]
            }
            devices.append(device)
        
        return devices
    
    def _generate_device_time_series(self, device: Dict, time_points: pd.DatetimeIndex) -> List[Dict]:
        """Generate time series data for a single device."""
        device_data = []
        config = device['config']
        
        for timestamp in time_points:
            # Base value with seasonal and daily patterns
            base_value = self._calculate_base_value(timestamp, config)
            
            # Add noise
            noise = np.random.normal(0, config['noise_factor'])
            
            # Calculate final value
            value = base_value + noise
            
            # Ensure value is within critical range
            value = max(config['critical_range'][0], min(config['critical_range'][1], value))
            
            # Generate additional metrics
            record = {
                'timestamp': timestamp,
                'device_id': device['device_id'],
                'device_name': device['device_name'],
                'device_type': device['device_type'],
                'location': device['location'],
                'location_x': device['location_x'],
                'location_y': device['location_y'],
                'zone': device['zone'],
                'value': round(value, 3),
                'unit': self._get_unit_for_device_type(device['device_type']),
                'status': 'normal',
                'quality': random.uniform(0.95, 1.0),  # Data quality score
                'signal_strength': random.randint(80, 100),
                'battery_level': random.uniform(0.7, 1.0) if 'wireless' in device.get('connection_type', '') else None
            }
            
            # Add device-specific metrics
            if device['device_type'] == 'temperature_sensor':
                # Generate and add humidity first
                humidity = random.uniform(40, 70)
                record.update({
                    'humidity': humidity,
                    'heat_index': self._calculate_heat_index(value, humidity)
                })
            elif device['device_type'] == 'vibration_sensor':
                record.update({
                    'frequency_hz': random.uniform(10, 100),
                    'amplitude_mm': value,
                    'rms_velocity': random.uniform(0.5, 5.0)
                })
            elif device['device_type'] == 'power_meter':
                record.update({
                    'voltage': random.uniform(220, 240),
                    'current': value / random.uniform(220, 240),
                    'power_factor': random.uniform(0.8, 1.0),
                    'energy_consumed': value * random.uniform(0.5, 2.0)
                })
            
            device_data.append(record)
        
        return device_data
    
    def _calculate_base_value(self, timestamp: pd.Timestamp, config: Dict) -> float:
        """Calculate base value with seasonal and daily patterns."""
        # Get normal range
        normal_min, normal_max = config['normal_range']
        base_value = (normal_min + normal_max) / 2
        
        # Add seasonal variation (yearly cycle)
        day_of_year = timestamp.dayofyear
        seasonal_factor = math.sin(2 * math.pi * day_of_year / 365.25)
        seasonal_adjustment = seasonal_factor * config['seasonal_amplitude']
        
        # Add daily variation
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        daily_factor = math.sin(2 * math.pi * hour_of_day / 24)
        daily_adjustment = daily_factor * config['daily_amplitude']
        
        # Add weekly variation (lower activity on weekends)
        weekly_factor = 1.0
        if timestamp.weekday() >= 5:  # Weekend
            weekly_factor = 0.7
        
        return base_value + seasonal_adjustment + daily_adjustment * weekly_factor
    
    def _get_unit_for_device_type(self, device_type: str) -> str:
        """Get measurement unit for device type."""
        unit_mapping = {
            'temperature_sensor': '°C',
            'pressure_sensor': 'hPa',
            'vibration_sensor': 'mm/s',
            'humidity_sensor': '%RH',
            'power_meter': 'W'
        }
        return unit_mapping.get(device_type, 'units')
    
    def _calculate_heat_index(self, temperature: float, humidity: float) -> float:
        """Calculate heat index from temperature and humidity."""
        if temperature < 26.7:  # Below 80°F
            return temperature
        
        # Simplified heat index calculation
        hi = (
            -42.379 + 
            2.04901523 * temperature +
            10.14333127 * humidity -
            0.22475541 * temperature * humidity -
            6.83783e-3 * temperature**2 -
            5.481717e-2 * humidity**2 +
            1.22874e-3 * temperature**2 * humidity +
            8.5282e-4 * temperature * humidity**2 -
            1.99e-6 * temperature**2 * humidity**2
        )
        return round(hi, 2)
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields to the dataset."""
        df = df.copy()
        
        # Health score based on value relative to normal range
        def calculate_health_score(row):
            device_type = row['device_type']
            value = row['value']
            
            if device_type in self.device_types:
                normal_range = self.device_types[device_type]['normal_range']
                normal_min, normal_max = normal_range
                
                if normal_min <= value <= normal_max:
                    return random.uniform(0.8, 1.0)  # Good health
                else:
                    # Degraded health based on how far from normal range
                    distance = min(abs(value - normal_min), abs(value - normal_max))
                    max_distance = max(abs(normal_min), abs(normal_max))
                    # Handle division by zero for ranges like (0,0)
                    if max_distance == 0:
                        return 0.1
                    return max(0.1, 1.0 - (distance / max_distance))
            
            return random.uniform(0.5, 1.0)
        
        df['health_score'] = df.apply(calculate_health_score, axis=1)
        
        # Efficiency score (inverse of deviation from optimal)
        def calculate_efficiency_score(row):
            health_score = row['health_score']
            quality = row['quality']
            
            # Base efficiency on health and quality
            efficiency = (health_score * 0.7) + (quality * 0.3)
            
            # Add some randomness
            efficiency += random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, efficiency))
        
        df['efficiency_score'] = df.apply(calculate_efficiency_score, axis=1)
        
        # Operating hours (cumulative)
        df = df.sort_values(['device_id', 'timestamp'])
        df['operating_hours'] = df.groupby('device_id').cumcount() * 0.083  # 5 minutes = 0.083 hours
        
        # Days since maintenance
        df['days_since_maintenance'] = (
            df['timestamp'] - df.groupby('device_id')['timestamp'].transform('min')
        ).dt.days
        
        return df
    
    def _add_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomalies to the dataset."""
        df = df.copy()
        anomaly_count = 0
        
        # Optimized anomaly generation to avoid iloc on a large DataFrame
        anomalous_indices = random.sample(
            range(len(df)), 
            int(len(df) * self.anomaly_probability)
        )

        for idx in anomalous_indices:
            device_type = df.loc[idx, 'device_type']
            
            if device_type in self.device_types:
                config = self.device_types[device_type]
                critical_range = config['critical_range']
                
                # Generate anomalous value
                if random.random() < 0.5:
                    # High anomaly
                    anomalous_value = random.uniform(
                        critical_range[1] * 0.8,
                        critical_range[1]
                    )
                else:
                    # Low anomaly
                    anomalous_value = random.uniform(
                        critical_range[0],
                        critical_range[0] + (critical_range[1] - critical_range[0]) * 0.2
                    )
                
                df.loc[idx, 'value'] = round(anomalous_value, 3)
                df.loc[idx, 'status'] = 'anomaly'
                df.loc[idx, 'health_score'] *= 0.3  # Reduce health score
                df.loc[idx, 'efficiency_score'] *= 0.4  # Reduce efficiency
                df.loc[idx, 'quality'] *= 0.7  # Reduce data quality
                
                anomaly_count += 1
        
        self.logger.info(f"Added {anomaly_count} anomalies to dataset")
        return df
    
    def _add_maintenance_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance events to the dataset."""
        df = df.copy()
        maintenance_count = 0
        
        maintenance_indices = random.sample(
            range(len(df)), 
            int(len(df) * self.maintenance_probability)
        )
        
        for idx in maintenance_indices:
            df.loc[idx, 'status'] = 'maintenance'
            df.loc[idx, 'value'] = 0.0  # Device offline during maintenance
            df.loc[idx, 'health_score'] = 1.0  # Perfect after maintenance
            df.loc[idx, 'efficiency_score'] = 0.0  # No efficiency during maintenance
            df.loc[idx, 'quality'] = 1.0
            
            maintenance_count += 1
        
        self.logger.info(f"Added {maintenance_count} maintenance events to dataset")
        return df
    
    def generate_system_metrics(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate system-level performance metrics."""
        try:
            self.logger.info(f"Generating system metrics for {days_of_data} days")
            
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='1H')
            
            system_data = []
            
            for timestamp in time_points:
                # Base system load with business hours pattern
                hour = timestamp.hour
                is_weekend = timestamp.weekday() >= 5
                
                # Business hours have higher load
                if 8 <= hour <= 18 and not is_weekend:
                    base_load = random.uniform(60, 90)
                elif 6 <= hour <= 22:
                    base_load = random.uniform(30, 70)
                else:
                    base_load = random.uniform(10, 40)
                
                # Add seasonal variation
                seasonal_factor = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
                load_adjustment = seasonal_factor * 10
                
                system_load = max(0, min(100, base_load + load_adjustment))
                
                record = {
                    'timestamp': timestamp,
                    'metric_type': 'system_performance',
                    'cpu_usage_percent': round(system_load + random.uniform(-10, 10), 2),
                    'memory_usage_percent': round(system_load * 0.8 + random.uniform(-5, 15), 2),
                    'disk_usage_percent': round(random.uniform(45, 75), 2),
                    'network_io_mbps': round(system_load * 0.5 + random.uniform(0, 20), 2),
                    'active_connections': int(system_load * 2 + random.randint(0, 50)),
                    'response_time_ms': round(max(50, 100 + (system_load - 50) * 2 + random.uniform(-20, 50)), 2),
                    'error_rate_percent': round(max(0, (system_load - 80) * 0.1 + random.uniform(0, 0.5)), 3),
                    'throughput_rps': round(max(10, 1000 - (system_load - 50) * 5 + random.uniform(-100, 200)), 2),
                    'availability_percent': round(max(95, 100 - (system_load - 70) * 0.1 + random.uniform(-1, 1)), 3)
                }
                
                system_data.append(record)
            
            df = pd.DataFrame(system_data)
            self.logger.info(f"Generated {len(df):,} system metric records")
            return df
            
        except Exception as e:
            self.logger.error(f"System metrics generation error: {e}")
            raise
    
    def generate_energy_data(self, days_of_data: int = 30) -> pd.DataFrame:
        """Generate energy consumption and efficiency data."""
        try:
            self.logger.info(f"Generating energy data for {days_of_data} days")
            
            start_time = datetime.now() - timedelta(days=days_of_data)
            time_points = pd.date_range(start_time, datetime.now(), freq='15min')
            
            energy_data = []
            cumulative_energy = 0
            
            for timestamp in time_points:
                # Energy consumption pattern
                hour = timestamp.hour
                is_weekend = timestamp.weekday() >= 5
                
                # Base consumption pattern
                if 6 <= hour <= 22 and not is_weekend:
                    base_consumption = random.uniform(800, 1500)
                elif 22 <= hour or hour <= 6:
                    base_consumption = random.uniform(200, 600)
                else:  # Weekend
                    base_consumption = random.uniform(300, 800)
                
                # Add seasonal variation (heating/cooling)
                seasonal_factor = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
                seasonal_adjustment = seasonal_factor * 300
                
                total_consumption = max(0, base_consumption + seasonal_adjustment)
                cumulative_energy += total_consumption * 0.25  # 15 minutes = 0.25 hours
                
                record = {
                    'timestamp': timestamp,
                    'metric_type': 'energy',
                    'power_consumption_kw': round(total_consumption, 2),
                    'energy_consumed_kwh': round(cumulative_energy, 2),
                    'voltage_v': round(random.uniform(220, 240), 1),
                    'current_a': round(total_consumption / random.uniform(220, 240), 2),
                    'power_factor': round(random.uniform(0.8, 0.95), 3),
                    'frequency_hz': round(random.uniform(49.9, 50.1), 2),
                    'energy_cost_usd': round(cumulative_energy * 0.12, 2),  # $0.12 per kWh
                    'carbon_footprint_kg': round(cumulative_energy * 0.4, 2),  # 0.4 kg CO2 per kWh
                    'efficiency_percent': round(random.uniform(85, 95), 2),
                    'renewable_percent': round(random.uniform(20, 40), 2)
                }
                
                energy_data.append(record)
            
            df = pd.DataFrame(energy_data)
            self.logger.info(f"Generated {len(df):,} energy records")
            return df
            
        except Exception as e:
            self.logger.error(f"Energy data generation error: {e}")
            raise
    
    def save_to_database(self, dataframes: Dict[str, pd.DataFrame]):
        """Save generated data to database."""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                for table_name, df in dataframes.items():
                    self.logger.info(f"Saving {len(df):,} records to table '{table_name}'")
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            self.logger.info(f"All data saved to database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database save error: {e}")
            raise
    
    def generate_complete_dataset(self, 
                                 device_count: int = 20,
                                 days_of_data: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset with all data types."""
        try:
            self.logger.info(f"Starting complete dataset generation")
            
            datasets = {}
            
            # Generate device sensor data
            datasets['device_data'] = self.generate_device_data(
                device_count=device_count,
                days_of_data=days_of_data
            )
            
            # Generate system metrics
            datasets['system_metrics'] = self.generate_system_metrics(
                days_of_data=days_of_data
            )
            
            # Generate energy data
            datasets['energy_data'] = self.generate_energy_data(
                days_of_data=days_of_data
            )
            
            # Save to database
            self.save_to_database(datasets)
            
            # Generate summary statistics
            summary = self.generate_dataset_summary(datasets)
            self.logger.info(f"Dataset generation completed. Summary: {summary}")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Complete dataset generation error: {e}")
            raise
    
    def generate_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for the datasets."""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'tables': {}
        }
        
        for table_name, df in datasets.items():
            table_summary = {
                'record_count': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns and not df.empty else None,
                    'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns and not df.empty else None
                },
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            }
            
            # Add specific statistics for different table types
            if table_name == 'device_data':
                table_summary.update({
                    'unique_devices': df['device_id'].nunique() if 'device_id' in df.columns else 0,
                    'device_types': df['device_type'].unique().tolist() if 'device_type' in df.columns else [],
                    'anomaly_count': len(df[df['status'] == 'anomaly']) if 'status' in df.columns else 0,
                    'maintenance_count': len(df[df['status'] == 'maintenance']) if 'status' in df.columns else 0
                })
            
            summary['tables'][table_name] = table_summary
        
        return summary

def main():
    """Main function to generate and save data."""
    print("🏭 Digital Twin Data Generation System")
    print("=" * 50)
    
    # Initialize generator
    generator = UnifiedDataGenerator()
    
    # Generate complete dataset
    print("Generating comprehensive dataset...")
    datasets = generator.generate_complete_dataset(
        device_count=25,
        days_of_data=45
    )
    
    # Print summary
    print("\n📊 Generation Summary:")
    for table_name, df in datasets.items():
        print(f"  {table_name}: {len(df):,} records")
    
    print(f"\n✅ Data generation completed!")
    print(f"📁 Database saved to: {generator.db_path}")
    
    # Display sample data
    print("\n🔍 Sample Device Data:")
    if 'device_data' in datasets and not datasets['device_data'].empty:
        sample = datasets['device_data'].head(3)
        for col in ['timestamp', 'device_id', 'device_type', 'value', 'status', 'health_score']:
            if col in sample.columns:
                print(f"  {col}: {sample[col].tolist()}")



if __name__ == "__main__":
    main()