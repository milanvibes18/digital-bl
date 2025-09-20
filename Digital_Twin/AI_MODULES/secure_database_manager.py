import sqlite3
import json
import logging
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class SecureDatabaseManager:
    """
    Advanced secure database manager with encryption, audit logging, and data integrity.
    """
    
    def __init__(self, db_path="DATABASE/secure_database.db", encryption_key_path="CONFIG/encryption.key", salt_key_path="CONFIG/salt.key"):
        self.db_path = db_path
        self.encryption_key_path = encryption_key_path
        self.salt_key_path = salt_key_path
        self.logger = self._setup_logging()
        
        # Ensure directories exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(encryption_key_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self.fernet = self._initialize_encryption()
        
        # Initialize database
        self._initialize_database()
        
    def _setup_logging(self):
        """Setup security-focused logging."""
        logger = logging.getLogger('SecureDatabaseManager')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("LOGS").mkdir(exist_ok=True)
        
        handler = logging.FileHandler('LOGS/digital_twin_security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _generate_key(self):
        """Generate a new encryption key."""
        return Fernet.generate_key()
    
    def _generate_salt(self):
        """Generate a new salt for key derivation."""
        return secrets.token_bytes(32)
    
    def _initialize_encryption(self):
        """Initialize encryption with key management."""
        try:
            # Load or generate encryption key
            if os.path.exists(self.encryption_key_path):
                with open(self.encryption_key_path, 'rb') as key_file:
                    key = key_file.read()
            else:
                key = self._generate_key()
                with open(self.encryption_key_path, 'wb') as key_file:
                    key_file.write(key)
                os.chmod(self.encryption_key_path, 0o600)  # Restrict permissions
                
            # Load or generate salt
            if os.path.exists(self.salt_key_path):
                with open(self.salt_key_path, 'rb') as salt_file:
                    salt = salt_file.read()
            else:
                salt = self._generate_salt()
                with open(self.salt_key_path, 'wb') as salt_file:
                    salt_file.write(salt)
                os.chmod(self.salt_key_path, 0o600)
                
            # Create Fernet instance
            fernet = Fernet(key)
            self.logger.info("Encryption initialized successfully")
            return fernet
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize the secure database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create health data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS health_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        encrypted_data TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create audit log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        user_id TEXT,
                        table_name TEXT,
                        record_id TEXT,
                        old_values TEXT,
                        new_values TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT
                    )
                ''')
                
                # Create device registry table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS device_registry (
                        device_id TEXT PRIMARY KEY,
                        device_name TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        location TEXT,
                        status TEXT DEFAULT 'active',
                        encrypted_config TEXT,
                        last_seen DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        encrypted_session_data TEXT,
                        expires_at DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def encrypt_data(self, data):
        """Encrypt data using Fernet encryption."""
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            elif not isinstance(data, (str, bytes)):
                data = str(data)
                
            if isinstance(data, str):
                data = data.encode()
                
            encrypted = self.fernet.encrypt(data)
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data using Fernet encryption."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def _calculate_hash(self, data):
        """Calculate SHA-256 hash of data for integrity verification."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)
            
        return hashlib.sha256(data.encode()).hexdigest()
    
    def log_audit_event(self, action, user_id=None, table_name=None, record_id=None, 
                       old_values=None, new_values=None, ip_address=None, user_agent=None):
        """Log security audit events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_log (action, user_id, table_name, record_id, 
                                         old_values, new_values, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (action, user_id, table_name, record_id, 
                      json.dumps(old_values) if old_values else None,
                      json.dumps(new_values) if new_values else None,
                      ip_address, user_agent))
                
                conn.commit()
                self.logger.info(f"Audit event logged: {action}")
                
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
    
    def insert_health_data(self, device_id, data, user_id=None):
        """Insert encrypted health data with integrity verification."""
        try:
            # Calculate hash before encryption
            data_hash = self._calculate_hash(data)
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO health_data (device_id, timestamp, encrypted_data, data_hash)
                    VALUES (?, ?, ?, ?)
                ''', (device_id, datetime.now(), encrypted_data, data_hash))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    action="INSERT",
                    user_id=user_id,
                    table_name="health_data",
                    record_id=str(record_id),
                    new_values={"device_id": device_id, "data_hash": data_hash}
                )
                
                self.logger.info(f"Health data inserted for device {device_id}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to insert health data: {e}")
            raise
    
    def get_health_data(self, device_id=None, start_date=None, end_date=None, limit=None):
        """Retrieve and decrypt health data with integrity verification."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT id, device_id, timestamp, encrypted_data, data_hash FROM health_data WHERE 1=1"
                params = []
                
                if device_id:
                    query += " AND device_id = ?"
                    params.append(device_id)
                    
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    record_id, dev_id, timestamp, encrypted_data, stored_hash = row
                    
                    try:
                        # Decrypt data
                        decrypted_data = self.decrypt_data(encrypted_data)
                        
                        # Parse JSON if applicable
                        try:
                            data = json.loads(decrypted_data)
                        except json.JSONDecodeError:
                            data = decrypted_data
                        
                        # Verify integrity
                        calculated_hash = self._calculate_hash(data)
                        if calculated_hash != stored_hash:
                            self.logger.warning(f"Data integrity check failed for record {record_id}")
                            continue
                        
                        results.append({
                            'id': record_id,
                            'device_id': dev_id,
                            'timestamp': timestamp,
                            'data': data
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process record {record_id}: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve health data: {e}")
            raise
    
    def register_device(self, device_id, device_name, device_type, location=None, config=None):
        """Register a new device with encrypted configuration."""
        try:
            encrypted_config = None
            if config:
                encrypted_config = self.encrypt_data(config)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO device_registry 
                    (device_id, device_name, device_type, location, encrypted_config, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (device_id, device_name, device_type, location, encrypted_config, datetime.now()))
                
                conn.commit()
                
                self.log_audit_event(
                    action="DEVICE_REGISTERED",
                    table_name="device_registry",
                    record_id=device_id,
                    new_values={"device_name": device_name, "device_type": device_type}
                )
                
                self.logger.info(f"Device registered: {device_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to register device: {e}")
            raise
    
    def get_devices(self):
        """Get list of registered devices."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT device_id, device_name, device_type, location, status, last_seen, created_at
                    FROM device_registry
                    ORDER BY device_name
                ''')
                
                rows = cursor.fetchall()
                
                devices = []
                for row in rows:
                    devices.append({
                        'device_id': row[0],
                        'device_name': row[1],
                        'device_type': row[2],
                        'location': row[3],
                        'status': row[4],
                        'last_seen': row[5],
                        'created_at': row[6]
                    })
                
                return devices
                
        except Exception as e:
            self.logger.error(f"Failed to get devices: {e}")
            raise
    
    def update_device_status(self, device_id, status):
        """Update device status and last seen timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE device_registry 
                    SET status = ?, last_seen = ?
                    WHERE device_id = ?
                ''', (status, datetime.now(), device_id))
                
                conn.commit()
                
                self.log_audit_event(
                    action="DEVICE_STATUS_UPDATED",
                    table_name="device_registry",
                    record_id=device_id,
                    new_values={"status": status}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update device status: {e}")
            raise
    
    def create_session(self, user_id, session_data, expires_at):
        """Create encrypted user session."""
        try:
            session_id = secrets.token_urlsafe(32)
            encrypted_session_data = self.encrypt_data(session_data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO user_sessions (session_id, user_id, encrypted_session_data, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_id, encrypted_session_data, expires_at))
                
                conn.commit()
                
                self.log_audit_event(
                    action="SESSION_CREATED",
                    user_id=user_id,
                    table_name="user_sessions",
                    record_id=session_id
                )
                
                return session_id
                
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id):
        """Retrieve and decrypt user session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT user_id, encrypted_session_data, expires_at
                    FROM user_sessions
                    WHERE session_id = ? AND expires_at > ?
                ''', (session_id, datetime.now()))
                
                row = cursor.fetchone()
                
                if row:
                    user_id, encrypted_data, expires_at = row
                    session_data = json.loads(self.decrypt_data(encrypted_data))
                    
                    return {
                        'user_id': user_id,
                        'session_data': session_data,
                        'expires_at': expires_at
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            raise
    
    def delete_session(self, session_id):
        """Delete user session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM user_sessions WHERE session_id = ?', (session_id,))
                conn.commit()
                
                self.log_audit_event(
                    action="SESSION_DELETED",
                    table_name="user_sessions",
                    record_id=session_id
                )
                
        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            raise
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM user_sessions WHERE expires_at < ?', (datetime.now(),))
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} expired sessions")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
    
    def get_audit_logs(self, limit=100, action_filter=None, start_date=None, end_date=None):
        """Retrieve audit logs with filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT action, user_id, table_name, record_id, old_values, 
                           new_values, timestamp, ip_address, user_agent
                    FROM audit_log WHERE 1=1
                '''
                params = []
                
                if action_filter:
                    query += " AND action = ?"
                    params.append(action_filter)
                    
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    logs.append({
                        'action': row[0],
                        'user_id': row[1],
                        'table_name': row[2],
                        'record_id': row[3],
                        'old_values': json.loads(row[4]) if row[4] else None,
                        'new_values': json.loads(row[5]) if row[5] else None,
                        'timestamp': row[6],
                        'ip_address': row[7],
                        'user_agent': row[8]
                    })
                
                return logs
                
        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            raise
    
    def backup_database(self, backup_path=None):
        """Create encrypted backup of the database."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"SECURITY/data_backups/backup_{timestamp}.db"
            
            # Create backup directory
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            
            self.logger.info(f"Database backed up to {backup_path}")
            
            self.log_audit_event(
                action="DATABASE_BACKUP",
                new_values={"backup_path": backup_path}
            )
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            raise
    
    def close(self):
        """Close database connections and cleanup."""
        try:
            self.logger.info("Database manager closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize secure database manager
    db_manager = SecureDatabaseManager()
    
    # Register a test device
    db_manager.register_device(
        device_id="sensor_001",
        device_name="Temperature Sensor 1",
        device_type="temperature",
        location="Factory Floor A",
        config={"sampling_rate": 60, "threshold": 75.0}
    )
    
    # Insert test data
    test_data = {
        "temperature": 72.5,
        "humidity": 45.2,
        "pressure": 1013.25,
        "vibration": 0.1
    }
    
    db_manager.insert_health_data("sensor_001", test_data)
    
    # Retrieve data
    data = db_manager.get_health_data("sensor_001", limit=10)
    print(f"Retrieved {len(data)} records")
    
    # Get devices
    devices = db_manager.get_devices()
    print(f"Registered devices: {len(devices)}")
    
    # Get audit logs
    logs = db_manager.get_audit_logs(limit=5)
    print(f"Recent audit events: {len(logs)}")
    
    # Create backup
    backup_path = db_manager.backup_database()
    print(f"Backup created: {backup_path}")
    
    # Close
    db_manager.close()