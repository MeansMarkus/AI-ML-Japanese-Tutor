"""
Configuration file for Japanese AI Tutor
Easily switch between different persistence methods and customize settings
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Persistence Settings
    PERSISTENCE_METHOD: str = "sqlite"  # Options: "json", "sqlite"
    DATA_DIRECTORY: str = "data"
    DATABASE_PATH: str = "data/japanese_tutor.db"
    
    # Auto-save Settings
    AUTO_SAVE_ENABLED: bool = True
    SAVE_FREQUENCY: int = 1  # Save after every N actions
    
    # Backup Settings
    AUTO_BACKUP_ENABLED: bool = True
    BACKUP_FREQUENCY_DAYS: int = 7
    MAX_BACKUP_FILES: int = 10
    
    # UI Settings
    PAGE_TITLE: str = "Japanese AI Tutor"
    PAGE_ICON: str = "🗾"
    THEME: str = "light"  # Options: "light", "dark", "auto"
    
    # Learning Settings
    DEFAULT_LEVEL: str = "Beginner"
    AVAILABLE_LEVELS: List[str] = None
    AVAILABLE_TOPICS: List[str] = None
    DAILY_GOAL_DEFAULT: int = 5
    
    # Review Settings
    REVIEW_ALGORITHM: str = "spaced_repetition"  # Options: "spaced_repetition", "simple"
    SHOW_STATISTICS: bool = True
    MAX_QUIZ_WORDS: int = 20
    
    # API Settings
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 150
    TEMPERATURE: float = 0.7
    
    # Advanced Settings
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    CACHE_ENABLED: bool = True
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.AVAILABLE_LEVELS is None:
            self.AVAILABLE_LEVELS = ["Beginner", "Intermediate", "Advanced"]
        
        if self.AVAILABLE_TOPICS is None:
            self.AVAILABLE_TOPICS = [
                "Daily Greetings", 
                "Food & Restaurants", 
                "Travel", 
                "Work & School", 
                "Hobbies", 
                "Free Conversation"
            ]

class ConfigManager:
    """Manages application configuration with environment variable overrides"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = AppConfig()
        self.config_file = config_file or "app_config.json"
        self.load_config()
        self.apply_env_overrides()
    
    def load_config(self):
        """Load configuration from file if it exists"""
        if os.path.exists(self.config_file):
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update config with loaded values
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            import json
            from dataclasses import asdict
            
            config_data = asdict(self.config)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'PERSISTENCE_METHOD': 'PERSISTENCE_METHOD',
            'DATA_DIRECTORY': 'DATA_DIRECTORY', 
            'OPENAI_MODEL': 'OPENAI_MODEL',
            'DEBUG_MODE': 'DEBUG_MODE',
            'AUTO_SAVE_ENABLED': 'AUTO_SAVE_ENABLED'
        }
        
        for config_key, env_key in env_mappings.items():
            env_value = os.getenv(f"JAPANESE_TUTOR_{env_key}")
            if env_value:
                # Convert string values to appropriate types
                if config_key in ['AUTO_SAVE_ENABLED', 'DEBUG_MODE', 'AUTO_BACKUP_ENABLED']:
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['SAVE_FREQUENCY', 'DAILY_GOAL_DEFAULT', 'MAX_TOKENS']:
                    env_value = int(env_value)
                elif config_key in ['TEMPERATURE']:
                    env_value = float(env_value)
                
                setattr(self.config, config_key, env_value)
    
    def get_persistence_manager(self):
        """Get the appropriate persistence manager based on configuration"""
        if self.config.PERSISTENCE_METHOD.lower() == "sqlite":
            from db_persistence import SQLiteDataPersistence
            return SQLiteDataPersistence(self.config.DATABASE_PATH)
        else:
            from data_persistence import DataPersistence
            return DataPersistence(self.config.DATA_DIRECTORY)
    
    def get_initialization_function(self):
        """Get the appropriate initialization function"""
        if self.config.PERSISTENCE_METHOD.lower() == "sqlite":
            from db_persistence import initialize_sqlite_session_state
            return initialize_sqlite_session_state
        else:
            from data_persistence import initialize_persistent_session_state
            return initialize_persistent_session_state
    
    def get_save_function(self):
        """Get the appropriate save function"""
        if self.config.PERSISTENCE_METHOD.lower() == "sqlite":
            from db_persistence import sync_to_database
            return sync_to_database
        else:
            from data_persistence import save_session_data
            return save_session_data

# Global configuration instance
config_manager = ConfigManager()
config = config_manager.config

# Convenience functions
def get_config():
    """Get the global configuration"""
    return config

def update_config(**kwargs):
    """Update configuration values"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config_manager.save_config()

def reset_config():
    """Reset configuration to defaults"""
    global config_manager, config
    config_manager = ConfigManager()
    config = config_manager.config

# Environment-specific configurations
def setup_development_config():
    """Setup configuration for development"""
    update_config(
        DEBUG_MODE=True,
        LOG_LEVEL="DEBUG",
        AUTO_BACKUP_ENABLED=False,
        PERSISTENCE_METHOD="json"
    )

def setup_production_config():
    """Setup configuration for production"""
    update_config(
        DEBUG_MODE=False,
        LOG_LEVEL="INFO",
        AUTO_BACKUP_ENABLED=True,
        PERSISTENCE_METHOD="sqlite",
        CACHE_ENABLED=True
    )

def setup_demo_config():
    """Setup configuration for demo/testing"""
    update_config(
        DATA_DIRECTORY="demo_data",
        DATABASE_PATH="demo_data/demo.db",
        AUTO_SAVE_ENABLED=False,
        MAX_BACKUP_FILES=3
    )

# Usage examples and documentation
USAGE_EXAMPLES = """
# Configuration Usage Examples

## Basic Usage
```python
from config import get_config, update_config

# Get current config
config = get_config()
print(f"Using {config.PERSISTENCE_METHOD} persistence")

# Update settings
update_config(PERSISTENCE_METHOD="sqlite", DEBUG_MODE=True)
```

## Environment Variables
Set these environment variables to override config:
- JAPANESE_TUTOR_PERSISTENCE_METHOD=sqlite
- JAPANESE_TUTOR_DEBUG_MODE=true  
- JAPANESE_TUTOR_DATA_DIRECTORY=/custom/path
- JAPANESE_TUTOR_OPENAI_MODEL=gpt-4
- JAPANESE_TUTOR_AUTO_SAVE_ENABLED=false

## In Your Main App
```python
from config import config_manager

# Initialize with config
initialize_func = config_manager.get_initialization_function()
save_func = config_manager.get_save_function()
persistence_manager = config_manager.get_persistence_manager()

# Use in your app
initialize_func()
# ... your app logic ...
save_func()  # Save when needed
```

## Environment Setup
Create a .env file:
```
OPENAI_API_KEY=your_api_key_here
JAPANESE_TUTOR_PERSISTENCE_METHOD=sqlite
JAPANESE_TUTOR_DEBUG_MODE=false
JAPANESE_TUTOR_DATA_DIRECTORY=data
```

## Different Environments
```python
# For development
from config import setup_development_config
setup_development_config()

# For production
from config import setup_production_config  
setup_production_config()

# For demo/testing
from config import setup_demo_config
setup_demo_config()
```
"""