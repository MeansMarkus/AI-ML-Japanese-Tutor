import json
import os
from datetime import datetime
import streamlit as st
import pandas as pd

class DataPersistence:
    def __init__(self, data_dir="data"):
        """Initialize data persistence with a data directory"""
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # File paths
        self.learned_words_file = os.path.join(data_dir, "learned_words.json")
        self.conversation_history_file = os.path.join(data_dir, "conversation_history.json")
        self.user_settings_file = os.path.join(data_dir, "user_settings.json")
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_learned_words(self, learned_words):
        """Save learned words to JSON file"""
        try:
            with open(self.learned_words_file, 'w', encoding='utf-8') as f:
                json.dump(learned_words, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving learned words: {e}")
            return False
    
    def load_learned_words(self):
        """Load learned words from JSON file"""
        try:
            if os.path.exists(self.learned_words_file):
                with open(self.learned_words_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading learned words: {e}")
            return []
    
    def save_conversation_history(self, conversation_history):
        """Save conversation history to JSON file"""
        try:
            with open(self.conversation_history_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving conversation history: {e}")
            return False
    
    def load_conversation_history(self):
        """Load conversation history from JSON file"""
        try:
            if os.path.exists(self.conversation_history_file):
                with open(self.conversation_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading conversation history: {e}")
            return []
    
    def save_user_settings(self, settings):
        """Save user settings like preferred level, topics, etc."""
        try:
            with open(self.user_settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving user settings: {e}")
            return False
    
    def load_user_settings(self):
        """Load user settings"""
        try:
            if os.path.exists(self.user_settings_file):
                with open(self.user_settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                "preferred_level": "Beginner",
                "favorite_topics": [],
                "daily_goal": 5
            }
        except Exception as e:
            st.error(f"Error loading user settings: {e}")
            return {}
    
    def export_all_data(self):
        """Export all data as a single JSON file for backup"""
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "learned_words": self.load_learned_words(),
                "conversation_history": self.load_conversation_history(),
                "user_settings": self.load_user_settings()
            }
            
            export_filename = f"japanese_tutor_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_path = os.path.join(self.data_dir, export_filename)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return export_path
        except Exception as e:
            st.error(f"Error exporting data: {e}")
            return None
    
    def import_data(self, import_file_path):
        """Import data from a backup file"""
        try:
            with open(import_file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate data structure
            if all(key in import_data for key in ["learned_words", "conversation_history", "user_settings"]):
                self.save_learned_words(import_data["learned_words"])
                self.save_conversation_history(import_data["conversation_history"])
                self.save_user_settings(import_data["user_settings"])
                return True
            else:
                st.error("Invalid backup file format")
                return False
        except Exception as e:
            st.error(f"Error importing data: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all stored data"""
        try:
            files_to_remove = [
                self.learned_words_file,
                self.conversation_history_file,
                self.user_settings_file
            ]
            
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return True
        except Exception as e:
            st.error(f"Error clearing data: {e}")
            return False
    
    def get_data_stats(self):
        """Get statistics about stored data"""
        stats = {
            "learned_words_count": len(self.load_learned_words()),
            "conversation_count": len(self.load_conversation_history()),
            "data_size_mb": 0
        }
        
        # Calculate total data size
        total_size = 0
        for file_path in [self.learned_words_file, self.conversation_history_file, self.user_settings_file]:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        stats["data_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats

# Helper functions for Streamlit integration
def initialize_persistent_session_state():
    """Initialize session state with persistent data"""
    data_manager = DataPersistence()
    
    # Load data if not already in session state
    if 'learned_words' not in st.session_state:
        st.session_state.learned_words = data_manager.load_learned_words()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = data_manager.load_conversation_history()
    
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = data_manager.load_user_settings()
    
    # Store data manager in session state for easy access
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = data_manager

def save_session_data():
    """Save current session state to persistent storage"""
    if 'data_manager' in st.session_state:
        data_manager = st.session_state.data_manager
        
        # Save all data
        data_manager.save_learned_words(st.session_state.learned_words)
        data_manager.save_conversation_history(st.session_state.conversation_history)
        data_manager.save_user_settings(st.session_state.user_settings)

def auto_save_callback():
    """Callback function for automatic saving"""
    save_session_data()

# Decorator for auto-saving after function execution
def auto_save(func):
    """Decorator to automatically save data after function execution"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        save_session_data()
        return result
    return wrapper