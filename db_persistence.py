import sqlite3
import json
import os
from datetime import datetime
import streamlit as st
import pandas as pd

class SQLiteDataPersistence:
    def __init__(self, db_path="data/japanese_tutor.db"):
        """Initialize SQLite database connection"""
        self.db_path = db_path
        self.ensure_database_directory()
        self.init_database()
    
    def ensure_database_directory(self):
        """Create database directory if it doesn't exist"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Learned words table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    japanese TEXT UNIQUE NOT NULL,
                    english TEXT NOT NULL,
                    added_date TEXT NOT NULL,
                    practice_count INTEGER DEFAULT 0,
                    review_sequence TEXT DEFAULT '[]',
                    last_reviewed TEXT,
                    next_review TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Conversation history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT,
                    topic TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_words_japanese ON learned_words(japanese)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_words_next_review ON learned_words(next_review)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversation_history(timestamp)')
            
            conn.commit()
        except Exception as e:
            st.error(f"Error initializing database: {e}")
        finally:
            conn.close()
    
    def save_learned_word(self, word_data):
        """Save or update a learned word"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Convert review_sequence to JSON string
            review_sequence_json = json.dumps(word_data.get('review_sequence', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO learned_words 
                (japanese, english, added_date, practice_count, review_sequence, 
                 last_reviewed, next_review, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                word_data['japanese'],
                word_data['english'],
                word_data.get('added_date', datetime.now().strftime("%Y-%m-%d")),
                word_data.get('practice_count', 0),
                review_sequence_json,
                word_data.get('last_reviewed'),
                word_data.get('next_review'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving word: {e}")
            return False
        finally:
            conn.close()
    
    def load_learned_words(self):
        """Load all learned words"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT japanese, english, added_date, practice_count, 
                       review_sequence, last_reviewed, next_review
                FROM learned_words
                ORDER BY added_date DESC
            ''')
            
            words = []
            for row in cursor.fetchall():
                word = {
                    'japanese': row[0],
                    'english': row[1],
                    'added_date': row[2],
                    'practice_count': row[3],
                    'review_sequence': json.loads(row[4]) if row[4] else [],
                    'last_reviewed': row[5],
                    'next_review': row[6]
                }
                words.append(word)
            
            return words
        except Exception as e:
            st.error(f"Error loading words: {e}")
            return []
        finally:
            conn.close()
    
    def delete_learned_word(self, japanese_word):
        """Delete a learned word"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM learned_words WHERE japanese = ?', (japanese_word,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            st.error(f"Error deleting word: {e}")
            return False
        finally:
            conn.close()
    
    def save_conversation(self, conversation_data):
        """Save a conversation entry"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversation_history 
                (user_message, ai_response, timestamp, level, topic)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                conversation_data['user'],
                conversation_data['ai'],
                conversation_data['timestamp'],
                conversation_data.get('level'),
                conversation_data.get('topic')
            ))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving conversation: {e}")
            return False
        finally:
            conn.close()
    
    def load_conversation_history(self, limit=None):
        """Load conversation history"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = '''
                SELECT user_message, ai_response, timestamp, level, topic
                FROM conversation_history
                ORDER BY created_at DESC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            
            conversations = []
            for row in cursor.fetchall():
                conv = {
                    'user': row[0],
                    'ai': row[1],
                    'timestamp': row[2],
                    'level': row[3],
                    'topic': row[4]
                }
                conversations.append(conv)
            
            # Reverse to get chronological order
            return list(reversed(conversations))
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            return []
        finally:
            conn.close()
    
    def save_user_setting(self, key, value):
        """Save a user setting"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Convert value to JSON string if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now().isoformat()))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving setting: {e}")
            return False
        finally:
            conn.close()
    
    def load_user_settings(self):
        """Load all user settings"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT setting_key, setting_value FROM user_settings')
            
            settings = {}
            for row in cursor.fetchall():
                key, value = row
                try:
                    # Try to parse as JSON
                    settings[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If not JSON, store as string
                    settings[key] = value
            
            # Set defaults if not present
            if 'preferred_level' not in settings:
                settings['preferred_level'] = 'Beginner'
            if 'favorite_topics' not in settings:
                settings['favorite_topics'] = []
            if 'daily_goal' not in settings:
                settings['daily_goal'] = 5
            
            return settings
        except Exception as e:
            st.error(f"Error loading settings: {e}")
            return {}
        finally:
            conn.close()
    
    def get_statistics(self):
        """Get database statistics"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Count total words
            cursor.execute('SELECT COUNT(*) FROM learned_words')
            total_words = cursor.fetchone()[0]
            
            # Count total conversations
            cursor.execute('SELECT COUNT(*) FROM conversation_history')
            total_conversations = cursor.fetchone()[0]
            
            # Count words with reviews
            cursor.execute("SELECT COUNT(*) FROM learned_words WHERE review_sequence != '[]'")
            words_with_reviews = cursor.fetchone()[0]
            
            # Get database file size
            db_size_mb = 0
            if os.path.exists(self.db_path):
                db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
            
            # Get today's activity
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute('SELECT COUNT(*) FROM conversation_history WHERE timestamp LIKE ?', (f"{today}%",))
            today_conversations = cursor.fetchone()[0]
            
            return {
                'total_words': total_words,
                'total_conversations': total_conversations,
                'words_with_reviews': words_with_reviews,
                'db_size_mb': db_size_mb,
                'today_conversations': today_conversations
            }
        except Exception as e:
            st.error(f"Error getting statistics: {e}")
            return {}
        finally:
            conn.close()
    
    def export_to_json(self):
        """Export all data to JSON format"""
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "learned_words": self.load_learned_words(),
                "conversation_history": self.load_conversation_history(),
                "user_settings": self.load_user_settings()
            }
            
            export_filename = f"japanese_tutor_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_path = os.path.join(os.path.dirname(self.db_path), export_filename)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return export_path
        except Exception as e:
            st.error(f"Error exporting data: {e}")
            return None
    
    def import_from_json(self, json_file_path):
        """Import data from JSON backup"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate data structure
            if not all(key in import_data for key in ["learned_words", "conversation_history", "user_settings"]):
                st.error("Invalid backup file format")
                return False
            
            # Clear existing data (optional - you might want to merge instead)
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Import learned words
            for word in import_data["learned_words"]:
                self.save_learned_word(word)
            
            # Import conversations
            for conv in import_data["conversation_history"]:
                self.save_conversation(conv)
            
            # Import settings
            for key, value in import_data["user_settings"].items():
                self.save_user_setting(key, value)
            
            return True
        except Exception as e:
            st.error(f"Error importing data: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all data from database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM learned_words')
            cursor.execute('DELETE FROM conversation_history')
            cursor.execute('DELETE FROM user_settings')
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error clearing data: {e}")
            return False
        finally:
            conn.close()
    
    def backup_database(self):
        """Create a backup of the entire database file"""
        try:
            backup_filename = f"japanese_tutor_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            backup_path = os.path.join(os.path.dirname(self.db_path), backup_filename)
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            return backup_path
        except Exception as e:
            st.error(f"Error backing up database: {e}")
            return None

# Helper functions for Streamlit integration with SQLite
def initialize_sqlite_session_state():
    """Initialize session state with SQLite data"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = SQLiteDataPersistence()
    
    # Load data if not already in session state
    if 'learned_words' not in st.session_state:
        st.session_state.learned_words = st.session_state.db_manager.load_learned_words()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = st.session_state.db_manager.load_conversation_history()
    
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = st.session_state.db_manager.load_user_settings()

def sync_to_database():
    """Sync current session state to database"""
    if 'db_manager' not in st.session_state:
        return
    
    db_manager = st.session_state.db_manager
    
    # Sync learned words
    for word in st.session_state.learned_words:
        db_manager.save_learned_word(word)
    
    # Note: Conversations are saved individually as they're created
    # to avoid duplicates, so we don't bulk sync them here
    
    # Sync user settings
    for key, value in st.session_state.user_settings.items():
        db_manager.save_user_setting(key, value)

def save_new_conversation(conversation_data):
    """Save a new conversation to database"""
    if 'db_manager' in st.session_state:
        return st.session_state.db_manager.save_conversation(conversation_data)
    return False

def save_new_word(word_data):
    """Save a new word to database"""
    if 'db_manager' in st.session_state:
        return st.session_state.db_manager.save_learned_word(word_data)
    return False

def update_word_in_database(word_data):
    """Update an existing word in database"""
    if 'db_manager' in st.session_state:
        return st.session_state.db_manager.save_learned_word(word_data)
    return False