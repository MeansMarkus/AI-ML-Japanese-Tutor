# Japanese AI Tutor 🗾

A comprehensive Streamlit-based application for learning Japanese through AI-powered conversations, vocabulary tracking, and intelligent spaced repetition review.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Integration](#api-integration)
- [Data Persistence](#data-persistence)
- [Machine Learning Features](#machine-learning-features)
- [T5 Transformer Model I trained](#T5-Transformer-info)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

### 🤖 AI-Powered Conversation Practice
- Interactive chat with OpenAI GPT models
- Level-appropriate responses (Beginner, Intermediate, Advanced)
- Topic-focused conversations (Greetings, Food, Travel, Work, Hobbies)
- Automatic vocabulary extraction from conversations
- Formatted responses with furigana, romanji, and English translations

### 📊 Progress Tracking
- Daily conversation statistics
- Learning streak tracking
- Progress visualization with charts
- Activity breakdown by level and topic
- Conversation history management

### 📝 Vocabulary Management
- Add custom vocabulary with Japanese and English meanings
- Search and filter vocabulary
- Sort by date added, alphabetically, or practice frequency
- Export vocabulary to CSV
- Success rate tracking per word

### 🧠 Intelligent Review System
- Machine learning-powered spaced repetition
- Adaptive review scheduling based on performance
- Success rate calculation and streak tracking
- Mixed quiz modes (Japanese→English, English→Japanese)
- Smart algorithm adjusts review frequency based on user performance

### 🔧 Flexible Configuration
- Multiple persistence methods (JSON files or SQLite database)
- Environment variable configuration
- Development/Production/Demo setup modes
- Customizable OpenAI model settings
- Auto-save functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MeansMarkus/AI-ML-Japanese-Tutor.git
cd AI-ML-Japanese-Tutor
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with these dependencies:

```txt
torch
transformers
wandb
python-dotenv
openai>=1.0.0
Streamlit
pandas
sqlite
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
JAPANESE_TUTOR_PERSISTENCE_METHOD=sqlite
JAPANESE_TUTOR_DEBUG_MODE=false
JAPANESE_TUTOR_DATA_DIRECTORY=data
```

## Configuration

The application uses a flexible configuration system that supports multiple persistence methods and environment-specific settings.

### Basic Configuration

```python
# config.py - Main configuration
@dataclass
class AppConfig:
    # Persistence Settings
    PERSISTENCE_METHOD: str = "sqlite"  # "json" or "sqlite"
    DATA_DIRECTORY: str = "data"
    DATABASE_PATH: str = "data/japanese_tutor.db"
    
    # Auto-save Settings
    AUTO_SAVE_ENABLED: bool = True
    
    # UI Settings
    PAGE_TITLE: str = "Japanese AI Tutor"
    PAGE_ICON: str = "🗾"
    
    # Learning Settings
    DEFAULT_LEVEL: str = "Beginner"
    AVAILABLE_LEVELS: List[str] = ["Beginner", "Intermediate", "Advanced"]
    AVAILABLE_TOPICS: List[str] = [
        "Daily Greetings", 
        "Food & Restaurants", 
        "Travel", 
        "Work & School", 
        "Hobbies", 
        "Free Conversation"
    ]
    
    # API Settings
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 150
    TEMPERATURE: float = 0.7
```

### Environment-Specific Setup

```python
# Development setup
from config import setup_development_config
setup_development_config()

# Production setup
from config import setup_production_config  
setup_production_config()

# Demo/Testing setup
from config import setup_demo_config
setup_demo_config()
```

### Environment Variables

Override configuration with environment variables:

```bash
export JAPANESE_TUTOR_PERSISTENCE_METHOD=sqlite
export JAPANESE_TUTOR_DEBUG_MODE=true
export JAPANESE_TUTOR_OPENAI_MODEL=gpt-4
export JAPANESE_TUTOR_AUTO_SAVE_ENABLED=false
```

## Usage

### Starting the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

### Basic Workflow

1. **Setup**: Configure your OpenAI API key in the sidebar
2. **Chat Practice**: Select your level and topic, then start conversing
3. **Vocabulary**: Add new words you learn during conversations
4. **Review**: Use the Smart Review system to practice vocabulary
5. **Progress**: Track your learning progress and statistics

### Chat Practice Example

```python
# System prompt format used internally
system_prompt = f"""You are a helpful Japanese language tutor. 
Student level: {level}
Topic: {topic}

Rules:
1. ALWAYS respond in this exact format:
   Japanese text with furigana [romanji] (English translation)
   Example: こんにちは [Konnichiwa] (Hello)

2. Introduce 1-2 new words naturally in conversations.
3. Correct mistakes gently and explain in English.
4. Keep responses concise and educational.
"""
```

### Vocabulary Management

```python
# Example vocabulary entry structure
word_entry = {
    "japanese": "こんにちは",
    "english": "hello",
    "added_date": "2024-01-01",
    "practice_count": 0,
    "review_sequence": [],
    "last_reviewed": None,
    "next_review": "2024-01-01"
}
```

## Architecture

### File Structure

```
japanese-ai-tutor/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration management
├── ml_utils.py            # Machine learning utilities
├── data_persistence.py    # JSON-based persistence
├── db_persistence.py      # SQLite-based persistence
├── style.css             # Custom CSS styling
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── data/                 # Data storage directory
    ├── learned_words.json
    ├── conversations.json
    ├── user_settings.json
    └── japanese_tutor.db
```

### Core Components

#### Configuration Manager
```python
class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config = AppConfig()
        self.load_config()
        self.apply_env_overrides()
    
    def get_persistence_manager(self):
        if self.config.PERSISTENCE_METHOD.lower() == "sqlite":
            return SQLiteDataPersistence(self.config.DATABASE_PATH)
        else:
            return DataPersistence(self.config.DATA_DIRECTORY)
```

#### Data Models
```python
# Conversation Entry
conversation_entry = {
    "user": str,           # User input
    "ai": str,             # AI response
    "timestamp": str,      # ISO format timestamp
    "level": str,          # User level
    "topic": str           # Conversation topic
}

# Vocabulary Entry
vocabulary_entry = {
    "japanese": str,       # Japanese word/phrase
    "english": str,        # English translation
    "added_date": str,     # Date added
    "practice_count": int, # Number of reviews
    "review_sequence": [],  # Review history
    "last_reviewed": str,  # Last review date
    "next_review": str     # Next scheduled review
}
```

## API Integration

### OpenAI Integration

```python
# Initialize client
client = OpenAI(api_key=api_key)

# Chat completion request
response = client.chat.completions.create(
    model=config.OPENAI_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ],
    max_tokens=config.MAX_TOKENS,
    temperature=config.TEMPERATURE
)
```

### API Key Configuration

The application supports multiple methods for API key configuration:

1. **Streamlit Secrets** (Production deployment):
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_api_key_here"
```

2. **Environment Variables** (Local development):
```bash
export OPENAI_API_KEY="your_api_key_here"
```

3. **.env File** (Local development):
```env
OPENAI_API_KEY=your_api_key_here
```

## Data Persistence

### JSON-Based Persistence

```python
class DataPersistence:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.learned_words_file = os.path.join(data_dir, "learned_words.json")
        self.conversations_file = os.path.join(data_dir, "conversations.json")
        self.settings_file = os.path.join(data_dir, "user_settings.json")
    
    def save_learned_words(self, words: List[dict]):
        with open(self.learned_words_file, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
```

### SQLite-Based Persistence

```python
class SQLiteDataPersistence:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learned_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            japanese TEXT UNIQUE NOT NULL,
            english TEXT NOT NULL,
            added_date TEXT NOT NULL,
            practice_count INTEGER DEFAULT 0,
            review_sequence TEXT DEFAULT '[]',
            last_reviewed TEXT,
            next_review TEXT NOT NULL
        )
        ''')
```

### Data Import/Export

```python
# Export data to JSON
def export_all_data():
    export_data = {
        "learned_words": load_learned_words(),
        "conversation_history": load_conversation_history(),
        "user_settings": load_user_settings(),
        "export_date": datetime.now().isoformat(),
        "app_version": "1.0.0"
    }
    
    filename = f"japanese_tutor_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(DATA_DIRECTORY, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return filepath
```

## Machine Learning Features

### Spaced Repetition Algorithm

```python
def calculate_next_review_date(word: dict, is_correct: bool) -> str:
    """Calculate next review date using spaced repetition algorithm"""
    base_intervals = [1, 3, 7, 14, 30, 90]  # days
    
    current_sequence = word.get('review_sequence', [])
    current_level = len([r for r in current_sequence if r.get('correct', False)])
    
    if is_correct:
        if current_level < len(base_intervals):
            interval = base_intervals[current_level]
        else:
            # For advanced words, increase interval exponentially
            interval = base_intervals[-1] * (2 ** (current_level - len(base_intervals) + 1))
    else:
        # Reset to first interval if incorrect
        interval = base_intervals[0]
    
    next_date = datetime.now() + timedelta(days=interval)
    return next_date.strftime("%Y-%m-%d")
```

### Performance Analytics

```python
def calculate_success_rate(word: dict) -> float:
    """Calculate success rate for a word"""
    reviews = word.get('review_sequence', [])
    if not reviews:
        return 0.0
    
    correct_count = sum(1 for review in reviews if review.get('correct', False))
    return correct_count / len(reviews)

def get_current_streak(word: dict) -> int:
    """Get current correct streak for a word"""
    reviews = word.get('review_sequence', [])
    if not reviews:
        return 0
    
    streak = 0
    for review in reversed(reviews):
        if review.get('correct', False):
            streak += 1
        else:
            break
    
    return streak
```

### Review Scheduling

```python
def get_words_due_for_review(learned_words: List[dict]) -> List[dict]:
    """Get words that are due for review today"""
    today = datetime.now().strftime("%Y-%m-%d")
    due_words = []
    
    for word in learned_words:
        next_review = word.get('next_review')
        if next_review and next_review <= today:
            due_words.append(word)
    
    # Sort by priority: words with lower success rates first
    due_words.sort(key=lambda w: calculate_success_rate(w))
    
    return due_words
```

## T5-Model-info
  
T5 Transformer Model

Architecture Overview
The application initially explored a custom fine-tuned T5 transformer model for Japanese grammar correction before adopting GPT-3.5 Turbo for production use.
Training Data Pipeline
The model utilized a comprehensive four-method data collection approach:

Data Generation Methods

```python
python# Four distinct data generation methods

def method_1_manual_expert_data():      # Expert annotations
def method_2_synthetic_data_generation(): # Template-based generation  
def method_3_lang8_style_data():        # Learner error simulation
def method_4_jlpt_based_data():         # Proficiency-graded examples
```
Expert-curated examples: High-quality grammar corrections focusing on common learner errors across multiple proficiency levels.
Synthetic data generation: Template-based error injection across multiple grammar patterns, simulating realistic student mistakes.
Learner simulation: L1 interference patterns and overgeneralization errors based on common Japanese language learning challenges.
JLPT-structured data: Difficulty-graded examples aligned with Japanese proficiency standards (N5-N3 coverage).
Model Architecture
Base Model: Google's MT5-small (multilingual T5)
Task Format: Text-to-text generation ("grammar: [incorrect]" → "[correct]")
Fine-tuning: Full parameter updates with AdamW optimizer
Evaluation: Validation loss tracking and qualitative correction assessment
Technical Implementation
Data Collection Pipeline

python# training_data_collection.py - Four distinct data generation methods


```python
class JapaneseGrammarDataGenerator:
    def generate_training_data(self):
        data = []
        data.extend(self.method_1_manual_expert_data())
        data.extend(self.method_2_synthetic_data_generation())
        data.extend(self.method_3_lang8_style_data())
        data.extend(self.method_4_jlpt_based_data())
        return data
```
Error Categories Covered

Particle usage: は/が, を, に/で distinctions
Tense consistency: Past, present, and future form corrections
Adjective conjugation: い-adjective and な-adjective forms
Word order patterns: SOV structure and modifier placement
Politeness level appropriateness: Formal vs informal register

Fine-tuning Pipeline
python# fine_tuning.py - Complete training pipeline

```python
class T5GrammarTrainer:
    def __init__(self, model_name="google/mt5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def train(self, training_data):
        # Custom PyTorch Dataset implementation
        # T5ForConditionalGeneration fine-tuning
        # Learning rate scheduling with warmup
        # Checkpoint management and model persistence
        # Validation loop with loss tracking
```
Results & Performance
Training Statistics

Total Examples: ~200+ carefully crafted training pairs ~200+ available training sentences from JCoLA (Japanese Corpus of Linguistic Acceptability)

Error Types: 15+ distinct grammatical error categories
Difficulty Levels: JLPT N5-N3 coverage
Training Time: 2-3 hours on GPU

Findings:
-Ran it for 3 Epochs
-Epoch 2 ended up being most accurate, so set that as a base and implemented edge cases based on things it missed such as particle usage, keigo usage, etc.

Model Capabilities
The fine-tuned model successfully corrects common Japanese grammar errors:
python# Example corrections
Input:  "grammar: 私が学生です。"
Output: "私は学生です。"        # Correct は/が particle usage

Input:  "grammar: 本が読みます。"  
Output: "本を読みます。"        # Correct object particle を

Input:  "grammar: 昨日映画を見ます。"
Output: "昨日映画を見ました。"    # Correct past tense
Production Decision
While this fine-tuning approach demonstrated technical feasibility, GPT-3.5 Turbo was ultimately chosen for production deployment due to several practical considerations:
Broader linguistic knowledge: Better handling of edge cases and contextual nuances beyond the training data scope.
Faster deployment: No infrastructure required for custom model hosting and serving.
Consistent performance: Reduced need for extensive validation datasets and model maintenance.

## Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set up secrets in the Streamlit dashboard

3. **Configure Secrets**:
```toml
# In Streamlit Cloud secrets management
OPENAI_API_KEY = "your_api_key_here"
```

### Local Deployment with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run Docker container
docker build -t japanese-ai-tutor .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key japanese-ai-tutor
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
```
❌ No API Key found. Please contact the app administrator.
```
**Solution**: Set up your OpenAI API key in `.env` file or Streamlit secrets.

2. **Database Connection Error**:
```
sqlite3.OperationalError: database is locked
```
**Solution**: Ensure only one instance of the app is running, or switch to JSON persistence.

3. **Import/Export Errors**:
```
Export error: [Errno 2] No such file or directory
```
**Solution**: Check that the data directory exists and has proper permissions.

### Debug Mode

Enable debug mode for detailed error information:

```python
# In config.py or environment
DEBUG_MODE = True
```

Or set environment variable:
```bash
export JAPANESE_TUTOR_DEBUG_MODE=true
```

### Performance Optimization

1. **Large Vocabulary Collections**:
   - Use SQLite persistence for better performance
   - Limit vocabulary display with pagination
   - Use database indexing

2. **Memory Usage**:
   - Enable auto-save to reduce session state size
   - Limit conversation history retention
   - Use data compression for exports

### Data Recovery

```python
# Backup current data before making changes
def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    shutil.copytree("data", backup_dir)
    return backup_dir

# Restore from backup
def restore_backup(backup_dir: str):
    if os.path.exists("data"):
        shutil.rmtree("data")
    shutil.copytree(backup_dir, "data")
```

## Contributing

### Development Setup

1. **Clone and setup development environment**:
```bash
git clone https://github.com/MeansMarkus/AI-ML-Japanese-Tutor.git
cd AI-ML-Japanese-Tutor
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Configure for development**:
```python
from config import setup_development_config
setup_development_config()
```

3. **Run tests**:
```bash
# Add test files and run
python -m pytest tests/
```


### Contribution Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Feature Requests

Submit feature requests by opening an issue with:
- Clear description of the feature
- Use cases and benefits
- Implementation suggestions (if any)
- Screenshots or mockups (if applicable)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI GPT API
- Streamlit for the web framework

---
