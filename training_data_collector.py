import requests
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
import json
import time
import random
from pathlib import Path

class JapaneseTrainingDataCollector:
    """Collect and prepare training data for Japanese grammar correction"""
    
    def __init__(self):
        self.db_path = "grammar_training_data.db"
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for storing training data"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS raw_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incorrect_text TEXT NOT NULL,
                correct_text TEXT NOT NULL,
                error_type TEXT,
                source TEXT,
                difficulty_level INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for processed/cleaned data ready for training
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                target_text TEXT NOT NULL,
                error_category TEXT,
                confidence_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def method_1_manual_expert_data(self):
        """Method 1: Create high-quality expert-curated data"""
        
        expert_data = [
            # Particle errors - most common for learners
            {"incorrect": "私が日本人です。", "correct": "私は日本人です。", "error_type": "particle_wa_ga", "level": 1},
            {"incorrect": "今日が暑いです。", "correct": "今日は暑いです。", "error_type": "particle_wa_ga", "level": 1},
            {"incorrect": "田中さんが先生です。", "correct": "田中さんは先生です。", "error_type": "particle_wa_ga", "level": 1},
            
            # Object particle errors
            {"incorrect": "本が読みます。", "correct": "本を読みます。", "error_type": "particle_wo", "level": 1},
            {"incorrect": "音楽が聞きます。", "correct": "音楽を聞きます。", "error_type": "particle_wo", "level": 1},
            {"incorrect": "映画が見ました。", "correct": "映画を見ました。", "error_type": "particle_wo", "level": 1},
            {"incorrect": "日本語が勉強します。", "correct": "日本語を勉強します。", "error_type": "particle_wo", "level": 1},
            
            # Location/direction particle errors
            {"incorrect": "学校で行きます。", "correct": "学校に行きます。", "error_type": "particle_ni_de", "level": 2},
            {"incorrect": "図書館に勉強します。", "correct": "図書館で勉強します。", "error_type": "particle_ni_de", "level": 2},
            {"incorrect": "東京で住んでいます。", "correct": "東京に住んでいます。", "error_type": "particle_ni_de", "level": 2},
            
            # Tense errors
            {"incorrect": "昨日映画を見ます。", "correct": "昨日映画を見ました。", "error_type": "tense_past", "level": 2},
            {"incorrect": "明日友達と会いました。", "correct": "明日友達と会います。", "error_type": "tense_future", "level": 2},
            {"incorrect": "今朝パンを食べます。", "correct": "今朝パンを食べました。", "error_type": "tense_past", "level": 2},
            
            # Adjective conjugation errors
            {"incorrect": "昨日は楽しいでした。", "correct": "昨日は楽しかったです。", "error_type": "adjective_past", "level": 3},
            {"incorrect": "この本は面白いでした。", "correct": "この本は面白かったです。", "error_type": "adjective_past", "level": 3},
            {"incorrect": "料理は美味しいでした。", "correct": "料理は美味しかったです。", "error_type": "adjective_past", "level": 3},
            
            # Word order errors
            {"incorrect": "学校に行きました昨日。", "correct": "昨日学校に行きました。", "error_type": "word_order", "level": 3},
            {"incorrect": "友達と映画を見ました今日。", "correct": "今日友達と映画を見ました。", "error_type": "word_order", "level": 3},
            {"incorrect": "日本語を勉強しています大学で。", "correct": "大学で日本語を勉強しています。", "error_type": "word_order", "level": 3},
            
            # Missing particles
            {"incorrect": "私田中です。", "correct": "私は田中です。", "error_type": "missing_particle", "level": 1},
            {"incorrect": "学校行きます。", "correct": "学校に行きます。", "error_type": "missing_particle", "level": 2},
            {"incorrect": "友達話しました。", "correct": "友達と話しました。", "error_type": "missing_particle", "level": 2},
            
            # Verb form errors
            {"incorrect": "日本に行くつもりでした。", "correct": "日本に行くつもりです。", "error_type": "verb_form", "level": 4},
            {"incorrect": "明日は忙しくなりました。", "correct": "明日は忙しくなります。", "error_type": "verb_form", "level": 4},
        ]
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        for item in expert_data:
            conn.execute('''
                INSERT INTO raw_training_data 
                (incorrect_text, correct_text, error_type, source, difficulty_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (item["incorrect"], item["correct"], item["error_type"], "expert_curated", item["level"]))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(expert_data)} expert-curated examples")
        return len(expert_data)
    
    def method_2_synthetic_data_generation(self):
        """Method 2: Generate synthetic error data using templates"""
        
        # Common sentence templates
        templates = [
            # Subject-verb patterns
            {"template": "{subject}は{object}を{verb}ます。", "error_template": "{subject}が{object}を{verb}ます。", "error_type": "particle_wa_ga"},
            {"template": "{subject}は{object}を{verb}ます。", "error_template": "{subject}は{object}が{verb}ます。", "error_type": "particle_wo"},
            {"template": "{place}で{activity}ます。", "error_template": "{place}に{activity}ます。", "error_type": "particle_ni_de"},
            {"template": "{place}に行きます。", "error_template": "{place}で行きます。", "error_type": "particle_ni_de"},
            
            # Tense patterns
            {"template": "昨日{object}を{past_verb}。", "error_template": "昨日{object}を{present_verb}。", "error_type": "tense_mismatch"},
            {"template": "明日{object}を{present_verb}。", "error_template": "明日{object}を{past_verb}。", "error_type": "tense_mismatch"},
        ]
        
        # Word lists for templates
        subjects = ["私", "友達", "先生", "学生", "母", "父", "姉", "弟"]
        objects = ["本", "映画", "音楽", "料理", "宿題", "手紙", "写真", "ゲーム"]
        verbs = ["読み", "見", "聞き", "作り", "し", "書き", "撮り", "遊び"]
        places = ["学校", "図書館", "公園", "家", "駅", "病院", "銀行", "店"]
        activities = ["勉強し", "読書し", "運動し", "買い物し", "食事し", "会議し"]
        past_verbs = ["読みました", "見ました", "聞きました", "作りました", "しました", "書きました"]
        present_verbs = ["読みます", "見ます", "聞きます", "作ります", "します", "書きます"]
        
        synthetic_data = []
        
        for template_info in templates:
            template = template_info["template"]
            error_template = template_info["error_template"]
            error_type = template_info["error_type"]
            
            # Generate variations
            for _ in range(10):  # Generate 10 examples per template
                if "{subject}" in template:
                    subject = random.choice(subjects)
                    obj = random.choice(objects)
                    verb = random.choice(verbs)
                    
                    correct = template.format(subject=subject, object=obj, verb=verb)
                    incorrect = error_template.format(subject=subject, object=obj, verb=verb)
                
                elif "{place}" in template:
                    place = random.choice(places)
                    if "{activity}" in template:
                        activity = random.choice(activities)
                        correct = template.format(place=place, activity=activity)
                        incorrect = error_template.format(place=place, activity=activity)
                    else:
                        correct = template.format(place=place)
                        incorrect = error_template.format(place=place)
                
                elif "昨日" in template:
                    obj = random.choice(objects)
                    past_verb = random.choice(past_verbs)
                    present_verb = random.choice(present_verbs)
                    
                    correct = template.format(object=obj, past_verb=past_verb)
                    incorrect = error_template.format(object=obj, present_verb=present_verb)
                
                synthetic_data.append({
                    "incorrect": incorrect,
                    "correct": correct,
                    "error_type": error_type,
                    "level": 2
                })
        
        # Store synthetic data
        conn = sqlite3.connect(self.db_path)
        for item in synthetic_data:
            conn.execute('''
                INSERT INTO raw_training_data 
                (incorrect_text, correct_text, error_type, source, difficulty_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (item["incorrect"], item["correct"], item["error_type"], "synthetic", item["level"]))
        
        conn.commit()
        conn.close()
        
        print(f"Generated {len(synthetic_data)} synthetic examples")
        return len(synthetic_data)
    
    def method_3_lang8_style_data(self):
        """Method 3: Create Lang-8 style learner correction data"""
        
        # Simulate common learner errors based on L1 interference
        lang8_style_data = [
            # English speakers common errors
            {"incorrect": "私は好きです寿司を。", "correct": "私は寿司が好きです。", "error_type": "word_order_english", "level": 2},
            {"incorrect": "私は行きます学校に車で。", "correct": "私は車で学校に行きます。", "error_type": "word_order_english", "level": 3},
            {"incorrect": "それは本です面白い。", "correct": "それは面白い本です。", "error_type": "adjective_placement", "level": 2},
            
            # Overgeneralization errors
            {"incorrect": "私は学生だます。", "correct": "私は学生です。", "error_type": "overgeneralization", "level": 1},
            {"incorrect": "今日は寒いますね。", "correct": "今日は寒いですね。", "error_type": "overgeneralization", "level": 2},
            
            # Literal translation errors
            {"incorrect": "私は18歳を持っています。", "correct": "私は18歳です。", "error_type": "literal_translation", "level": 2},
            {"incorrect": "私は名前を田中と呼ばれています。", "correct": "私の名前は田中です。", "error_type": "literal_translation", "level": 3},
            
            # Context-specific errors
            {"incorrect": "先生、おはよう！", "correct": "先生、おはようございます！", "error_type": "politeness_level", "level": 2},
            {"incorrect": "すみません、トイレはどこですか？", "correct": "すみません、お手洗いはどこですか？", "error_type": "politeness_level", "level": 3},
        ]
        
        # Store Lang-8 style data
        conn = sqlite3.connect(self.db_path)
        for item in lang8_style_data:
            conn.execute('''
                INSERT INTO raw_training_data 
                (incorrect_text, correct_text, error_type, source, difficulty_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (item["incorrect"], item["correct"], item["error_type"], "lang8_style", item["level"]))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(lang8_style_data)} Lang-8 style examples")
        return len(lang8_style_data)
    
    def method_4_jlpt_based_data(self):
        """Method 4: Create JLPT level-based grammar correction data"""
        
        jlpt_data = [
            # N5 level errors (basic)
            {"incorrect": "これが私の本です。", "correct": "これは私の本です。", "error_type": "demonstrative_particle", "jlpt_level": "N5"},
            {"incorrect": "私の名前田中です。", "correct": "私の名前は田中です。", "error_type": "topic_particle", "jlpt_level": "N5"},
            {"incorrect": "今何時ですか？", "correct": "今何時ですか？", "error_type": "correct", "jlpt_level": "N5"},  # Correct example
            
            # N4 level errors (intermediate basic)
            {"incorrect": "友達と映画を見に行くつもりでした。", "correct": "友達と映画を見に行くつもりです。", "error_type": "intention_tense", "jlpt_level": "N4"},
            {"incorrect": "雨が降っているので、家にいるつもりでした。", "correct": "雨が降っているので、家にいるつもりです。", "error_type": "intention_tense", "jlpt_level": "N4"},
            
            # N3 level errors (intermediate)
            {"incorrect": "彼は来るかもしれませんでした。", "correct": "彼は来るかもしれませんでした。", "error_type": "correct", "jlpt_level": "N3"},
            {"incorrect": "この問題は難しすぎて解けませんでした。", "correct": "この問題は難しすぎて解けませんでした。", "error_type": "correct", "jlpt_level": "N3"},
        ]
        
        # Store JLPT-based data
        conn = sqlite3.connect(self.db_path)
        for item in jlpt_data:
            conn.execute('''
                INSERT INTO raw_training_data 
                (incorrect_text, correct_text, error_type, source, difficulty_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (item["incorrect"], item["correct"], item["error_type"], f"jlpt_{item['jlpt_level']}", 
                 {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}.get(item["jlpt_level"], 3)))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(jlpt_data)} JLPT-based examples")
        return len(jlpt_data)
    
    def process_and_format_data(self):
        """Process raw data into training format"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get all raw training data
        df = pd.read_sql_query('''
            SELECT incorrect_text, correct_text, error_type, difficulty_level 
            FROM raw_training_data
        ''', conn)
        
        processed_data = []
        
        for _, row in df.iterrows():
            # Format for T5: "grammar: <incorrect>" -> "<correct>"
            input_text = f"grammar: {row['incorrect_text']}"
            target_text = row['correct_text']
            
            processed_data.append({
                'input_text': input_text,
                'target_text': target_text,
                'error_category': row['error_type'],
                'confidence_score': 1.0 if row['difficulty_level'] <= 2 else 0.8
            })
        
        # Store processed data
        for item in processed_data:
            conn.execute('''
                INSERT INTO processed_training_data 
                (input_text, target_text, error_category, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (item['input_text'], item['target_text'], 
                  item['error_category'], item['confidence_score']))
        
        conn.commit()
        conn.close()
        
        print(f"Processed {len(processed_data)} training examples")
        return len(processed_data)
    
    def export_training_data(self, output_file="japanese_grammar_training.json"):
        """Export processed data for fine-tuning"""
        
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query('''
            SELECT input_text, target_text, error_category, confidence_score
            FROM processed_training_data
        ''', conn)
        
        conn.close()
        
        # Convert to format expected by fine-tuning script
        training_data = []
        for _, row in df.iterrows():
            training_data.append({
                "input_text": row['input_text'],
                "target_text": row['target_text'],
                "error_category": row['error_category'],
                "confidence": row['confidence_score']
            })
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(training_data)} examples to {output_file}")
        return output_file
    
    def get_training_stats(self):
        """Get statistics about collected training data"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Total counts
        total_raw = conn.execute("SELECT COUNT(*) FROM raw_training_data").fetchone()[0]
        total_processed = conn.execute("SELECT COUNT(*) FROM processed_training_data").fetchone()[0]
        
        # Error type distribution
        error_dist = pd.read_sql_query('''
            SELECT error_type, COUNT(*) as count 
            FROM raw_training_data 
            GROUP BY error_type 
            ORDER BY count DESC
        ''', conn)
        
        # Difficulty distribution  
        diff_dist = pd.read_sql_query('''
            SELECT difficulty_level, COUNT(*) as count 
            FROM raw_training_data 
            GROUP BY difficulty_level 
            ORDER BY difficulty_level
        ''', conn)
        
        conn.close()
        
        print(f"\n=== Training Data Statistics ===")
        print(f"Total raw examples: {total_raw}")
        print(f"Total processed examples: {total_processed}")
        print(f"\nError Type Distribution:")
        print(error_dist.to_string(index=False))
        print(f"\nDifficulty Level Distribution:")
        print(diff_dist.to_string(index=False))
        
        return {
            'total_raw': total_raw,
            'total_processed': total_processed,
            'error_distribution': error_dist,
            'difficulty_distribution': diff_dist
        }

# Main execution
def collect_all_training_data():
    """Collect training data from all methods"""
    
    collector = JapaneseTrainingDataCollector()
    
    print("Collecting Japanese Grammar Correction Training Data...")
    print("=" * 60)
    
    # Method 1: Expert curated data
    print("Method 1: Expert Curated Data")
    collector.method_1_manual_expert_data()
    
    # Method 2: Synthetic data generation  
    print("\nMethod 2: Synthetic Data Generation")
    collector.method_2_synthetic_data_generation()
    
    # Method 3: Lang-8 style data
    print("\nMethod 3: Lang-8 Style Learner Data")
    collector.method_3_lang8_style_data()
    
    # Method 4: JLPT-based data
    print("\nMethod 4: JLPT-Based Data")
    collector.method_4_jlpt_based_data()
    
    # Process all data
    print("\nProcessing Data for Training...")
    collector.process_and_format_data()
    
    # Export for fine-tuning
    print("\nExporting Training Data...")
    output_file = collector.export_training_data()
    
    # Show statistics
    collector.get_training_stats()
    
    print(f"\n✅ Training data ready! Use '{output_file}' for fine-tuning.")
    return output_file

if __name__ == "__main__":
    training_file = collect_all_training_data()