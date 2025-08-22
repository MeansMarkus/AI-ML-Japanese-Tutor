import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    AdamW,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import sqlite3

class GrammarCorrectionDataset(Dataset):
    """Custom Dataset for Japanese Grammar Correction Fine-tuning"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Format: "correct: <incorrect_sentence>" -> "<correct_sentence>"
        source_text = f"correct: {item['incorrect_text']}"
        target_text = item['correct_text']
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target 
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # For T5, labels are the target token IDs
        labels = target_encoding['input_ids']
        # Replace padding token id with -100 (ignored in loss calculation)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class JapaneseGrammarFineTuner:
    """Fine-tune T5 model specifically for Japanese grammar correction"""
    
    def __init__(self, model_name="sonoisa/t5-base-japanese", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        print(f"Model loaded: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_training_data(self):
        """Create comprehensive training dataset for Japanese grammar correction"""
        
        # Expanded training data with common Japanese grammar mistakes
        training_examples = [
            # Particle errors (は/が confusion)
            {"incorrect_text": "私が学生です。", "correct_text": "私は学生です。", "error_type": "particle"},
            {"incorrect_text": "日本語が好きです。", "correct_text": "日本語が好きです。", "error_type": "correct"},
            {"incorrect_text": "今日は天気は良いです。", "correct_text": "今日は天気が良いです。", "error_type": "particle"},
            
            # を/が confusion
            {"incorrect_text": "本が読みます。", "correct_text": "本を読みます。", "error_type": "particle"},
            {"incorrect_text": "音楽が聞きます。", "correct_text": "音楽を聞きます。", "error_type": "particle"},
            {"incorrect_text": "映画が見ました。", "correct_text": "映画を見ました。", "error_type": "particle"},
            
            # Word order errors
            {"incorrect_text": "学校に行きました昨日私は。", "correct_text": "私は昨日学校に行きました。", "error_type": "word_order"},
            {"incorrect_text": "友達と会います明日。", "correct_text": "明日友達と会います。", "error_type": "word_order"},
            {"incorrect_text": "日本語を勉強しています大学で。", "correct_text": "大学で日本語を勉強しています。", "error_type": "word_order"},
            
            # Tense errors
            {"incorrect_text": "昨日映画を見ます。", "correct_text": "昨日映画を見ました。", "error_type": "tense"},
            {"incorrect_text": "明日東京に行きました。", "correct_text": "明日東京に行きます。", "error_type": "tense"},
            {"incorrect_text": "今朝パンを食べます。", "correct_text": "今朝パンを食べました。", "error_type": "tense"},
            
            # Adjective inflection
            {"incorrect_text": "この本は面白いでした。", "correct_text": "この本は面白かったです。", "error_type": "adjective"},
            {"incorrect_text": "昨日は寒いでした。", "correct_text": "昨日は寒かったです。", "error_type": "adjective"},
            {"incorrect_text": "料理は美味しいでした。", "correct_text": "料理は美味しかったです。", "error_type": "adjective"},
            
            # Missing particles
            {"incorrect_text": "私田中です。", "correct_text": "私は田中です。", "error_type": "missing_particle"},
            {"incorrect_text": "学校行きます。", "correct_text": "学校に行きます。", "error_type": "missing_particle"},
            {"incorrect_text": "友達話します。", "correct_text": "友達と話します。", "error_type": "missing_particle"},
            
            # Verb form errors
            {"incorrect_text": "日本語を勉強します毎日。", "correct_text": "毎日日本語を勉強します。", "error_type": "word_order"},
            {"incorrect_text": "本を読んでいました昨日。", "correct_text": "昨日本を読んでいました。", "error_type": "word_order"},
        ]
        
        # Add synthetic variations for robustness
        additional_examples = []
        base_sentences = [
            "私は{subject}です。",
            "{object}を{verb}ます。", 
            "{time}に{place}に行きます。"
        ]
        
        # Generate variations (simplified for demo)
        subjects = ["学生", "先生", "医者", "エンジニア"]
        objects = ["本", "映画", "音楽", "料理"]
        verbs = ["読み", "見", "聞き", "作り"]
        times = ["朝", "昼", "夜", "明日"]
        places = ["学校", "図書館", "映画館", "レストラン"]
        
        for subject in subjects[:2]:  # Limited for demo
            for obj, verb in zip(objects[:2], verbs[:2]):
                # Create incorrect version (common mistake pattern)
                incorrect = f"{obj}が{verb}ます。"  # Wrong particle
                correct = f"{obj}を{verb}ます。"    # Correct particle
                additional_examples.append({
                    "incorrect_text": incorrect,
                    "correct_text": correct, 
                    "error_type": "particle"
                })
        
        all_examples = training_examples + additional_examples
        return pd.DataFrame(all_examples)
    
    def prepare_data(self, df, test_size=0.2, batch_size=4):
        """Prepare data loaders for training and validation"""
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Create datasets
        train_dataset = GrammarCorrectionDataset(train_df, self.tokenizer)
        val_dataset = GrammarCorrectionDataset(val_df, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def fine_tune(self, train_loader, val_loader, epochs=5, learning_rate=1e-4):
        """Actually fine-tune the model weights"""
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="japanese-grammar-finetuning",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "model": "t5-base-japanese"
            }
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - THIS IS THE ACTUAL FINE-TUNING
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass - Update model weights
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Log to wandb
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # Validation after each epoch
            val_loss = self.validate(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            wandb.log({"val_loss": val_loss, "epoch": epoch})
            
            # Save checkpoint
            self.save_model(f"./checkpoints/grammar-correction-epoch-{epoch+1}")
    
    def validate(self, val_loader):
        """Validate the fine-tuned model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save_model(self, path):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def test_correction(self, incorrect_sentence):
        """Test the fine-tuned model on new sentences"""
        self.model.eval()
        
        input_text = f"correct: {incorrect_sentence}"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected

# Training Pipeline
def main():
    """Complete fine-tuning pipeline"""
    
    # Initialize fine-tuner
    fine_tuner = JapaneseGrammarFineTuner()
    
    # Create training data
    print("Creating training dataset...")
    df = fine_tuner.create_training_data()
    print(f"Created {len(df)} training examples")
    
    # Prepare data loaders
    train_loader, val_loader = fine_tuner.prepare_data(df, batch_size=4)
    
    # Fine-tune the model (THIS IS THE REAL FINE-TUNING)
    print("Starting fine-tuning process...")
    fine_tuner.fine_tune(train_loader, val_loader, epochs=3, learning_rate=1e-4)
    
    # Test the fine-tuned model
    test_sentences = [
        "私が学生です",
        "本が読みます", 
        "昨日映画を見ます"
    ]
    
    print("\nTesting fine-tuned model:")
    for sentence in test_sentences:
        corrected = fine_tuner.test_correction(sentence)
        print(f"Original: {sentence}")
        print(f"Corrected: {corrected}")
        print("-" * 40)

if __name__ == "__main__":
    main()