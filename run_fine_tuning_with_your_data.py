import json
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import os

class GrammarCorrectionDataset(Dataset):
    """Custom Dataset for Japanese Grammar Correction Fine-tuning"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Use your data format: "grammar: ..." -> "correct text"
        source_text = item['input_text']  # Already formatted as "grammar: ..."
        target_text = item['target_text']
        
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
    """Fine-tune T5 model with YOUR collected training data"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Use a smaller model for faster training on CPU/limited GPU
        model_name = "google/mt5-small"  # Multilingual T5 that supports Japanese
        
        print(f"🚀 Using device: {self.device}")
        print(f"📥 Loading model: {model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_your_training_data(self, json_file="japanese_grammar_training.json"):
        """Load the training data YOU created"""
        
        print(f"📂 Loading your training data from {json_file}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {json_file} not found!")
            print("💡 Make sure you ran 'python training_data_collection.py' first")
            return None
        
        print(f"✅ Loaded {len(data)} training examples")
        
        # Convert to DataFrame for easier handling
        df_data = []
        for item in data:
            df_data.append({
                'input_text': item['input_text'],    # "grammar: ..."
                'target_text': item['target_text'],  # "correct sentence"
                'error_category': item.get('error_category', 'unknown')
            })
        
        df = pd.DataFrame(df_data)
        
        # Show sample data
        print("\n📋 Sample of your training data:")
        print("=" * 80)
        for i, row in df.head(3).iterrows():
            print(f"Input:  {row['input_text']}")
            print(f"Target: {row['target_text']}")
            print(f"Type:   {row['error_category']}")
            print("-" * 80)
        
        return df
    
    def prepare_data(self, df, test_size=0.2, batch_size=4):
        """Prepare data loaders for training and validation"""
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        
        print(f"\n🔄 Data split:")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        
        # Create datasets
        train_dataset = GrammarCorrectionDataset(train_df, self.tokenizer)
        val_dataset = GrammarCorrectionDataset(val_df, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def fine_tune(self, train_loader, val_loader, epochs=3, learning_rate=5e-5):
        """Actually fine-tune the model weights - THIS IS THE REAL FINE-TUNING!"""
        
        print(f"\n🏋️ Starting ACTUAL fine-tuning process...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Total training steps: {len(train_loader) * epochs}")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=10,
            num_training_steps=total_steps
        )
        
        # Initialize wandb for experiment tracking (optional)
        try:
            wandb.init(
                project="japanese-grammar-finetuning",
                config={
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "model": "mt5-small"
                }
            )
            use_wandb = True
        except:
            print("⚠️  Wandb not available, continuing without logging")
            use_wandb = False
        
        self.model.train()
        
        print("🚀 Fine-tuning started! This will take a while...")
        print("⏱️  Estimated time: 15-30 minutes depending on your hardware")
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - THIS IS THE ACTUAL FINE-TUNING!
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass - UPDATE MODEL WEIGHTS!
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()  # <-- THIS UPDATES THE WEIGHTS!
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
            
            # Validation after each epoch
            val_loss = self.validate(val_loader)
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"  Training Loss: {total_loss/len(train_loader):.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            
            if use_wandb:
                wandb.log({"val_loss": val_loss, "epoch": epoch})
            
            # Save checkpoint after each epoch
            checkpoint_dir = f"./checkpoints/grammar-correction-epoch-{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.save_model(checkpoint_dir)
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
        
        print("🎉 Fine-tuning completed successfully!")
        
        # Save final model
        final_model_dir = "./fine-tuned-japanese-grammar-model"
        self.save_model(final_model_dir)
        print(f"💾 Final model saved: {final_model_dir}")
    
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
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✅ Model saved to {path}")
    
    def test_fine_tuned_model(self, test_sentences=None):
        """Test the fine-tuned model"""
        
        if test_sentences is None:
            test_sentences = [
                "grammar: 私が学生です。",
                "grammar: 本が読みます。", 
                "grammar: 昨日映画を見ます。",
                "grammar: 学校に行きました昨日。",
                "grammar: 私の名前田中です。"
            ]
        
        print("\n🧪 Testing your fine-tuned model:")
        print("=" * 80)
        
        self.model.eval()
        
        for sentence in test_sentences:
            inputs = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )
            
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Input:     {sentence}")
            print(f"Corrected: {corrected}")
            print("-" * 80)
        
        print("✅ Testing completed!")

# MAIN EXECUTION - RUN YOUR FINE-TUNING!
def run_fine_tuning_with_your_data():
    """Run the complete fine-tuning pipeline with YOUR training data"""
    
    print("🎌 Japanese Grammar Correction Fine-Tuning")
    print("Using YOUR collected training data!")
    print("=" * 60)
    
    # Initialize fine-tuner
    fine_tuner = JapaneseGrammarFineTuner()
    
    # Load YOUR training data
    df = fine_tuner.load_your_training_data("japanese_grammar_training.json")
    if df is None:
        print("❌ Could not load training data. Exiting.")
        return
    
    # Prepare data loaders
    train_loader, val_loader = fine_tuner.prepare_data(df, batch_size=2)  # Small batch for memory
    
    # Run the ACTUAL fine-tuning process
    print("\n" + "="*60)
    print("🚀 STARTING ACTUAL FINE-TUNING - THIS WILL UPDATE MODEL WEIGHTS!")
    print("="*60)
    
    fine_tuner.fine_tune(
        train_loader, 
        val_loader, 
        epochs=3,           # Start with 3 epochs
        learning_rate=5e-5  # Conservative learning rate
    )
    
    # Test the fine-tuned model
    print("\n" + "="*60)
    print("🧪 TESTING YOUR FINE-TUNED MODEL")
    print("="*60)
    fine_tuner.test_fine_tuned_model()
    
    print("\n🎉 CONGRATULATIONS, successfully fine-tuned model")

if __name__ == "__main__":
    run_fine_tuning_with_your_data()