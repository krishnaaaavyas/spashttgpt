import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import time
import os  # Added for saving

class RiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RiskClassifier(nn.Module):
    def __init__(self, n_classes, pre_trained_model):
        super(RiskClassifier, self).__init__()
        self.bert = pre_trained_model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output.last_hidden_state[:, 0, :])
        return self.out(output)

def train_model(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    
    for batch_idx, d in enumerate(progress_bar):
        input_ids = d["input_ids"].to(device, non_blocking=True)
        attention_mask = d["attention_mask"].to(device, non_blocking=True)
        labels = d["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        current_acc = correct_predictions.double() / ((batch_idx + 1) * data_loader.batch_size)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}',
                'GPU': f'{gpu_memory:.1f}GB'
            })
        else:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def predict(model, text, tokenizer, device, max_len, label_mapping):
    model.eval()
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    texts = [text]
    labels = [0]  # Dummy label

    dataset = RiskDataset(texts, labels, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs, dim=1)

    return reverse_label_mapping.get(prediction.item(), "Unknown")

if __name__ == "__main__":
    # ⚡ FAST TRAINING PARAMETERS
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LEN = 128
    BATCH_SIZE = 20  # Increased for better GPU utilization
    EPOCHS = 3       # 🔥 REDUCED FROM 5 TO 3 FOR FASTER TRAINING
    LEARNING_RATE = 2e-5

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("🚀 MEDBERT FAST TRAINING SETUP")
    print("="*60)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️  No GPU detected - training will be slow")
        BATCH_SIZE = 8
    
    print("="*60)

    try:
        print("Loading tokenizer and pre-trained MedBERT model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        pre_trained_bert = AutoModel.from_pretrained(MODEL_NAME)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

    try:
        print("\nLoading data from healthcare.csv...")
        df = pd.read_csv('healthcare.csv')

        df['labels'] = df['labels'].fillna('unknown').str.lower().str.strip()
        unique_labels = sorted(df['labels'].unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        print(f"Found labels: {unique_labels}")
        df['labels'] = df['labels'].map(label_mapping)

        if df['labels'].isna().any():
            df = df.dropna(subset=['labels'])

        texts = df['text'].tolist()
        labels = df['labels'].astype(int).tolist()

        # 🔥 OPTIONAL: Use smaller dataset for SUPER FAST testing
        USE_SUBSET = True  # Change to False for full dataset
        if USE_SUBSET and len(texts) > 8000:
            texts = texts[:8000]
            labels = labels[:8000]
            print(f"🚀 Using subset of {len(texts)} samples for FAST training")

        print(f"✅ Total samples: {len(texts)}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    except FileNotFoundError:
        print("❌ Error: healthcare.csv not found.")
        exit()

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    num_classes = len(label_mapping)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Batches per epoch: {len(train_texts) // BATCH_SIZE}")

    # Create optimized data loaders
    train_dataset = RiskDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Move model to device
    print(f"\n🚀 Moving model to {device}...")
    model = RiskClassifier(n_classes=num_classes, pre_trained_model=pre_trained_bert)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if torch.cuda.is_available():
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 🔥 FAST TRAINING LOOP (3 EPOCHS)
    print("\n" + "="*60)
    print("🚀 STARTING FAST TRAINING (3 EPOCHS)")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch + 1}/{EPOCHS}")
        epoch_start = time.time()
        
        train_acc, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, len(train_texts)
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"✅ Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Time: {epoch_time:.1f}s")
        
        if torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    total_time = time.time() - start_time
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"⏰ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Quick evaluation
    test_dataset = RiskDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model.eval()
    y_pred, y_test = [], []

    print("\n📊 Evaluating model...")
    with torch.no_grad():
        for d in tqdm(test_data_loader, desc="Testing"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_test.extend(labels.cpu().tolist())

    # Calculate final accuracy
    final_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    unique_test_labels = sorted(set(y_test))
    test_target_names = [reverse_label_mapping[label] for label in unique_test_labels]

    print(f"🎯 Accuracy: {final_accuracy:.4f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=test_target_names, labels=unique_test_labels))

    # 💾 SAVE THE TRAINED MODEL
    print("\n" + "="*60)
    print("💾 SAVING TRAINED MODEL")
    print("="*60)
    
    # Create model directory
    model_dir = "saved_medbert_model"
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Save the BERT model and tokenizer
        print("Saving BERT model and tokenizer...")
        model.bert.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Save the classifier weights and metadata
        print("Saving classifier weights and metadata...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_mapping': label_mapping,
            'num_classes': num_classes,
            'model_config': {
                'max_len': MAX_LEN,
                'model_name': MODEL_NAME,
                'batch_size': BATCH_SIZE,
                'epochs_trained': EPOCHS,
                'learning_rate': LEARNING_RATE
            },
            'training_info': {
                'final_accuracy': final_accuracy,
                'training_time_seconds': total_time,
                'training_samples': len(train_texts),
                'test_samples': len(test_texts),
                'device_used': str(device)
            }
        }, f"{model_dir}/classifier_weights.pth")
        
        print(f"✅ Model successfully saved to '{model_dir}' directory!")
        print(f"📁 Files saved:")
        print(f"   - pytorch_model.bin (BERT weights)")
        print(f"   - config.json (model configuration)")
        print(f"   - tokenizer files")
        print(f"   - classifier_weights.pth (custom classifier + metadata)")
        print(f"📊 Final accuracy: {final_accuracy:.4f}")
        print(f"⏰ Training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    # Test prediction
    print("\n🔬 PREDICTION TEST")
    print("="*60)
    new_note = "Patient requests specific controlled substances by name without examination."
    risk_level = predict(model, new_note, tokenizer, device, MAX_LEN, label_mapping)
    print(f"Clinical Note: '{new_note}'")
    print(f"🎯 Predicted Risk Level: {risk_level}")
    
    print("\n✅ Training complete! Model ready for use.")
    print(f"💾 Your trained model is saved in: {os.path.abspath(model_dir)}")
