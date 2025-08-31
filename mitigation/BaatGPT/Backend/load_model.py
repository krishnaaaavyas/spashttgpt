import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import os

# Import your custom classes (make sure they match your original definitions)
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

def load_trained_model(model_dir="../saved_medbert_model"):
    """Load the trained model from the saved directory"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_dir}...")
    print(f"Using device: {device}")
    
    # Load the saved metadata and weights
    checkpoint_path = os.path.join(model_dir, "classifier_weights.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    label_mapping = checkpoint['label_mapping']
    num_classes = checkpoint['num_classes']
    model_config = checkpoint['model_config']
    
    print(f"Number of classes: {num_classes}")
    print(f"Labels: {list(label_mapping.keys())}")
    
    # Load tokenizer and BERT model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert_model = AutoModel.from_pretrained(model_dir)
    
    # Recreate the classifier
    model = RiskClassifier(n_classes=num_classes, pre_trained_model=bert_model)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    return model, tokenizer, label_mapping, model_config, device

def predict_single(model, tokenizer, text, label_mapping, max_len, device):
    """Make a prediction on a single text input"""
    model.eval()
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Prepare the input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)
        confidence = probabilities[0][prediction].item()
    
    predicted_label = reverse_label_mapping.get(prediction.item(), "Unknown")
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()

def predict_batch(model, tokenizer, texts, label_mapping, max_len, device, batch_size=16):
    """Make predictions on a batch of texts"""
    model.eval()
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Create dataset and dataloader
    dummy_labels = [0] * len(texts)  # Dummy labels for prediction
    dataset = RiskDataset(texts, dummy_labels, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, batch_predictions = torch.max(outputs, dim=1)
            
            for i, pred in enumerate(batch_predictions):
                predicted_label = reverse_label_mapping.get(pred.item(), "Unknown")
                confidence = probabilities[i][pred].item()
                predictions.append(predicted_label)
                confidences.append(confidence)
    
    return predictions, confidences
