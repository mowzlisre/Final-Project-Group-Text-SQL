import torch
from transformers import BertTokenizer
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertCustomModel(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.5):
        super(BertCustomModel, self).__init__()
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        
        # Attention Layer
        self.attention = nn.Linear(768, 1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(768 + 768 + 128, 512)  # Combine CLS and attention outputs
        self.fc2 = nn.Linear(512, num_labels)
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask):
        # Extract BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token (batch_size, 768)
        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        
        # Attention Mechanism
        attention_weights = self.attention(token_embeddings)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize weights
        weighted_embeddings = (token_embeddings * attention_weights).sum(dim=1)  # (batch_size, 768)
        
        # Pass through convolutional layers
        token_embeddings_permuted = token_embeddings.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        x = self.conv1(token_embeddings_permuted)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.mean(x, dim=2)  # Global average pooling (batch_size, 128)
        
        # Combine CLS token and attention output
        combined = torch.cat((cls_token, weighted_embeddings, x), dim=1)  # (batch_size, 768 + 768 + 128)
        
        # Pass through fully connected layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

label_columns = ['SELECT', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'ASC', 'DESC',
       'LIMIT', 'OFFSET', 'LIKE', 'BETWEEN', 'IN', 'IS NULL', 'IS NOT NULL']

device = "cuda" if torch.cuda.is_available() else "cpu"
global model
model = BertCustomModel(num_labels=len(label_columns)).to(device)
model.load_state_dict(torch.load("./models/BERT_SQL_MODEL.pth", map_location=device))
model.eval()

# Load the tokenizer
global tokenizer
tokenizer = BertTokenizer.from_pretrained("./models/BERT_SQL_TOKENIZER")
# Prediction function
def predict(model, tokenizer, phrase, max_length=128, threshold=0.5):
    model.eval()
    predictions = []
    with torch.no_grad():
        # Tokenize input
        encoding = tokenizer(phrase, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Model forward pass
        outputs = model(input_ids, attention_mask)
        logits = torch.sigmoid(outputs).cpu().numpy()
        
        # Apply threshold
        preds = (logits > threshold).astype(int)
        predictions.append(preds.flatten())
    
    return np.array(predictions)

def get_labels(prompt):
    prediction = predict(model, tokenizer, prompt)
    return prediction[0]