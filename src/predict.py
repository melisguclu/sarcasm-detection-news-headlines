import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model
model_path = "./models/sarcasm_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return "Sarcastic 🤨" if predicted_class == 1 else "Not Sarcastic ✅"
