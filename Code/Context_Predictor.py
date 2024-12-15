import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .Label_Predictor import get_labels
from .lib.similarity import process_prompt
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the state_dict
model_save_path = "./models/context_model.pth"  # Path to your saved model
state_dict = torch.load(model_save_path, map_location=device)

# Load weights into the model
model.load_state_dict(state_dict)

# Move model to the appropriate device
model.to(device)

# Set the model to evaluation mode
model.eval()

def predict_sql(model, tokenizer, phrase, labels, label_columns, max_length=128):
    label_text = ", ".join([f"{label}:{value}" for label, value in zip(label_columns, labels)])
    input_text = f"Prompt: {phrase} Using Labels: {label_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    output_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    label_columns = ['SELECT', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'ASC', 'DESC',
       'LIMIT', 'OFFSET', 'LIKE', 'BETWEEN', 'IN', 'IS NULL', 'IS NOT NULL']

    # Test a prompt
    prompt = "What are the top expensive products"
    redefined_prompt = process_prompt(prompt)
    # Process the prompt
    if redefined_prompt == "Nothing":
        print("The prompt does not match the schema. Exiting.")
    else:
        labels = get_labels(redefined_prompt)
        predicted_sql = predict_sql(model, tokenizer, redefined_prompt, labels, label_columns)
        print(f"{predicted_sql}")

