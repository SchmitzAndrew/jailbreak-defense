from datasets import Dataset
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

malicious_df = pd.read_csv('forbidden_question_set.csv')

url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
response = requests.get(url)
alpaca_data = response.json()

# Filter benign examples
benign_examples = [
    example.get("instruction", "") for example in alpaca_data
    if any(keyword in example.get("instruction", "").lower() 
           for keyword in ["explain", "how to", "what is"])
]

# Sample same number as malicious examples
num_malicious = len(malicious_df)
benign_examples_sampled = pd.Series(benign_examples).sample(n=num_malicious, random_state=42).tolist()

# Create BALANCED benign DataFrame
benign_df = pd.DataFrame({
    "content_policy_id": 1,
    "content_policy_name": "Benign",
    "q_id": range(num_malicious),
    "question": benign_examples_sampled
})
# ===== Compact Taxonomy =====
SAFETY_LABELS = {
    "malicious": 0,
    "benign": 1  # Now only 2 classes
}

# ===== Data Preparation =====
# For malicious data
malicious_df["label"] = SAFETY_LABELS["malicious"]

# For benign data
benign_df["label"] = SAFETY_LABELS["benign"]

# Combine datasets
balanced_df = pd.concat([malicious_df, benign_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final class distribution:\n{balanced_df['content_policy_name'].value_counts()}")


# Load tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
# ===== Model Setup =====
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small",
    num_labels=2  # Matches the 2 classes
)

# Tokenization with labels
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["question"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    tokenized["labels"] = examples["content_policy_id"]
    return tokenized

# Convert DataFrames to Dataset objects# Create train, validation, and test splits
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, stratify=balanced_df['content_policy_id'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['content_policy_id'])

# Add test dataset creation
train_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    logging_steps=10,  # Log every 10 steps
    report_to="tensorboard",  # Or "wandb"
    per_device_train_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train!
trainer.train()
# After training, evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")