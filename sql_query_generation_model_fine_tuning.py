# Import necessary libraries
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the Spider dataset
dataset = load_dataset("xlangai/spider")

# Load the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Preprocess function to prepare the data
def preprocess_data(batch):
    # Add "Translate to SQL: " to each question in the batch and tokenize
    inputs = tokenizer(["Translate to SQL: " + q for q in batch['question']],
                       max_length=512, truncation=True, padding="max_length")

    # Tokenize each SQL query in the batch
    targets = tokenizer(batch['query'], max_length=512, truncation=True, padding="max_length")

    # Set labels for training
    inputs['labels'] = targets['input_ids']
    return inputs


# Apply the preprocessing to the entire dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./sql_gen_t5",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    no_cuda=True,  # Train on CPU
)


# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)


# Start training
trainer.train()


# Define a sample prompt for testing
prompt = "Show all employees with a salary over 3000."

# Tokenize the input prompt
inputs = tokenizer("Translate to SQL: " + prompt, return_tensors="pt")

# Generate the SQL query
outputs = model.generate(**inputs)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated SQL: {sql_query}")


# Save the model and tokenizer
model.save_pretrained("./sql_gen_t5")
tokenizer.save_pretrained("./sql_gen_t5")
