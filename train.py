import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments
)

# Force single GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load and prepare model
print("Loading model...")
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)

repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
   repo_id, 
   quantization_config=bnb_config,
   device_map="auto"
)

# Prepare for QLoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# Configure LoRA
config = LoraConfig(
    r=8,                   
    lora_alpha=16,
    bias="none",           
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)
model = get_peft_model(model, config)

# Display parameter stats
trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f'Trainable parameters: {trainable_parms/1e6:.2f}M')
print(f'Total parameters: {tot_parms/1e6:.2f}M')
print(f'Percentage trainable: {100*trainable_parms/tot_parms:.2f}%')

# Load dataset
print("Loading dataset...")
dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
dataset = dataset.rename_column("sentence", "prompt")
dataset = dataset.rename_column("translation_extra", "completion")
dataset = dataset.remove_columns(["translation"])

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a simple function to format chat and apply tokenization
def tokenize_function(examples):
    # For each prompt/completion pair
    inputs = []
    for i in range(len(examples["prompt"])):
        # Format as chat messages
        messages = [
            {"role": "user", "content": examples["prompt"][i]},
            {"role": "assistant", "content": examples["completion"][i]}
        ]
        
        # Apply the chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs.append(text)
    
    # Tokenize all texts together
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors=None  # Return lists, not tensors
    )
    
    # Use the input_ids as labels too for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Apply preprocessing to the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    remove_columns=dataset.column_names
)

# Print sample to validate format
print("\nSample dataset entry:")
sample_input = tokenized_dataset[0]["input_ids"]
print(f"Input type: {type(sample_input)}")
print(f"Input length: {len(sample_input)}")
print(f"Sample decoded: {tokenizer.decode(sample_input)}")

# Create data collator that handles tensors properly
class CustomDataCollator:
    def __call__(self, features):
        # Convert lists to tensors
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features])
        }
        return batch

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./phi3-mini-yoda-adapter",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    report_to="none",
)

# Initialize trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=CustomDataCollator()
)

# Train
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
model.save_pretrained("./phi3-mini-yoda-adapter")
print("Training complete!")
