Phi-3 Yoda Speech Fine-Tuning
This project demonstrates how to fine-tune Microsoft's Phi-3-mini-4k-instruct model to mimic Yoda's speech patterns using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
Project Overview
This repository contains scripts to:

Fine-tune the Phi-3-mini model using the "dvgodoy/yoda_sentences" dataset
Make the model respond to user inputs in Yoda's distinctive speech style
Predict responses using the fine-tuned model

Requirements
The project requires the following Python packages:
Copiertransformers==4.46.2
peft==0.13.2
accelerate==1.1.1
trl==0.12.1
bitsandbytes==0.45.2
datasets==3.1.0
huggingface-hub==0.26.2
safetensors==0.4.5
pandas==2.2.2
matplotlib==3.8.0
numpy==1.26.4
You can install all dependencies using:
bashCopierpip install -r requirements.txt
Model Architecture
This project uses:

Base Model: microsoft/Phi-3-mini-4k-instruct
Fine-tuning Method: Low-Rank Adaptation (LoRA)
Quantization: 4-bit quantization with NF4 format
Target Modules: 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'

Training
The train.py script handles the fine-tuning process:
bashCopierpython train.py
Key training parameters:

Learning rate: 3e-4
Epochs: 10
Batch size: Auto-determined with gradient accumulation
Optimizer: 8-bit AdamW with paging

The script will save the fine-tuned LoRA adapter to "./phi3-mini-yoda-adapter".
Making Predictions
After training, you can use the predict.py script to generate Yoda-style responses:
bashCopierpython predict.py
The script provides an interactive prompt where you can type sentences and receive Yoda-style responses.
Example Usage
pythonCopier# Load the model and tokenizer
model, tokenizer = load_model("./phi3-mini-yoda-adapter")

# Generate a Yoda-style response
sentence = "The Force is strong in you!"
prompt = gen_prompt(tokenizer, sentence)
output = generate(model, tokenizer, prompt)
print(output)  # Will output Yoda-style speech
Training Results
The model achieves a loss of approximately 0.11-0.15 during training, indicating successful adaptation to Yoda's speech patterns. The fine-tuning process modifies only about 1-2% of the model parameters while retaining the base model's capabilities.
Credits

The Yoda sentences dataset: "dvgodoy/yoda_sentences" on Hugging Face
Microsoft's Phi-3-mini-4k-instruct model
Hugging Face's Transformers, PEFT, and TRL libraries
