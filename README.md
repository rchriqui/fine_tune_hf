# 🧠 Phi-3 Yoda Speech Fine-Tuning

This project demonstrates how to fine-tune Microsoft's `phi-3-mini-4k-instruct` model to mimic **Yoda's** speech patterns using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**.

---

## 🗂 Project Overview

This repository includes scripts to:

- Fine-tune the Phi-3-mini model using the [`dvgodoy/yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences) dataset  
- Make the model respond to user inputs in **Yoda's iconic speech style**
- Generate predictions using the fine-tuned model

---

## 📦 Requirements

You'll need the following Python packages:

transformers==4.46.2
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

csharp
Copy
Edit

Install them with:

```bash
pip install -r requirements.txt
🏗️ Model Architecture
Base Model: microsoft/phi-3-mini-4k-instruct

Fine-tuning Method: LoRA (Low-Rank Adaptation)

Quantization: 4-bit quantization (NF4 format)

Target Modules: 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'

🧪 Training
The train.py script handles the full fine-tuning process:

bash
Copy
Edit
python train.py
Key training params:

Learning rate: 3e-4

Epochs: 10

Batch size: Auto-determined (with gradient accumulation)

Optimizer: 8-bit AdamW with paging

Output: The LoRA adapter will be saved to:

bash
Copy
Edit
./phi3-mini-yoda-adapter
🤖 Making Predictions
After training, use predict.py to generate Yoda-style responses:

bash
Copy
Edit
python predict.py
The script gives you an interactive prompt to type normal sentences and get responses like a Jedi Master.

💡 Example Usage
python
Copy
Edit
# Load the model and tokenizer
model, tokenizer = load_model("./phi3-mini-yoda-adapter")

# Generate a Yoda-style response
sentence = "The Force is strong in you!"
prompt = gen_prompt(tokenizer, sentence)
output = generate(model, tokenizer, prompt)
print(output)  # Expect some wise Yoda-speak
📈 Training Results
Final training loss: ~0.11–0.15

Only ~1–2% of the model parameters were updated (via LoRA)

The base model's performance is preserved while gaining Yoda-style generation

🙏 Credits
🧙 Yoda Dataset: dvgodoy/yoda_sentences

🤖 Base Model: microsoft/phi-3-mini-4k-instruct

🧠 Libraries: Transformers, PEFT, TRL by Hugging Face
