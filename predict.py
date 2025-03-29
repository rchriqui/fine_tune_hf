import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the model and tokenizer
def load_model(model_path="./phi3-mini-yoda-adapter"):
    # Define quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )
    
    # Load base model
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer

# Format the input prompt
def gen_prompt(tokenizer, sentence):
    converted_sample = [
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(converted_sample, 
                                           tokenize=False, 
                                           add_generation_prompt=True)
    return prompt

# Generate text from the model
def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    model.eval()
    generation_output = model.generate(**tokenized_input,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=max_new_tokens)
    
    output = tokenizer.batch_decode(generation_output, 
                                    skip_special_tokens=skip_special_tokens)
    return output[0]

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test with a sentence
    sentence = "The Force is strong in you!"
    prompt = gen_prompt(tokenizer, sentence)
    output = generate(model, tokenizer, prompt)
    
    print(f"Input: {sentence}")
    print(f"Output: {output}")
    
    # Interactive mode
    while True:
        user_input = input("\nEnter a sentence (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        prompt = gen_prompt(tokenizer, user_input)
        output = generate(model, tokenizer, prompt)
        print(f"Yoda says: {output}")
