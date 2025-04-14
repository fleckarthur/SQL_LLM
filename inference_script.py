import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# 🚀 Paths to the model and tokenizer
MODEL_PATH = "/home/ayman/workspace/LLM_Project/qlora_finetuned_model"  # Replace with your actual save path

# 💻 Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,  # Ensure it matches the max_seq_length during training
    device_map="auto"  # Automatically maps to available GPU or CPU
)

# Set the model to evaluation mode
model.eval()

# 🧐 Define the inference function
def generate_response(prompt, max_length=2048, temperature=0.7, top_p=0.9):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Move input tensors to the appropriate device
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,  # Enables sampling for diverse outputs
        )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🧾 Inference example
if __name__ == "__main__":
    prompt = "<|user|>\nExplain the advantages of using LoRA for fine-tuning large language models.\n<|assistant|>\n"
    response = generate_response(prompt)
    print("Response:", response)
