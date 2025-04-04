import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ====== Memory Optimization Settings ======
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ====== Configuration ======
base_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
output_dir = "./LLM_Project/qlora_finetuned_model"
dataset_path = "/home/ayman/LLM_Project/Data_Preparation/fine_tuning_train.jsonl"
eval_dataset_path = "/home/ayman/LLM_Project/Data_Preparation/fine_tuning_eval.jsonl"

# ====== Load Dataset ======
dataset = load_dataset("json", data_files=dataset_path, split="train")
eval_dataset = load_dataset("json", data_files=eval_dataset_path, split="train")
print(f"📦 Loaded {len(dataset)} training and {len(eval_dataset)} evaluation samples.")

# ====== Tokenizer Setup ======
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ====== Dynamic Tokenization ======
def tokenize(example):
    prompt = (
        f"<instruction>\n{example['instruction']}\n</instruction>\n"
        f"<schema>\n{example['input']}\n</schema>\n"
        f"<response> {example['output']}"
    )
    return tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
    )

# ====== Tokenize and Filter Datasets ======
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized_dataset = tokenized_dataset.filter(lambda x: 10 < len(x["input_ids"]) < 2048)

tokenized_eval = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)
tokenized_eval = tokenized_eval.filter(lambda x: 10 < len(x["input_ids"]) < 2048)

# ====== 4-bit Quantization Config ======
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ====== Load Base Model in 4-bit ======
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# ====== Prepare Model for QLoRA ======
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# ====== LoRA Configuration ======
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====== Training Arguments ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    optim="adamw_bnb_8bit",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    report_to="none",
    dataloader_num_workers=4,
)

# ====== Data Collator ======
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ====== Trainer Setup ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ====== Free Up Memory ======
torch.cuda.empty_cache()

# ====== Start Training ======
print("🚀 Starting QLoRA fine-tuning on DeepSeek 6.7B...")
try:
    trainer.train()
except Exception as e:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    partial_dir = f"{output_dir}_partial_{timestamp}"
    print(f"❌ Training failed: {str(e)}")
    print(f"💾 Saving partial model to {partial_dir}")
    model.save_pretrained(partial_dir)
    tokenizer.save_pretrained(partial_dir)

# ====== Save Final Model ======
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Fine-tuning complete and model saved to disk!")
