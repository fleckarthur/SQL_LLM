import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== Load Base and Fine-Tuned Model ======
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "./LLM_Project/qlora_finetuned_model")
model.eval()

# ====== Load Tokenizer ======
tokenizer = AutoTokenizer.from_pretrained("./LLM_Project/qlora_finetuned_model")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ====== Inference Function ======
def generate_sql(instruction: str, schema: str = "") -> str:
    prompt = (
        f"<instruction>\n{instruction.strip()}\n</instruction>\n"
        f"<schema>\n{schema.strip()}\n</schema>\n"
        f"<response>"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== Interactive CLI ======
if __name__ == "__main__":
    print("💬 Welcome to the SQL Query Generator!")
    print("Type your natural language instruction (type 'exit' to quit):")

    while True:
        instruction = input("\n📝 Instruction: ").strip()
        if instruction.lower() in {"exit", "quit"}:
            print("👋 Exiting. Have a great day!")
            break

        schema = input("📘 (Optional) Schema: ").strip()
        print("\n⏳ Generating SQL Query...\n")
        sql = generate_sql(instruction, schema)
        print("🧾 Generated SQL Query:\n")
        print(sql)
