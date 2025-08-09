from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ckiplab/gpt2-base-chinese"
save_path = "./models/gpt2_zh"

print("🚀 開始下載模型...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
print("✅ 模型載入成功，準備儲存...", flush=True)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"📦 已成功儲存 GPT2 模型到：{save_path}", flush=True)
