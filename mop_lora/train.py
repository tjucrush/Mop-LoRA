import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from utils import inject_mop_lora, print_trainable_parameters

def main():
    parser = argparse.ArgumentParser(description="Fine-tune with MoP-LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-7B")
    parser.add_argument("--dataset_path", type=str, default="data/400k_mixed_data")
    parser.add_argument("--r", type=int, default=8, help="MoP-LoRA rank")
    parser.add_argument("--num_paths", type=int, default=4, help="Number of MoP paths")
    parser.add_argument("--output_dir", type=str, default="./mop_lora_output")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Automatically set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Injecting MoP-LoRA modules (r={args.r}, paths={args.num_paths})...")
    # Commonly targeted projection modules for Llama / DeepSeek / Qwen
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    replaced = inject_mop_lora(model, target_modules=target_modules, r=args.r, num_paths=args.num_paths)
    print(f"Replaced {replaced} linear layers with MoP-LoRA.")
    
    print_trainable_parameters(model)

    # Basic data handling placeholder
    # dataset = load_dataset("json", data_files=args.dataset_path)
    # def tokenize_fn(examples):
    #     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)
    # tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="epoch",
        bf16=True,  # recommended for modern LLaMA & DeepSeek models
        report_to="none"
    )

    # Initialize HuggingFace Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    # )
    
    print("Ready for supervised fine-tuning (SFT) over 400k mixed data! \nRun trainer.train() to begin.")

if __name__ == "__main__":
    main()
