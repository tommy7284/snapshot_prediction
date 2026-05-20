import os
import glob
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Model ID applied for on Hugging Face
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-snapshot-finetuned"

def read_all_csv(dir_path):
    search_pattern = os.path.join(dir_path, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Warning: No '*.csv' files found in '{dir_path}'.")
        return pd.DataFrame()

    print(f"📂 Loading {len(csv_files)} CSV files...")
    dfs_list = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty: continue
            df['source_file'] = os.path.basename(file_path)
            dfs_list.append(df)
        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"Error: Failed to read '{file_path}': {e}")

    if not dfs_list:
        return pd.DataFrame()

    return pd.concat(dfs_list, ignore_index=True)

def prepare_llama_dataset(X_data, y_data, tokenizer):
    """
    Takes lists created with Pandas and converts them into chat format for Llama training.
    """
    formatted_data = []
    for text, label in zip(X_data, y_data):
        prompt = (
            "You are an expert software engineer.\n"
            "Predict whether this commit diff will be reverted (1) or not (0).\n\n"
            f"{text}\n\n"
            f"[Prediction]\n{label}"
            f"{tokenizer.eos_token}" # Signal for end of response
        )
        formatted_data.append({"text": prompt})
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))


def main():
    DATA_REPO = "file_data" # Path to data directory
    
    # ---------------------------------------------------------
    # 1. Data Loading and Preprocessing
    # ---------------------------------------------------------
    df = read_all_csv(DATA_REPO)
    if df.empty: return
    df.dropna(subset=['commit_message'], inplace=True)
    df['commit_message'] = df['commit_message'].fillna('')
    df['diff_text'] = df['diff_text'].fillna('')
    df['text'] = "\n\n[commit_message]\n" + df['commit_message'].astype(str).str[:1500] + "\n\n[Diff]\n" + df['diff_text'].astype(str).str[:1500]
    df = df[['text', 'reverted']]

    # ---------------------------------------------------------
    # 2. Data Splitting and Undersampling
    # ---------------------------------------------------------
    df_train, df_val = train_test_split(df, test_size=0.3, random_state=777, stratify=df['reverted'])

    X_val = df_val['text'].tolist()
    y_val = df_val['reverted'].tolist()

    df_train_true = df_train[df_train['reverted'] == 1]
    df_train_false = df_train[df_train['reverted'] == 0]
    
    df_train_false_undersampled = df_train_false.sample(n=len(df_train_true), random_state=557)
    df_train_balanced = pd.concat([df_train_true, df_train_false_undersampled]).sample(frac=1, random_state=72).reset_index(drop=True)

    X_train = df_train_balanced['text'].tolist()
    y_train = df_train_balanced['reverted'].tolist()

    print(f"✅ Data preparation complete!")
    print(f"  - Training data (balanced): {len(X_train)} records (1: {sum(y_train)}, 0: {len(y_train)-sum(y_train)})")
    print(f"  - Evaluation data (original proportions): {len(X_val)} records (1: {sum(y_val)}, 0: {len(y_val)-sum(y_val)})")

    # ---------------------------------------------------------
    # 3. Conversion to Llama Dataset and Splitting
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_llama_dataset(X_train, y_train, tokenizer)
    val_dataset = prepare_llama_dataset(X_val, y_val, tokenizer)

    print("\n🚀 3. Loading the main model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    print("\n🚀 4. Applying LoRA configurations...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    print("\n🚀 5. Starting the training process...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  
        learning_rate=2e-4,             
        logging_steps=10,
        max_steps=200,                  
        save_steps=50,
        eval_strategy="steps",          
        eval_steps=50,                  
        bf16=True,                      
        report_to="none",
        dataset_text_field="text",
        max_seq_length=1024,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    print("\n🎉 6. Training complete! Saving custom model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}!")

    # =========================================================
    # ▼▼▼ Evaluation (Inference) Phase ▼▼▼
    # =========================================================
    print("\n🚀 7. Transitioning directly to evaluation (inference) phase...")
    
    model.eval()
    y_pred = []
    
    for text in tqdm(X_val, desc="Evaluating"):
        prompt = (
            "You are an expert software engineer.\n"
            "Predict whether this commit diff will be reverted (1) or not (0).\n\n"
            f"{text}\n\n"
            "[Prediction]\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if "1" in answer:
            y_pred.append(1)
        elif "0" in answer:
            y_pred.append(0)
        else:
            y_pred.append(0)

    # ---------------------------------------------------------
    # Outputting Evaluation Results
    # ---------------------------------------------------------
    print("\n🎉 Evaluation complete! Displaying results.")
    print("="*50)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    
    try:
        auc = roc_auc_score(y_val, y_pred)
        print(f"AUC (ROC-AUC)  : {auc:.4f}")
    except ValueError:
        print("AUC (ROC-AUC)  : Cannot be calculated (only one class present in y_true)")

    print("\n[Classification Report]")
    print(classification_report(y_val, y_pred))
    
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(y_val, y_pred)
    print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predict 0', 'Predict 1']))
    print("="*50)
    
if __name__ == "__main__":
    main()