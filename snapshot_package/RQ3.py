import os
import sys
import glob
import warnings

# --- TensorFlow version compatibility setting ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- Project path setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    average_precision_score, 
    matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from transformers import AutoTokenizer, TFAutoModel

warnings.filterwarnings("ignore")

DATA_DIR = "file_data"

def calculate_reversion_ratio():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{DATA_DIR}'.")
        return

    print(f"📂 Scanning {len(csv_files)} files...")
    df_list = []
    empty_files_count = 0

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                empty_files_count += 1
                continue
            df_list.append(df)
        except pd.errors.EmptyDataError:
            empty_files_count += 1
            continue
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")

    if not df_list:
        print("No CSV files containing valid data (rows) were found.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    total_count = len(full_df)
    reverted_count = full_df['reverted'].sum()
    not_reverted_count = total_count - reverted_count
    ratio = (reverted_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "="*40)
    print("📊 Reversion Ratio Summary")
    print("="*40)
    print(f"Files loaded: {len(df_list)}")
    print(f"Empty files skipped: {empty_files_count}")
    print("-" * 40)
    print(f"Total data count (PRs): {total_count:,}")
    print("-" * 40)
    print(f"✅ Reverted (reverted=1): {reverted_count:,}")
    print(f"❌ Not reverted (reverted=0): {not_reverted_count:,}")
    print("-" * 40)
    print(f"📈 Reversion rate: {ratio:.2f}%")
    print("="*40)

def read_all_csv(dir_path):
    search_pattern = os.path.join(dir_path, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Warning: No '*.csv' files found in directory '{dir_path}'.")
        return pd.DataFrame()

    print(f"Loading {len(csv_files)} CSV files:")
    dfs_list = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            dfs_list.append(df)
        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"Error: Failed to read file '{file_path}'. Skipping. Reason: {e}")

    if not dfs_list:
        print("Warning: No readable CSV files were found.")
        return pd.DataFrame()

    combined_df = pd.concat(dfs_list, ignore_index=True)
    return combined_df

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')

    for k, text in enumerate(texts):
        tok_text = tokenizer.tokenize(text)
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        attention_mask[k, :input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }
    
def plot_roc_curve(y_true, y_probs, save_path, roc_auc):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to '{save_path}'.")

def plot_confusion_matrix(X_test, y_test, model):
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm, index=label_names, columns=label_names)
    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
def print_advanced_metrics(model_name, y_true, y_pred, y_pred_probs):
    """Calculates and prints PR-AUC, MCC, and G-Mean metrics."""
    
    # 1. PR-AUC (Precision-Recall Area Under Curve)
    # Note: Uses predicted probabilities instead of predicted labels (0/1)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    
    # 2. MCC (Matthews Correlation Coefficient)
    # Note: Ranges from -1 to 1. 0 is random, 1 is perfect prediction.
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 3. G-Mean (Geometric Mean)
    # Note: Computed using True Positives, etc.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)
    
    print(f"\n--- {model_name} Advanced Metrics ---")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"MCC    : {mcc:.4f}")
    print(f"G-Mean : {g_mean:.4f}")
    print("-" * 35)

def build_model(model_name):
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')
        
        model = TFAutoModel.from_pretrained(model_name, from_pt=True)
        model.trainable = True
        
        outputs = model({
            "input_ids": input_word_ids,
            "attention_mask": input_mask
        })

        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]
        x = tf.keras.layers.Dropout(0.1)(cls_token)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

def training(DATA_REPO):
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    EPOCHS = 3
    output_dir = "output_RQ3"
    os.makedirs(output_dir, exist_ok=True) 

    # --- 1. Data Preparation ---
    df = read_all_csv(DATA_REPO)
    print("Loaded dataframe.")
    df.dropna(subset=['commit_message'], inplace=True)
    print(sum(df['reverted'].value_counts()))

    df['commit_message'] = df['commit_message'].fillna('')
    df['pr_description'] = df['pr_description'].fillna('')
    df['diff_text'] = df['diff_text'].fillna('')
    
    # Fix: Corrected syntax error (++) to proper string concatenation
    df['text'] = df['pr_description'] + "\n" + df['diff_text']
    df = df[['text', 'reverted']]
    
    # 1. Fetch all data
    X_data_all = df['text'].to_numpy()
    y_data_all = df['reverted'].to_numpy()

    # 2. Split first (X_test_text and y_test are finalized here with real-world proportions)
    # Using stratify=y_data_all maintains the original reversion ratio during the split
    X_train_raw, X_test_text, y_train_raw, y_test = train_test_split(
        X_data_all, y_data_all, test_size=0.3, random_state=777, stratify=y_data_all
    )

    # 3. Undersample only the training data
    df_train = pd.DataFrame({'text': X_train_raw, 'reverted': y_train_raw})
    
    df_true = df_train[df_train['reverted'] == 1]
    df_false = df_train[df_train['reverted'] == 0]
    df_false_undersampled = df_false.sample(n=len(df_true), random_state=557)
    df_final = pd.concat([df_true, df_false_undersampled]).sample(frac=1, random_state=72).reset_index(drop=True)

    # 4. Store undersampled training data back into primary variables
    X_train_text = df_final['text'].to_numpy()
    y_train = df_final['reverted'].to_numpy()

    target_names = ['0', '1']

    # --- 2. CodeBERT Preparation and Training ---
    print("\n" + "="*40)
    print("🤖 CodeBERT Training and Prediction")
    print("="*40)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    X_train = roberta_encode(X_train_text, tokenizer)
    X_test = roberta_encode(X_test_text, tokenizer)

    y_train = np.asarray(y_train, dtype='float32')
    y_test = np.asarray(y_test, dtype='float32')

    with strategy.scope():
        model = build_model(MODEL_NAME)
        # model.summary() # Uncomment if needed

    print('Training CodeBERT...')
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(X_test, y_test))

    # Save accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(10, 8))
    ax_acc.set_title('Training and Validation Accuracy')
    xaxis = np.arange(len(history.history['accuracy']))
    ax_acc.plot(xaxis, history.history['accuracy'], label='Train set')
    ax_acc.plot(xaxis, history.history['val_accuracy'], label='Validation set')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)
    save_path = os.path.join(output_dir, 'no_diff_accuracy_plot.png')
    fig_acc.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig_acc)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CodeBERT Accuracy: %.2f%%" % (scores[1] * 100))

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n--- CodeBERT Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    roc_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"CodeBERT ROC-AUC: {roc_auc:.4f}")

    print_advanced_metrics("CodeBERT", y_test, y_pred, y_pred_probs)
    
    plot_roc_curve(y_test, y_pred_probs, os.path.join(output_dir, 'diff_roc_curve.png'), roc_auc)
    
    plot_confusion_matrix(X_test, y_test, model)
    plt.savefig(os.path.join(output_dir, 'diff_confusion_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3. Random Forest (For Comparison) ---
    print("\n" + "="*40)
    print("🌲 Random Forest (TF-IDF) Prediction Results")
    print("="*40)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)

    rf_y_pred = rf_model.predict(X_test_tfidf)
    rf_y_pred_probs = rf_model.predict_proba(X_test_tfidf)[:, 1]

    print("\n--- RF Classification Report ---")
    print(classification_report(y_test, rf_y_pred, target_names=target_names))
    rf_roc_auc = roc_auc_score(y_test, rf_y_pred_probs)
    print(f"RF ROC-AUC: {rf_roc_auc:.4f}")
    print_advanced_metrics("RandomForest", y_test, rf_y_pred, rf_y_pred_probs)
    
    # --- 4. Dummy Classifier (Baseline) ---
    print("\n" + "="*40)
    print("🎲 Random Prediction (Dummy Classifier) Results")
    print("="*40)
    dummy_model = DummyClassifier(strategy="uniform", random_state=42)
    dummy_model.fit(X_train_tfidf, y_train)

    dummy_y_pred = dummy_model.predict(X_test_tfidf)
    dummy_y_pred_probs = dummy_model.predict_proba(X_test_tfidf)[:, 1]

    print("\n--- Random Classification Report ---")
    print(classification_report(y_test, dummy_y_pred, target_names=target_names))
    dummy_roc_auc = roc_auc_score(y_test, dummy_y_pred_probs)
    print(f"Random ROC-AUC: {dummy_roc_auc:.4f}")
    print_advanced_metrics("Random Dummy", y_test, dummy_y_pred, dummy_y_pred_probs)
    
    # --- 5. RoBERTa (For Comparison) ---
    print("\n" + "="*40)
    print("📚 RoBERTa (roberta-base) Training and Prediction")
    print("="*40)
    tf.keras.backend.clear_session() # Free up memory used by CodeBERT
    
    roberta_name = 'roberta-base'
    tokenizer_roberta = AutoTokenizer.from_pretrained(roberta_name)
    X_train_rob = roberta_encode(X_train_text, tokenizer_roberta)
    X_test_rob = roberta_encode(X_test_text, tokenizer_roberta)

    with strategy.scope():
        model_roberta = build_model(model_name=roberta_name)
        
    model_roberta.fit(X_train_rob, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # Fetch probabilities
    prob_roberta = model_roberta.predict(X_test_rob).flatten()

    # Generate predicted labels: 1 if probability > 0.5, else 0
    pred_roberta = (prob_roberta > 0.5).astype(int)
    
    print("\n--- RoBERTa Classification Report ---")
    print(classification_report(y_test, pred_roberta, target_names=target_names))

    rob_roc_auc = roc_auc_score(y_test, prob_roberta)
    print(f"RoBERTa ROC-AUC: {rob_roc_auc:.4f}")
    print_advanced_metrics("RoBERTa", y_test, pred_roberta, prob_roberta)

    # --- 6. Save Data (CSV) ---
    # 6.1 Integrated data for comparing models
    roc_plot_df = pd.DataFrame({
        "text": X_test_text,
        "actual_label": y_test,
        "prob_CodeBERT": y_pred_probs.flatten(),
        "prob_RoBERTa": prob_roberta,
        "prob_RandomForest": rf_y_pred_probs,
        "prob_Random": dummy_y_pred_probs
    })
    roc_csv_path = os.path.join(output_dir, 'all_models_roc_data.csv')
    roc_plot_df.to_csv(roc_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved integrated data for future comparison (CSV): {roc_csv_path}")

    # 6.2 Detailed analysis data for CodeBERT
    analysis_df = pd.DataFrame({"text": X_test_text}).copy()
    analysis_df['actual_label'] = np.array(y_test).flatten()
    analysis_df['predicted_prob'] = y_pred_probs.flatten()
    analysis_df['predicted_label'] = y_pred.flatten()
    analysis_df['is_correct'] = analysis_df['actual_label'] == analysis_df['predicted_label']
    
    analysis_df_sorted = analysis_df.sort_values(by='predicted_prob', ascending=False)
    analysis_csv_path = os.path.join(output_dir, 'prediction_analysis.csv')
    analysis_df_sorted.to_csv(analysis_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved CodeBERT detailed analysis data (CSV): {analysis_csv_path}")

    print("\n🎉 Finish: All processes completed!")

if __name__ == "__main__":
    print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
    MODEL_NAME = 'microsoft/codebert-base'
    MAX_LEN = 256

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
        
    print("Start")
    training(DATA_DIR)