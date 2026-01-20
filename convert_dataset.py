"""
Convert v5-sections CSV dataset to DPGNN format

This script converts the WorkfitAI v5-sections dataset format to the format
required by the original DPGNN implementation.

Required output files:
- dataset/data.train_all - All training interactions
- dataset/data.train_all_add - Training with additional edges
- dataset/data.valid_g / data.valid_j - Validation sets
- dataset/data.test_g / data.test_j - Test sets
- dataset/data.user_add - User-initiated interactions
- dataset/data.job_add - Job-initiated interactions
- dataset/geek.token - List of geek (resume) IDs
- dataset/job.token - List of job IDs
- dataset/geek.bert.npy - BERT embeddings for resumes
- dataset/job.bert.npy - BERT embeddings for jobs

Usage:
    python convert_dataset.py --train_csv train.csv --test_csv test.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel


def combine_sections(row, cols):
    """Combine multiple text columns into a single string"""
    texts = []
    for col in cols:
        text = str(row.get(col, ''))
        if text and text.lower() != 'nan' and text.strip():
            texts.append(text)
    return ' [SEP] '.join(texts)


def encode_texts_bert(texts, tokenizer, model, device, max_length=512, batch_size=16):
    """Encode texts using BERT"""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description='Convert v5-sections to DPGNN format')
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--skip_bert', action='store_true', help='Skip BERT encoding')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading CSV files...")
    train_df = pd.read_csv(args.train_csv).fillna('')
    test_df = pd.read_csv(args.test_csv).fillna('')

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Define columns
    resume_cols = ['resume_summary', 'resume_experience', 'resume_skills', 'resume_education']
    job_cols = ['jd_overview', 'jd_responsibilities', 'jd_requirements', 'jd_preferred']

    # Build unique geeks (resumes) based on original_index
    print("\nBuilding unique geeks (resumes)...")
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    geek_texts = {}  # geek_token -> combined text
    for idx, row in all_df.iterrows():
        geek_token = str(row['original_index'])
        if geek_token not in geek_texts:
            geek_texts[geek_token] = combine_sections(row, resume_cols)

    geek_tokens = sorted(geek_texts.keys(), key=lambda x: int(x))
    print(f"Found {len(geek_tokens)} unique geeks")

    # Build unique jobs based on JD content hash
    print("Building unique jobs...")
    job_hashes = {}  # hash -> job_token
    job_texts = {}   # job_token -> combined text
    row_to_job = {}  # row_index -> job_token

    for idx, row in all_df.iterrows():
        jd_content = '|'.join([str(row.get(col, '')) for col in job_cols])
        jd_hash = hash(jd_content)

        if jd_hash not in job_hashes:
            job_token = str(len(job_hashes))
            job_hashes[jd_hash] = job_token
            job_texts[job_token] = combine_sections(row, job_cols)

        row_to_job[idx] = job_hashes[jd_hash]

    job_tokens = sorted(job_texts.keys(), key=lambda x: int(x))
    print(f"Found {len(job_tokens)} unique jobs")

    # Write token files
    print("\nWriting token files...")
    with open(os.path.join(args.output_dir, 'geek.token'), 'w') as f:
        for token in geek_tokens:
            f.write(f"{token}\n")

    with open(os.path.join(args.output_dir, 'job.token'), 'w') as f:
        for token in job_tokens:
            f.write(f"{token}\n")

    # Process train data
    print("\nProcessing training data...")
    train_interactions = []
    positive_interactions = []
    user_add_interactions = []  # User initiated (all interactions from user perspective)
    job_add_interactions = []   # Job initiated (we'll use same as user since no explicit direction)

    for idx, row in train_df.iterrows():
        geek_token = str(row['original_index'])
        # Find job_token for this row
        jd_content = '|'.join([str(row.get(col, '')) for col in job_cols])
        jd_hash = hash(jd_content)
        job_token = job_hashes[jd_hash]

        label_str = str(row['label']).strip().lower()
        label = 1 if label_str == 'good fit' else 0

        train_interactions.append((geek_token, job_token, label))

        if label == 1:
            positive_interactions.append((geek_token, job_token, label))
            user_add_interactions.append((geek_token, job_token, label))
            job_add_interactions.append((geek_token, job_token, label))

    # Split validation from training (using positive samples)
    print("Creating validation split...")
    np.random.seed(42)
    n_valid = int(len(positive_interactions) * args.validation_split)
    indices = np.random.permutation(len(positive_interactions))
    valid_indices = set(indices[:n_valid])

    valid_interactions = []
    train_final = []

    for i, inter in enumerate(positive_interactions):
        if i in valid_indices:
            valid_interactions.append(inter)
        else:
            train_final.append(inter)

    # Process test data
    print("Processing test data...")
    test_interactions = []
    for idx, row in test_df.iterrows():
        geek_token = str(row['original_index'])
        jd_content = '|'.join([str(row.get(col, '')) for col in job_cols])
        jd_hash = hash(jd_content)
        job_token = job_hashes[jd_hash]

        label_str = str(row['label']).strip().lower()
        label = 1 if label_str == 'good fit' else 0
        test_interactions.append((geek_token, job_token, label))

    # Write interaction files
    print("\nWriting interaction files...")

    def write_interactions(filepath, interactions):
        with open(filepath, 'w') as f:
            for geek, job, label in interactions:
                f.write(f"{geek}\t{job}\t{label}\n")

    # data.train_all - All training (positives only for BPR)
    write_interactions(
        os.path.join(args.output_dir, 'data.train_all'),
        train_final
    )
    print(f"  data.train_all: {len(train_final)} samples")

    # data.train_all_add - Same as train_all for our case
    write_interactions(
        os.path.join(args.output_dir, 'data.train_all_add'),
        train_final
    )
    print(f"  data.train_all_add: {len(train_final)} samples")

    # data.user_add - User initiated interactions
    write_interactions(
        os.path.join(args.output_dir, 'data.user_add'),
        user_add_interactions
    )
    print(f"  data.user_add: {len(user_add_interactions)} samples")

    # data.job_add - Job initiated interactions
    write_interactions(
        os.path.join(args.output_dir, 'data.job_add'),
        job_add_interactions
    )
    print(f"  data.job_add: {len(job_add_interactions)} samples")

    # Validation sets (same for both perspectives in our case)
    write_interactions(
        os.path.join(args.output_dir, 'data.valid_g'),
        valid_interactions
    )
    write_interactions(
        os.path.join(args.output_dir, 'data.valid_j'),
        valid_interactions
    )
    print(f"  data.valid_g/valid_j: {len(valid_interactions)} samples")

    # Test sets (same for both perspectives)
    write_interactions(
        os.path.join(args.output_dir, 'data.test_g'),
        test_interactions
    )
    write_interactions(
        os.path.join(args.output_dir, 'data.test_j'),
        test_interactions
    )
    print(f"  data.test_g/test_j: {len(test_interactions)} samples")

    # Generate BERT embeddings
    if not args.skip_bert:
        print("\n" + "=" * 50)
        print("Generating BERT embeddings...")
        print("=" * 50)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        print(f"Loading model: {args.bert_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model = AutoModel.from_pretrained(args.bert_model).to(device)

        # Encode geeks
        print("\nEncoding geeks (resumes)...")
        geek_text_list = [geek_texts[t] for t in geek_tokens]
        geek_embeddings = encode_texts_bert(
            geek_text_list, tokenizer, model, device,
            max_length=args.max_length, batch_size=args.batch_size
        )

        # Format: first column is ID, rest is embedding
        geek_ids = np.array([int(t) for t in geek_tokens]).reshape(-1, 1)
        geek_data = np.hstack([geek_ids, geek_embeddings])
        np.save(os.path.join(args.output_dir, 'geek.bert.npy'), geek_data)
        print(f"Saved geek.bert.npy: shape {geek_data.shape}")

        # Encode jobs
        print("\nEncoding jobs...")
        job_text_list = [job_texts[t] for t in job_tokens]
        job_embeddings = encode_texts_bert(
            job_text_list, tokenizer, model, device,
            max_length=args.max_length, batch_size=args.batch_size
        )

        # Format: first column is ID, rest is embedding
        job_ids = np.array([int(t) for t in job_tokens]).reshape(-1, 1)
        job_data = np.hstack([job_ids, job_embeddings])
        np.save(os.path.join(args.output_dir, 'job.bert.npy'), job_data)
        print(f"Saved job.bert.npy: shape {job_data.shape}")

    # Summary
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Unique geeks: {len(geek_tokens)}")
    print(f"Unique jobs: {len(job_tokens)}")
    print(f"Training interactions: {len(train_final)}")
    print(f"Validation interactions: {len(valid_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f}: {size_str}")


if __name__ == '__main__':
    main()
