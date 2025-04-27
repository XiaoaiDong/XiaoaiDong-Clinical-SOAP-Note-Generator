# -*- coding: utf-8 -*-
# File 1: create_vectorstore.py
# Purpose: Load data, process text, generate embeddings (using CUDA), and save FAISS vector store.

print("--- Starting execution: create_vectorstore.py ---")

# --- 1. Install necessary libraries (if not already installed) ---
# pip install datasets pandas langchain sentence-transformers faiss-gpu torch transformers accelerate
# (Ensure torch is the CUDA version, and faiss-gpu is installed correctly)

print("\n--- 1. Importing required libraries ---")
import os
import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import time

# --- 2. Configuration and Checks ---
# Define the embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Define the FAISS index save path
FAISS_INDEX_PATH = "faiss_index_mental_health_gpu"
# Define the amount of data to process (set to None to process all, or a number to limit)
NUM_SAMPLES_TO_PROCESS = None # None to process all, or e.g., 1000 to process the first 1000 entries

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Warning: CUDA not available, will use CPU.")
    device = 'cpu'

# --- 3. Load and Prepare Data ---
print("\n--- 3. Loading and preparing data ---")
dataset_name = "Amod/mental_health_counseling_conversations"
try:
    ds = load_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' loaded successfully!")
    split_name = 'train' # Or other available split
    if split_name not in ds:
        split_name = list(ds.keys())[0] # Try using the first split

    # Convert to Pandas DataFrame and create 'full_conversation' column
    df = ds[split_name].to_pandas()
    context_col = 'Context'
    response_col = 'Response'
    if context_col in df.columns and response_col in df.columns:
        df['full_conversation'] = df[context_col].astype(str) + "\n---\n" + df[response_col].astype(str)
        df['full_conversation'] = df['full_conversation'].str.strip()
        print("Created 'full_conversation' column.")
    else:
        raise ValueError(f"Dataset is missing '{context_col}' or '{response_col}' column.")

    # Get text list and handle null values
    texts = df['full_conversation'].fillna('').astype(str).tolist()
    texts = [text for text in texts if text.strip()]

    if not texts:
        raise ValueError("No valid conversation texts available after processing.")

    # Process subset or all data based on the setting
    if NUM_SAMPLES_TO_PROCESS is not None and NUM_SAMPLES_TO_PROCESS < len(texts):
        texts_to_process = texts[:NUM_SAMPLES_TO_PROCESS]
        print(f"Will process the first {NUM_SAMPLES_TO_PROCESS} texts.")
    else:
        texts_to_process = texts
        print(f"Will process all {len(texts_to_process)} valid texts.")

except Exception as e:
    print(f"Error loading or preparing data: {e}")
    exit()

# --- 4. Text Splitting ---
print("\n--- 4. Splitting texts ---")
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    documents = text_splitter.create_documents(texts_to_process)
    print(f"Texts split into {len(documents)} document chunks.")
    if not documents:
        raise ValueError("No document chunks were generated after text splitting.")
except Exception as e:
    print(f"Error during text splitting: {e}")
    exit()

# --- 5. Load Embedding Model (using specified device) ---
print(f"\n--- 5. Loading embedding model '{EMBEDDING_MODEL_NAME}' to {device} ---")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device} # Use the detected device ('cuda' or 'cpu')
    )
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error: Failed to load embedding model. Error message: {e}")
    exit()

# --- 6. Compute Embeddings and Build FAISS Index ---
print("\n--- 6. Computing embeddings and building FAISS index ---")
print("(This might take some time depending on data size and hardware...)")
start_time = time.time()
try:
    vectorstore = FAISS.from_documents(documents, embeddings)
    end_time = time.time()
    print(f"FAISS vector database built successfully! Time taken: {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"Error: Failed to build FAISS vector database. Error message: {e}")
    exit()

# --- 7. Save FAISS Index Locally ---
print(f"\n--- 7. Saving FAISS index to '{FAISS_INDEX_PATH}' ---")
try:
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index successfully saved to '{FAISS_INDEX_PATH}' folder.")
except Exception as e:
    print(f"Error: Failed to save FAISS index. Error message: {e}")
    exit()

print("\n--- Execution finished: create_vectorstore.py ---")