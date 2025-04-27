# -*- coding: utf-8 -*-
# File: query_rag_openai.py
# Description: Core RAG logic using OpenAI API (GPT-4o) for SOAP note generation.
# Designed to be imported and used by a Flask application.

import os
import re # For cleaning output
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_openai import ChatOpenAI # New standard import
except ImportError:
    print("Warning: Could not import ChatOpenAI from langchain_openai, trying from langchain.chat_models...")
    try:
        from langchain.chat_models import ChatOpenAI # Fallback for older versions
    except ImportError:
        print("ERROR: Failed to import ChatOpenAI. Ensure 'langchain-openai' or 'langchain' (older) is installed.")
        raise # Re-raise the error to stop execution if import fails
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index_mental_health_gpu" # Path to your FAISS index
LLM_MODEL_NAME = "gpt-4o" # OpenAI model
NUM_RETRIEVED_DOCS = 10 # Number of documents to retrieve for context

# --- Global Variables (to be initialized by load_rag_components) ---
embeddings = None
vectorstore = None
llm = None
qa_chain = None

# --- Helper Function for Output Cleaning (adapted from app.py) ---
def clean_llm_output(raw_output: str) -> str:
    """Cleans LLM output, removing potential Markdown titles and stray asterisks."""
    cleaned = raw_output
    # Remove potential Markdown title like "### SOAP Note" (case-insensitive, multiline) and leading whitespace
    cleaned = re.sub(r'^#+\s*SOAP\s*Note\s*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Remove double asterisks (likely failed Markdown bolding)
    cleaned = cleaned.replace('**', '')
    # Remove leading/trailing whitespace again
    cleaned = cleaned.strip()
    return cleaned

# --- Load RAG Components Function ---
def load_rag_components_openai():
    """Loads embedding model, FAISS index, initializes OpenAI LLM, and creates the RAG chain."""
    global embeddings, vectorstore, llm, qa_chain
    print("--- Loading RAG Components (OpenAI Version) ---")

    # Check for OpenAI API Key
    if "OPENAI_API_KEY" not in os.environ:
        error_msg = "ERROR: OPENAI_API_KEY environment variable not set!"
        print(error_msg)
        raise EnvironmentError(error_msg)
    else:
        print("OpenAI API Key found.")

    # Check CUDA for embeddings
    if torch.cuda.is_available():
        print(f"CUDA available! Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("Warning: CUDA not available, using CPU for embeddings.")
        device = 'cpu'

    # Check FAISS index
    if not os.path.exists(FAISS_INDEX_PATH):
        error_msg = f"ERROR: FAISS index not found at '{FAISS_INDEX_PATH}'! Run create_vectorstore.py first."
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Load Embeddings
    try:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' to {device}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise

    # Load FAISS index
    try:
        print(f"Loading FAISS index from '{FAISS_INDEX_PATH}'...")
        # Ensure dangerous deserialization is allowed if the index was created with it
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Set based on how index was saved
        )
        print("FAISS index loaded.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise

    # Initialize OpenAI LLM
    try:
        print(f"Initializing OpenAI LLM: {LLM_MODEL_NAME}...")
        llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.7)
        print(f"OpenAI LLM '{LLM_MODEL_NAME}' initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI LLM '{LLM_MODEL_NAME}': {e}")
        print("   Check your API key, network connection, and OpenAI service status.")
        raise

    # Define Prompt Template (English Version, adapted from app.py)
    print("Defining SOAP Prompt Template (English)...")
    prompt_template_str = """
Based on the **main dialogue content** provided in the "Query/Topic" section below, and referencing the additional "Context Snippets" (from a relevant knowledge base), generate a structured clinical SOAP note.

**Primary Task:** Analyze the dialogue in "Query/Topic", extract information, and organize it into the SOAP format.
**Secondary Task:** The "Context Snippets" may offer linguistic patterns or suggestions from similar situations for reference, but the **primary source of truth should be the dialogue in "Query/Topic"**.

**Background:** The dialogue in "Query/Topic" is the core content to be documented. The "Context Snippets" come from informal exchanges on an online mental health support platform.

**Instructions:**
1.  Carefully read the main dialogue in "Query/Topic".
2.  Fill in the four sections of the SOAP note based *only* on the main dialogue content:
    * **S (Subjective):** The client's (requester in the dialogue) own statements about feelings, experiences, symptoms, concerns, goals, etc. Quote or paraphrase directly from the main dialogue.
    * **O (Objective):** Objective observations or behaviors that can be **inferred** from the main dialogue text (e.g., intensity of emotional expression like "very agitated", "cried"; described behaviors like "unable to sleep", "avoiding social events"; speech patterns). If objective cues are minimal in the text, state "Insufficient information". **Do not fabricate clinical observations.**
    * **A (Assessment):** A **preliminary, informal** assessment or summary of the client's current state based on the S and O from the main dialogue (e.g., main problems, emotional state, potential risks if clearly mentioned). **Avoid formal diagnostic labels** unless explicitly mentioned in the dialogue. State that this is an initial impression based on the informal dialogue.
    * **P (Plan):** Coping strategies, suggestions, next steps, or goals mentioned *within the main dialogue*. These could be mentioned by the client or the responder. If no clear plan is mentioned, state "No specific plan mentioned".
3.  When generating, you may refer to the "Context Snippets" for similar phrasing or suggestions, but do not directly copy irrelevant content.

**Format Requirements:**
Strictly adhere to the following format, with each section labeled:
Subjective: [Subjective information]
Objective: [Objective information or state insufficient information]
Assessment: [Assessment or preliminary impression]
Plan: [Plan or state no specific plan mentioned]

**Context Snippets (for reference):**
{context}

**Generate a SOAP note based on the following main dialogue:**
Query/Topic: {question}

**Generated SOAP Note:**
**IMPORTANT: Directly output ONLY the SOAP note content (Subjective, Objective, Assessment, Plan sections). Do NOT include any thinking process, explanations, comments, or any other text outside the note itself.**
"""
    SOAP_PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )
    print("SOAP Prompt defined.")

    # Create RAG Chain
    print("Creating RAG chain (OpenAI)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is common for putting context directly in prompt
        retriever=retriever,
        return_source_documents=False, # Set to False for cleaner web app output
        chain_type_kwargs={"prompt": SOAP_PROMPT}
    )
    print("RAG chain created successfully (using OpenAI).")
    print("--- RAG Components Loaded (OpenAI Version) ---")

def get_qa_chain():
    """Returns the initialized QA chain."""
    if qa_chain is None:
        raise RuntimeError("RAG components not loaded. Call load_rag_components_openai() first.")
    return qa_chain

# Example of how to run it standalone (optional, for testing)
if __name__ == '__main__':
    print("Running query_rag_openai.py in standalone test mode...")
    try:
        load_rag_components_openai()
        test_chain = get_qa_chain()

        # Example: Read from a test file (replace with your test file)
        test_file_path = 'test.txt' # Make sure this file exists [cite: 5]
        if os.path.exists(test_file_path):
            print(f"Reading test dialogue from: {test_file_path}")
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_dialogue = f.read()

            print("\n--- Generating SOAP note for test dialogue ---")
            start_time = time.time()
            result = test_chain.invoke({"query": test_dialogue})
            end_time = time.time()

            raw_llm_result = result.get('result', 'Error: No result key found')
            cleaned_note = clean_llm_output(raw_llm_result)

            print("\n--- Generated SOAP Note (Cleaned) ---")
            print(cleaned_note)
            print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
        else:
            print(f"Test file '{test_file_path}' not found. Skipping standalone test.")

    except (FileNotFoundError, EnvironmentError, RuntimeError, Exception) as e:
        print(f"\nError during standalone test: {e}")