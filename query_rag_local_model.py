# -*- coding: utf-8 -*-
# File 2: query_rag_model.py (Modified version: Using Ollama + Output Cleaning)
# Purpose: Load index, set up RAG chain (including Prompt), connect to a model running on local Ollama service,
#          read dialogue content from a TXT file, generate SOAP notes, and clean the output format.

print("--- Starting execution: query_rag_model.py (Ollama + Output Cleaning Version) ---")

# --- 1. Import required libraries ---
print("\n--- 1. Importing required libraries ---")
import os
import time
import re # <-- Added import re for regular expression cleaning
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch

# --- 2. Configuration and Checks ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index_mental_health_gpu"
# **Ensure this is the correct model name in your Ollama setup**
LLM_MODEL_NAME = "deepseek-r1:14b"
NUM_RETRIEVED_DOCS = 10

# (API Key check removed)

if torch.cuda.is_available():
    print(f"CUDA available! Using GPU (for embedding model): {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Warning: CUDA not available, embedding model will use CPU.")
    device = 'cpu'

if not os.path.exists(FAISS_INDEX_PATH):
    print(f"Error: FAISS index folder '{FAISS_INDEX_PATH}' not found.")
    print("Please run 'create_vectorstore.py' first to generate and save the index.")
    exit()

# --- 3. Load Embedding Model ---
# (Same as previous version)
print(f"\n--- 3. Loading embedding model '{EMBEDDING_MODEL_NAME}' to {device} ---")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error: Failed to load embedding model. Error message: {e}")
    exit()

# --- 4. Load FAISS Index ---
# (Same as previous version)
print(f"\n--- 4. Loading FAISS index from '{FAISS_INDEX_PATH}' ---")
print("   (This index contains vectorized content from the Amod dataset and will be used for assistance)")
try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully!")
except Exception as e:
    print(f"Error: Failed to load FAISS index. Error message: {e}")
    exit()

# --- 5. Initialize LLM (Connect to Ollama) ---
# (Same as previous version)
print(f"\n--- 5. Connecting to model in Ollama service: {LLM_MODEL_NAME} ---")
print("   (Ensure Ollama service is running in the background and the model is downloaded)")
try:
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.7)
    print(f"Connected to Ollama model '{LLM_MODEL_NAME}' successfully.")
except Exception as e:
    print(f"Error: Failed to initialize or connect to Ollama LLM ({LLM_MODEL_NAME})...") # Keep previous error hint
    exit()

# --- 6. Define Prompt Template (Slightly adjusted) ---
print("\n--- 6. Defining SOAP Prompt Template ---")
# Added a clear instruction at the end of the prompt to not output the thinking process
prompt_template = """
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
""" # <-- Added explicit instruction at the end

SOAP_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
print("SOAP Prompt defined (with added output cleaning instruction).")

# --- 7. Create RAG Chain ---
# (Same as previous version)
print("\n--- 7. Creating RetrievalQA chain ---")
retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": SOAP_PROMPT}
)
print(f"RAG chain created, will use Ollama model '{LLM_MODEL_NAME}' and retrieve top {NUM_RETRIEVED_DOCS} relevant document chunks.")


# --- Added: Define cleaning function ---
def clean_llm_output(raw_output: str) -> str:
    """Cleans LLM output, removing <think> blocks and potential Markdown titles."""
    # Remove <think>...</think> blocks and subsequent whitespace
    # re.DOTALL makes . match newlines
    cleaned = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL)

    # Remove potential Markdown titles, like "### SOAP Note" or similar format, and subsequent whitespace
    # ^ matches start, #+ matches one or more #, \s* matches zero or more whitespace, SOAP Note matches text, \s* again matches whitespace
    cleaned = re.sub(r'^#+\s*SOAP\s*Note\s*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove leading and trailing excess whitespace
    cleaned = cleaned.strip()

    return cleaned

# --- 8. Read Dialogue TXT File and Execute ---
# (Same as previous TXT version)
print("\n--- 8. Reading dialogue TXT file and generating SOAP note ---")
dialogue_txt_path = input("Please enter the full path to the TXT file containing the dialogue: ")

if not os.path.exists(dialogue_txt_path):
    print(f"Error: File not found '{dialogue_txt_path}'")
    exit()
if not dialogue_txt_path.lower().endswith('.txt'):
    print(f"Error: File '{dialogue_txt_path}' is not a .txt file.")
    exit()

try:
    with open(dialogue_txt_path, 'r', encoding='utf-8') as f:
        dialogue_text = f.read()
    if not dialogue_text.strip():
        print("Error: TXT file content is empty.")
        exit()
    print(f"Successfully loaded dialogue text from '{dialogue_txt_path}'.")
except Exception as e:
    print(f"An error occurred while reading the TXT file: {e}")
    exit()

# --- 9. Construct Query and Execute RAG Chain ---
print(f"\n--- 9. Generating SOAP note for the provided dialogue (Using Ollama: {LLM_MODEL_NAME}) ---")
query_for_rag = dialogue_text

print(f"Calling RAG chain to process dialogue (via Ollama), please wait...")
try:
    start_time = time.time()
    result = qa_chain.invoke({"query": query_for_rag})
    end_time = time.time()

    # --- Modified: Clean the result before printing ---
    raw_llm_result = result['result']
    cleaned_result = clean_llm_output(raw_llm_result) # Call the cleaning function

    print("\n--- RAG Generated Result (SOAP Note) ---")
    print(cleaned_result) # <-- Print the cleaned result
    print(f"\n(Processing time: {end_time - start_time:.2f} seconds)")

    # (Optional) Print source document info
    # print("\n--- Retrieved Auxiliary Document Snippets (from Amod dataset index) ---")
    # for i, doc in enumerate(result['source_documents']):
    #     print(f"Snippet {i+1}:\n{doc.page_content[:150]}...\n---")

except Exception as e:
    print(f"\nError processing dialogue: {e}")
    print(f"   Possible causes: Ollama service not running, model '{LLM_MODEL_NAME}' not loaded or OOM, error on Ollama side, network connection issue, content generation timeout, etc.")

print("\n--- Program finished ---")