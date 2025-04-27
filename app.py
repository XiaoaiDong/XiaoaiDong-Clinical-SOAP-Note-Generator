# -*- coding: utf-8 -*-
# File: app.py
# Description: Flask web application for SOAP note generation using RAG.
#              Supports both local Ollama (DeepSeek) and OpenAI API (GPT-4o).
#              Includes SSE progress updates via background thread.

import os
import re
import time
import threading # For background processing
import queue # For passing progress messages
from flask import Flask, request, render_template, redirect, url_for, flash, Response, stream_with_context, session, jsonify
from werkzeug.utils import secure_filename

# LangChain and related imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
try:
    # Use the newer import structure
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Warning: Could not import ChatOpenAI from langchain_openai, trying older import...")
    try:
        # Fallback for older langchain versions
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print("ERROR: Failed to import ChatOpenAI. Ensure 'langchain-openai' is installed.")
        # Depending on deployment strategy, you might exit or just disable OpenAI functionality.
        # For now, we'll allow the app to start but OpenAI will fail later if not available.
        ChatOpenAI = None

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Keep original upload folder
ALLOWED_EXTENSIONS = {'txt'}
TEMPLATE_FOLDER = 'templates' # Use original templates folder
STATIC_FOLDER = 'static' # Use original static folder

# RAG Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index_mental_health_gpu"
# Model Names
LOCAL_LLM_MODEL_NAME = "deepseek-r1:14b" # Ollama model
OPENAI_LLM_MODEL_NAME = "gpt-4o" # OpenAI model
NUM_RETRIEVED_DOCS = 10

# --- Flask App Setup ---
# Use original template/static folders
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = os.urandom(24) # Needed for flashing messages and session

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Global Variables ---
embeddings = None
vectorstore = None
llm_local = None # For Ollama/DeepSeek
llm_openai = None # For OpenAI API
qa_chain_local = None # RAG chain using local LLM
qa_chain_openai = None # RAG chain using OpenAI LLM
rag_components_loaded = threading.Event()
rag_load_lock = threading.Lock()
openai_available = False # Flag to track if OpenAI setup succeeded

# --- Global Variables for Progress and Results ---
progress_queue = queue.Queue()
final_results = {}
processing_active = threading.Event()

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_llm_output(raw_output: str) -> str:
    """Cleans LLM output."""
    if not isinstance(raw_output, str): # Handle potential non-string inputs gracefully
        return "Error: Invalid model output type."
    cleaned = raw_output
    # Remove <think> blocks (Ollama specific, but harmless for others)
    cleaned = re.sub(r'<think>.*?</think>\s*', '', cleaned, flags=re.DOTALL)
    # Remove potential Markdown titles
    cleaned = re.sub(r'^#+\s*SOAP\s*Note\s*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Remove double asterisks (common formatting artifact)
    cleaned = cleaned.replace('**', '')
    cleaned = cleaned.strip()
    return cleaned

# --- Background Processing Function ---
# MODIFIED: Accepts model_choice
def process_files_background(files_data, session_id, model_choice):
    """Processes file contents in the background using the chosen model."""
    global final_results, qa_chain_local, qa_chain_openai, progress_queue, processing_active
    results_list = []
    processing_errors = []
    total_files = len(files_data)
    model_name_display = "Local (DeepSeek)" if model_choice == 'local' else "OpenAI API (GPT-4o)"
    print(f"[{session_id}] Background Task: Started for {total_files} files using {model_name_display}.")
    start_total_time = time.time()

    # Wait for RAG components to load
    if not rag_components_loaded.wait(timeout=60): # Wait up to 60 seconds
         error_msg = "RAG components did not load in time."
         print(f"[{session_id}] Background Task: ERROR - {error_msg}")
         progress_queue.put(f"event: error\ndata: {error_msg}\n\n")
         processing_active.clear(); return

    # Select the appropriate QA chain based on user choice
    selected_qa_chain = None
    if model_choice == 'openai':
        if qa_chain_openai:
            selected_qa_chain = qa_chain_openai
        else:
            error_msg = "OpenAI model/chain not available or failed to load. Check API key and server logs."
            print(f"[{session_id}] Background Task: ERROR - {error_msg}")
            progress_queue.put(f"event: error\ndata: {error_msg}\n\n")
            processing_active.clear(); return
    else: # Default to local
        if qa_chain_local:
            selected_qa_chain = qa_chain_local
        else:
             error_msg = "Local model/chain not available or failed to load. Check Ollama setup and server logs."
             print(f"[{session_id}] Background Task: ERROR - {error_msg}")
             progress_queue.put(f"event: error\ndata: {error_msg}\n\n")
             processing_active.clear(); return

    # Send initial progress message
    progress_queue.put(f"event: progress\ndata: Starting processing for {total_files} files using {model_name_display}...\n\n")

    for i, file_data in enumerate(files_data):
        filename = file_data['filename']
        dialogue_text = file_data['content']
        progress_update = f"Processing file {i+1}/{total_files} ({model_name_display}): {filename}"
        # Send progress update before starting work on the file
        progress_queue.put(f"event: progress\ndata: {progress_update}\n\n")
        print(f"[{session_id}] Background Task: {progress_update}")

        try:
            if not dialogue_text.strip():
                warning_msg = f"File '{filename}' content is empty. Skipping."
                print(f"[{session_id}] Background Task: Warning - {warning_msg}")
                processing_errors.append(warning_msg)
                progress_queue.put(f"event: warning\ndata: {warning_msg}\n\n")
                continue

            query_for_rag = dialogue_text
            start_time = time.time()
            # Invoke the SELECTED QA chain
            result = selected_qa_chain.invoke({"query": query_for_rag})
            end_time = time.time()

            raw_llm_result = result.get('result', None)
            if raw_llm_result is None:
                 cleaned_note = f"Error: No result from {model_name_display} model."
                 error_msg = f"No 'result' key found in RAG output for {filename} using {model_name_display}."
                 print(f"[{session_id}] Background Task: Warning - {error_msg}")
                 processing_errors.append(error_msg)
                 progress_queue.put(f"event: warning\ndata: {error_msg}\n\n")
            else:
                 cleaned_note = clean_llm_output(raw_llm_result)

            results_list.append({
                'filename': filename,
                'soap_note': cleaned_note,
                'time': f"{end_time - start_time:.2f}",
                'model_used': model_name_display # Add which model was used
            })
            print(f"[{session_id}] Background Task: Finished {filename} ({model_name_display}) in {end_time - start_time:.2f}s")

        except Exception as e:
            error_msg = f"Error processing '{filename}' with {model_name_display}: {str(e)}"
            # Check for common OpenAI API key error
            if model_choice == 'openai' and ('api_key' in str(e).lower() or 'openai_api_key' in str(e).lower()):
                 error_msg += " (Check if OPENAI_API_KEY environment variable is set correctly)"
            print(f"[{session_id}] Background Task: ERROR - {error_msg}")
            processing_errors.append(error_msg)
            progress_queue.put(f"event: error\ndata: {error_msg}\n\n")
            results_list.append({
                'filename': filename,
                'soap_note': f"Error processing this file's content with {model_name_display}: {str(e)}",
                'time': "N/A",
                'model_used': model_name_display
            })

    end_total_time = time.time()
    print(f"[{session_id}] Background Task: Storing results for session.")
    final_results[session_id] = {
        'results': results_list,
        'errors': processing_errors,
        'timestamp': time.time()
    }
    print(f"[{session_id}] Background Task: Results stored. Keys in final_results: {list(final_results.keys())}")

    print(f"[{session_id}] Background Task: Finished all files ({model_name_display}) in {end_total_time - start_total_time:.2f}s.")
    progress_queue.put(f"event: complete\ndata: Processing complete ({model_name_display}). Redirecting to results...\n\n")
    processing_active.clear()


# --- Load RAG Components Function ---
# MODIFIED: Loads both Ollama and OpenAI components
def load_rag_components():
    global embeddings, vectorstore, llm_local, llm_openai, qa_chain_local, qa_chain_openai, rag_components_loaded, openai_available
    if rag_components_loaded.is_set():
        print("RAG components already loaded or loading attempt finished.")
        return
    rag_components_loaded.clear()
    openai_available = False # Reset flag
    print("--- Loading RAG Components (Local & OpenAI) ---")

    # --- Shared Components ---
    try:
        # Check CUDA for embeddings
        if torch.cuda.is_available(): device = 'cuda'; print(f"Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
        else: device = 'cpu'; print("Warning: Using CPU for embeddings.")

        # Check FAISS index
        if not os.path.exists(FAISS_INDEX_PATH): raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

        # Load Embeddings
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})
        print("Embedding model loaded.")

        # Load FAISS index
        print(f"Loading FAISS index: {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")

        # Define SOAP Prompt Template (shared for both models)
        print("Defining SOAP Prompt Template...")
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
**IMPORTANT: Directly output ONLY the SOAP note content (Subjective, Objective, Assessment, Plan sections). Do NOT include any thinking process (e.g., <think>...</think> tags), explanations, comments, or any other text outside the note itself.**
"""
        SOAP_PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
        print("SOAP Prompt defined.")

        # Create Retriever (shared)
        retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})
        print(f"Retriever created (k={NUM_RETRIEVED_DOCS}).")

    except Exception as e:
        print(f"CRITICAL ERROR during shared RAG component loading: {e}")
        # Mark loading as complete (failed) and exit the function
        rag_components_loaded.set()
        return

    # --- Load Local (Ollama) Components ---
    try:
        print(f"Connecting to local Ollama model: {LOCAL_LLM_MODEL_NAME}...")
        llm_local = ChatOllama(model=LOCAL_LLM_MODEL_NAME, temperature=0.7)
        # Test connection (optional, but good practice)
        # llm_local.invoke("Hello")
        print(f"Connected to Ollama model.")

        print("Creating RAG chain (Local)...")
        qa_chain_local = RetrievalQA.from_chain_type(
            llm=llm_local,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False, # Keep false for web app
            chain_type_kwargs={"prompt": SOAP_PROMPT}
        )
        print("RAG chain created successfully (Local).")
    except Exception as e:
        print(f"ERROR during Local (Ollama) component loading: {e}")
        print("   (Ensure Ollama service is running and the model is pulled)")
        qa_chain_local = None # Mark as unavailable

    # --- Load OpenAI Components ---
    if ChatOpenAI is None:
        print("Skipping OpenAI component loading because ChatOpenAI could not be imported.")
    else:
        try:
            # Check for API Key before trying to initialize
            if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
                print("Warning: OPENAI_API_KEY environment variable not set or empty. OpenAI functionality will be disabled.")
                raise EnvironmentError("OpenAI API Key missing or empty.") # Raise to skip OpenAI setup

            print(f"Initializing OpenAI LLM: {OPENAI_LLM_MODEL_NAME}...")
            llm_openai = ChatOpenAI(model_name=OPENAI_LLM_MODEL_NAME, temperature=0.7)
             # Test connection (optional)
            # llm_openai.invoke("Hello")
            print(f"OpenAI LLM '{OPENAI_LLM_MODEL_NAME}' initialized.")

            print("Creating RAG chain (OpenAI)...")
            qa_chain_openai = RetrievalQA.from_chain_type(
                llm=llm_openai,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False, # Keep false for web app
                chain_type_kwargs={"prompt": SOAP_PROMPT}
            )
            print("RAG chain created successfully (OpenAI).")
            openai_available = True # Mark OpenAI as ready

        except Exception as e:
            print(f"ERROR during OpenAI component loading: {e}")
            print("   (Check API key, network connection, 'langchain-openai' installation)")
            qa_chain_openai = None # Mark as unavailable
            openai_available = False

    # --- Finalize Loading ---
    if qa_chain_local or qa_chain_openai:
        print("--- RAG Components Loading Attempt Finished ---")
        if qa_chain_local: print("Local model ready.")
        if qa_chain_openai: print("OpenAI model ready.")
    else:
        print("--- CRITICAL: NO RAG models could be loaded. Application may not function. ---")

    rag_components_loaded.set() # Signal that loading process (even if partial) is complete


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    if 'process_id' not in session:
        session['process_id'] = os.urandom(8).hex()
        print(f"New session started. ID: {session['process_id']}")
    else:
        print(f"Existing session. ID: {session['process_id']}")

    # Pass model availability flags directly to the template
    # These flags are set during load_rag_components()
    # Ensure components are loaded before checking flags
    if not rag_components_loaded.is_set():
        # Wait a very short time or flash a message if loading takes too long
        rag_components_loaded.wait(timeout=0.5) # Non-blocking wait

    # Now check flags (they might be None if loading failed)
    local_is_ready = bool(qa_chain_local)
    openai_is_ready = openai_available

    # Flash message if still loading
    if not rag_components_loaded.is_set():
        flash("System is initializing, please wait a moment...", "warning")

    return render_template('index.html',
                           local_ready=local_is_ready,
                           openai_ready=openai_is_ready)

# MODIFIED: Accepts model_choice from form
@app.route('/process', methods=['POST'])
def process_files_start():
    """Handles file uploads, gets model choice, starts background processing, returns JSON."""
    global processing_active, progress_queue, final_results, qa_chain_local, qa_chain_openai

    # Get or create session ID
    if 'process_id' not in session:
         session['process_id'] = os.urandom(8).hex()
         print(f"New session started in /process. ID: {session['process_id']}")
    session_id = session['process_id']
    print(f"[{session_id}] Request Handler: POST /process")

    # Get model choice from form data
    model_choice = request.form.get('model_choice', 'local') # Default to 'local' if not provided
    print(f"[{session_id}] Request Handler: Model choice from form: {model_choice}")

    # Check RAG status before proceeding
    if not rag_components_loaded.is_set():
        return jsonify({'success': False, 'message': 'RAG components still loading. Please wait and try again.'}), 503 # Service Unavailable

    # Check if the chosen model is actually available
    if model_choice == 'openai' and not qa_chain_openai:
         return jsonify({'success': False, 'message': 'OpenAI model selected, but it is not available. Check API key and server logs.'}), 500
    if model_choice == 'local' and not qa_chain_local:
         return jsonify({'success': False, 'message': 'Local model selected, but it is not available. Check Ollama setup and server logs.'}), 500
    if processing_active.is_set():
        return jsonify({'success': False, 'message': 'A processing task is already running. Please wait.'}), 429 # Too Many Requests

    # Check for files
    if 'dialogue_files' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request.'}), 400 # Bad Request
    files = request.files.getlist('dialogue_files')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'No files selected.'}), 400

    # Read files and validate
    valid_files_data = []; invalid_files_messages = []
    print(f"[{session_id}] Request Handler: Reading files...")
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                content = file.read().decode('utf-8')
                valid_files_data.append({'filename': filename, 'content': content})
            except Exception as e:
                err_msg = f"Error reading '{filename}': {e}."
                invalid_files_messages.append(err_msg)
        elif file and file.filename != '':
            err_msg = f"Invalid file type '{secure_filename(file.filename)}'. Only .txt allowed."
            invalid_files_messages.append(err_msg)

    if not valid_files_data:
         error_message = "No valid .txt files could be read."
         if invalid_files_messages:
             error_message += " Issues: " + "; ".join(invalid_files_messages)
         return jsonify({'success': False, 'message': error_message}), 400

    # --- Start background processing ---
    final_results.pop(session_id, None) # Clear previous results for this ID
    while not progress_queue.empty(): # Clear queue
         try: progress_queue.get_nowait()
         except queue.Empty: break
    print(f"[{session_id}] Request Handler: Cleared previous results/queue.")

    print(f"[{session_id}] Request Handler: Starting background thread ({len(valid_files_data)} files, model: {model_choice}).")
    processing_active.set()

    # Pass model_choice to the background function
    thread = threading.Thread(target=process_files_background, args=(valid_files_data, session_id, model_choice))
    thread.daemon = True; thread.start()

    # Return success JSON
    model_name_display = "Local (DeepSeek)" if model_choice == 'local' else "OpenAI API (GPT-4o)"
    return jsonify({
        'success': True,
        'message': f'Processing started using {model_name_display}.',
        'invalid_files': invalid_files_messages
    })


# --- SSE Stream Route (No changes needed from original app.py) ---
@app.route('/stream')
def stream():
    """SSE endpoint."""
    if 'process_id' not in session: return Response("event: error\ndata: Missing session ID\n\n", mimetype="text/event-stream")
    session_id = session['process_id']
    print(f"[{session_id}] SSE Handler: Connection established.")
    @stream_with_context
    def event_stream():
        is_active = processing_active.is_set()
        print(f"[{session_id}] SSE Handler: Stream started. Active: {is_active}")
        if is_active:
            yield f"event: progress\ndata: Processing already in progress...\n\n"

        while is_active:
            try:
                message = progress_queue.get(timeout=0.5)
                yield message
            except queue.Empty:
                if not processing_active.is_set():
                    is_active = False
                    print(f"[{session_id}] SSE Handler: Processing stopped.")
                else:
                    yield ": keepalive\n\n" # Send keepalive comment
                    continue # Continue checking queue
        print(f"[{session_id}] SSE Handler: Sending remaining messages.")
        while not progress_queue.empty():
            try: message = progress_queue.get_nowait(); yield message
            except queue.Empty: break
        print(f"[{session_id}] SSE Handler: Stream closing.")
    return Response(event_stream(), mimetype="text/event-stream")

# --- Results Route (No changes needed from original app.py, but template is updated) ---
@app.route('/results')
def show_results():
    """Displays results page."""
    if 'process_id' not in session:
         print("[Results Handler] Error: No process_id in session.")
         flash('Session invalid or expired. Please start a new upload.', 'error')
         return redirect(url_for('index'))
    session_id = session['process_id']
    print(f"[{session_id}] Results Handler: Fetching results. Keys: {list(final_results.keys())}")

    results_data = final_results.get(session_id)

    # Optional: Clear results from memory after fetching
    # final_results.pop(session_id, None)
    # print(f"[{session_id}] Results Handler: Results removed from memory after retrieval.")


    if results_data is None:
        print(f"[{session_id}] Results Handler: No results found for ID.")
        # Distinguish between ongoing processing and truly missing results
        if processing_active.is_set():
            flash("Processing is still ongoing. Please wait for completion.", 'warning')
            print(f"[{session_id}] Results Handler: Redirecting - processing active.")
            # Redirect back to index where the loading indicator is shown
            return redirect(url_for('index'))
        else:
            flash("No results found for your session, or they may have expired. Please try uploading again.", 'error')
            print(f"[{session_id}] Results Handler: Redirecting - no results, inactive.")
            return redirect(url_for('index'))

    print(f"[{session_id}] Results Handler: Results found. Timestamp: {results_data.get('timestamp')}")
    # Display any errors that occurred during background processing
    for error in results_data.get('errors', []):
        flash(error, 'error') # Use 'error' category for errors

    # Render the results template (results.html needs to be updated to show 'model_used')
    return render_template('results.html', results=results_data.get('results', []))


# --- Initialization using before_request ---
# Ensures RAG components are loaded before handling requests
@app.before_request
def ensure_rag_loaded_before_request():
    # Only attempt load if not already loaded/attempted
    if not rag_components_loaded.is_set():
        with rag_load_lock: # Prevent multiple threads trying to load simultaneously
            if not rag_components_loaded.is_set():
                print("RAG components not loaded, attempting load via before_request...")
                try:
                    load_rag_components()
                except Exception as e:
                    print(f"ERROR during RAG loading (before_request): {e}")
                    # Set the event even on failure so we don't retry indefinitely
                    rag_components_loaded.set()

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask server (supporting Local and OpenAI models)...")
    # Ensure RAG components load at startup in the main thread if possible
    # This might block startup but ensures readiness for first request
    if not rag_components_loaded.is_set():
        print("Initial RAG component load initiated at startup...")
        load_rag_components()

    # Run the Flask app
    # Use threaded=True for background tasks and SSE
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)