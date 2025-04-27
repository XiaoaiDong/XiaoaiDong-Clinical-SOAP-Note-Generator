# -*- coding: utf-8 -*-
# File: app_openai.py
# Description: Flask web application for SOAP note generation using RAG + OpenAI API (GPT-4o).
# Imports logic from query_rag_openai.py

import os
import time
from flask import Flask, request, render_template, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename

# Import the RAG logic and cleaning function from the file you provided
# Ensure query_rag_openai.py is in the same directory or Python path
try:
    from query_rag_openai import load_rag_components_openai, get_qa_chain, clean_llm_output
except ImportError as e:
    print(f"ERROR: Could not import from query_rag_openai.py - {e}")
    print("Ensure query_rag_openai.py is in the same directory as app_openai.py.")
    # Exit if the essential module can't be imported
    exit(1)


# --- Configuration ---
# Using different names for folders to avoid conflicts if running both apps
UPLOAD_FOLDER = 'uploads_openai'
ALLOWED_EXTENSIONS = {'txt'}
TEMPLATE_FOLDER = 'templates_openai'
STATIC_FOLDER = 'static_openai'

# --- Flask App Setup ---
# Explicitly set template and static folders
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = os.urandom(24) # Necessary for flashing messages

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Variable for QA Chain ---
# Will be initialized at startup
qa_chain_instance = None

# --- Helper Function ---
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index_openai():
    """Renders the main upload page."""
    # Renders index_openai.html from the templates_openai folder
    return render_template('index_openai.html')

@app.route('/process_openai', methods=['POST'])
def process_files_openai():
    """Handles file uploads, processes them using the OpenAI RAG chain, and shows results."""
    global qa_chain_instance # Access the globally initialized chain

    # Check if the RAG chain is ready
    if qa_chain_instance is None:
        flash('RAG System Error: Chain not initialized. Please check server logs and restart.')
        # Redirect back to the index page as processing cannot continue
        return redirect(url_for('index_openai'))

    # Check if the post request has the file part
    if 'dialogue_files' not in request.files:
        flash('No file part selected in the form.')
        return redirect(request.url) # Redirect back to the same page (index)

    files = request.files.getlist('dialogue_files') # Get all uploaded files

    # Check if any files were actually selected
    if not files or files[0].filename == '':
        flash('No files selected for upload.')
        return redirect(request.url) # Redirect back to the index page

    results = []
    processing_errors = []

    print(f"\n--- Processing {len(files)} files (OpenAI Backend) ---")
    total_start_time = time.time()

    for file in files:
        # Ensure the file object is valid and has an allowed extension
        if file and allowed_file(file.filename):
            # Secure the filename before using it (though not saving permanently here)
            filename = secure_filename(file.filename)
            print(f"Processing file: {filename}")
            try:
                # Read file content directly from the file stream
                # Use 'utf-8' decoding, handle potential errors
                dialogue_text = file.read().decode('utf-8')

                # Check if the file content is empty after reading
                if not dialogue_text.strip():
                    print(f"Warning: File '{filename}' is empty or contains only whitespace. Skipping.")
                    processing_errors.append(f"File '{filename}' is empty.")
                    continue # Skip to the next file

                # --- RAG Chain Invocation ---
                query_for_rag = dialogue_text
                start_time = time.time()
                # Invoke the QA chain (already initialized)
                result = qa_chain_instance.invoke({"query": query_for_rag})
                end_time = time.time()
                # --- End RAG Chain Invocation ---

                # Safely get and clean the LLM output
                raw_llm_result = result.get('result', None) # Use .get() for safety
                if raw_llm_result is None:
                     print(f"Warning: No 'result' key found in RAG chain output for {filename}.")
                     cleaned_note = "Error: Could not generate SOAP note (missing result)."
                     processing_errors.append(f"Failed to generate note for '{filename}': Missing result from model.")
                else:
                    cleaned_note = clean_llm_output(raw_llm_result) # Clean the output

                # Append results for display
                results.append({
                    'filename': filename,
                    'soap_note': cleaned_note,
                    'time': f"{end_time - start_time:.2f}"
                })
                print(f"Finished processing {filename} in {end_time - start_time:.2f}s")

            except UnicodeDecodeError:
                 print(f"Error: Could not decode file '{filename}' as UTF-8. Skipping.")
                 processing_errors.append(f"Error decoding file '{filename}'. Please ensure it is UTF-8 encoded.")
            except Exception as e:
                # Catch other potential errors during processing
                print(f"Error processing file {filename} with OpenAI: {e}")
                processing_errors.append(f"Unexpected error processing file '{filename}': {e}")
                # Optionally add the error note to results for visibility
                results.append({
                    'filename': filename,
                    'soap_note': f"Error processing this file: {e}",
                    'time': "N/A"
                })

        elif file.filename != '': # Catch files that were selected but had wrong extension
            print(f"Warning: File type not allowed for '{file.filename}'. Skipping.")
            processing_errors.append(f"File type not allowed for '{file.filename}'. Only .txt files are accepted.")

    total_end_time = time.time()
    print(f"--- Finished processing all files in {total_end_time - total_start_time:.2f}s ---")

    # Flash any accumulated errors to the user
    for error in processing_errors:
        flash(error)

    # Render the results page, passing the generated notes (and any errors within them)
    # Renders results_openai.html from the templates_openai folder
    return render_template('results_openai.html', results=results)


# Serve static files correctly (needed if CSS isn't loading)
@app.route('/static_openai/<path:filename>')
def static_files(filename):
    return app.send_static_file(filename)


# --- Initialization Block ---
def initialize_app():
    """Loads RAG components before the first request."""
    global qa_chain_instance
    print("Attempting to initialize RAG components for OpenAI...")
    try:
        load_rag_components_openai() # Function from query_rag_openai.py
        qa_chain_instance = get_qa_chain() # Function from query_rag_openai.py
        print("RAG components loaded successfully.")
    except (FileNotFoundError, EnvironmentError, RuntimeError, ImportError, Exception) as e:
         print(f"FATAL: Initialization Error - {e}. Cannot start server.")
         # In a real deployment, you might want more robust error handling or logging here
         # For now, we print the error and exit
         exit(1) # Stop the application if critical components fail

# --- Main Execution ---
if __name__ == '__main__':
    initialize_app() # Load components before starting the server
    print("Starting Flask server for OpenAI SOAP Note Generator...")
    # Use a different port (e.g., 5002) to avoid conflict if the other app is running
    # host='0.0.0.0' makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0', port=5002)