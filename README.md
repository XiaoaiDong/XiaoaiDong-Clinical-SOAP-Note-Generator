# Clinical-SOAP-Note-Generator

## Overview

This project provides a two-part workflow to automatically generate clinical SOAP notes from audio recordings of therapy or counseling sessions.

1.  **Speech-to-Text:** Transcribes audio files into text format, including speaker diarization (identifying who spoke when) and optional translation.
2.  **SOAP Note Generation:** Uses the generated text transcript and a Retrieval-Augmented Generation (RAG) approach to create a structured SOAP note (Subjective, Objective, Assessment, Plan).

## Features

* **Local Speech-to-Text:** Utilizes a local Whisper model for accurate transcription without sending audio data externally.
* **Speaker Diarization:** Identifies different speakers in the audio using Pyannote.audio (requires setup and Hugging Face token).
* **Optional Translation:** Can translate non-Chinese transcripts into Chinese using the OpenAI API (requires API key).
* **RAG-based SOAP Note Generation:** Leverages a knowledge base built from mental health counseling conversations and Large Language Models (LLMs) to generate relevant and structured SOAP notes.
* **Flexible LLM Choice:** Supports both a local LLM (e.g., DeepSeek via Ollama) and the OpenAI API (GPT-4o) for SOAP note generation.
* **Web Interface:** Provides a simple web interface (Flask app) for uploading transcripts and viewing generated SOAP notes.
* **GUI for Transcription:** Includes a desktop GUI application (PyQt6) for easy audio file processing.
* **Multiple Output Formats:** Transcripts can be saved as TXT or PDF. SOAP Notes can be saved as PDF from the web interface.

## Prerequisites & Setup

### General:

* Python 3.x
* Git (for cloning repositories if needed)

### Part 1: Speech-to-Text (`audio_to_text_local`)

* **Core:**
    * `openai-whisper`: For transcription.
    * `torch`: Required by Whisper and Pyannote. Install the version compatible with your hardware (CPU or CUDA).
    * `PyQt6`: For the GUI application.
    * `fpdf2`: For saving transcripts as PDF.
* **Optional (Highly Recommended):**
    * `ffmpeg`: Required by Whisper and Pydub for handling various audio formats. Ensure it's installed and in your system's PATH.
    * `pydub`: For audio format conversion.
    * `pyannote.audio`: For speaker diarization. Requires accepting model terms on Hugging Face (`pyannote/speaker-diarization-3.1`).
    * `Hugging Face Hub Account & Token`: Required for downloading the Pyannote model. Set the token in `backend.py` or as an environment variable.
* **Optional (Translation):**
    * `openai` Python library: For using the translation feature.
    * `OpenAI API Key`: Required for translation. Set the `OPENAI_API_KEY` environment variable.
* **Fonts:** Ensure the specified `.ttf` font files (`yahei.ttf`, `yahei_bold.ttf`) exist at the paths configured in `backend.py` or update the paths if necessary. Arial will be used as a fallback if they are not found, which may cause issues with non-Latin characters.

### Part 2: SOAP Note Generation (`create_vectorstore.py` & `app.py`)

* **Vector Store Creation (`create_vectorstore.py`):**
    * `datasets`: To download the `Amod/mental_health_counseling_conversations` dataset.
    * `pandas`: Data manipulation.
    * `langchain`, `langchain-community`: Core framework for RAG.
    * `sentence-transformers`: To download and use the embedding model (`all-MiniLM-L6-v2`).
    * `faiss-cpu` or `faiss-gpu`: For creating and saving the vector index. `faiss-gpu` requires CUDA.
    * `torch`: Required by sentence-transformers.
    * `transformers`, `accelerate`: Dependencies for Hugging Face models.
* **Web Application (`app.py`):**
    * `Flask`: Web framework.
    * `langchain`, `langchain-community`, `langchain-openai`: Core framework for RAG.
    * `faiss-cpu` or `faiss-gpu`: To load the vector index.
    * `sentence-transformers`, `torch`: To load the embedding model.
    * **Optional (Local LLM):**
        * `Ollama`: Needs to be installed and running separately. Pull the desired model (e.g., `ollama pull deepseek-r1:14b`). Ensure the model name in `app.py` (`LOCAL_LLM_MODEL_NAME`) matches the pulled model.
    * **Optional (OpenAI LLM):**
        * `openai` Python library (likely installed via `langchain-openai`).
        * `OpenAI API Key`: Required for using the OpenAI model. Set the `OPENAI_API_KEY` environment variable.

## Usage

### Part 1: Transcribing Audio (`audio_to_text_local`)

1.  **Configure `backend.py`:**
    * Set the `OUTPUT_DIR` variable to your desired default location for saving transcripts.
    * Set `HUGGING_FACE_TOKEN` if needed for Pyannote.
    * Ensure `FONT_PATH_REGULAR` and `FONT_PATH_BOLD` point to valid font files.
    * Set the `OPENAI_API_KEY` environment variable if you intend to use the translation feature.
2.  **Run the GUI:**
    ```bash
    python audio_to_text_local/gui_app.py
    ```
3.  **Select File:** Click the "Select File" button and choose your audio recording (e.g., `.mp3`, `.wav`, `.m4a`).
4.  **Choose Options:** Check the "Enable Chinese Translation" box if needed.
5.  **Start Processing:** Click the "Start Processing" button. The application will:
    * Convert the audio to a suitable format (if needed, requires `pydub`/`ffmpeg`).
    * Perform speaker diarization (if `pyannote.audio` is installed and configured).
    * Transcribe the audio using the local Whisper model.
    * Align speaker information with the transcript.
    * Filter potentially repetitive or meaningless segments.
    * Translate segments to Chinese (if enabled and source is not Chinese).
    * Display the results in the text area.
6.  **Save Transcript:** Once processing is complete, click "Save TXT" or "Save PDF" to save the formatted transcript. The TXT file is typically used as input for the next part.

### Part 2: Generating SOAP Notes

1.  **Create Vector Store (Run Once):**
    * Execute the `create_vectorstore.py` script:
        ```bash
        python create_vectorstore.py
        ```
    * This script downloads the `Amod/mental_health_counseling_conversations` dataset, processes it, generates embeddings using `all-MiniLM-L6-v2`, and saves a FAISS vector index to the `faiss_index_mental_health_gpu` folder (or the path specified in `FAISS_INDEX_PATH`). This index acts as the knowledge base for the RAG system. Ensure you have sufficient disk space and potentially GPU resources for faster processing.
2.  **Run the SOAP Note Web Application:**
    * **Ensure Prerequisites:**
        * Make sure the FAISS index folder (`faiss_index_mental_health_gpu`) exists in the same directory as `app.py`.
        * If using the local LLM option, ensure Ollama is running and the specified model (e.g., `deepseek-r1:14b`) is pulled.
        * If using the OpenAI option, ensure the `OPENAI_API_KEY` environment variable is set.
    * **Start the Flask App:**
        ```bash
        python app.py
        ```
    * **Access the Web UI:** Open your web browser and go to `http://127.0.0.1:5001` (or the address provided in the console output).
    * **Upload Transcript:** Select the `.txt` transcript file(s) generated in Part 1.
    * **Choose Model:** Select either the "Local (DeepSeek)" or "OpenAI API (GPT-4o)" option for processing. Availability depends on your setup.
    * **Generate Notes:** Click "Generate SOAP Notes". The application will process each transcript using the selected RAG pipeline. Progress updates will be shown.
    * **View Results:** You will be redirected to a results page displaying the generated SOAP note for each uploaded file.
    * **Save Results:** You can save individual SOAP notes as PDF files directly from the results page.

## File Structure (Key Files)

**SOAP note/**
   **audio_to_text_local/** — Part 1: Speech-to-Text GUI application
    - `backend.py` — Core logic for transcription, diarization, translation, saving
    - `gui_app.py` — PyQt6 GUI application
  - `create_vectorstore.py` — Part 2: Script to build the FAISS knowledge base
  - `app.py` — Part 2: Main Flask web application for SOAP note generation
   **templates/** — HTML templates for the Flask app (`app.py`)
    - `index.html` — Main upload page
    - `results.html` — Results display page
   **static/** — Static files (CSS) for the Flask app (`app.py`)
    - `style.css`
   **faiss_index_mental_health_gpu/** — Default output folder for the FAISS index (created by `create_vectorstore.py`)
   **uploads/** — Default folder for temporary uploads by `app.py` (can be configured)

 **Intermediate/Alternative Files (Not part of the main workflow)**
  - `query_rag_local_model.py` — Standalone script testing RAG with local Ollama model
  - `query_rag_openai.py` — Standalone script/module testing RAG with OpenAI API
  - `app_openai.py` — Alternative Flask app using only OpenAI
   **templates_openai/** — Templates for `app_openai.py`
   **static_openai/** — Static files for `app_openai.py`
  - (Other generated files like `.pdf`, `.txt` transcripts)
    
## Notes

* The files `query_rag_local_model.py`, `query_rag_openai.py`, and `app_openai.py` (along with their respective `templates_openai` and `static_openai` folders) represent intermediate development steps or alternative implementations focusing solely on one LLM type. The primary workflow uses `app.py` which integrates both local and OpenAI options.
* Ensure all dependencies are correctly installed for each part of the workflow. Dependency conflicts might arise, consider using virtual environments (e.g., `venv`, `conda`).
* Speaker diarization accuracy depends heavily on audio quality and the Pyannote model's performance.
* Translation quality depends on the OpenAI API and the chosen model (`gpt-4o-mini` by default).
* SOAP note generation quality depends on the transcript quality, the richness of the vector store, and the chosen LLM's capabilities. The prompt used guides the LLM to extract information primarily from the provided transcript.
