import io
import os
import datetime
import time
import traceback
from pathlib import Path
from openai import OpenAI # Synchronous client
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re # For filtering repetitive content

# --- Local Whisper Library ---
import whisper
import torch

# --- Try importing other dependencies ---
try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    PYANNOTE_INSTALLED = True
    print("Pyannote.audio imported successfully.")
except ImportError:
    PYANNOTE_INSTALLED = False
    print("Warning: PyTorch or pyannote.audio not found.")
    Pipeline = None

try:
    from pydub import AudioSegment
    try:
        # Set converter path explicitly if needed, or rely on PATH
        AudioSegment.converter = AudioSegment.ffmpeg if AudioSegment.ffmpeg else AudioSegment.avconv
        PYDUB_READY = True
        print("pydub loaded successfully.")
    except Exception as e_pydub_check:
        print(f"Warning: pydub might not work correctly (ffmpeg/ffprobe config issue): {e_pydub_check}")
        PYDUB_READY = False
    PYDUB_INSTALLED = True
except ImportError:
    PYDUB_INSTALLED = False
    PYDUB_READY = False
    print("Warning: pydub library not found.")
    AudioSegment = None

# --- Configuration Section ---
OUTPUT_DIR = "D:/transcript" # Default output directory
API_KEY = os.getenv("OPENAI_API_KEY") # For translation
TRANSLATION_MODEL = "gpt-4o-mini" # OpenAI model for translation
FONT_PATH_REGULAR = "D:/yahei.ttf" # Path to regular font file for PDF
FONT_PATH_BOLD = "D:/yahei_bold.ttf" # Path to bold font file for PDF
HUGGING_FACE_TOKEN = "set to your own token" # Needed for Pyannote model download
SPEAKER_SEGMENT_TIME_LIMIT_SEC = 30 # Max seconds gap before splitting same speaker segment

# --- Global OpenAI Client (Synchronous version) ---
try:
    if API_KEY and API_KEY.startswith("sk-"): # Basic key validation
        client = OpenAI(api_key=API_KEY)
        print("OpenAI synchronous client initialized successfully (for translation).")
    else:
        client = None; raise ValueError("Invalid OpenAI API key provided.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}"); client = None

# --- Helper Functions ---
def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS.fff string."""
    if seconds is None: return "N/A"
    # Secure calculation avoiding potential float precision issues
    total_microseconds = int(seconds * 1_000_000)
    # Extract components
    td = datetime.timedelta(microseconds=total_microseconds)
    hours, remainder_seconds = divmod(td.seconds, 3600)
    minutes, seconds_part = divmod(remainder_seconds, 60)
    milliseconds = td.microseconds // 1000
    # Add days if necessary (for very long audio)
    hours += td.days * 24
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds_part):02}.{int(milliseconds):03}"

def convert_to_mono_flac(input_path: Path, output_dir: Path) -> Path | None:
    """Converts input audio file to mono FLAC format (16kHz) using pydub."""
    if not PYDUB_READY or not AudioSegment:
        print("Error: pydub library is not ready or installed. Cannot convert audio."); return None
    # print(f"Attempting to convert '{input_path.name}' to mono FLAC (16kHz)...") # Debugging info
    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        # Convert to mono and set frame rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        # Define output path and filename
        flac_filename = input_path.stem + "_converted.flac"
        output_path = output_dir / flac_filename
        # Remove existing converted file to prevent errors
        if output_path.exists():
            try: os.remove(output_path)
            except Exception as e_del: print(f"Warning: Failed to delete existing converted file '{output_path}': {e_del}")
        # Export to FLAC format
        audio.export(output_path, format="flac")
        # print(f"Successfully converted to FLAC: '{output_path}'") # Debugging info
        return output_path
    except FileNotFoundError: # Specifically catch if ffmpeg/ffprobe isn't found
        print(f"Error: Cannot find ffmpeg/ffprobe. Please ensure it's installed and added to the system PATH."); return None
    except Exception as e:
        # Catch other potential pydub/ffmpeg errors
        print(f"Error: Failed to convert file '{input_path}' using pydub: {e}"); traceback.print_exc(); return None

# --- PDF Generation Class ---
class PDF(FPDF):
    """Handles PDF generation with custom fonts."""
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.fonts_loaded = {'regular': False, 'bold': False}
        # Load regular font
        if os.path.exists(FONT_PATH_REGULAR):
            try:
                self.add_font('yahei', '', FONT_PATH_REGULAR) # '' for regular style
                self.fonts_loaded['regular'] = True
            except Exception as e: print(f"PDF Warning: Failed to load regular font '{FONT_PATH_REGULAR}': {e}")
        else: print(f"PDF Warning: Regular font file not found '{FONT_PATH_REGULAR}'")
        # Load bold font
        if os.path.exists(FONT_PATH_BOLD):
            try:
                self.add_font('yahei', 'B', FONT_PATH_BOLD) # 'B' for bold style
                self.fonts_loaded['bold'] = True
            except Exception as e: print(f"PDF Warning: Failed to load bold font '{FONT_PATH_BOLD}': {e}")
        else: print(f"PDF Warning: Bold font file not found '{FONT_PATH_BOLD}'")
        # Set default fonts, falling back to Arial if YaHei failed
        self.default_font = 'yahei' if self.fonts_loaded['regular'] else 'Arial'
        self.default_font_bold = 'yahei' if self.fonts_loaded['bold'] else 'Arial'
        # Warn if fallback is used, as Chinese might not render
        if 'yahei' not in self.default_font:
             print("PDF Warning: YaHei font not loaded, using Arial. Chinese characters might not display correctly!")

    def header(self):
        """Defines the PDF header."""
        try: self.set_font(self.default_font, '', 12) # Use default regular font
        except Exception: self.set_font('Arial', '', 12) # Ultimate fallback
        self.cell(0, 10, 'Audio Transcription, Translation & Speaker Log', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C'); self.ln(10)

    def footer(self):
        """Defines the PDF footer."""
        self.set_y(-15); # Position 1.5 cm from bottom
        try: self.set_font(self.default_font, '', 8) # Use default regular font, small size
        except Exception: self.set_font('Arial', '', 8) # Ultimate fallback
        # Align page number to the center
        self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def add_aligned_entry(self, start_time_str, end_time_str, speaker, text, translation=None):
        """Adds a formatted entry (speaker, text, optional translation) to the PDF."""
        font_name = self.default_font; font_name_bold = self.default_font_bold
        try:
            # --- Speaker Info Line (Bold) ---
            try: self.set_font(font_name_bold, 'B', 10) # Try setting bold font
            except RuntimeError: self.set_font(font_name, '', 10) # Fallback to regular if bold failed
            self.cell(0, 5, f"[{start_time_str} --> {end_time_str}] {speaker}:"); self.ln(5) # Add speaker line

            # --- Original Text (Regular) ---
            self.set_font(font_name, '', 10) # Set regular font for text
            # Use multi_cell for potential line breaks, indent text
            self.multi_cell(0, 5, f"  {text}"); self.ln(1) # Small space after text

            # --- Translation (Regular, smaller font, indented) ---
            if translation:
                self.set_font(font_name, '', 9) # Use regular font, slightly smaller size
                # Indent translation further and add prefix
                self.multi_cell(0, 5, f"    Translation: {translation}"); self.ln(1) # Small space after translation

            self.ln(2) # Extra space between entries

        except Exception as e:
             print(f"PDF Warning: Error adding entry to PDF: {e}"); traceback.print_exc()
             # Attempt to add entry using Arial as a fallback
             try:
                 self.set_font('Arial', 'B', 10); self.cell(0, 5, f"[{start_time_str} --> {end_time_str}] {speaker}:"); self.ln(5)
                 self.set_font('Arial', '', 10); self.multi_cell(0, 5, f"  {text[:100]}..."); self.ln(1) # Show partial text only
                 if translation:
                     self.set_font('Arial', '', 9); self.multi_cell(0, 5, f"    Translation: [Content might not display correctly due to font issue]"); self.ln(1)
                 self.ln(2)
             except Exception as e_arial: print(f"Error: Arial font fallback also failed: {e_arial}")

# --- Main File Processing Class ---
class FileTranscriberTranslator:
    """Handles the entire audio processing pipeline: conversion, transcription, diarization, alignment, translation, and saving."""
    def __init__(self, translate=False, hf_token=None, status_callback=None, progress_callback=None):
        """Initializes the processor, loads models, and sets up callbacks."""
        self.translate_enabled = translate
        # Determine Hugging Face token (priority: argument > config > environment)
        self.hf_token = hf_token or HUGGING_FACE_TOKEN or os.getenv("HUGGING_FACE_TOKEN")
        # Initialize state variables
        self.full_transcript_result = None; self.diarization = None
        self.aligned_transcript = None; self.final_output_segments = None
        self.diarization_pipeline = None; self.detected_language = None
        # Store callbacks for UI updates
        self.status_callback = status_callback; self.progress_callback = progress_callback
        self.local_whisper_model = None # Initialize Whisper model variable

        # --- Load Local Whisper Model ---
        try:
            # Choose model size ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
            model_size = "large" # Adjust as needed for performance/accuracy trade-off
            device = "cuda" if torch.cuda.is_available() else "cpu" # Auto-detect GPU
            self._update_status(f"Loading local Whisper model ({model_size}, {device}). This may take a moment...")
            self.local_whisper_model = whisper.load_model(model_size, device=device)
            self._update_status(f"Local Whisper model ({model_size}, {device}) loaded successfully.")
        except Exception as e:
            self._update_status(f"Error: Failed to load local Whisper model: {e}. Check installation and model files.", is_error=True)
            traceback.print_exc()

        # --- Load Pyannote Diarization Model ---
        if PYANNOTE_INSTALLED:
            self._load_diarization_pipeline() # Attempt to load if library is present
        else:
            self._update_status("Warning: Pyannote/PyTorch not installed. Speaker diarization will be disabled.")

        # --- Check OpenAI Client for Translation ---
        global client # Access the global client instance
        if not client and self.translate_enabled:
             # Warn if translation is requested but client failed
             self._update_status("Warning: Translation enabled but OpenAI client failed to initialize (check API Key). Translation will fail.", is_error=True)
        elif not client:
             # Info message if client is not needed or failed silently
             self._update_status("Info: OpenAI client not initialized (API Key missing/invalid). Translation is disabled or will fail if enabled later.")

    def _update_status(self, message, is_error=False):
        """Safely calls the status callback function if provided."""
        # print(f"Status ({'Error' if is_error else 'Info'}): {message}") # Optional console logging
        if self.status_callback:
            try:
                # Try calling with the is_error flag (newer interface)
                self.status_callback(message, is_error)
            except TypeError:
                 # Fallback for older interface without is_error
                 try: self.status_callback(message)
                 except Exception as e_cb_old: print(f"Error: Calling legacy status callback failed: {e_cb_old}")
            except Exception as e: print(f"Error: Calling status callback failed: {e}")

    def _update_progress(self, value):
        """Safely calls the progress callback function if provided."""
        if self.progress_callback:
            try:
                # Ensure value is within 0-100 range
                progress_value = max(0, min(100, int(value)))
                self.progress_callback(progress_value)
            except Exception as e: print(f"Error: Calling progress callback failed: {e}")

    def _load_diarization_pipeline(self):
        """Loads the Pyannote speaker diarization pipeline from Hugging Face."""
        if not Pipeline: return # Should not happen if PYANNOTE_INSTALLED is True, but safeguard
        if not self.hf_token:
            self._update_status("Warning: Hugging Face Token not provided. Pyannote model download might fail if the model requires authentication.");
            # Depending on the model, loading might still work for public models
            # If using a gated model (like speaker-diarization-3.1), this will likely fail.
        try:
            self._update_status("Loading Pyannote speaker diarization model (may require download)...")
            # Ensure you have accepted the terms on Hugging Face Hub for this model
            # Model: pyannote/speaker-diarization-3.1 (or others like pyannote/speaker-diarization)
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token # Use token if provided, necessary for gated models
            )
            # Move model to GPU if available for faster processing
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            if torch.cuda.is_available(): self.diarization_pipeline.to(torch.device("cuda"))
            self._update_status(f"Pyannote model loaded successfully and running on {device_type}.")
        except Exception as e:
            # Catch errors like authentication failure, download issues, etc.
            self._update_status(f"Error: Failed to load Pyannote model: {e}. Check token, internet connection, and model terms acceptance.", is_error=True); traceback.print_exc();
            self.diarization_pipeline = None # Ensure pipeline is None on failure

    def _translate_text(self, text_to_translate):
        """Translates a given text string to Chinese using the OpenAI API."""
        global client; # Use the globally initialized client
        # Check if translation should be skipped
        if not client:
            # Avoid repeated status updates if client is known to be unavailable
            # self._update_status("Error: OpenAI client not initialized, cannot translate.", is_error=True);
            return None
        if not text_to_translate or not self.translate_enabled or len(text_to_translate.strip()) < 3:
             # print("Skipping translation for short/empty text.") # Debug info
             return None # Don't attempt to translate very short or empty strings

        # print(f"Translating: {text_to_translate[:50]}...") # Debug: Show text being translated
        try:
            # Call the OpenAI Chat Completions API
            completion = client.chat.completions.create(
                model=TRANSLATION_MODEL, # Use configured translation model
                messages=[
                    {"role": "system", "content": "You are a helpful translator. Please translate the following text into Chinese, keeping the meaning and tone as close as possible."},
                    {"role": "user", "content": text_to_translate}
                ],
                temperature=0.3, # Lower temperature for more deterministic translation
                max_tokens=4000, # Set a reasonable limit based on expected text length
                timeout=600.0 # Generous timeout for API call
            )
            # Extract the translated text
            translation = completion.choices[0].message.content.strip()
            # Basic validation: return None if translation is empty or identical to original
            return translation if translation and translation != text_to_translate else None
        except Exception as e: # Catch potential API errors (rate limits, connection issues, etc.)
            self._update_status(f"[Translation API Error]: {e}", is_error=True); traceback.print_exc();
            return None # Return None on error

    def _transcribe_entire_audio(self, audio_path: Path):
        """Transcribes the entire audio file using the loaded local Whisper model."""
        if not self.local_whisper_model:
            self._update_status("Error: Local Whisper model is not loaded. Cannot transcribe.", is_error=True); return False
        self.full_transcript_result = None # Reset result from previous runs

        try:
            self._update_status(f"Starting local Whisper transcription for: {audio_path.name}...")
            audio_path_str = str(audio_path) # Whisper expects a string path
            # Determine if half-precision floating point can be used (typically faster on GPU)
            use_fp16 = True if self.local_whisper_model.device.type == 'cuda' else False

            # --- Call Whisper's transcribe method ---
            # For more options, see: https://github.com/openai/whisper#available-models-and-languages
            start_time = time.time()
            response = self.local_whisper_model.transcribe(
                audio_path_str,
                verbose=None, # Set to True/False for different levels of console output during transcription
                fp16=use_fp16, # Use FP16 if on GPU
                # beam_size=5, # Optional: Adjust beam size (can affect accuracy/speed)
                # best_of=5, # Optional: Number of candidates for beam search
                condition_on_previous_text=True, # Helps with consistency across chunks (less relevant here as we process whole file)
                no_speech_threshold=0.5, # Probability threshold below which audio is considered non-speech
                # word_timestamps=False, # Set True for word-level timestamps (slower)
                # language=None # Set to a specific language code (e.g., "en", "zh") to force it
            )
            end_time = time.time()
            self._update_status(f"Local Whisper transcription finished. Duration: {end_time - start_time:.2f}s")

            # --- Process the response ---
            if response and 'language' in response:
                self.detected_language = response['language'] # Store detected language code
            else: self.detected_language = "Unknown" # Fallback if language detection fails
            self._update_status(f"Whisper detected language: {self.detected_language}")

            # Check if segments with timestamps were produced
            if response and 'segments' in response and response['segments']:
                self.full_transcript_result = response # Store the entire result dictionary
                return True
            else:
                # Handle cases where transcription might have failed or produced no segments
                self._update_status(f"Warning: Local Whisper did not return valid timestamped segments.")
                # Check if there's at least the full text available
                if response and 'text' in response and response['text'].strip():
                     # Create a fallback result structure if only text is present
                     self.full_transcript_result = {'text': response['text'].strip(), 'segments': []}
                     self._update_status("Warning: No timestamped segments found. Using full transcribed text only.")
                     return True # Consider it a success if text exists, though alignment won't work well
                # No segments and no text -> Failure
                self._update_status("Error: Transcription produced no text.", is_error=True)
                return False

        except Exception as e:
            # Catch any other exception during transcription
            self._update_status(f"[Local Whisper Transcription Error]: {e}", is_error=True); traceback.print_exc();
            return False

    def _diarize_audio(self, audio_path: Path):
        """Performs speaker diarization using the loaded Pyannote pipeline."""
        # Check if diarization should be skipped (pipeline not loaded or failed)
        if not self.diarization_pipeline:
            # Message already sent during init or loading attempt
            if PYANNOTE_INSTALLED: self._update_status("Skipping diarization: Pyannote model not loaded.", is_error=True)
            else: self._update_status("Skipping diarization: Pyannote not installed.")
            self.diarization = [] # Ensure diarization is an empty list
            return True # Return True as skipping is not a processing failure

        # Avoid re-running diarization if results already exist
        if self.diarization is not None:
            # print("Diarization result already exists, skipping.") # Debug message
            return True

        try:
            self._update_status("Running Pyannote speaker diarization (can be slow)...")
            start_time = time.time()
            # --- Perform diarization ---
            # The pipeline takes the file path and returns a pyannote.core.Annotation object
            diarization_result = self.diarization_pipeline(str(audio_path))
            end_time = time.time()
            self._update_status(f"Speaker diarization completed. Duration: {end_time - start_time:.2f}s")

            # --- Extract speaker turns from the Annotation object ---
            # itertracks yields (segment, track_name, speaker_label) tuples
            turns = [
                {'start': turn.start, 'end': turn.end, 'speaker': speaker}
                for turn, _, speaker in diarization_result.itertracks(yield_label=True)
            ]

            if not turns:
                self._update_status("Warning: Pyannote did not identify any speaker segments in this audio.")
            else:
                # Optional: Log number of speakers found
                unique_speakers = {turn['speaker'] for turn in turns}
                self._update_status(f"Pyannote identified {len(unique_speakers)} potential speakers ({', '.join(sorted(list(unique_speakers)))}).")

            self.diarization = turns # Store the list of turn dictionaries
            return True
        except Exception as e:
            # Catch errors during diarization (e.g., memory issues, model errors)
            self._update_status(f"[Pyannote Speaker Diarization Error]: {e}", is_error=True); traceback.print_exc();
            self.diarization = [] # Set to empty list on error to allow fallback
            return False # Indicate that diarization failed

    def _align_full_results(self):
        """Aligns Whisper transcription segments with Pyannote diarization turns."""
        # --- Pre-checks ---
        # Check if transcription results (segments) are available
        if not self.full_transcript_result or 'segments' not in self.full_transcript_result or not self.full_transcript_result['segments']:
            self._update_status("Error: No valid transcription segments available for alignment.", is_error=True)
            # Handle fallback if only full text exists
            if self.full_transcript_result and 'text' in self.full_transcript_result:
                 # Create a single segment representing the full text
                 self.aligned_transcript = [{'start': 0.0, 'end': None, 'speaker': 'UNKNOWN', 'text': self.full_transcript_result['text']}]
                 self._update_status("Alignment fallback: Using full text as one segment (no speaker info).")
                 return True # Treat as successful alignment for fallback
            return False # Cannot align without segments

        # Check diarization results (can proceed with 'UNKNOWN' if empty/failed)
        if self.diarization is None: # If diarization hasn't run or failed critically before this point
             self._update_status("Warning: Diarization results are missing. Marking all segments as 'UNKNOWN'.", is_error=True)
             self.diarization = [] # Ensure it's an empty list for processing
        elif not self.diarization: # If diarization ran but found no turns
             self._update_status("Info: No speaker turns found by Pyannote. Marking all transcription segments as 'UNKNOWN'.")
             # self.diarization is already [], no action needed

        # --- Alignment Logic ---
        self.aligned_transcript = [] # Initialize list for aligned results
        transcript_segments = self.full_transcript_result['segments']
        diarization_turns = self.diarization # Use the potentially empty list

        # Iterate through each transcription segment from Whisper
        for segment in transcript_segments:
            t_start = segment.get('start') # Start time of Whisper segment
            t_end = segment.get('end')     # End time of Whisper segment
            text = segment.get('text','').strip() # Text content of the segment

            # Skip segment if essential info is missing
            if t_start is None or t_end is None or not text: continue
            # Skip segment if duration is invalid
            segment_duration = t_end - t_start
            if segment_duration <= 0: continue

            # --- Find Overlapping Speaker ---
            speaker_overlaps = {} # Dictionary to store overlap duration per speaker for this segment
            if diarization_turns: # Only calculate overlaps if we have diarization data
                for turn in diarization_turns:
                    s_start, s_end, speaker = turn['start'], turn['end'], turn['speaker']
                    # Calculate the time overlap between the Whisper segment and the speaker turn
                    overlap_start = max(t_start, s_start)
                    overlap_end = min(t_end, s_end)
                    overlap_duration = max(0, overlap_end - overlap_start) # Ensure non-negative

                    # Add overlap duration to the speaker's total for this segment
                    # Require a small minimum overlap to count (e.g., 10ms) to avoid noise
                    if overlap_duration > 0.01:
                        speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap_duration

            # --- Determine Best Speaker ---
            if speaker_overlaps:
                # Find the speaker with the maximum overlap duration
                best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                # Alternative: Sort and pick top one (handles ties consistently)
                # best_speaker = sorted(speaker_overlaps.items(), key=lambda item: item[1], reverse=True)[0][0]
            else:
                # If no speakers overlap (or no diarization data), assign 'UNKNOWN'
                best_speaker = "UNKNOWN"

            # --- Append Aligned Segment ---
            self.aligned_transcript.append({
                'start': t_start,
                'end': t_end,
                'speaker': best_speaker,
                'text': text
            })

        # Final check if alignment produced any results
        if not self.aligned_transcript:
            self._update_status("Warning: Alignment process resulted in an empty transcript.", is_error=True);
            return False # Indicate potential issue

        # self._update_status("Alignment of transcription and diarization completed.") # Status update moved to main process flow
        return True


    def _segment_output_by_speaker_and_time(self):
        """Merges consecutive segments from the same speaker unless a time gap limit is exceeded."""
        if not self.aligned_transcript:
             self.final_output_segments = [] # Ensure it's an empty list if no input
             self._update_status("No aligned transcript data available to segment.")
             return

        self._update_status("Segmenting and merging aligned results...")
        segmented_output = []; # Initialize list for final output segments
        if not self.aligned_transcript: # Double check, should be handled above
             self.final_output_segments = segmented_output; return

        # --- Check for valid time info in the first segment ---
        first_segment = self.aligned_transcript[0]
        if first_segment.get('start') is None or first_segment.get('end') is None:
             self._update_status("Warning: First aligned segment lacks time information. Segmentation might be inaccurate.", is_error=True)
             # Fallback: Try to process remaining segments or treat all as one?
             # For simplicity, we'll stop segmentation here if the first segment is bad.
             # A more robust approach might skip bad segments.
             # If the only segment is the fallback full text, keep it.
             self.final_output_segments = self.aligned_transcript if len(self.aligned_transcript) == 1 else []
             return

        # --- Initialize with the first segment ---
        current_segment_data = first_segment
        current_speaker = current_segment_data['speaker']
        current_text_parts = [current_segment_data['text']] # Store text pieces to join later
        current_start_time = current_segment_data['start']
        last_segment_end_time = current_segment_data['end'] # Track the end of the last merged piece

        # --- Iterate through remaining aligned segments ---
        for i in range(1, len(self.aligned_transcript)):
            next_segment = self.aligned_transcript[i]
            next_speaker = next_segment['speaker']
            next_start_time = next_segment.get('start')
            next_end_time = next_segment.get('end')
            next_text = next_segment.get('text','')

            # Skip segment if critical time info is missing
            if next_start_time is None or current_start_time is None or last_segment_end_time is None or next_end_time is None:
                # print(f"Warning: Skipping segment {i} during merge due to missing time info.") # Debug
                continue

            # --- Determine if a new segment should start ---
            speaker_changed = (next_speaker != current_speaker)
            # Check if time gap exceeds the limit (only if speaker is the same)
            time_gap = next_start_time - last_segment_end_time # Use end time of last piece
            time_limit_exceeded = (not speaker_changed) and (time_gap > SPEAKER_SEGMENT_TIME_LIMIT_SEC)

            # --- Split Condition Met ---
            if speaker_changed or time_limit_exceeded:
                # Finalize and store the previous segment
                if current_text_parts: # Check if there's text accumulated
                    final_text = " ".join(current_text_parts).strip() # Join text parts
                    if final_text: # Only add if the joined text is not empty
                        segmented_output.append({
                            'start': current_start_time,
                            'end': last_segment_end_time, # Use the end time of the last piece included
                            'speaker': current_speaker,
                            'text': final_text
                        })

                # Start the new segment with the current 'next_segment' data
                current_speaker = next_speaker
                current_text_parts = [next_text] # Reset text parts
                current_start_time = next_start_time
                last_segment_end_time = next_end_time
            # --- Merge Condition Met ---
            else:
                # Append text and update the end time of the current merged segment
                current_text_parts.append(next_text)
                last_segment_end_time = next_end_time # Update to the end of the latest piece

        # --- Add the very last accumulated segment ---
        if current_text_parts:
            final_text = " ".join(current_text_parts).strip()
            if final_text:
                segmented_output.append({
                    'start': current_start_time,
                    'end': last_segment_end_time,
                    'speaker': current_speaker,
                    'text': final_text
                })

        self.final_output_segments = segmented_output # Store the final list
        # self._update_status(f"Segmentation complete. Produced {len(self.final_output_segments)} final segments.") # Status update moved

    def _filter_repetitive_segments(self):
        """Filters out segments that are likely short, repetitive, or meaningless based on duration and content patterns."""
        if not self.final_output_segments: return # Exit if there's nothing to filter

        self._update_status("Filtering repetitive or potentially meaningless segments...")
        filtered_segments = [] # List to hold segments that pass filters
        # --- Define Regular Expressions for common fillers/noises ---
        # (Adjust these patterns based on observed noise in your specific use case)
        repetitive_patterns = [
            # Common English fillers (case-insensitive)
            r"^(uh huh|uh-huh|um|uh|ah|er|hmm|yeah|ok|okay|right|so|well|you know|i see|like)[\s.,?!]*$",
            # Very short words (likely transcription errors or single sounds)
            r"^\w{1,2}$", # Matches 1 or 2 letter words (adjust length threshold if needed)
            # Single character repeated (e.g., "a a a", "...")
            r"^(.| )\s?(\1\s?)+$", # Matches a character (or space) followed by itself one or more times
            # Specific repetitive phrases sometimes generated by models (add more examples)
            r"^(Thanks for watching\.?[\s]*)+$",
            r"^(Please subscribe\.?[\s]*)+$",
            r"^(Bye\.?[\s]*)+$",
            # Just punctuation (likely errors)
            r"^[.,?!]+$"
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in repetitive_patterns] # Compile for efficiency
        # --- Define Duration Threshold ---
        min_duration_threshold = 0.4 # Segments shorter than this (in seconds) are filtered

        # --- Iterate and Filter ---
        for segment in self.final_output_segments:
            text = segment['text'].strip()
            start = segment.get('start')
            end = segment.get('end')

            # --- Basic Validity Checks ---
            if not text: continue # Skip empty text
            if start is None or end is None: continue # Skip if time is missing

            # --- Duration Check ---
            duration = end - start
            if duration < min_duration_threshold:
                 # print(f"Filtering short segment ({duration:.2f}s): {text}") # Debug
                 continue # Skip short segments

            # --- Content Pattern Check ---
            is_repetitive = False
            for pattern in compiled_patterns:
                if pattern.match(text):
                    # print(f"Filtering repetitive segment: {text}") # Debug
                    is_repetitive = True; break # Stop checking patterns if one matches

            # --- Keep Segment if Not Filtered ---
            if not is_repetitive:
                filtered_segments.append(segment)

        # --- Log Filtering Results and Update ---
        original_count = len(self.final_output_segments)
        filtered_count = len(filtered_segments)
        if original_count > filtered_count:
             self._update_status(f"Filtered out {original_count - filtered_count} repetitive or short segments.")
        else:
             self._update_status("No segments were filtered.") # Info message if nothing changed

        self.final_output_segments = filtered_segments # Replace original list with filtered one


    def process_audio_file(self, file_path_str: str):
        """Main method to process a single audio file through the entire pipeline."""
        start_process_time = time.time() # Track total processing time
        self._update_status(f"Starting processing for: {Path(file_path_str).name}")
        self._update_progress(0) # Initialize progress

        # Reset state for the new file
        self.full_transcript_result = None; self.diarization = None
        self.aligned_transcript = None; self.final_output_segments = None
        self.detected_language = None
        temp_files_to_delete = [] # List to track temporary files for cleanup

        original_path = Path(file_path_str.replace('\\', '/')) # Normalize path separators
        # --- File Existence Check ---
        if not original_path.is_file():
            self._update_status(f"Error: Input file not found -> {original_path}", is_error=True); return False

        # --- Step 0: Audio Format Conversion (if necessary) ---
        processing_path = original_path # Assume original file is usable initially
        needs_conversion = original_path.suffix.lower() not in ['.wav', '.flac']
        # More robust check could involve ffprobe to verify codec/sample rate/channels
        if needs_conversion:
            self._update_status("Input format requires conversion to FLAC (16kHz, mono)...")
            if not PYDUB_READY:
                self._update_status("Error: pydub/ffmpeg not available, cannot convert audio.", is_error=True); return False
            # Perform conversion
            converted_flac_path = convert_to_mono_flac(original_path, original_path.parent)
            if converted_flac_path and converted_flac_path.is_file():
                processing_path = converted_flac_path # Use the converted file path
                temp_files_to_delete.append(converted_flac_path) # Add to cleanup list
                self._update_status("Audio format conversion successful.")
            else:
                self._update_status("Error: Audio conversion failed.", is_error=True)
                return False # Stop if conversion fails
        else:
             self._update_status("Input audio format (WAV/FLAC) is suitable, skipping conversion.")
             # Optional Warning: Check if it's actually 16kHz mono?
             # self._update_status("Warning: If file is not 16kHz mono, accuracy may be affected.")
        self._update_progress(10) # Update progress after conversion step

        # --- Main Processing Pipeline ---
        success = False # Flag to track overall success
        try:
            # --- Step 1: Speaker Diarization ---
            self._update_status("Performing speaker diarization...")
            diarize_ok = self._diarize_audio(processing_path)
            # Note: diarize_ok=False is handled internally by setting self.diarization=[]
            # The process continues, but speakers will be UNKNOWN.
            self._update_progress(30)

            # --- Step 2: Transcription ---
            self._update_status("Performing speech transcription (using local Whisper)...")
            transcribe_ok = self._transcribe_entire_audio(processing_path)
            if not transcribe_ok: # Transcription failure is critical
                self._update_status("Error: Speech transcription step failed.", is_error=True); return False
            self._update_progress(70)

            # --- Step 3: Alignment ---
            self._update_status("Aligning transcription and speaker data...")
            align_ok = self._align_full_results()
            # Alignment failure might be recoverable if fallback text exists
            if not align_ok and not self.aligned_transcript:
                 self._update_status("Error: Alignment failed critically.", is_error=True); return False
            elif not align_ok: # Warning if alignment failed but fallback might exist
                 self._update_status("Warning: Alignment produced limited results (e.g., missing timestamps).", is_error=True)
            self._update_progress(80)

            # --- Step 4: Segmentation / Merging ---
            self._segment_output_by_speaker_and_time()
            self._update_progress(85)

            # --- Step 5: Filtering ---
            self._filter_repetitive_segments()
            self._update_progress(90)

            # --- Step 6: Translation (Optional) ---
            if self.translate_enabled and self.final_output_segments:
                self._update_status("Performing translation (using OpenAI API)...")
                # Check if translation is necessary based on detected language
                if self.detected_language and self.detected_language.lower().startswith('zh'):
                     self._update_status(f"Detected language is Chinese ('{self.detected_language}'), skipping translation.")
                elif not client: # Check if translation client is available
                     self._update_status("Translation skipped: OpenAI client not available.", is_error=True)
                else:
                    # Perform translation for each segment
                    translations = []; start_transl_time = time.time()
                    for entry in self.final_output_segments:
                        translation_result = self._translate_text(entry['text'])
                        translations.append(translation_result) # Append result (could be None)
                    end_transl_time = time.time();
                    self._update_status(f"Translation processing finished. Duration: {end_transl_time - start_transl_time:.2f}s")
                    # Add translation results back to the segments
                    for i, entry in enumerate(self.final_output_segments):
                        entry['translation'] = translations[i] # Assign result
                self._update_progress(95)
            else:
                 # Skip translation or no segments to translate
                 self._update_progress(95) # Still mark progress

            # --- Finalization ---
            if not self.final_output_segments:
                 # This might happen if filtering removed everything
                 self._update_status("Processing completed, but the final result contains no segments.", is_error=True) # Consider this a warning/error?
            end_process_time = time.time()
            self._update_status(f"Processing finished successfully for: {original_path.name}. Total time: {end_process_time - start_process_time:.2f}s")
            success = True # Mark overall success

        except Exception as e:
            # Catch any unexpected errors in the pipeline
            self._update_status(f"An critical unexpected error occurred during processing: {e}", is_error=True); traceback.print_exc();
            success = False
        finally:
            # --- Cleanup Temporary Files ---
            if temp_files_to_delete:
                # print("Cleaning up temporary audio files...") # Debug
                for f_path in temp_files_to_delete:
                    if f_path and f_path.exists():
                        try: os.remove(f_path)
                        except Exception as e_del: self._update_status(f"Warning: Failed to delete temporary file '{f_path}': {e_del}", is_error=True)
                # print("Cleanup complete.")

        self._update_progress(100); # Ensure progress reaches 100%
        return success # Return the overall success status

    def save_to_pdf(self, output_filename: str | None = None):
        """Saves the final processed segments to a PDF file."""
        if not self.final_output_segments:
            self._update_status("Cannot save PDF: No final results available.", is_error=True); return

        # Determine save directory and filename
        save_dir = Path(OUTPUT_DIR) # Use default output directory
        if output_filename:
            # Use filename provided by caller (e.g., GUI)
            final_filename = Path(output_filename)
            save_dir = final_filename.parent # Extract directory from provided path
        else:
            # Generate a default filename if none is provided
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Consider basing filename on original audio file if known? (Requires storing it)
            final_filename = save_dir / f"transcript_{timestamp}.pdf"

        # Ensure the target directory exists
        try: save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             self._update_status(f"Error: Failed to create output directory '{save_dir}': {e}", is_error=True); return

        self._update_status(f"Generating PDF file: {final_filename.name}...")
        try:
            pdf = PDF(); # Create PDF instance (initializes fonts)
            # Check if essential fonts were loaded
            if not pdf.fonts_loaded['regular']:
                 self._update_status("Error: Cannot generate PDF because required fonts failed to load.", is_error=True); return
            # Setup PDF document properties
            pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()

            # Add each processed segment to the PDF
            for entry in self.final_output_segments:
                pdf.add_aligned_entry(
                    format_timestamp(entry.get('start')), # Format start time
                    format_timestamp(entry.get('end')),   # Format end time
                    entry.get('speaker', 'UNKNOWN'),      # Get speaker (default UNKNOWN)
                    entry.get('text', ''),                # Get text (default empty)
                    entry.get('translation')              # Get translation (can be None)
                )

            # Save the PDF file
            pdf.output(str(final_filename))
            self._update_status(f"PDF file saved successfully to: {final_filename}")
        except Exception as e:
            # Catch errors during PDF generation/saving
            self._update_status(f"A critical error occurred while saving the PDF: {e}", is_error=True); traceback.print_exc()

    # *** NEW METHOD: save_to_txt ***
    def save_to_txt(self, output_filename: str | None = None):
        """Saves the final processed segments to a plain text (TXT) file."""
        if not self.final_output_segments:
            self._update_status("Cannot save TXT: No final results available.", is_error=True); return

        # Determine save directory and filename (similar logic to PDF)
        save_dir = Path(OUTPUT_DIR)
        if output_filename:
            final_filename = Path(output_filename)
            save_dir = final_filename.parent
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_filename = save_dir / f"transcript_{timestamp}.txt" # .txt extension

        # Ensure the target directory exists
        try: save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             self._update_status(f"Error: Failed to create output directory '{save_dir}': {e}", is_error=True); return

        self._update_status(f"Generating TXT file: {final_filename.name}...")
        try:
            # Open the file in write mode with UTF-8 encoding
            with open(final_filename, 'w', encoding='utf-8') as f:
                # Write a simple header (optional)
                f.write(f"Audio Transcription & Translation\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")

                # Iterate through final segments and write formatted text
                for entry in self.final_output_segments:
                    start_str = format_timestamp(entry.get('start'))
                    end_str = format_timestamp(entry.get('end'))
                    speaker = entry.get('speaker', 'UNKNOWN')
                    text = entry.get('text', '')
                    translation = entry.get('translation') # Will be None if not available/enabled

                    # Write the main entry line
                    f.write(f"[{start_str} --> {end_str}] {speaker}:\n")
                    # Write the transcribed text, indented
                    f.write(f"  {text}\n")
                    # Write the translation line only if translation exists
                    if translation:
                        f.write(f"    Translation: {translation}\n")
                    # Add a separator line between entries
                    f.write("-" * 60 + "\n\n")

            self._update_status(f"TXT file saved successfully to: {final_filename}")
        except Exception as e:
            # Catch errors during file writing
            self._update_status(f"An error occurred while saving the TXT file: {e}", is_error=True); traceback.print_exc()


# --- Command-line Test Entry Point (Not used by GUI but good for testing) ---
# Example: python backend.py your_audio.mp3 --translate
# (Requires adding argument parsing logic if run directly)
# async def main():
#     # Add argparse here for command-line execution
#     pass
# if __name__ == "__main__":
#     # asyncio.run(main()) # If using async main
#     pass