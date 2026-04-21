import streamlit as st
import os
import json
import time
from PIL import Image
import io
import base64
from datetime import datetime
import mimetypes
from collections import Counter

# Import Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google Gemini SDK not installed. Run: pip install google-genai")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Manuscript Transcription Assistant",
    page_icon="📜",
    layout="wide"
)

# Available Gemini models
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

def compute_bow_f1(ai_text, correct_text):
    """Word-level F1: how much of the correct text the AI captured, regardless of order."""
    ai_words = Counter(ai_text.lower().split())
    correct_words = Counter(correct_text.lower().split())
    common = sum((ai_words & correct_words).values())
    if common == 0:
        return 0.0
    precision = common / sum(ai_words.values())
    recall = common / sum(correct_words.values())
    return round(2 * precision * recall / (precision + recall), 3)

def compute_wer(ai_text, correct_text):
    """Word Error Rate: fraction of words that are wrong (insertions + deletions + substitutions)."""
    ref = correct_text.lower().split()
    hyp = ai_text.lower().split()
    if len(ref) == 0:
        return 0.0
    # Dynamic programming edit distance on word level
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return round(d[len(ref)][len(hyp)] / len(ref), 3)

# Initialize Gemini client
@st.cache_resource
def get_gemini_client(_api_key):
    """Initialize Gemini client with provided API key"""
    if not GEMINI_AVAILABLE:
        raise ValueError("Google Gemini SDK not installed.")
    return genai.Client(api_key=_api_key)

# Convert PIL image to bytes for Gemini API
def image_to_bytes(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_workflow_stage" not in st.session_state:
    st.session_state.current_workflow_stage = "upload"
if "current_iteration" not in st.session_state:
    st.session_state.current_iteration = 0
if "default_prompt" not in st.session_state:
    st.session_state.default_prompt = "Transcribe the handwritten text in the image as accurately as possible. Read line by line, word by word. Return only the transcription, nothing else!"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "training"  # "training" or "direct"
if "direct_mode_type" not in st.session_state:
    st.session_state.direct_mode_type = "Single page"  # "Single page" or "Bulk transcription (multiple pages)"
if "training_metadata" not in st.session_state:
    st.session_state.training_metadata = {
        "name": "Unnamed training session",
        "description": "No description",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": 0
    }
if "selected_model" not in st.session_state:
    st.session_state.selected_model = GEMINI_MODELS[0]  # Default to first model
if "quality_scores" not in st.session_state:
    st.session_state.quality_scores = []  # list of {"iteration": n, "bow_f1": x, "wer": x}

# Get API key from secrets or user input
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    GEMINI_API_KEY = None

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.session_state.user_api_key or None

API_KEY_READY = bool(GEMINI_API_KEY)

# Function to save training history to JSON
def save_training_history():
    data = {
        "conversation_history": st.session_state.conversation_history,
        "metadata": st.session_state.training_metadata
    }
    return json.dumps(data, ensure_ascii=False)

# Function to load training history from JSON
def load_training_history(json_string):
    try:
        data = json.loads(json_string)
        st.session_state.conversation_history = data["conversation_history"]
        st.session_state.training_metadata = data["metadata"]
        return True
    except Exception as e:
        st.error(f"Error loading training history: {str(e)}")
        return False

# Function to process transcription with Gemini
def process_transcription_gemini(image, prompt, update_history=True):
    client = get_gemini_client(GEMINI_API_KEY)
    image_bytes = image_to_bytes(image)
    mime_type = "image/png"
    
    # Build content - simplified approach for first message
    if len(st.session_state.conversation_history) == 0:
        # Simple first message without history
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        ]
    else:
        # Build content from conversation history
        contents = []
        
        # Add all previous conversation history
        for msg in st.session_state.conversation_history:
            role = msg["role"]
            if role == "assistant":
                role = "model"  # Gemini uses "model" instead of "assistant"
            
            # Handle message content
            if isinstance(msg["content"], str):
                # Simple text message
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )
            elif isinstance(msg["content"], list):
                # Complex message with image and/or text
                parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        parts.append(types.Part.from_text(text=part["text"]))
                    elif part["type"] == "image":
                        # We need to reconstruct the image from base64 if it's stored that way
                        # For now, we'll skip images in history as they're already processed
                        # Gemini should remember from the conversation context
                        pass
                
                if parts:  # Only add if we have parts
                    contents.append(
                        types.Content(
                            role=role,
                            parts=parts
                        )
                    )
        
        # Add current message with image
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        )
    
    # Call Gemini API with selected model
    response = client.models.generate_content(
        model=st.session_state.selected_model,
        contents=contents
    )
    
    transcription = response.text
    
    # If we should update the history (training mode), add the exchange to conversation history
    if update_history:
        # Store user message
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("utf-8")
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        st.session_state.conversation_history.append(user_message)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": transcription
        })
    
    return transcription

# Wrapper function for consistency
def process_transcription(image, prompt, update_history=True):
    return process_transcription_gemini(image, prompt, update_history)

# Header
st.title("📜 Handwritten Manuscript Transcription Assistant")
st.markdown("Use Gemini to transcribe handwritten documents with iterative training.")

if not API_KEY_READY:
    st.info("To get started, enter your Gemini API key in the sidebar. You can get one for free at [Google AI Studio](https://aistudio.google.com/app/apikey).", icon="🔑")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    st.session_state.selected_model = st.selectbox(
        "Choose Gemini model:",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(st.session_state.selected_model)
    )
    
    # API key status
    if API_KEY_READY:
        st.success("✓ API key ready")
    else:
        st.warning("No API key configured.")
        user_key = st.text_input(
            "Enter your Gemini API key:",
            type="password",
            placeholder="AIza..."
        )
        if user_key:
            st.session_state.user_api_key = user_key
            GEMINI_API_KEY = user_key
            API_KEY_READY = True
            st.rerun()
    
    st.divider()
    
    # Mode selection
    st.subheader("Work Mode")
    mode = st.radio(
        "Select mode:",
        ["Training Mode", "Direct Transcription"],
        index=0 if st.session_state.app_mode == "training" else 1
    )
    
    if mode == "Training Mode":
        st.session_state.app_mode = "training"
    else:
        st.session_state.app_mode = "direct"
        # Show direct mode type selector
        st.session_state.direct_mode_type = st.radio(
            "Type of direct transcription:",
            ["Single page", "Bulk transcription (multiple pages)"]
        )
    
    st.divider()
    
    # Training session metadata (only in training mode)
    if st.session_state.app_mode == "training":
        st.subheader("Session Information")
        st.session_state.training_metadata["name"] = st.text_input(
            "Session name:",
            value=st.session_state.training_metadata["name"]
        )
        st.session_state.training_metadata["description"] = st.text_area(
            "Description:",
            value=st.session_state.training_metadata["description"],
            height=100
        )
        
        st.divider()
    
    # Save and load training history
    if st.session_state.app_mode == "training":
        st.subheader("Manage Training History")
        
        if st.button("💾 Download Training History"):
            history_json = save_training_history()
            st.download_button(
                label="Download JSON",
                data=history_json,
                file_name=f"training_{st.session_state.training_metadata['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        uploaded_history = st.file_uploader(
            "Upload training history (JSON)", 
            type=["json"]
        )
        
        if uploaded_history is not None:
            history_content = uploaded_history.read().decode("utf-8")
            if st.button("Load Training History"):
                if load_training_history(history_content):
                    st.success("Training history loaded!")
                    st.rerun()
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Training Session", type="secondary"):
            st.session_state.conversation_history = []
            st.session_state.current_workflow_stage = "upload"
            st.session_state.current_iteration = 0
            st.session_state.training_metadata["iterations"] = 0
            st.session_state.quality_scores = []
            st.success("Session reset!")
            st.rerun()

# Main content area
if st.session_state.app_mode == "training":
    # Training mode workflow
    st.header("Training Mode")
    st.info(f"🎯 Model: **{st.session_state.selected_model}** | 📊 Iteration: {st.session_state.current_iteration}")
    
    # Stage 1: Upload image
    if st.session_state.current_workflow_stage == "upload":
        st.subheader("Step 1: Upload Manuscript Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a handwritten manuscript", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)
            
            # Store in session state
            st.session_state.training_image = image
            
            # Show the prompt input field
            training_prompt = st.text_area(
                "Prompt for Gemini:", 
                value=st.session_state.default_prompt,
                height=150
            )
            
            # Button to get first transcription
            if st.button("Get AI Transcription", disabled=not API_KEY_READY):
                with st.spinner("Gemini is transcribing the manuscript..."):
                    try:
                        # Get transcription from AI
                        transcription = process_transcription(
                            st.session_state.training_image, 
                            training_prompt
                        )
                        
                        # Store the transcription
                        st.session_state.ai_transcription = transcription
                        
                        # Move to next stage
                        st.session_state.current_workflow_stage = "review"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
    
    # Stage 2: Review transcription and provide feedback
    elif st.session_state.current_workflow_stage == "review":
        st.subheader("Step 2: Review and Correct the Transcription")
        
        # Show the image again
        if "training_image" in st.session_state:
            st.image(st.session_state.training_image, caption="Manuscript", use_column_width=True)
        
        # Show AI's transcription
        st.write("**Gemini's transcription:**")
        st.text_area(
            "AI transcription",
            value=st.session_state.ai_transcription,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
        
        # Input for correct transcription
        st.write("**Enter the correct transcription:**")
        correct_transcription = st.text_area(
            "Correct transcription",
            height=200,
            placeholder="Enter the correct transcription here...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✓ Submit Feedback and Continue Training", disabled=not API_KEY_READY):
                if correct_transcription.strip():
                    with st.spinner("Gemini is reflecting on the feedback..."):
                        try:
                            # Compute quality scores before storing feedback
                            bow = compute_bow_f1(st.session_state.ai_transcription, correct_transcription)
                            wer = compute_wer(st.session_state.ai_transcription, correct_transcription)

                            # Create feedback message
                            feedback_prompt = f"Here is the correct transcription:\n\n{correct_transcription}\n\nCompare this with your previous transcription and explain what needed to be corrected and why. Be specific about what mistakes you made."

                            # Get AI's reflection
                            reflection = process_transcription(
                                st.session_state.training_image,
                                feedback_prompt
                            )

                            # Update iteration count
                            st.session_state.current_iteration += 1
                            st.session_state.training_metadata["iterations"] = st.session_state.current_iteration

                            # Store quality scores
                            st.session_state.quality_scores.append({
                                "iteration": st.session_state.current_iteration,
                                "bow_f1": bow,
                                "wer": wer
                            })

                            # Go back to upload stage for next iteration
                            st.session_state.current_workflow_stage = "upload"

                            # Clear the stored transcription
                            if "ai_transcription" in st.session_state:
                                del st.session_state.ai_transcription

                            st.success(f"Training iteration {st.session_state.current_iteration} completed!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"An error occurred during reflection: {str(e)}")
                else:
                    st.warning("Please enter the correct transcription.")
        
        with col2:
            if st.button("← Back to Upload"):
                st.session_state.current_workflow_stage = "upload"
                st.rerun()

else:  # Direct mode
    st.header("Direct Transcription")
    st.info(f"🎯 Model: **{st.session_state.selected_model}** | 📚 Training iterations: {st.session_state.current_iteration}")
    
    if st.session_state.direct_mode_type == "Single page":
        st.subheader("Transcribe Single Manuscript")
        
        # File uploader for direct mode
        uploaded_file = st.file_uploader(
            "Choose an image of a handwritten manuscript", 
            type=["jpg", "jpeg", "png"],
            key="direct_upload"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)
            
            # Store in session state
            st.session_state.direct_mode_image = image
            
            # Show the prompt input field
            direct_prompt = st.text_area(
                "Prompt for Gemini:", 
                value=st.session_state.default_prompt,
                height=150
            )
            
            # Button to start direct transcription
            if st.button("Start Direct Transcription", disabled=not API_KEY_READY):
                with st.spinner("Gemini is transcribing the manuscript..."):
                    try:
                        # Get transcription from AI using all previous training
                        # Pass update_history=False to prevent adding to conversation history in direct mode
                        transcription = process_transcription(
                            st.session_state.direct_mode_image, 
                            direct_prompt,
                            update_history=False
                        )
                        
                        # Display the result
                        st.session_state.direct_transcription = transcription
                        
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
            
            # Show the direct transcription result if available
            if "direct_transcription" in st.session_state:
                st.subheader("Transcription Result")
                st.text_area(
                    "Gemini's transcription:", 
                    value=st.session_state.direct_transcription, 
                    height=300
                )
                
                # Option to copy to clipboard
                if st.button("Copy to Clipboard"):
                    st.code(st.session_state.direct_transcription)
                    st.success("Transcription copied to clipboard!")
                
                # Button to clear for next transcription
                if st.button("Clear and Transcribe a New Image"):
                    if "direct_transcription" in st.session_state:
                        del st.session_state.direct_transcription
                    if "direct_mode_image" in st.session_state:
                        del st.session_state.direct_mode_image
                    st.rerun()
        else:
            st.info("Upload an image to start transcribing.")
    
    else:  # Bulk transcription mode
        st.subheader("Bulk Transcription of Multiple Manuscripts")
        
        # Initialize session state for bulk transcription if not already done
        if "bulk_transcription_results" not in st.session_state:
            st.session_state.bulk_transcription_results = []
        
        if "bulk_transcription_completed" not in st.session_state:
            st.session_state.bulk_transcription_completed = False
        
        if "bulk_transcription_progress" not in st.session_state:
            st.session_state.bulk_transcription_progress = 0
        
        # Allow uploading multiple files
        uploaded_files = st.file_uploader(
            "Choose multiple images of handwritten manuscripts", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        # Show the prompt input field
        bulk_prompt = st.text_area(
            "Prompt for Gemini (used for all images):", 
            value=st.session_state.default_prompt,
            height=150
        )
        
        # Button to start bulk transcription
        if uploaded_files and st.button("Start Bulk Transcription", disabled=not API_KEY_READY):
            # Reset results
            st.session_state.bulk_transcription_results = []
            st.session_state.bulk_transcription_completed = False
            st.session_state.bulk_transcription_progress = 0
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_percent = int(100 * i / len(uploaded_files))
                progress_bar.progress(progress_percent)
                status_text.text(f"Transcribing file {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    # Process the image
                    image = Image.open(uploaded_file)
                    
                    # Get transcription from AI
                    transcription = process_transcription(
                        image, 
                        bulk_prompt,
                        update_history=False
                    )
                    
                    # Store the result
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": transcription
                    })
                    
                except Exception as e:
                    # Store the error
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": f"ERROR: {str(e)}"
                    })
            
            # Complete the progress bar
            progress_bar.progress(100)
            status_text.text(f"Transcription complete! {len(uploaded_files)} files processed.")
            
            # Mark as completed
            st.session_state.bulk_transcription_completed = True
            st.rerun()
        
        # If bulk transcription has been completed, show results and download option
        if st.session_state.bulk_transcription_completed and st.session_state.bulk_transcription_results:
            st.subheader("Bulk Transcription Results")
            
            # Create a DataFrame from the results
            import pandas as pd
            results_df = pd.DataFrame(st.session_state.bulk_transcription_results)
            
            # Display the results in a table
            st.dataframe(results_df)
            
            # Create CSV for download
            csv = results_df.to_csv(index=False)
            
            # Create a download button
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="transcription_results.csv",
                mime="text/csv"
            )
            
            # Option to clear results
            if st.button("Clear Results and Transcribe New Files"):
                st.session_state.bulk_transcription_results = []
                st.session_state.bulk_transcription_completed = False
                st.rerun()

# Show quality score trend
if st.session_state.quality_scores:
    with st.expander("📊 Transcription Quality Over Iterations", expanded=True):
        st.caption(
            "**BoW F1** (Bag of Words F1, 0–1): measures word overlap between AI and correct text, ignoring order. "
            "Higher is better. "
            "**WER** (Word Error Rate, 0–1+): fraction of words that were wrong. "
            "Lower is better."
        )
        import pandas as pd
        scores_df = pd.DataFrame(st.session_state.quality_scores).set_index("iteration")
        scores_df.columns = ["BoW F1 (higher=better)", "WER (lower=better)"]
        st.line_chart(scores_df)
        st.dataframe(scores_df)

# Show training history
with st.expander("View Training History"):
    if len(st.session_state.conversation_history) > 0:
        iteration = 1
        message_index = 0
        
        while message_index < len(st.session_state.conversation_history):
            # Try to find a complete iteration (4 messages)
            if message_index + 3 < len(st.session_state.conversation_history):
                # Get the messages
                image_msg = st.session_state.conversation_history[message_index]
                transcription_msg = st.session_state.conversation_history[message_index + 1]
                
                # Check if this is a training iteration with feedback
                if message_index + 3 < len(st.session_state.conversation_history) and "Here is the correct transcription" in st.session_state.conversation_history[message_index + 2]["content"]:
                    # This is a training iteration with feedback
                    feedback_msg = st.session_state.conversation_history[message_index + 2]
                    reflection_msg = st.session_state.conversation_history[message_index + 3]
                    
                    st.write(f"### Training Iteration {iteration}")
                    
                    # Display transcription
                    st.write("**AI's transcription:**")
                    st.write(transcription_msg["content"])
                    
                    # Display reflection
                    st.write("**AI's reflection:**")
                    st.write(reflection_msg["content"])
                    
                    st.divider()
                    
                    # Move to next iteration
                    iteration += 1
                    message_index += 4
                else:
                    # Skip to next message - we shouldn't have direct mode messages in history anymore
                    message_index += 1
            else:
                # Handle remaining messages
                st.write("*Incomplete training iteration*")
                break
    else:
        st.write("No training history yet.")
