import streamlit as st
from PIL import Image
import io
import pandas as pd

# Import Gemini SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Google Gemini SDK not installed. Run: pip install google-genai")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Manuscript Transcription Assistant",
    page_icon="📜",
    layout="wide"
)

# Gemini models that accept image input via generate_content.
# Verified against the Gemini API model list (Nov 2026). Excluded: TTS, image
# generation, embedding, live/audio, music, agent, robotics and
# Interactions-API-only models, which cannot be used for image transcription here.
GEMINI_MODELS = [
    # Gemini 4
    "gemini-4-argon",
    "gemini-4-carbon",
    "gemini-4-hydrogen",
    "gemini-4-neon",
    # Gemini 3.x
    "gemini-3.1-pro-preview",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    # Gemini 2.5
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # Aliases that always point to Google's current model in each tier
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
DEFAULT_MODEL = "gemini-3.1-pro-preview"

DEFAULT_PROMPT = (
    "Transcribe the handwritten text in the image as accurately as possible. "
    "Read line by line, word by word. Return only the transcription, nothing else!"
)


@st.cache_resource
def get_gemini_client(_api_key):
    """Initialize Gemini client with provided API key"""
    return genai.Client(api_key=_api_key)


def image_to_bytes(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()


def process_transcription(image, prompt):
    """Send one image + prompt to Gemini and return the transcription."""
    client = get_gemini_client(GEMINI_API_KEY)
    response = client.models.generate_content(
        model=st.session_state.selected_model,
        contents=[
            types.Part.from_bytes(data=image_to_bytes(image), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
    )
    return response.text


# Initialize session state variables
if "selected_model" not in st.session_state or st.session_state.selected_model not in GEMINI_MODELS:
    st.session_state.selected_model = DEFAULT_MODEL
if "mode_type" not in st.session_state:
    st.session_state.mode_type = "Single page"
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# Get API key from secrets or user input
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.session_state.user_api_key or None

API_KEY_READY = bool(GEMINI_API_KEY)

# Header
st.title("📜 Handwritten Manuscript Transcription Assistant")
st.markdown("Use Gemini to transcribe handwritten documents.")

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
            st.rerun()

    st.divider()

    # Mode selection
    st.subheader("Work Mode")
    st.session_state.mode_type = st.radio(
        "Type of transcription:",
        ["Single page", "Bulk transcription (multiple pages)"],
        index=0 if st.session_state.mode_type == "Single page" else 1
    )

# Main content area
st.info(f"🎯 Model: **{st.session_state.selected_model}**")

if st.session_state.mode_type == "Single page":
    st.subheader("Transcribe Single Manuscript")

    uploaded_file = st.file_uploader(
        "Choose an image of a handwritten manuscript",
        type=["jpg", "jpeg", "png"],
        key="single_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        prompt = st.text_area(
            "Prompt for Gemini:",
            value=DEFAULT_PROMPT,
            height=150
        )

        if st.button("Start Transcription", disabled=not API_KEY_READY):
            with st.spinner("Gemini is transcribing the manuscript..."):
                try:
                    st.session_state.single_transcription = process_transcription(image, prompt)
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")

        if "single_transcription" in st.session_state:
            st.subheader("Transcription Result")
            st.text_area(
                "Gemini's transcription:",
                value=st.session_state.single_transcription,
                height=300
            )
            st.caption("Copy the text with the copy icon in the top-right corner below:")
            st.code(st.session_state.single_transcription, language=None)

            if st.button("Clear and Transcribe a New Image"):
                del st.session_state.single_transcription
                st.rerun()
    else:
        st.info("Upload an image to start transcribing.")

else:  # Bulk transcription mode
    st.subheader("Bulk Transcription of Multiple Manuscripts")

    if "bulk_transcription_results" not in st.session_state:
        st.session_state.bulk_transcription_results = []

    uploaded_files = st.file_uploader(
        "Choose multiple images of handwritten manuscripts",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    bulk_prompt = st.text_area(
        "Prompt for Gemini (used for all images):",
        value=DEFAULT_PROMPT,
        height=150
    )

    if uploaded_files and st.button("Start Bulk Transcription", disabled=not API_KEY_READY):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(int(100 * i / len(uploaded_files)))
            status_text.text(f"Transcribing file {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
            try:
                transcription = process_transcription(Image.open(uploaded_file), bulk_prompt)
            except Exception as e:
                transcription = f"ERROR: {str(e)}"
            results.append({"filename": uploaded_file.name, "transcription": transcription})

        progress_bar.progress(100)
        status_text.text(f"Transcription complete! {len(uploaded_files)} files processed.")
        st.session_state.bulk_transcription_results = results
        st.rerun()

    if st.session_state.bulk_transcription_results:
        st.subheader("Bulk Transcription Results")
        results_df = pd.DataFrame(st.session_state.bulk_transcription_results)
        st.dataframe(results_df)

        st.download_button(
            label="Download Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="transcription_results.csv",
            mime="text/csv"
        )

        if st.button("Clear Results and Transcribe New Files"):
            st.session_state.bulk_transcription_results = []
            st.rerun()
