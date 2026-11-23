import streamlit as st
from PIL import Image
import io
import pandas as pd
import re
import hashlib

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

# --- Model list -------------------------------------------------------------
# The list is fetched live from the Gemini API (cached for 24 h), so new models
# show up automatically. Rules:
#  * Any model with "flash" in its name (Flash and Flash-Lite tiers, the cheap
#    ones) is included automatically.
#  * Expensive models (output >= $10 per 1M tokens: all Pro models and Gemini 4
#    Argon/Carbon/Neon) are excluded. Because other new model families can't be
#    priced from the API, they are only shown if listed in EXTRA_MODELS.
#  * Models that cannot transcribe images (TTS, image generation, live/audio,
#    Omni, etc.) are filtered out.
EXTRA_MODELS = {"gemini-4-hydrogen"}  # non-Flash models checked to be affordable ($1.50/$7.50)
EXCLUDE_KEYWORDS = ("tts", "image", "live", "audio", "omni", "embedding",
                    "robotics", "computer-use", "customtools", "transcribe")

# Used if the live model list cannot be fetched.
FALLBACK_MODELS = [
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


def _model_sort_key(name):
    """Newest version first; Flash before Flash-Lite; -latest aliases last."""
    m = re.match(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else -1.0
    return (-version, "lite" in name, "preview" in name, name)


def _is_allowed(name):
    if not name.startswith("gemini-"):
        return False
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return "flash" in name or name in EXTRA_MODELS


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_available_models(_api_key, key_id):
    """Fetch image-capable, affordable Gemini models from the API."""
    try:
        client = genai.Client(api_key=_api_key)
        names = [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        models = sorted({n for n in names if _is_allowed(n)}, key=_model_sort_key)
        return models or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


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
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
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

if API_KEY_READY:
    GEMINI_MODELS = get_available_models(
        GEMINI_API_KEY, hashlib.sha256(GEMINI_API_KEY.encode()).hexdigest()
    )
else:
    GEMINI_MODELS = FALLBACK_MODELS

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
    current = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "Choose Gemini model:",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(current) if current in GEMINI_MODELS else None,
        placeholder="Select a model..."
    )
    st.caption("The list updates automatically. The most expensive models (Pro, Gemini 4 Argon/Carbon/Neon) are excluded.")

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
MODEL_READY = st.session_state.selected_model is not None
if MODEL_READY:
    st.info(f"🎯 Model: **{st.session_state.selected_model}**")
else:
    st.warning("Select a Gemini model in the sidebar to start transcribing.")
CAN_RUN = API_KEY_READY and MODEL_READY

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

        if st.button("Start Transcription", disabled=not CAN_RUN):
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

    if uploaded_files and st.button("Start Bulk Transcription", disabled=not CAN_RUN):
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
