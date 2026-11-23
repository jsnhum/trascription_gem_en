# Handwritten Manuscript Transcription Assistant

A Streamlit application that uses Google Gemini to transcribe handwritten documents.

## Features

- **Single page**: Transcribe one page at a time
- **Bulk transcription**: Process multiple pages in one go and export results as CSV
- **Editable prompt**: Adjust the instructions sent to Gemini
- **Model selection**: No default model; choose one before transcribing.
  The list is fetched live from the Gemini API (refreshed every 24 h), so new
  Flash and Flash-Lite models appear automatically. The most expensive models
  (output >= $10 per 1M tokens: all Pro models and Gemini 4 Argon/Carbon/Neon)
  are excluded. Other new model families must be added to `EXTRA_MODELS` in the
  code after checking their price.

## Installation

### Requirements
- Python 3.8 or later
- Google Gemini API key

### Quick Start

1. **Clone or download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

4. **Run the app**
   ```bash
   streamlit run transcription_app_english.py
   ```

5. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`

## Usage

1. Choose a model in the sidebar
2. Choose "Single page" or "Bulk transcription (multiple pages)"
3. Upload image(s) and adjust the prompt if needed
4. Click "Start Transcription" / "Start Bulk Transcription"
5. For bulk, download the results as CSV

## Deployment

See [DEPLOYMENT_EN.md](DEPLOYMENT_EN.md) for complete instructions on:
- Streamlit Cloud deployment
- Docker deployment
- Heroku deployment
- Troubleshooting

## File Structure

```
project/
├── transcription_app_english.py  # Main application
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── secrets.toml             # API key (not in git!)
├── .gitignore                   # Git ignore rules
└── README_EN.md                 # This file
```

## Security

⚠️ **IMPORTANT**: NEVER expose your API key!
- Always add `.streamlit/secrets.toml` to `.gitignore`
- Use Streamlit Cloud secrets for deployment
- Never share API keys publicly

## Tech Stack

- **Frontend/Backend**: Streamlit
- **AI Model**: Google Gemini (multiple variants)
- **Image Handling**: Pillow (PIL)
- **Data Export**: Pandas

## License

[Your desired license here]

## Support

For questions or issues, [open an issue](link-to-issues) or contact [your-contact-info].
