# Handwritten Manuscript Transcription Assistant

A Streamlit application that uses Google Gemini to transcribe handwritten documents with iterative training.

## Features

### 📚 Training Mode
- Upload handwritten manuscripts
- Get AI transcription from Gemini
- Provide feedback with correct transcription
- AI reflects and learns from mistakes
- Save and load training sessions as JSON

### ⚡ Direct Transcription
- **Single page**: Transcribe one page at a time
- **Bulk transcription**: Process multiple pages simultaneously
- Uses all previous training for better results
- Export results as CSV

### 🎯 Model Selection
Choose from four Gemini models:
- `gemini-3-pro-preview` - Most powerful
- `gemini-3-flash-preview` - Fast performance
- `gemini-2.5-flash` - Balanced
- `gemini-2.5-pro` - Optimized precision

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

### Training Mode

1. Select "Training Mode" in the sidebar
2. Upload an image of a handwritten manuscript
3. Click "Get AI Transcription"
4. Review the result and enter the correct transcription
5. AI reflects and learns
6. Repeat to improve accuracy

**Tip**: Save your training session regularly via "Download Training History"

### Direct Transcription

1. Train the app first (recommended)
2. Select "Direct Transcription" in the sidebar
3. Choose between single page or bulk transcription
4. Upload image(s)
5. Click "Start Direct Transcription"
6. Export results if bulk

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
