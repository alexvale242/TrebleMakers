# Music Transcription to Score Prototype

This prototype converts MP3 audio files into musical scores using AI-powered transcription.

## Features

- 🎵 Audio transcription to MIDI using Basic Pitch
- 📝 Note sequence extraction
- 🎼 Piano roll visualization
- 📊 Basic music analysis
- 💾 Download MIDI and score images

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `twinkle.mp3` file in the same directory as `test.py`

## Usage

Run the Streamlit app:
```bash
streamlit run test.py
```

The app will automatically process the `twinkle.mp3` file and display:
- Note sequence transcription
- Music analysis (note count, most common notes)
- Piano roll visualization
- Download options for MIDI and score images

You can also upload your own MP3 files using the file uploader.

## How it works

1. **Audio Transcription**: Uses Basic Pitch to convert audio to MIDI
2. **Note Extraction**: Parses MIDI to extract individual notes with timing
3. **Score Rendering**: Creates a piano roll visualization
4. **Analysis**: Provides basic statistics about the music

## Dependencies

- `basic-pitch`: AI-powered audio transcription
- `pretty-midi`: MIDI file processing
- `librosa`: Audio analysis and visualization
- `matplotlib`: Plotting and image generation
- `streamlit`: Web interface 