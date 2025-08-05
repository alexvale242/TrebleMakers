import streamlit as st
import os
import tempfile
from pathlib import Path
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
import soundfile as sf
import io
from basic_pitch.inference import predict
import basic_pitch
import yt_dlp
import re
import subprocess
import platform

# Fix for missing scipy.signal.gaussian in newer versions
if not hasattr(signal, 'gaussian'):
    def gaussian(M, std, sym=True):
        """Create a Gaussian window."""
        from scipy.signal.windows import gaussian as gaussian_window
        return gaussian_window(M, std, sym=sym)
    signal.gaussian = gaussian

def download_youtube_audio(url):
    """Download audio from YouTube URL and return the file path"""
    try:
        # Configure yt-dlp options for audio extraction
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': '%(title)s.%(ext)s',
            'noplaylist': True,  # Only download the specific video, not the playlist
            'extract_flat': False,  # Don't extract playlist info
            'quiet': True,  # Reduce verbose output
            'no_warnings': True,  # Suppress warnings
            'ignoreerrors': False,  # Don't ignore errors
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'youtube_video')
            
            # Download the audio
            ydl.download([url])
            
            # Find the downloaded file
            downloaded_file = None
            current_files = os.listdir('.')
            
            # Look for mp3 files that match the video title
            for file in current_files:
                if file.endswith('.mp3'):
                    # Check if the video title is in the filename (case insensitive)
                    if video_title.lower().replace(' ', '_').replace('-', '_') in file.lower().replace(' ', '_').replace('-', '_'):
                        downloaded_file = file
                        break
            
            if downloaded_file:
                return downloaded_file
            else:
                # Fallback: find any mp3 file that was recently created
                mp3_files = [f for f in current_files if f.endswith('.mp3')]
                if mp3_files:
                    # Get the most recently modified mp3 file
                    mp3_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    return mp3_files[0]
                else:
                    raise Exception("Could not find downloaded audio file")
                    
    except Exception as e:
        st.error(f"Error downloading YouTube audio: {str(e)}")
        return None

def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL"""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

def clean_youtube_url(url):
    """Clean YouTube URL to extract just the video ID"""
    # Remove playlist parameters and other extra parameters
    url = re.sub(r'&list=[^&]*', '', url)
    url = re.sub(r'&index=[^&]*', '', url)
    url = re.sub(r'&start_radio=[^&]*', '', url)
    url = re.sub(r'&t=[^&]*', '', url)
    return url

def midi_to_audio(midi_path):
    """Convert MIDI file to audio for playback"""
    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Generate audio from MIDI
        audio_data = midi_data.synthesize(fs=22050)
        
        # Save as temporary WAV file
        audio_path = midi_path.replace('.mid', '_audio.wav')
        sf.write(audio_path, audio_data, 22050)
        
        return audio_path
    except Exception as e:
        st.error(f"Error converting MIDI to audio: {str(e)}")
        return None

def transcribe_to_midi(audio_file):
    """Convert audio file to MIDI using basic-pitch library"""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name
    
    try:
        # Transcribe audio to MIDI using basic-pitch
        midi_path = audio_path.replace('.mp3', '.mid')
        
        # Use the predict function which returns model outputs, MIDI data, and notes
        # Try using the ONNX model which should work better
        model_path_str = str(basic_pitch.ICASSP_2022_MODEL_PATH)
        onnx_model_path = model_path_str.replace('/nmp', '/nmp.onnx')
        model_outputs, midi_data, notes = predict(audio_path, onnx_model_path)
        
        # Save the MIDI file
        midi_data.write(midi_path)
        
        # Clean up temporary audio file
        os.unlink(audio_path)
        
        return midi_path
        
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        # Clean up temporary audio file
        try:
            os.unlink(audio_path)
        except:
            pass
        return None

def extract_notes(midi_path):
    """Extract note sequence from MIDI file"""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_name = pretty_midi.note_number_to_name(note.pitch)
                duration = note.end - note.start
                notes.append({
                    'note': note_name,
                    'start': note.start,
                    'end': note.end,
                    'duration': duration,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        
        # Create a simple string representation
        note_sequence = []
        for note in notes:
            note_sequence.append(f"{note['note']}({note['duration']:.2f}s)")
        
        return " ".join(note_sequence)
    
    except Exception as e:
        return f"Error extracting notes: {str(e)}"

def render_score(midi_path):
    """Render MIDI as a simple piano roll visualization"""
    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Create piano roll
        piano_roll = midi_data.get_piano_roll(fs=10)  # 10 Hz resolution
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display piano roll
        librosa.display.specshow(
            piano_roll,
            x_axis='time',
            y_axis='cqt_note',
            ax=ax,
            cmap='viridis'
        )
        
        ax.set_title('Piano Roll Visualization')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Note')
        
        # Save to temporary file
        score_path = midi_path.replace('.mid', '_score.png')
        plt.savefig(score_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return score_path
    
    except Exception as e:
        st.error(f"Error rendering score: {str(e)}")
        return None

def analyze_music(note_sequence):
    """Simple music analysis"""
    analysis = []
    
    # Count notes
    notes = note_sequence.split()
    analysis.append(f"Total notes: {len(notes)}")
    
    # Find common notes
    note_names = [note.split('(')[0] for note in notes]
    from collections import Counter
    most_common = Counter(note_names).most_common(5)
    analysis.append(f"Most common notes: {', '.join([f'{note}({count})' for note, count in most_common])}")
    
    return "\n".join(analysis)

def open_in_musescore(midi_path):
    """Convert MIDI to MuseScore format and open it"""
    try:
        # Create a more sensible output directory
        output_dir = os.path.expanduser("~/Desktop/shoehorners_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the MIDI file to the output directory with a better name
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_midi_path = os.path.join(output_dir, f"transcribed_music_{timestamp}.mid")
        output_mscz_path = os.path.join(output_dir, f"transcribed_music_{timestamp}.mscz")
        
        # Copy the file
        shutil.copy2(midi_path, output_midi_path)
        
        # Convert MIDI to MuseScore format
        if platform.system() == "Windows":
            musescore_exe = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
            if not os.path.exists(musescore_exe):
                musescore_exe = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
            
            if os.path.exists(musescore_exe):
                # Convert MIDI to MSCZ
                result = subprocess.run([musescore_exe, "-o", output_mscz_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the MSCZ file
                    subprocess.Popen([musescore_exe, output_mscz_path])
                else:
                    st.error(f"Error converting MIDI to MSCZ: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on Windows")
                return
                
        elif platform.system() == "Darwin": # macOS
            # Try different possible app names and paths for macOS
            possible_paths = [
                "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
                "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
                "/Applications/MuseScore.app/Contents/MacOS/mscore"
            ]
            
            # Check if any of the possible paths exist
            musescore_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    musescore_path = path
                    break
            
            if musescore_path:
                # Convert MIDI to MSCZ
                result = subprocess.run([musescore_path, "-o", output_mscz_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the MSCZ file
                    subprocess.run(["open", "-a", "/Applications/MuseScore 4.app", "--args", output_mscz_path])
                else:
                    st.error(f"Error converting MIDI to MSCZ: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on macOS")
                return
                
        else: # Linux
            # Try different Linux commands
            musescore_cmds = ["musescore4", "musescore3", "musescore"]
            musescore_cmd = None
            
            for cmd in musescore_cmds:
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, check=True)
                    musescore_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if musescore_cmd:
                # Convert MIDI to MSCZ
                result = subprocess.run([musescore_cmd, "-o", output_mscz_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the MSCZ file
                    subprocess.Popen([musescore_cmd, output_mscz_path])
                else:
                    st.error(f"Error converting MIDI to MSCZ: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on Linux")
                return
        
        st.success(f"Converted and opened {output_mscz_path} in MuseScore.")
        st.info(f"Files saved to: {output_dir}")
        
    except Exception as e:
        st.error(f"Error converting MIDI to MuseScore: {str(e)}")
        st.info("Make sure MuseScore is installed in /Applications/")

def open_pdf_from_midi(midi_path):
    """Convert MIDI to PDF and open it"""
    try:
        # Create a more sensible output directory
        output_dir = os.path.expanduser("~/Desktop/shoehorners_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the MIDI file to the output directory with a better name
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_midi_path = os.path.join(output_dir, f"transcribed_music_{timestamp}.mid")
        output_mscz_path = os.path.join(output_dir, f"transcribed_music_{timestamp}.mscz")
        output_pdf_path = os.path.join(output_dir, f"transcribed_music_{timestamp}.pdf")
        
        # Copy the file
        shutil.copy2(midi_path, output_midi_path)
        
        # Convert MIDI to PDF using MuseScore
        if platform.system() == "Windows":
            musescore_exe = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
            if not os.path.exists(musescore_exe):
                musescore_exe = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
            
            if os.path.exists(musescore_exe):
                # Convert MIDI to PDF
                result = subprocess.run([musescore_exe, "-o", output_pdf_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the PDF file
                    subprocess.Popen([musescore_exe, output_pdf_path])
                else:
                    st.error(f"Error converting MIDI to PDF: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on Windows")
                return
                
        elif platform.system() == "Darwin": # macOS
            # Try different possible app names and paths for macOS
            possible_paths = [
                "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
                "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
                "/Applications/MuseScore.app/Contents/MacOS/mscore"
            ]
            
            # Check if any of the possible paths exist
            musescore_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    musescore_path = path
                    break
            
            if musescore_path:
                # Convert MIDI to PDF
                result = subprocess.run([musescore_path, "-o", output_pdf_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the PDF file
                    subprocess.run(["open", output_pdf_path])
                else:
                    st.error(f"Error converting MIDI to PDF: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on macOS")
                return
                
        else: # Linux
            # Try different Linux commands
            musescore_cmds = ["musescore4", "musescore3", "musescore"]
            musescore_cmd = None
            
            for cmd in musescore_cmds:
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, check=True)
                    musescore_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if musescore_cmd:
                # Convert MIDI to PDF
                result = subprocess.run([musescore_cmd, "-o", output_pdf_path, output_midi_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Open the PDF file
                    subprocess.Popen([musescore_cmd, output_pdf_path])
                else:
                    st.error(f"Error converting MIDI to PDF: {result.stderr}")
                    return
            else:
                st.error("MuseScore not found on Linux")
                return
        
        st.success(f"Converted and opened {output_pdf_path} in PDF viewer.")
        st.info(f"Files saved to: {output_dir}")
        
    except Exception as e:
        st.error(f"Error converting MIDI to PDF: {str(e)}")
        st.info("Make sure MuseScore is installed in /Applications/")

# Streamlit app
st.title("🎵 Music Transcription")
st.write("Upload an MP3 file or provide a YouTube link to convert it to MIDI and analyze the music")

# YouTube link input
st.subheader("🎥 YouTube Link")
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
st.caption("Supports: youtube.com/watch?v=..., youtu.be/..., and other YouTube video formats. Playlist URLs will download only the first video.")

if youtube_url and is_valid_youtube_url(youtube_url):
    # Clean the URL to remove playlist parameters
    cleaned_url = clean_youtube_url(youtube_url)
    st.info(f"Processing: {cleaned_url}")
    with st.spinner("Downloading audio from YouTube..."):
        audio_file_path = download_youtube_audio(cleaned_url)
    
    if audio_file_path and os.path.exists(audio_file_path):
        st.success(f"Successfully downloaded: {audio_file_path}")
        
        # Process the downloaded audio file
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        
        # Create a file-like object for processing
        class FileLikeObject:
            def __init__(self, data):
                self.data = data
                self.position = 0
            
            def read(self):
                return self.data
            
            def seek(self, position):
                self.position = position
        
        audio_file = FileLikeObject(audio_data)
        
        # Process the audio
        with st.spinner("Transcribing audio to MIDI..."):
            midi_path = transcribe_to_midi(audio_file)
        
        if midi_path:
            with st.spinner("Extracting notes..."):
                note_sequence = extract_notes(midi_path)
            
            # Display results
            st.subheader("📝 Note Sequence")
            st.text(note_sequence)
            
            # Music analysis
            st.subheader("🎼 Music Analysis")
            analysis = analyze_music(note_sequence)
            st.text(analysis)
            
            # Render piano roll
            st.subheader("🎼 Piano Roll Visualization")
            score_path = render_score(midi_path)
            if score_path and os.path.exists(score_path):
                st.image(score_path, caption="Piano Roll Visualization")
            
            # Convert MIDI to audio for playback
            st.subheader("🎵 Play Transcribed Music")
            audio_path = midi_to_audio(midi_path)
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                st.audio(audio_bytes, format='audio/wav')
                
                # Download buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with open(midi_path, 'rb') as f:
                        st.download_button(
                            label="Download MIDI",
                            data=f.read(),
                            file_name="youtube_transcribed_music.mid",
                            mime="audio/midi"
                        )
                
                with col2:
                    with open(audio_path, 'rb') as f:
                        st.download_button(
                            label="Download Audio",
                            data=f.read(),
                            file_name="youtube_transcribed_music.wav",
                            mime="audio/wav"
                        )
                
                with col3:
                    if st.button("🎼 Open in MuseScore"):
                        open_in_musescore(midi_path)
                
                with col4:
                    if st.button("📄 Open PDF"):
                        open_pdf_from_midi(midi_path)
            
            # Clean up temporary files
            try:
                os.unlink(midi_path)
                if score_path and os.path.exists(score_path):
                    os.unlink(score_path)
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                # Clean up downloaded YouTube file
                if audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except Exception as e:
                st.warning(f"Note: Some temporary files couldn't be cleaned up: {str(e)}")
elif youtube_url and not is_valid_youtube_url(youtube_url):
    st.error("Please enter a valid YouTube URL")

st.subheader("📁 Or upload your own MP3 file:")

# Check if twinkle.mp3 exists
twinkle_path = "twinkle.mp3"
if os.path.exists(twinkle_path):
    st.success(f"Found {twinkle_path} in the current directory!")
    
    # Process the file
    with open(twinkle_path, 'rb') as f:
        audio_data = f.read()
    
    # Create a file-like object for processing
    class FileLikeObject:
        def __init__(self, data):
            self.data = data
            self.position = 0
        
        def read(self):
            return self.data
        
        def seek(self, position):
            self.position = position
    
    audio_file = FileLikeObject(audio_data)
    
    # Process the audio
    with st.spinner("Transcribing audio to MIDI..."):
        midi_path = transcribe_to_midi(audio_file)
    
    with st.spinner("Extracting notes..."):
        note_sequence = extract_notes(midi_path)
    
    # Display results
    st.subheader("📝 Note Sequence")
    st.text(note_sequence)
    
    # Music analysis
    st.subheader("🎼 Music Analysis")
    analysis = analyze_music(note_sequence)
    st.text(analysis)
    
    # Render piano roll
    st.subheader("🎼 Piano Roll Visualization")
    score_path = render_score(midi_path)
    if score_path and os.path.exists(score_path):
        st.image(score_path, caption="Piano Roll Visualization")
    
    # Convert MIDI to audio for playback
    st.subheader("🎵 Play Transcribed Music")
    audio_path = midi_to_audio(midi_path)
    if audio_path and os.path.exists(audio_path):
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        st.audio(audio_bytes, format='audio/wav')
        
        # Download buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with open(midi_path, 'rb') as f:
                st.download_button(
                    label="Download MIDI",
                    data=f.read(),
                    file_name="transcribed_music.mid",
                    mime="audio/midi"
                )
        
        with col2:
            with open(audio_path, 'rb') as f:
                st.download_button(
                    label="Download Audio",
                    data=f.read(),
                    file_name="transcribed_music.wav",
                    mime="audio/wav"
                )
        
        with col3:
            if st.button("🎼 Open in MuseScore"):
                open_in_musescore(midi_path)
        
        with col4:
            if st.button("📄 Open PDF"):
                open_pdf_from_midi(midi_path)
    
    # Clean up temporary files
    try:
        os.unlink(midi_path)
        if score_path and os.path.exists(score_path):
            os.unlink(score_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
    except:
        pass

else:
    st.error(f"Could not find {twinkle_path} in the current directory.")
    st.info("Please make sure the twinkle.mp3 file is in the same directory as this script.")

uploaded_file = st.file_uploader("Upload MP3", type=['mp3'])

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        midi_path = transcribe_to_midi(uploaded_file)
        note_sequence = extract_notes(midi_path)
        
        st.subheader("📝 Note Sequence")
        st.text(note_sequence)
        
        # Music analysis
        st.subheader("🎼 Music Analysis")
        analysis = analyze_music(note_sequence)
        st.text(analysis)
        
        # Render piano roll
        st.subheader("🎼 Piano Roll Visualization")
        score_path = render_score(midi_path)
        if score_path and os.path.exists(score_path):
            st.image(score_path, caption="Piano Roll Visualization")
        
        # Convert MIDI to audio for playback
        st.subheader("🎵 Play Transcribed Music")
        audio_path = midi_to_audio(midi_path)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            st.audio(audio_bytes, format='audio/wav')
            
            # Download buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with open(midi_path, 'rb') as f:
                    st.download_button(
                        label="Download MIDI",
                        data=f.read(),
                        file_name="transcribed_music.mid",
                        mime="audio/midi"
                    )
            
            with col2:
                with open(audio_path, 'rb') as f:
                    st.download_button(
                        label="Download Audio",
                        data=f.read(),
                        file_name="transcribed_music.wav",
                        mime="audio/wav"
                    )
            
            with col3:
                if st.button("🎼 Open in MuseScore"):
                    open_in_musescore(midi_path)
            
            with col4:
                if st.button("📄 Open PDF"):
                    open_pdf_from_midi(midi_path)
