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
    """Convert audio file to MIDI using librosa pitch detection with proper note grouping"""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Extract pitch using librosa with higher resolution
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, hop_length=512)
    
    # Create MIDI file
    midi_path = audio_path.replace('.mp3', '.mid')
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.Instrument(program=0)  # Piano
    
    # Parameters for note detection
    min_note_duration = 0.1  # Minimum note duration in seconds
    pitch_tolerance = 2  # Semitones tolerance for grouping same notes
    silence_threshold = 0.05  # Threshold for detecting silence
    silence_duration = 0.05  # Minimum silence duration to separate notes
    
    # Process pitch data to group into notes
    notes = []
    current_note = None
    silence_start = None
    
    for t in range(pitches.shape[1]):
        pitch = pitches[:, t]
        magnitude = magnitudes[:, t]
        current_time = t * librosa.get_duration(y=y, sr=sr) / pitches.shape[1]
        
        # Check if this is silence
        is_silence = np.max(magnitude) <= silence_threshold
        
        if is_silence:
            # Track silence duration
            if silence_start is None:
                silence_start = current_time
            
            # If we have a current note and silence is long enough, finalize the note
            if current_note is not None and (current_time - silence_start) >= silence_duration:
                if current_note['end'] - current_note['start'] >= min_note_duration:
                    notes.append(current_note)
                current_note = None
        else:
            # Reset silence tracking
            silence_start = None
            
            # Find the strongest pitch at this time
            pitch_idx = np.argmax(magnitude)
            pitch_value = pitch[pitch_idx]
            
            if pitch_value > 0:
                # Convert frequency to MIDI note number
                midi_note = int(round(12 * np.log2(pitch_value / 440) + 69))
                
                # Ensure note is in valid range
                if 21 <= midi_note <= 108:
                    # Calculate velocity (ensure it's in valid MIDI range 0-127)
                    velocity = int(np.clip(np.max(magnitude) * 127, 1, 127))
                    
                    if current_note is None:
                        # Start new note
                        current_note = {
                            'pitch': midi_note,
                            'start': current_time,
                            'end': current_time,  # Initialize end time
                            'velocity': velocity
                        }
                    elif abs(midi_note - current_note['pitch']) <= pitch_tolerance:
                        # Continue current note
                        current_note['end'] = current_time
                    else:
                        # Different note detected, finalize current note and start new one
                        if current_note['end'] - current_note['start'] >= min_note_duration:
                            notes.append(current_note)
                        
                        current_note = {
                            'pitch': midi_note,
                            'start': current_time,
                            'end': current_time,  # Initialize end time
                            'velocity': velocity
                        }
                else:
                    # Finalize current note if we have one
                    if current_note is not None:
                        if current_note['end'] - current_note['start'] >= min_note_duration:
                            notes.append(current_note)
                        current_note = None
            else:
                # Finalize current note if we have one
                if current_note is not None:
                    if current_note['end'] - current_note['start'] >= min_note_duration:
                        notes.append(current_note)
                    current_note = None
    
    # Add final note if exists
    if current_note is not None:
        if current_note['end'] - current_note['start'] >= min_note_duration:
            notes.append(current_note)
    
    # Create MIDI notes
    for note_info in notes:
        note = pretty_midi.Note(
            velocity=note_info['velocity'],
            pitch=note_info['pitch'],
            start=note_info['start'],
            end=note_info['end']
        )
        piano_program.notes.append(note)
    
    midi_data.instruments.append(piano_program)
    midi_data.write(midi_path)
    
    # Clean up temporary audio file
    os.unlink(audio_path)
    
    return midi_path

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

# Streamlit app
st.title("🎵 Music Transcription to Score")
st.write("Upload an MP3 file to convert it to a musical score")

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
    
    # Render score
    st.subheader("🎼 Musical Score")
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
            col1, col2, col3 = st.columns(3)
            with col1:
                with open(midi_path, 'rb') as f:
                    st.download_button(
                        label="Download MIDI",
                        data=f.read(),
                        file_name="transcribed_music.mid",
                        mime="audio/midi"
                    )
            
            with col2:
                with open(score_path, 'rb') as f:
                    st.download_button(
                        label="Download Score Image",
                        data=f.read(),
                        file_name="musical_score.png",
                        mime="image/png"
                    )
            
            with col3:
                with open(audio_path, 'rb') as f:
                    st.download_button(
                        label="Download Audio",
                        data=f.read(),
                        file_name="transcribed_music.wav",
                        mime="audio/wav"
                    )
    
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
    st.error(f"Could not find {twinkle.mp3} in the current directory.")
    st.info("Please make sure the twinkle.mp3 file is in the same directory as this script.")

# Also provide file upload option
st.subheader("Or upload your own MP3 file:")
uploaded_file = st.file_uploader("Upload MP3", type=['mp3'])

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        midi_path = transcribe_to_midi(uploaded_file)
        note_sequence = extract_notes(midi_path)
        
        st.subheader("📝 Note Sequence")
        st.text(note_sequence)
        
        st.subheader("🎼 Musical Score")
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
