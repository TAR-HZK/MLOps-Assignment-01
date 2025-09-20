import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

class AudioProcessor:
    
    # Handles audio processing tasks such as applying effects
    #  for keeping audio under control
    #  removing distortions, pitches shift handles
    
    @staticmethod
    def apply_pitch_shift(y, sr, n_steps):
        """Apply pitch shift to audio"""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def apply_time_stretch(y, rate):
        """Apply time stretching to audio"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    @staticmethod
    def apply_reverb(y, intensity=0.3, delay=4000):
        """Apply reverb effect to audio"""
        y_reversed = np.flipud(y)
        y_reversed = y_reversed * intensity
        y_shifted = np.zeros(len(y) + delay)
        y_shifted[:len(y)] = y
        y_shifted[delay:delay+len(y_reversed)] += y_reversed
        return y_shifted[:len(y)]
    
    @staticmethod
    def apply_tremolo(y, sr, rate=4.0, depth=0.2):
        """Apply tremolo effect to audio"""
        tremolo = 1 + depth * np.sin(2 * np.pi * rate * np.arange(len(y)) / sr)
        return y * tremolo
    
    @staticmethod
    def apply_distortion(y, gain=2.0, threshold=0.8):
        """Apply distortion effect to audio"""
        return np.clip(y * gain, -threshold, threshold)
    
    #  controlling the emotional effect in the audio
    @staticmethod
    def apply_emotion_effects(y, sr, emotion):
        """Apply emotion-specific audio effects"""
        if emotion == 'sadness':
            y = AudioProcessor.apply_reverb(y)
        elif emotion == 'fear':
            y = AudioProcessor.apply_tremolo(y, sr)
        elif emotion == 'anger':
            y = AudioProcessor.apply_distortion(y)
        return y
    
    #  at last releasing it
    #  and saving it in audio file for later use
    @staticmethod
    def process_audio(audio_path, emotion, settings):
        """
        Process audio file with emotion-specific effects
        Returns path to processed audio file
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply pitch shift
        if settings['pitch_shift'] != 0:
            y = AudioProcessor.apply_pitch_shift(y, sr, settings['pitch_shift'])
        
        # Apply speed change
        if settings['speed'] != 1.0:
            y = AudioProcessor.apply_time_stretch(y, settings['speed'])
        
        # Apply volume adjustment
        if settings['energy'] != 1.0:
            y = y * settings['energy']
        
        # Apply emotion-specific effects
        y = AudioProcessor.apply_emotion_effects(y, sr, emotion)
        
        # Save processed audio
        processed_path = audio_path.replace('.wav', f'_{emotion}_processed.wav')
        sf.write(processed_path, y, sr)
        
        return processed_path
    