import os
import time
import pyttsx3
import librosa
import soundfile as sf
import numpy as np
from TTS_Optimizer import TTSGeneticAlgorithm
from AudioProcessor import AudioProcessor

class TTSEngine:
    # Enhanced TTS Engine with Genetic Algorithm Optimization
    #  GA from Optimizer file 
    
    def __init__(self, emotion_classifier=None):
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.ga_optimizer = TTSGeneticAlgorithm(
            population_size=30,
            generations=20,
            mutation_rate=0.15
        )
        self.emotion_classifier = emotion_classifier
        
        # Default voice parameters
        self.current_params = {
            'pitch': 120,        # Fundamental frequency (Hz)
            'speed': 1.0,        # Speaking rate multiplier (0.5-2.0)
            'volume': 1.0,       # Volume level (0.0-1.0)
            'voice_clarity': 0.8,# Articulation strength (0.1-1.0)
            'pause_duration': 0.5# Pause length multiplier (0.1-1.5)
        }
        
        # Apply initial parameters
        self._apply_current_parameters()

    def _apply_current_parameters(self):
        """Apply current parameters to the TTS engine"""
        try:
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', 175 * self.current_params['speed'])
            
            # Set volume level
            self.engine.setProperty('volume', self.current_params['volume'])
            
            # Select first available voice
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
                
        except Exception as e:
            print(f"Parameter application error: {str(e)}")

    def optimize_parameters(self, text):
        """Optimize TTS parameters for given text"""
        try:
            # Get emotion prediction
            emotion = "neutral"
            if self.emotion_classifier:
                # Handle tuple return (emotion_str, probabilities_dict)
                emotion_result = self.emotion_classifier.predict(text)
                emotion = emotion_result[0] if isinstance(emotion_result, tuple) else "neutral"

            # Get target parameters for detected emotion
            target_params = self._get_emotion_targets(emotion.lower())
            
            # Run genetic optimization
            optimized_params = self.ga_optimizer.evolve(text, target_params)
            
            # Update and apply new parameters
            self.current_params.update(optimized_params)
            self._apply_current_parameters()
            return True
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return False

    def synthesize(self, text, output_file=None):
        """Convert text to speech with current parameters"""
        try:
            # Generate output filename
            if not output_file:
                output_dir = "output_audio"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"tts_{int(time.time())}.wav")
            
            # Generate raw speech
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            
            # Apply audio processing
            return self._process_audio(output_file)
            
        except Exception as e:
            print(f"Synthesis failed: {str(e)}")
            return None

    def _process_audio(self, input_file):
        """Apply post-processing effects"""
        try:
            # Load generated audio
            y, sr = librosa.load(input_file, sr=None)
            
            # Apply pitch shifting
            if 'pitch' in self.current_params:
                base_pitch = 120  # Default female voice
                n_steps = 12 * np.log2(self.current_params['pitch'] / base_pitch)
                y = self.audio_processor.apply_pitch_shift(y, sr, n_steps)
            
            # Apply time stretching
            if 'speed' in self.current_params:
                rate = 1.0 / self.current_params['speed']
                y = self.audio_processor.apply_time_stretch(y, rate)
            
            # Save processed audio
            processed_file = input_file.replace('.wav', '_processed.wav')
            sf.write(processed_file, y, sr)
            return processed_file
            
        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
            return input_file

    def _get_emotion_targets(self, emotion):
        """Get target parameters for different emotions"""
        return {
            'happy': {'pitch': 180, 'speed': 1.2, 'voice_clarity': 0.9},
            'sad': {'pitch': 100, 'speed': 0.8, 'voice_clarity': 0.7},
            'anger': {'pitch': 150, 'speed': 1.3, 'voice_clarity': 0.85},
            'neutral': {'pitch': 120, 'speed': 1.0, 'voice_clarity': 0.8}
        }.get(emotion, {})  # Returns empty dict for unknown emotions

    def get_current_parameters(self):
        """Return current parameter set"""
        return self.current_params.copy()