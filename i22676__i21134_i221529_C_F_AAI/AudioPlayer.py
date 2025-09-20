# Handles audio playback using pygame
# Controls volume, pause/play, etc.


import pygame
import time
#  controlls all the audio related stuff 
#  audio play , pause , picthes , speed etc
class AudioPlayer:
    """
    Handles audio playback functionality
    """
    def __init__(self):
        """Initialize pygame mixer for audio playback"""
        pygame.init()
        pygame.mixer.init()
    
    def play(self, filename):
        """
        Play the audio file using pygame
        """
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def stop(self):
        """Stop current audio playback"""
        pygame.mixer.music.stop()
    
    def set_volume(self, volume):
        """Set playback volume (0.0 to 1.0)"""
        pygame.mixer.music.set_volume(volume)
    
    def pause(self):
        """Pause playback"""
        pygame.mixer.music.pause()
    
    def unpause(self):
        """Resume playback"""
        pygame.mixer.music.unpause()
    
    def cleanup(self):
        """Clean up pygame resources"""
        pygame.mixer.quit()
        pygame.quit()