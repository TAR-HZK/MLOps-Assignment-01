import os
import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QCheckBox, QMessageBox, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# Custom OCR imports
from main_script import process_image
from paddle_ocr import visualize_histogram_equalization, apply_histogram_equalization

# TTS System imports
from TTS_Engine import TTSEngine
from EmotionClassifier import EmotionClassifier
from AudioPlayer import AudioPlayer

#  hand writing recognition function
#  take hand writing as input and process on that
class HandwritingRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.results_text = None
        self.initUI()
        
        # Initializing the components
        #  for text to voice part
        self.audio_player = AudioPlayer()
        self.emotion_classifier = EmotionClassifier()
        self.tts_engine = TTSEngine(self.emotion_classifier)
        
        # Load emotion classifier
        #  sentiment analysis
        self.load_classifier()

    #  for emotions training on specific dataset
    def load_classifier(self):
        dataset_path = "Dataset"
        train_file = os.path.join(dataset_path, 'train.txt')
        val_file = os.path.join(dataset_path, 'val.txt')
        
        if os.path.exists(train_file):
            self.emotion_classifier.train(
                train_file, 
                val_file if os.path.exists(val_file) else None
            )
    
    # Styling and UI part 
    #  buttons etc
    def initUI(self):
        self.setWindowTitle("Handwriting Recognition & Speech Synthesis")
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Control buttons
        self.select_btn = QPushButton("Select Image")
        self.select_btn.setFont(QFont("Arial", 14))
        self.select_btn.setFixedHeight(50)
        self.select_btn.clicked.connect(self.select_image)
        
        # Processing options
        self.spell_check = QCheckBox("Enable Spell Correction")
        self.spell_check.setFont(QFont("Arial", 14))
        
        self.hist_eq_check = QCheckBox("Enable Histogram Equalization")
        self.hist_eq_check.setFont(QFont("Arial", 14))
        
        # Action buttons
        self.process_btn = QPushButton("Process Image")
        self.process_btn.setFont(QFont("Arial", 14))
        self.process_btn.setFixedHeight(50)
        self.process_btn.clicked.connect(self.process_current_image)
        self.process_btn.setEnabled(False)
        
        self.optimize_btn = QPushButton("Optimize Speech Parameters")
        self.optimize_btn.setFont(QFont("Arial", 14))
        self.optimize_btn.setFixedHeight(50)
        self.optimize_btn.clicked.connect(self.optimize_parameters)
        self.optimize_btn.setEnabled(False)
        
        self.tts_btn = QPushButton("Text to Speech")
        self.tts_btn.setFont(QFont("Arial", 14))
        self.tts_btn.setFixedHeight(50)
        self.tts_btn.clicked.connect(self.speak_text)
        self.tts_btn.setEnabled(False)
        
        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Arial", 14))
        
        # Layout organization
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.spell_check)
        left_layout.addWidget(self.hist_eq_check)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.optimize_btn)
        left_layout.addWidget(self.tts_btn)
        left_layout.addWidget(QLabel("Recognized Text:"))
        left_layout.addWidget(self.text_display)
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(400)
        
        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No image selected")
        
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setText("No processed image yet")
        
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.processed_image_label)
        right_panel.setLayout(right_layout)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    #  selection of new image
    #  for processing
    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path)
            self.process_btn.setEnabled(True)
            self.processed_image_label.setText("Click 'Process Image' to start recognition")

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            max_size = 500
            if h > max_size or w > max_size:
                if h > w:
                    new_h, new_w = max_size, int(w * max_size / h)
                else:
                    new_h, new_w = int(h * max_size / w), max_size
                image = cv2.resize(image, (new_w, new_h))
            
            h, w, c = image.shape
            q_img = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.image_label.setScaledContents(True)
        else:
            self.image_label.setText("Failed to load image")

    def process_current_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
        
        enable_spell = self.spell_check.isChecked()
        enable_hist_eq = self.hist_eq_check.isChecked()
        
        try:
            if enable_hist_eq:
                image = cv2.imread(self.current_image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                equalized = apply_histogram_equalization(gray)
                fig = visualize_histogram_equalization(gray, equalized)
                viz_path = "histogram_comparison.png"
                fig.savefig(viz_path)
                self.show_processed_image(viz_path)
            
            # Original OCR processing
            self.results_text = process_image(
                self.current_image_path, 
                enable_spell_correction=enable_spell,
                enable_histogram_eq=enable_hist_eq
            )
            
            self.text_display.setText(self.results_text)
            self.optimize_btn.setEnabled(True)
            self.tts_btn.setEnabled(True)
            self.show_processed_image("recognition_results.png")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")

    def show_processed_image(self, image_path):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if pixmap.width() > 800 or pixmap.height() > 600:
                pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio)
            self.processed_image_label.setPixmap(pixmap)
        else:
            self.processed_image_label.setText("Processed image not found")

    #  optimizing the current for text to voice
    #  to get GA
    def optimize_parameters(self):
        if self.results_text:
            try:
                self.tts_engine.optimize_parameters(self.results_text)
                QMessageBox.information(self, "Optimized", 
                    "TTS parameters optimized for current text!")
            except Exception as e:
                QMessageBox.critical(self, "Optimization Error", str(e))
        else:
            QMessageBox.warning(self, "Warning", "No text to optimize")
    
    #  final text to voice conversion call
    def speak_text(self):
        if self.results_text:
            try:
                output_dir = "output_audio"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"output_{int(time.time())}.wav")
                processed_file = self.tts_engine.synthesize(self.results_text, output_file)
                self.audio_player.play(processed_file)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Speech generation failed: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No text to speak")

#  for showing the whole display
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingRecognitionApp()
    window.show()
    sys.exit(app.exec_())