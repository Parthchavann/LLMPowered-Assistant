"""
Multi-Modal Document Processing

Supports processing and understanding of:
- Images (screenshots, diagrams, UI elements)
- Audio files (support calls, voice messages)
- Combined text + image documents
- Video transcription and analysis
"""

import logging
import base64
import io
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import ollama
from utils.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced image processing for document analysis"""
    
    def __init__(self, ollama_client=None):
        self.client = ollama_client or ollama.Client()
        self.model_name = "llava:7b"  # Vision-language model
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text using OCR"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Enhance image for better OCR
            image = self._enhance_image_for_ocr(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, lang='eng')
            
            logger.info(f"Extracted {len(text)} characters from image")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Resize if too small
        width, height = image.size
        if width < 800 or height < 600:
            scale_factor = max(800/width, 600/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def analyze_image_content(self, image_path: str, context: str = "") -> str:
        """Analyze image content using vision-language model"""
        try:
            # Check if vision model is available
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.warning(f"Vision model {self.model_name} not found. Attempting to pull...")
                try:
                    self.client.pull(self.model_name)
                except Exception as e:
                    logger.error(f"Failed to pull vision model: {str(e)}")
                    return self.extract_text_from_image(image_path)
            
            # Encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = f"""Analyze this image in the context of customer support documentation. 
            
            Context: {context if context else 'General support document'}
            
            Please describe:
            1. What type of content this image shows (screenshot, diagram, UI, etc.)
            2. Key information visible in the image
            3. Any text content that should be extracted
            4. How this relates to customer support or troubleshooting
            
            Be detailed and specific:"""
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_data],
                options={'temperature': 0.3}
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {str(e)}")
            # Fallback to OCR
            return self.extract_text_from_image(image_path)
    
    def detect_ui_elements(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect UI elements in screenshots"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect buttons (rounded rectangles)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            # Detect text regions
            text_regions = self._detect_text_regions(gray)
            
            # Detect rectangular elements (buttons, input fields)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ui_elements = []
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Classify element type
                    element_type = "unknown"
                    if 2 < aspect_ratio < 6 and 20 < h < 60:
                        element_type = "button"
                    elif 1 < aspect_ratio < 8 and 25 < h < 45:
                        element_type = "input_field"
                    elif aspect_ratio > 8 and h < 30:
                        element_type = "menu_bar"
                    
                    ui_elements.append({
                        'type': element_type,
                        'bbox': [x, y, w, h],
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
            
            logger.info(f"Detected {len(ui_elements)} UI elements")
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {str(e)}")
            return []
    
    def _detect_text_regions(self, gray_image):
        """Detect text regions in image"""
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)
        
        # Filter regions by size and aspect ratio
        text_regions = []
        for region in regions:
            if len(region) > 10:  # Minimum points
                x, y, w, h = cv2.boundingRect(region)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 10 and h > 10:  # Text-like aspect ratio
                    text_regions.append([x, y, w, h])
        
        return text_regions

class AudioProcessor:
    """Audio processing for support calls and voice messages"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def transcribe_audio(self, audio_path: str, language: str = "en-US") -> Dict[str, Any]:
        """Transcribe audio to text"""
        try:
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Record the audio
                audio_data = self.recognizer.record(source)
            
            # Transcribe using Google Speech Recognition (free tier)
            try:
                text = self.recognizer.recognize_google(audio_data, language=language)
                
                return {
                    'transcription': text,
                    'confidence': 'high',  # Google API doesn't return confidence
                    'language': language,
                    'method': 'google'
                }
                
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return {'transcription': '', 'confidence': 'low', 'error': 'unintelligible'}
                
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition error: {str(e)}")
                
                # Fallback to offline recognition
                return self._transcribe_offline(audio_data)
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {'transcription': '', 'error': str(e)}
    
    def _transcribe_offline(self, audio_data) -> Dict[str, Any]:
        """Fallback offline transcription using CMU Sphinx"""
        try:
            text = self.recognizer.recognize_sphinx(audio_data)
            return {
                'transcription': text,
                'confidence': 'medium',
                'method': 'sphinx_offline'
            }
        except Exception as e:
            logger.error(f"Offline transcription failed: {str(e)}")
            return {'transcription': '', 'error': 'transcription_failed'}
    
    def analyze_audio_sentiment(self, transcription: str) -> Dict[str, Any]:
        """Analyze sentiment of transcribed audio"""
        # Simple keyword-based sentiment analysis
        positive_words = {
            'thank', 'thanks', 'good', 'great', 'excellent', 'helpful',
            'appreciate', 'satisfied', 'happy', 'pleased'
        }
        
        negative_words = {
            'frustrated', 'angry', 'upset', 'disappointed', 'terrible',
            'awful', 'horrible', 'useless', 'hate', 'annoyed'
        }
        
        urgent_words = {
            'urgent', 'emergency', 'critical', 'immediately', 'asap',
            'broken', 'down', 'not working', 'error', 'problem'
        }
        
        words = transcription.lower().split()
        word_set = set(words)
        
        positive_score = len(word_set & positive_words)
        negative_score = len(word_set & negative_words)
        urgent_score = len(word_set & urgent_words)
        
        # Determine overall sentiment
        if negative_score > positive_score:
            sentiment = 'negative'
        elif positive_score > negative_score:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        urgency = 'high' if urgent_score > 2 else 'medium' if urgent_score > 0 else 'low'
        
        return {
            'sentiment': sentiment,
            'urgency': urgency,
            'scores': {
                'positive': positive_score,
                'negative': negative_score,
                'urgent': urgent_score
            },
            'confidence': min(1.0, (positive_score + negative_score + urgent_score) / 10)
        }
    
    def extract_key_topics(self, transcription: str) -> List[str]:
        """Extract key topics from transcribed audio"""
        # Common support topics
        topic_keywords = {
            'account': ['account', 'login', 'password', 'username', 'profile'],
            'billing': ['bill', 'charge', 'payment', 'refund', 'subscription', 'cost'],
            'technical': ['error', 'bug', 'crash', 'slow', 'loading', 'connection'],
            'feature': ['feature', 'function', 'how to', 'tutorial', 'guide'],
            'cancellation': ['cancel', 'terminate', 'close', 'delete', 'remove']
        }
        
        words = transcription.lower().split()
        word_set = set(words)
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in transcription.lower() for keyword in keywords):
                score = sum(1 for keyword in keywords if keyword in transcription.lower())
                detected_topics.append({
                    'topic': topic,
                    'score': score,
                    'keywords_found': [kw for kw in keywords if kw in transcription.lower()]
                })
        
        # Sort by relevance score
        detected_topics.sort(key=lambda x: x['score'], reverse=True)
        
        return [topic['topic'] for topic in detected_topics[:3]]  # Top 3 topics

class VideoProcessor:
    """Video processing for screen recordings and video tutorials"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio track from video"""
        try:
            if output_path is None:
                output_path = str(Path(video_path).with_suffix('.wav'))
            
            # Load video and extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is not None:
                audio.write_audiofile(output_path, verbose=False, logger=None)
                audio.close()
            
            video.close()
            
            logger.info(f"Extracted audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio from video: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        """Extract representative frames from video"""
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Calculate frame times
            frame_times = np.linspace(0, duration - 1, num_frames)
            
            frame_paths = []
            temp_dir = Path(tempfile.mkdtemp())
            
            for i, time_point in enumerate(frame_times):
                frame = video.get_frame(time_point)
                frame_path = temp_dir / f"frame_{i:03d}.jpg"
                
                # Convert numpy array to PIL Image and save
                pil_image = Image.fromarray(frame.astype('uint8'))
                pil_image.save(frame_path)
                
                frame_paths.append(str(frame_path))
            
            video.close()
            
            logger.info(f"Extracted {len(frame_paths)} frames")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def analyze_video_content(self, video_path: str) -> Dict[str, Any]:
        """Comprehensive video analysis"""
        try:
            # Extract audio and transcribe
            audio_path = self.extract_audio_from_video(video_path)
            audio_analysis = self.audio_processor.transcribe_audio(audio_path)
            
            # Analyze audio sentiment and topics
            if audio_analysis.get('transcription'):
                sentiment_analysis = self.audio_processor.analyze_audio_sentiment(
                    audio_analysis['transcription']
                )
                topics = self.audio_processor.extract_key_topics(
                    audio_analysis['transcription']
                )
            else:
                sentiment_analysis = {}
                topics = []
            
            # Extract and analyze key frames
            frame_paths = self.extract_frames(video_path)
            frame_analyses = []
            
            for frame_path in frame_paths[:5]:  # Analyze first 5 frames
                frame_analysis = self.image_processor.analyze_image_content(
                    frame_path, 
                    context="Video frame from support content"
                )
                frame_analyses.append(frame_analysis)
            
            return {
                'audio_transcription': audio_analysis.get('transcription', ''),
                'transcription_confidence': audio_analysis.get('confidence', 'low'),
                'sentiment_analysis': sentiment_analysis,
                'key_topics': topics,
                'frame_analyses': frame_analyses,
                'total_frames_extracted': len(frame_paths),
                'processing_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            return {
                'processing_status': 'error',
                'error': str(e)
            }

class MultiModalDocumentProcessor:
    """Main class for processing multi-modal documents"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_audio_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def process_multimodal_file(self, file_path: str, context: str = "") -> Dict[str, Any]:
        """Process a multi-modal file and extract comprehensive information"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in self.supported_image_formats:
                return self._process_image_file(str(file_path), context)
            elif file_extension in self.supported_audio_formats:
                return self._process_audio_file(str(file_path), context)
            elif file_extension in self.supported_video_formats:
                return self._process_video_file(str(file_path), context)
            else:
                return {
                    'error': f'Unsupported file format: {file_extension}',
                    'supported_formats': {
                        'images': list(self.supported_image_formats),
                        'audio': list(self.supported_audio_formats),
                        'video': list(self.supported_video_formats)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing multimodal file {file_path}: {str(e)}")
            return {'error': str(e), 'file_path': str(file_path)}
    
    def _process_image_file(self, file_path: str, context: str) -> Dict[str, Any]:
        """Process image files"""
        # Extract text using OCR
        ocr_text = self.image_processor.extract_text_from_image(file_path)
        
        # Analyze content using vision model
        content_analysis = self.image_processor.analyze_image_content(file_path, context)
        
        # Detect UI elements if it's a screenshot
        ui_elements = self.image_processor.detect_ui_elements(file_path)
        
        # Combine all text content
        combined_text = f"{content_analysis}\n\nExtracted Text (OCR):\n{ocr_text}"
        
        # Generate embedding for the combined content
        embedding = self.embedding_generator.generate_single_embedding(combined_text)
        
        return {
            'type': 'image',
            'file_path': file_path,
            'ocr_text': ocr_text,
            'content_analysis': content_analysis,
            'ui_elements': ui_elements,
            'combined_text': combined_text,
            'embedding': embedding,
            'metadata': {
                'has_text': len(ocr_text.strip()) > 0,
                'ui_elements_count': len(ui_elements),
                'content_type': 'screenshot' if ui_elements else 'document'
            }
        }
    
    def _process_audio_file(self, file_path: str, context: str) -> Dict[str, Any]:
        """Process audio files"""
        # Transcribe audio
        transcription_result = self.audio_processor.transcribe_audio(file_path)
        
        transcription = transcription_result.get('transcription', '')
        
        # Analyze sentiment and topics if transcription is available
        if transcription:
            sentiment_analysis = self.audio_processor.analyze_audio_sentiment(transcription)
            topics = self.audio_processor.extract_key_topics(transcription)
            
            # Create comprehensive text for embedding
            combined_text = f"Audio Content: {transcription}\n\nContext: {context}"
            if topics:
                combined_text += f"\n\nKey Topics: {', '.join(topics)}"
            
            # Generate embedding
            embedding = self.embedding_generator.generate_single_embedding(combined_text)
        else:
            sentiment_analysis = {}
            topics = []
            combined_text = f"Audio file (transcription failed): {context}"
            embedding = self.embedding_generator.generate_single_embedding(combined_text)
        
        return {
            'type': 'audio',
            'file_path': file_path,
            'transcription': transcription,
            'transcription_confidence': transcription_result.get('confidence', 'unknown'),
            'sentiment_analysis': sentiment_analysis,
            'key_topics': topics,
            'combined_text': combined_text,
            'embedding': embedding,
            'metadata': {
                'has_transcription': len(transcription.strip()) > 0,
                'transcription_method': transcription_result.get('method', 'unknown'),
                'sentiment': sentiment_analysis.get('sentiment', 'neutral'),
                'urgency': sentiment_analysis.get('urgency', 'low')
            }
        }
    
    def _process_video_file(self, file_path: str, context: str) -> Dict[str, Any]:
        """Process video files"""
        # Analyze video content
        video_analysis = self.video_processor.analyze_video_content(file_path)
        
        if video_analysis.get('processing_status') == 'error':
            combined_text = f"Video file (processing failed): {context}\nError: {video_analysis.get('error', 'Unknown error')}"
            embedding = self.embedding_generator.generate_single_embedding(combined_text)
            
            return {
                'type': 'video',
                'file_path': file_path,
                'error': video_analysis.get('error'),
                'combined_text': combined_text,
                'embedding': embedding,
                'metadata': {'processing_failed': True}
            }
        
        # Extract key information
        transcription = video_analysis.get('audio_transcription', '')
        topics = video_analysis.get('key_topics', [])
        frame_analyses = video_analysis.get('frame_analyses', [])
        
        # Create comprehensive text combining all analyses
        combined_text = f"Video Content Analysis:\n\nAudio Transcription: {transcription}\n\n"
        
        if topics:
            combined_text += f"Key Topics: {', '.join(topics)}\n\n"
        
        if frame_analyses:
            combined_text += "Visual Analysis:\n"
            for i, frame_analysis in enumerate(frame_analyses[:3], 1):
                combined_text += f"Frame {i}: {frame_analysis[:200]}...\n"
        
        combined_text += f"\nContext: {context}"
        
        # Generate embedding
        embedding = self.embedding_generator.generate_single_embedding(combined_text)
        
        return {
            'type': 'video',
            'file_path': file_path,
            'transcription': transcription,
            'key_topics': topics,
            'sentiment_analysis': video_analysis.get('sentiment_analysis', {}),
            'frame_analyses': frame_analyses,
            'combined_text': combined_text,
            'embedding': embedding,
            'metadata': {
                'has_audio': len(transcription.strip()) > 0,
                'frames_analyzed': len(frame_analyses),
                'sentiment': video_analysis.get('sentiment_analysis', {}).get('sentiment', 'neutral'),
                'urgency': video_analysis.get('sentiment_analysis', {}).get('urgency', 'low')
            }
        }
    
    def create_multimodal_chunks(self, processed_data: Dict[str, Any], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Create chunks from processed multimodal data"""
        combined_text = processed_data.get('combined_text', '')
        
        if len(combined_text) <= chunk_size:
            return [{
                **processed_data,
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_text': combined_text
            }]
        
        # Split text into chunks
        chunks = []
        text_chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
        
        for i, chunk_text in enumerate(text_chunks):
            # Generate embedding for this chunk
            chunk_embedding = self.embedding_generator.generate_single_embedding(chunk_text)
            
            chunk_data = {
                **processed_data,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_text': chunk_text,
                'embedding': chunk_embedding
            }
            chunks.append(chunk_data)
        
        return chunks