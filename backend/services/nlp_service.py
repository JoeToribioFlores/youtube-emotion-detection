import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, List, Any
from transformers import pipeline
from tqdm import tqdm

from app.config import Config

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, language: str = "spanish"):
        self.language = language
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(self.language))
        except Exception as e:
            logger.warning(f"Error downloading NLTK resources: {str(e)}")
            self.stop_words = set()

    def preprocess(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        try:
            text = text.lower()
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)

            try:
                tokens = word_tokenize(text, language=self.language)
            except:
                tokens = text.split()

            tokens = [word for word in tokens if word not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text

class EmotionAnalyzer:
    def __init__(self, model_name: str = None, language: str = None):
        self.language = language or Config.DEFAULT_LANGUAGE
        self.model_name = model_name or (
            Config.MODEL_SPANISH if self.language == "spanish" 
            else Config.MODEL_ENGLISH
        )
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True
            )
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'all_emotions': []
            }

        try:
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            result = self.classifier(text)
            main_emotion = max(result[0], key=lambda x: x['score'])

            return {
                'emotion': main_emotion['label'],
                'confidence': float(main_emotion['score']),
                'all_emotions': [
                    {'emotion': e['label'], 'score': float(e['score'])}
                    for e in result[0]
                ]
            }
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'all_emotions': [],
                'error': str(e)
            }