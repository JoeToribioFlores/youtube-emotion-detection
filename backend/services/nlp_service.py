import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

# Descarga recursos NLTK solo una vez
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, language: str = "spanish"):
        self.language = language
        self.stop_words = set(stopwords.words(self.language)) if language in stopwords.fileids() else set()

    def preprocess(self, text: str) -> str:
        """Preprocesamiento de texto optimizado"""
        if not isinstance(text, str):
            return ""

        text = text.lower().strip()
        text = re.sub(r'<[^>]+>|https?://\S+|www\.\S+|[^\w\s]|\d+', ' ', text)
        
        try:
            tokens = word_tokenize(text, language=self.language)
        except:
            tokens = text.split()

        return ' '.join(word for word in tokens if word not in self.stop_words)

class EmotionAnalyzer:
    def __init__(self):
        """Carga el modelo solo cuando sea necesario"""
        self.model = None
        self.model_loaded = False

    def _load_model(self):
        """Carga perezosa del modelo para ahorrar memoria"""
        if not self.model_loaded:
            from transformers import pipeline
            model_name = os.getenv('NLP_MODEL', 'finiteautomata/beto-sentiment-analysis')
            try:
                self.model = pipeline(
                    "text-classification",
                    model=model_name,
                    return_all_scores=True,
                    device=-1  # Usa CPU para compatibilidad con Vercel
                )
                self.model_loaded = True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """Análisis de emociones con manejo de errores"""
        if not isinstance(text, str) or not text.strip():
            return {'emotion': 'neutral', 'confidence': 0.5}

        try:
            self._load_model()
            text = text[:512]  # Limita el tamaño para el modelo
            
            result = self.model(text)[0]
            main_emotion = max(result, key=lambda x: x['score'])
            
            return {
                'emotion': main_emotion['label'],
                'confidence': float(main_emotion['score']),
                'details': [
                    {'label': e['label'], 'score': float(e['score'])}
                    for e in result
                ]
            }
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {'emotion': 'error', 'confidence': 0.0, 'error': str(e)}