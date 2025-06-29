import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("YOUTUBE_API_KEY", "")
    MODEL_SPANISH = "finiteautomata/beto-emotion-analysis"
    MODEL_ENGLISH = "j-hartmann/emotion-english-distilroberta-base"
    DEFAULT_LANGUAGE = "spanish"
    OUTPUT_DIR = "output"
    MAX_COMMENTS = 500