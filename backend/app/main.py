from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import logging
import pandas as pd
import os

from app.services.youtube_service import YouTubeService
from app.services.nlp_service import TextPreprocessor, EmotionAnalyzer
from app.services.visualization import VisualizationService
from app.config import Config
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Emotion Detection API",
    description="API for analyzing emotions in YouTube comments",
    version="1.0.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze")
async def analyze_video(
    video_url: str = Query(..., description="YouTube video URL"),
    max_comments: Optional[int] = Query(100, description="Maximum number of comments to analyze"),
    chart_type: Optional[str] = Query("bar", description="Type of chart to generate (bar or pie)")
):
    try:
        # Extraer el ID del video de la URL
        video_id = extract_video_id(video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Inicializar servicios
        youtube = YouTubeService()
        preprocessor = TextPreprocessor()
        analyzer = EmotionAnalyzer()
        visualizer = VisualizationService()

        # Obtener comentarios
        comments = youtube.extract_comments(video_id, max_comments)
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this video")

        # Analizar emociones
        results = []
        for comment in comments:
            processed_text = preprocessor.preprocess(comment['comment'])
            emotion_result = analyzer.analyze(processed_text)
            
            results.append({
                'author': comment['author'],
                'comment': comment['comment'],
                'processed_comment': processed_text,
                'emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'date': comment['date'],
                'likes': comment['likes']
            })

        df = pd.DataFrame(results)

        # Generar visualización
        video_details = youtube.get_video_details(video_id)
        title = f"Emotions in comments for: {video_details.get('title', video_id)}" if video_details else None

        if chart_type.lower() == "pie":
            chart_path = visualizer.create_pie_chart(df, title)
        else:
            chart_path = visualizer.create_emotion_distribution_plot(df, title)

        if not chart_path:
            raise HTTPException(status_code=500, detail="Error generating visualization")

        # Devolver la imagen generada
        return FileResponse(chart_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_video_id(url: str) -> Optional[str]:
    """Extrae el ID de un video de YouTube desde su URL"""
    import re
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\/]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.get("/")
async def root():
    return {"message": "YouTube Emotion Detection API is running"}