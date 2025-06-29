from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import pandas as pd
import re
import asyncio
from tempfile import NamedTemporaryFile
import os

# Manejo de imports para compatibilidad con Vercel
try:
    from services.youtube_service import YouTubeService
    from services.nlp_service import TextPreprocessor, EmotionAnalyzer
    from services.visualization import VisualizationService
    from config import Config
except ImportError:
    from .services.youtube_service import YouTubeService
    from .services.nlp_service import TextPreprocessor, EmotionAnalyzer
    from .services.visualization import VisualizationService
    from .config import Config

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Emotion Detection API",
    description="API for analyzing emotions in YouTube comments",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/analyze")
async def analyze_video(
    video_url: str = Query(..., description="YouTube video URL"),
    max_comments: Optional[int] = Query(50, description="Maximum number of comments to analyze (max: 50)"),
    chart_type: Optional[str] = Query("bar", description="Type of chart to generate (bar or pie)")
):
    """
    Analyze emotions in YouTube comments and return visualization
    """
    try:
        return await asyncio.wait_for(
            _perform_analysis(video_url, max_comments, chart_type),
            timeout=25.0
        )
    except asyncio.TimeoutError:
        logger.error("Analysis timed out")
        raise HTTPException(status_code=504, detail="Analysis timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

async def _perform_analysis(video_url: str, max_comments: int, chart_type: str):
    """Core analysis logic"""
    logger.info(f"Starting analysis for video: {video_url}")
    
    video_id = extract_video_id(video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    youtube = YouTubeService()
    preprocessor = TextPreprocessor()
    analyzer = EmotionAnalyzer()
    visualizer = VisualizationService()

    # Obtener comentarios
    comments = youtube.extract_comments(video_id, min(max_comments, 50))
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video")

    # Procesar comentarios
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
    video_details = youtube.get_video_details(video_id)
    title = f"Emotions in comments for: {video_details.get('title', video_id)}" if video_details else None

    # Generar gráfico en memoria
    with NamedTemporaryFile(delete=True, suffix='.png') as tmp_file:
        chart_path = tmp_file.name
        
        try:
            if chart_type.lower() == "pie":
                visualizer.create_pie_chart(df, title, output_path=chart_path)
            else:
                visualizer.create_emotion_distribution_plot(df, title, output_path=chart_path)
            
            with open(chart_path, 'rb') as f:
                image_data = f.read()
            
            logger.info(f"Analysis completed for video: {video_id}")
            return Response(content=image_data, media_type='image/png')
        
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating visualization")

def extract_video_id(url: str) -> Optional[str]:
    """Extrae el ID de un video de YouTube desde su URL"""
    if not url or not isinstance(url, str):
        return None
        
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\/]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^\/\?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return JSONResponse(
        content={
            "message": "YouTube Emotion Detection API",
            "endpoints": {
                "/api/analyze": {
                    "description": "Analyze YouTube comments",
                    "parameters": {
                        "video_url": "string (required)",
                        "max_comments": "int (optional, default: 50)",
                        "chart_type": "string (optional, 'bar' or 'pie')"
                    }
                },
                "/api/health": "GET - Service health check",
                "/api/docs": "Interactive API documentation",
                "/api/redoc": "Alternative API documentation"
            }
        }
    )