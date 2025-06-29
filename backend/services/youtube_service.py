import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not configured")
        
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API client initialized")
        except Exception as e:
            logger.error(f"Error initializing YouTube API: {str(e)}")
            raise

    def extract_comments(self, video_id: str, max_results: int = 50) -> List[Dict]:
        """Extrae comentarios de YouTube con manejo optimizado de memoria"""
        comments = []
        if not video_id:
            return comments

        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results),
                textFormat="plainText"
            )

            while request and len(comments) < max_results:
                response = request.execute()
                
                for item in response.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'author': snippet['authorDisplayName'],
                        'comment': snippet['textDisplay'],
                        'date': snippet['publishedAt'],
                        'likes': snippet.get('likeCount', 0)
                    })
                    
                    if len(comments) >= max_results:
                        break

                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response.get('nextPageToken'),
                    maxResults=min(100, max_results - len(comments)),
                    textFormat="plainText"
                ) if response.get('nextPageToken') else None

        except HttpError as e:
            error_msg = f"YouTube API error: {str(e)}"
            if "quotaExceeded" in str(e):
                error_msg += " - Daily quota exceeded"
            logger.error(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

        return comments[:max_results]

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Obtiene detalles del video con caché básica"""
        if not video_id:
            return None

        try:
            response = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()

            if response.get('items'):
                item = response['items'][0]
                return {
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'view_count': item['statistics'].get('viewCount', 0),
                    'like_count': item['statistics'].get('likeCount', 0)
                }
        except Exception as e:
            logger.error(f"Error getting video details: {str(e)}")
        
        return None