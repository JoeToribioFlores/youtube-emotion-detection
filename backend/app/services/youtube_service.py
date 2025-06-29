import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
from typing import List, Dict, Optional

from app.config import Config

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.API_KEY
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API connection established")
        except Exception as e:
            logger.error(f"Error connecting to YouTube API: {str(e)}")
            raise

    def extract_comments(self, video_id: str, max_results: int = None) -> List[Dict]:
        max_results = max_results or Config.MAX_COMMENTS
        comments = []

        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText"
            )

            with tqdm(total=max_results, desc="Extracting comments") as pbar:
                while request and len(comments) < max_results:
                    response = request.execute()

                    for item in response['items']:
                        snippet = item['snippet']['topLevelComment']['snippet']
                        comment = {
                            'author': snippet['authorDisplayName'],
                            'comment': snippet['textDisplay'],
                            'date': snippet['publishedAt'],
                            'likes': snippet.get('likeCount', 0),
                            'id': item['id']
                        }
                        comments.append(comment)
                        pbar.update(1)

                        if len(comments) >= max_results:
                            break

                    if 'nextPageToken' in response and len(comments) < max_results:
                        request = self.youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            pageToken=response['nextPageToken'],
                            maxResults=100,
                            textFormat="plainText"
                        )
                    else:
                        request = None

        except HttpError as e:
            logger.error(f"YouTube API HTTP error: {str(e)}")
            if "quotaExceeded" in str(e):
                logger.error("API quota exceeded. Try tomorrow or use another API key.")
        except Exception as e:
            logger.error(f"Unexpected error extracting comments: {str(e)}")

        logger.info(f"Extracted {len(comments)} comments from video {video_id}")
        return comments

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        try:
            response = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()

            if response['items']:
                video_data = response['items'][0]
                return {
                    'title': video_data['snippet']['title'],
                    'channel': video_data['snippet']['channelTitle'],
                    'published_at': video_data['snippet']['publishedAt'],
                    'view_count': video_data['statistics'].get('viewCount', 0),
                    'like_count': video_data['statistics'].get('likeCount', 0),
                    'comment_count': video_data['statistics'].get('commentCount', 0)
                }
            return None
        except Exception as e:
            logger.error(f"Error getting video details: {str(e)}")
            return None