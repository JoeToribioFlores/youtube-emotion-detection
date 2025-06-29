import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional

from app.config import Config

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or Config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def create_emotion_distribution_plot(self, data: pd.DataFrame, title: str = None, filename: str = None) -> Optional[str]:
        try:
            plt.figure(figsize=(12, 8))
            
            if 'emotion' not in data.columns:
                if 'emocion' in data.columns:
                    data = data.rename(columns={'emocion': 'emotion'})
                else:
                    raise ValueError("No emotion column found in data")

            emotion_counts = data['emotion'].value_counts()
            ax = sns.barplot(x=emotion_counts.values, y=emotion_counts.index)

            plt.xlabel('Number of comments')
            plt.ylabel('Emotion')
            plt.title(title or 'Emotion Distribution in Comments')

            for i, v in enumerate(emotion_counts.values):
                ax.text(v + 0.5, i, str(v), va='center')

            plt.tight_layout()

            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_distribution_{timestamp}.png"

            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            logger.info(f"Emotion distribution plot saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return None

    def create_pie_chart(self, data: pd.DataFrame, title: str = None, filename: str = None) -> Optional[str]:
        try:
            plt.figure(figsize=(10, 10))
            
            if 'emotion' not in data.columns:
                if 'emocion' in data.columns:
                    data = data.rename(columns={'emocion': 'emotion'})
                else:
                    raise ValueError("No emotion column found in data")

            emotion_counts = data['emotion'].value_counts()
            plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(title or 'Emotion Distribution in Comments')

            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_pie_{timestamp}.png"

            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            logger.info(f"Emotion pie chart saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return None