# YouTube Emotion Analysis API

## Deploy en Vercel

1. Conecta tu repositorio GitHub a Vercel
2. Asegúrate que las variables de entorno estén configuradas:
   - YOUTUBE_API_KEY
   - OTRA_VARIABLE
3. El despliegue se activará automáticamente

## Desarrollo local
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload