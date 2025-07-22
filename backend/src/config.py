import os

from dotenv import load_dotenv

load_dotenv('backend.env')

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")