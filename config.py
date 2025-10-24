"""
Configuration settings for the Advanced RAG application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# UI Configuration
PAGE_TITLE = "HR Onboarding Assistant"
PAGE_ICON = "🔎"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# File Processing Configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
CHROMA_COLLECTION_NAME = "rag-chroma"
CHROMA_PERSIST_DIR = "./.chroma"

# Model Configuration
LLM_TEMPERATURE = 0

# Supported File Types
SUPPORTED_EXTENSIONS =  ["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]

# UI Messages
UPLOAD_PLACEHOLDER_TITLE = "📤 Upload a document to get started"
UPLOAD_PLACEHOLDER_TEXT = "Once you upload a file, you'll be able to ask questions about its content."
QUESTION_PLACEHOLDER = "What is the main topic of this document?"

