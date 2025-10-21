"""
Multi-Format Document Loader for Advanced RAG System
Handles loading PDFs and images only, with OCR for images using Mistral OCR
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import logging
from mistralai import Mistral
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import sys
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ocr_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
# Supported file extensions
SUPPORTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp"}


class MultiFormatDocumentLoader:
    """Handles loading PDFs and image files (with OCR for images)"""
    
    def __init__(self):
        self.loaders = {
            "pdf": PyPDFLoader,
        }
        self.ocr_client = ocr_client

    def get_file_extension(self, file_path: Union[str, Path]) -> str:
        return Path(file_path).suffix[1:].lower()

    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        return self.get_file_extension(file_path) in SUPPORTED_EXTENSIONS

    def perform_ocr(self, file_path: str, model: str = "mistral-ocr-latest") -> Dict[str, Any]:
        """
        Perform OCR on an image file using the Mistral OCR client
        """
        if self.ocr_client is None:
            raise ValueError("OCR client is not provided")
        
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except FileNotFoundError:
            logger.error(f"The file '{file_path}' was not found.")
            return {"text": ""}

        # Upload file for OCR
        uploaded_file = self.ocr_client.files.upload(
            file={"file_name": file_path, "content": file_bytes},
            purpose="ocr"
        )

        file_signed_url = self.ocr_client.files.get_signed_url(file_id=uploaded_file.id)
        file_url = file_signed_url.url

        response = self.ocr_client.ocr.process(
            model=model,
            document={"type": "document_url", "document_url": file_url},
            include_image_base64=True
        )

        # Extract text from response
        return response

    def get_markdown_from_ocr(self,response):
        resp_dict = response.model_dump()
        pages = resp_dict.get("pages", [])
        # Return list of tuples (page_number, markdown_text)
        all_md = [(i+1, p.get("markdown", "")) for i, p in enumerate(pages) if p.get("markdown")]
        return all_md
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = self.get_file_extension(file_path)
        if not self.is_supported_format(file_path):
            raise ValueError(f"Unsupported file type: {extension}")

        logger.info(f"Loading document: {file_path} (format: {extension})")
        documents: List[Document] = []

        if extension == "pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            # Print the number of chunks/pages loaded
            print(f"Loaded {len(documents)} document chunks")

            # Inspect each chunk
            for i, doc in enumerate(documents):
                print(f"\n--- Document chunk {i+1} ---")
                print("Text snippet:", doc.page_content[:])  # show first 500 chars
                print("Metadata:", doc.metadata)
        else:
            # Image → perform OCR
            ocr_result = self.perform_ocr(str(file_path))
            md_text = self.get_markdown_from_ocr(ocr_result)
            # Combine all markdown text into a single string
            combined_md = "\n\n".join([md for _, md in md_text])
    
            print(f"OCR extracted text length: {len(combined_md)} characters and this is the text: {combined_md[:]}")
            documents.append(Document(page_content=combined_md))

        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": str(file_path),
                "file_type": extension,
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
            })

        logger.info(f"Loaded {len(documents)} document chunks from {file_path}")
        return documents

    def load_multiple_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        all_documents: List[Document] = []
        failed_files = []

        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                failed_files.append(str(file_path))

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        logger.info(f"Successfully loaded {len(all_documents)} document chunks")
        return all_documents

    def load_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> List[Document]:
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        pattern = "**/*" if recursive else "*"
        all_files = [f for f in directory_path.glob(pattern) if f.is_file() and self.is_supported_format(f)]
        logger.info(f"Found {len(all_files)} supported files in {directory_path}")

        return self.load_multiple_documents(all_files)

    def get_supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def get_document_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": "File not found"}

        extension = self.get_file_extension(file_path)
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_extension": extension,
            "file_size": file_path.stat().st_size,
            "is_supported": self.is_supported_format(file_path),
            "loader_type": self.loaders.get(extension, "OCR Loader").__name__ if extension in self.loaders else "OCR Loader"
        }
