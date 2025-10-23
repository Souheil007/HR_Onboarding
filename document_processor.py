"""
Document processing module for the Advanced RAG application
"""
import streamlit as st
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from utils import get_file_key
from ui_components import render_file_analysis
from sentence_transformers import CrossEncoder
from typing import List

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


class DocumentProcessor:
    """Processes documents and creates embeddings for the vector database"""
    
    def __init__(self, document_loader):
        self.document_loader = document_loader
        self.embedding_function = embedding_function
        self._chroma_db = None
        self._bm25_retriever = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",device="cpu" )
    def process_file(self, user_file):
        """
        Processes an uploaded file and creates embeddings
        Returns retriever or None if processing fails
        """
        if user_file is None:
            return None
        
        # Check if file already processed
        current_file_key = get_file_key(user_file)
        if st.session_state.get('processed_file') == current_file_key:
            st.info(f"💡 The file '{user_file.name}' is already uploaded.")
            return st.session_state.get('retriever')
        
        try:
            return self._process_new_file(user_file, current_file_key)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please make sure your file is in a supported format and try again.")
            return None
    
    def _process_new_file(self, user_file, current_file_key):
        """Processes a new file that hasn't been processed before"""
        # Get file info and display analysis
        file_info = self.document_loader.get_upload_info(user_file)
        render_file_analysis(file_info)
        
        # Check if file type is supported
        if not file_info['is_supported']:
            st.error(f"❌ Unsupported file type: .{file_info['extension']}")
            st.info(f"📋 Supported formats: {self.document_loader.get_supported_extensions_display()}")
            return None
        
        # Process the file
        return self._execute_processing_pipeline(user_file, file_info, current_file_key)
    
    def _execute_processing_pipeline(self, user_file, file_info, current_file_key):
        st.markdown("### 🔄 Processing Status")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load and split
            status_text.text("🔄 Loading document...")
            progress_bar.progress(25)
            documents = self.document_loader.load_uploaded_file(user_file)

            status_text.text("🔍 Extracting content...")
            progress_bar.progress(50)
            st.success(f"✅ Successfully extracted content from {file_info['filename']}")

            status_text.text("✂️ Splitting into chunks...")
            progress_bar.progress(75)
            doc_splits = self._create_document_chunks(documents, current_file_key)

            status_text.text("🧠 Creating embeddings...")
            progress_bar.progress(90)
            self._chroma_db = self._create_vector_database(doc_splits, current_file_key)
            self._bm25_retriever = BM25Retriever.from_documents(
                doc_splits, k=5, preprocess_func=lambda text: text.split()
            )

            # Store retriever in session
            st.session_state.processed_file = current_file_key
            st.session_state.retriever = self.hybrid_search

            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            return self.hybrid_search

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            raise e
    
    def _create_document_chunks(self, documents, current_file_key):
        """Splits documents into smaller chunks and adds metadata"""
        # Ensure all are Document objects
        documents = [
            doc if isinstance(doc, Document) else Document(page_content=doc, metadata={})
            for doc in documents
        ]

        # Initialize splitters
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on, 
            return_each_line=False,
            strip_headers=False  # Keep headers in the content
        )
        
        # Fallback splitter for documents without markdown headers
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Split each document separately and keep metadata
        expanded_docs = []

        for idx, doc in enumerate(documents):
            # Clean text but PRESERVE newlines for markdown headers
            text = doc.page_content.replace("\u200b", "")  # Remove zero-width spaces
            # DON'T collapse all whitespace - keep newlines for markdown
            text = text.replace("\r\n", "\n")  # Normalize line endings
            # Only collapse multiple spaces on the same line
            import re
            text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
            text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines → double newline

            print(f"\n--- Processing document {idx} ---")
            print(f"First 200 chars: {text[:200]}")

            # Try markdown splitting first
            chunks = []
            try:
                chunks = markdown_splitter.split_text(text)
                print(f"Markdown splitter produced {len(chunks)} chunks")
                
                # If no chunks or all empty, fall back to recursive splitter
                if not chunks:
                    print(f"No chunks from markdown splitter, using recursive splitter")
                    chunks = recursive_splitter.split_text(text)
                    if chunks and isinstance(chunks[0], str):
                        chunks = [Document(page_content=chunk) for chunk in chunks]
                else:
                    print(f"Successfully split by markdown headers!")
                    
            except Exception as e:
                print(f"Markdown splitting failed: {e}, using recursive splitter")
                chunks = recursive_splitter.split_text(text)
                if chunks and isinstance(chunks[0], str):
                    chunks = [Document(page_content=chunk) for chunk in chunks]

            # Extract text content from chunks
            chunk_texts = []
            for chunk in chunks:
                if isinstance(chunk, Document):
                    chunk_texts.append(chunk.page_content)
                elif isinstance(chunk, str):
                    chunk_texts.append(chunk)
                else:
                    chunk_texts.append(str(chunk))

            print(f"Document {idx}: Created {len(chunk_texts)} chunks")

            # Expand chunks up to 3 consecutive parts
            for i in range(len(chunk_texts)):
                for window_size in range(1, 4):  # 1 to 3 consecutive chunks
                    end = i + window_size
                    if end <= len(chunk_texts):
                        # Join with newlines to preserve structure
                        combined_text = "\n\n".join(chunk_texts[i:end])

                        # Skip empty chunks
                        if not combined_text.strip():
                            continue

                        metadata = doc.metadata.copy()
                        metadata.update({
                            "chunk_id_start": i,
                            "chunk_id_end": end - 1,
                            "window_size": window_size,
                            "total_chunks": len(chunk_texts),
                            "chunk_size": len(combined_text),
                            "file_key": current_file_key
                        })

                        expanded_docs.append(Document(page_content=combined_text, metadata=metadata))

        print(f"Total expanded chunks created: {len(expanded_docs)}")
        
        if not expanded_docs:
            raise ValueError("No document chunks were created. The document may be empty or improperly formatted.")
        
        return expanded_docs



    def _create_vector_database(self, doc_splits, current_file_key):
        """Creates a ChromaDB vector database from document chunks, avoids duplicates"""
        """Creates a ChromaDB vector database from document chunks"""
        chroma_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )

        # Check for duplicates based on file_key
        existing_docs = chroma_db.get(include=["metadatas"])
        existing_keys = [meta.get("file_key") for meta in existing_docs["metadatas"]]

        if current_file_key in existing_keys:
            st.info("This file is already in the database. Skipping embedding creation.")
            return chroma_db
        else:
            return Chroma.from_documents(
                documents=doc_splits, 
                collection_name=CHROMA_COLLECTION_NAME, 
                embedding=self.embedding_function,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
    def hybrid_search(self, query: str, top_k: int = 10,top_reranked = 5 ,weights=(0.7, 0.3),rerank: bool = True):
        """
        Self-contained hybrid search combining Chroma semantic search and BM25 keyword search.
        No need to pass retrievers manually.
        """
        if self._chroma_db is None or self._bm25_retriever is None:
            raise ValueError("No retrievers available. Process a file first.")

        # --- Step 1: Retrieve from both retrievers ---
        semantic_retriever = self._chroma_db.as_retriever(search_kwargs={"k": top_k})
        semantic_docs = semantic_retriever.invoke(query)
        bm25_docs_list = self._bm25_retriever.invoke(query)[:top_k]

        # --- Step 2: Combine results using weights ---
        combined = {}
        for doc in semantic_docs:
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            combined[key] = {
                "score": combined.get(key, {"score": 0})["score"] + weights[0],
                "doc": doc
            }

        for doc in bm25_docs_list:
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            combined[key] = {
                "score": combined.get(key, {"score": 0})["score"] + weights[1],
                "doc": doc
            }

        # --- Step 3: Sort combined docs ---
        sorted_combined = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        top_candidates = [entry["doc"] for entry in sorted_combined[:top_reranked]]

        # Step 4: Remove redundant chunks (where one chunk contains another)
        top_candidates = self._remove_redundant_chunks(top_candidates)
        print(f"After removing redundant chunks: {len(top_candidates)} documents remain")

        # Step 5: Optional reranking (only if reranker is available)
        if rerank and top_candidates and self.reranker is not None:
            try:
                pairs = [(query, doc.page_content) for doc in top_candidates]
                scores = self.reranker.predict(pairs)
                reranked = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in reranked]
            except Exception as e:
                print(f"Reranking failed: {e}, returning hybrid results")
                return top_candidates

        # Step 6: Return hybrid-only results
        return top_candidates
    
    def _remove_redundant_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Remove redundant chunks where one chunk's content is fully contained in another.
        Keeps the larger chunk when there's containment.
        """
        if not documents:
            return documents
        
        # Sort by content length (longest first)
        sorted_docs = sorted(documents, key=lambda d: len(d.page_content), reverse=True)
        
        non_redundant = []
        
        for i, doc in enumerate(sorted_docs):
            is_redundant = False
            doc_content = doc.page_content.strip()
            
            # Check if this document is contained in any larger document we've already kept
            for kept_doc in non_redundant:
                kept_content = kept_doc.page_content.strip()
                
                # If current doc is fully contained in a kept doc, it's redundant
                if doc_content in kept_content and doc_content != kept_content:
                    is_redundant = True
                    print(f"Removing redundant chunk (size {len(doc_content)}) - contained in larger chunk (size {len(kept_content)})")
                    break
            
            if not is_redundant:
                non_redundant.append(doc)
        
        return non_redundant