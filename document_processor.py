"""
Document processing module for the Advanced RAG application
"""
import streamlit as st
import time

from langchain_text_splitters import CharacterTextSplitter
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
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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

        # Extract text for splitting
        texts = [doc.page_content for doc in documents]
        # Initialize splitter
        splitter = CharacterTextSplitter(
            chunk_size=500,   # characters
            chunk_overlap=50,
            separator=""  # split strictly by characters
        )

        # Split each document separately and keep metadata
        doc_splits = []
        for idx, text in enumerate(texts):
            text = text.replace("\n", " ").replace("\u200b", "")  # remove zero-width spaces
            text = " ".join(text.split())  # collapse multiple spaces

            chunks = splitter.split_text(text)  # returns list of strings
            for i, chunk in enumerate(chunks):
                metadata = documents[idx].metadata.copy()
                metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "file_key": current_file_key
                })
                doc_splits.append(Document(page_content=chunk, metadata=metadata))

        return doc_splits



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

        # --- Step 4: Optional reranking ---
        if rerank and top_candidates and hasattr(self, "reranker"):
            pairs = [(query, doc.page_content) for doc in top_candidates]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in reranked]

        # --- Step 5: Return hybrid-only results ---
        return top_candidates