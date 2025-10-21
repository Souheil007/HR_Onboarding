"""
Document processing module for the Advanced RAG application
"""
import streamlit as st
import time
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from utils import get_file_key
from ui_components import render_file_analysis

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class DocumentProcessor:
    """Processes documents and creates embeddings for the vector database"""
    
    def __init__(self, document_loader):
        self.document_loader = document_loader
        self.embedding_function = embedding_function
    
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
        """Runs the complete processing pipeline"""
        st.markdown("### 🔄 Processing Status")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load document
            status_text.text("🔄 Loading document...")
            progress_bar.progress(25)
            documents = self.document_loader.load_uploaded_file(user_file)
            
            # Step 2: Extract content
            status_text.text("🔍 Extracting content...")
            progress_bar.progress(50)
            st.success(f"✅ Successfully extracted content from {file_info['filename']}")
            
            # Step 3: Split into chunks
            progress_bar.progress(75)
            status_text.text("✂️ Splitting into chunks...")
            doc_splits = self._create_document_chunks(documents, current_file_key)
    
            # Step 4: Create embeddings
            progress_bar.progress(90)
            status_text.text("🧠 Creating embeddings...")
            chroma_db = self._create_vector_database(doc_splits, current_file_key)

            
            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Clean up UI
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Store in session state
            retriever = chroma_db.as_retriever()
            st.session_state.processed_file = current_file_key
            st.session_state.retriever = retriever
            
            # Debug: Confirm retriever creation and test it
            print(f"Retriever created successfully: {retriever is not None}")
            print(f"Session state updated with file key: {current_file_key}")
            
            # Test the retriever with a simple query
            try:
                test_docs = retriever.invoke("test")
                print(f"Retriever test successful - found {len(test_docs)} documents")
            except Exception as test_error:
                print(f"Retriever test failed: {test_error}")
            
            return retriever
            
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