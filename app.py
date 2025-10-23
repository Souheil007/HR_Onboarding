"""
Advanced RAG System with LangGraph - Chat History with Context Awareness

This is the main Streamlit application that demonstrates how to build a RAG system
using LangGraph for workflow management with full chat history support and context awareness.

Key components:
- Document processing and chunking
- LangGraph workflow for RAG operations with conversation history
- Question answering with fallback to online search
- Evaluation and quality assessment
- Real-time user interface with chat history
"""
import streamlit as st
import pandas as pd
# Local imports
from config import QUESTION_PLACEHOLDER
from utils import clear_chroma_db, initialize_session_state
from ui_components import (
    setup_page_config, render_header, render_uploaded_files, 
    render_upload_section, render_upload_placeholder,
    render_question_section, render_chat_history, list_stored_files
)
from document_loader import MultiModalDocumentLoader
from document_processor import DocumentProcessor
from rag_workflow import RAGWorkflow
from topic_router import TopicRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()
topic_router = TopicRouter()


def format_chat_history_for_context():
    """Format chat history into a context string for the AI"""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return ""
    
    # Get last 5 exchanges to avoid overwhelming the context
    recent_history = st.session_state.chat_history[-5:]
    
    context_parts = ["Previous conversation:"]
    for i, chat in enumerate(recent_history, 1):
        context_parts.append(f"\nQ{i}: {chat['question']}")
        context_parts.append(f"A{i}: {chat['answer']}")
    
    context_parts.append("\n\nCurrent question:")
    return "\n".join(context_parts)


def handle_question_processing(question):
    """Handle the Q&A processing workflow with conversation history"""
    # Debug info
    print(f"Processing question: {question}")
    # Detect topic
    topic = topic_router.detect_topic(question)
    print(f"Detected topic: {topic}")
    
    # Format the question with chat history context
    chat_context = format_chat_history_for_context()
    
    # Create enhanced question with context
    if chat_context:
        contextualized_question = f"{chat_context}\n{question}"
        print(f"Added chat history context to question")
    else:
        contextualized_question = question
    
    with st.container():
        with st.spinner(f'🧠 Analyzing your question in topic "{topic}" and retrieving relevant information...'):
            # Process the question with context - workflow will handle retriever automatically
            result = rag_workflow.process_question(contextualized_question)
        
        # Add to chat history (store original question, not contextualized)
        if result and 'solution' in result:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                'question': question,  # Store original question
                'answer': result['solution'],
                'metadata': result  # Store full result for detailed view
            })
        
        # Show success message
        st.success("✅ Answer generated! Check the chat history above.")
        
        st.session_state.latest_evaluation = result

def handle_user_interaction(user_file):
    """Handle user interactions for Q&A"""
    retriever_available = "retriever" in st.session_state

    if not retriever_available:
        # Check if there are stored files in ChromaDB
        stored_files = list_stored_files()
        if stored_files:
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"} 
            )
            chroma_db = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function,
                persist_directory=CHROMA_PERSIST_DIR
            )

            # Get chunks from Chroma for BM25
            chroma_docs_raw = chroma_db.get(include=['metadatas','documents'])['documents']
            doc_chunks = [Document(page_content=d['page_content'], metadata=d['metadata']) 
                          for d in chroma_docs_raw]

            if doc_chunks:
                # BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(doc_chunks, k=5, preprocess_func=lambda text: text.split())
                document_processor._bm25_retriever = bm25_retriever
                document_processor._chroma_db = chroma_db
                document_processor.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            

                st.session_state.retriever = document_processor.hybrid_search
                retriever_available = True
                print(f"Hybrid retriever initialized with {len(doc_chunks)} chunks")
            else:
                st.session_state.retriever = None
                print("No documents in ChromaDB, retriever not created")

    # If no file uploaded AND no retriever available, show placeholder
    if not user_file and not retriever_available:
        render_upload_placeholder()

    # Render question input and button (with chat history)
    question, ask_button = render_question_section(user_file)

    # Process question if submitted
    if ask_button and question.strip():
        handle_question_processing(question)
        st.rerun()
    elif ask_button and not question.strip():
        st.warning("Please enter a question before clicking Ask.")

def init_retriever_from_chroma():
    """Initialize retriever from ChromaDB if not already in session_state"""
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        chroma_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )

        # Fetch docs and metadata
        data = chroma_db.get(include=['documents','metadatas'])
        documents = data['documents']      # list of strings
        metadatas = data['metadatas']      # list of dicts

        # Convert to Document objects
        doc_chunks = [Document(page_content=text, metadata=meta) 
                      for text, meta in zip(documents, metadatas)]

        if doc_chunks:
            # BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(
                doc_chunks, k=5, preprocess_func=lambda text: text.split()
            )
            document_processor._bm25_retriever = bm25_retriever
            document_processor._chroma_db = chroma_db
            document_processor.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            st.session_state.retriever = document_processor.hybrid_search
            print(f"Hybrid retriever initialized from ChromaDB with {len(doc_chunks)} chunks")
        else:
            st.session_state.retriever = None
            print("No documents in ChromaDB, retriever not created")


def main():
    """Main application function"""
    # Initialize session state and clear DB only once
    initialize_session_state()
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Clear ChromaDB only on first run
    if 'db_cleared' not in st.session_state:
        clear_chroma_db()
        st.session_state.db_cleared = True
        print("ChromaDB cleared on app startup")
    
    # Setup page and render UI
    setup_page_config()
    render_header()
    
    # Handle file upload
    user_file = render_upload_section(document_loader)
    
    # Initialize retriever if no new upload
    init_retriever_from_chroma()
    
    # Show list of previously uploaded files in sidebar
    render_uploaded_files()
    
    # Process uploaded file
    if user_file:
        retriever = document_processor.process_file(user_file)
        if retriever:
            st.session_state.retriever = retriever
            print(f"File processed, retriever stored in session state")
        else:
            print(f"File processing failed - no retriever created")
    
    # Handle user interactions
    handle_user_interaction(user_file)


if __name__ == "__main__":
    main()