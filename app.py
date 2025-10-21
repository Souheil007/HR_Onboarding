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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()


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
    
    # Format the question with chat history context
    chat_context = format_chat_history_for_context()
    
    # Create enhanced question with context
    if chat_context:
        contextualized_question = f"{chat_context}\n{question}"
        print(f"Added chat history context to question")
    else:
        contextualized_question = question
    
    with st.container():
        with st.spinner('🧠 Analyzing your question and retrieving relevant information...'):
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
        
        # Show evaluation scores and system information
        render_evaluation_section(result)


def render_evaluation_section(result):
    """Render the evaluation metrics section"""
    if not result:
        return
        
    st.markdown("---")
    st.markdown("### 📊 Latest Answer Evaluation")
    
    # Create summary table
    summary_data = []
    
    # Document Evaluations Summary
    if 'document_evaluations' in result and result['document_evaluations']:
        evaluations = result['document_evaluations']
        relevant_count = sum(1 for eval in evaluations if eval.score.lower() == 'yes')
        total_count = len(evaluations)
        summary_data.append(["📋 Document Relevance", f"{relevant_count}/{total_count} relevant"])
        
        # Show average relevance score if available
        if hasattr(evaluations[0], 'relevance_score'):
            avg_score = sum(eval.relevance_score for eval in evaluations) / len(evaluations)
            summary_data.append(["📊 Avg. Doc Relevance", f"{avg_score:.2f}"])
    
    # Question-Answer Match
    if 'question_relevance_score' in result:
        q_relevance = result['question_relevance_score']
        if hasattr(q_relevance, 'binary_score'):
            match_text = "✅ Well Matched" if q_relevance.binary_score else "❌ Poor Match"
            summary_data.append(["❓ Question Match", match_text])
        if hasattr(q_relevance, 'relevance_score'):
            summary_data.append(["📈 Question Score", f"{q_relevance.relevance_score:.2f}"])
        if hasattr(q_relevance, 'completeness'):
            summary_data.append(["📝 Completeness", q_relevance.completeness])
    
    # Document Relevance Grading
    if 'document_relevance_score' in result:
        doc_relevance = result['document_relevance_score']
        if hasattr(doc_relevance, 'binary_score'):
            grounding_text = "✅ Well Grounded" if doc_relevance.binary_score else "❌ Not Grounded"
            summary_data.append(["🎯 Answer Grounding", grounding_text])
        if hasattr(doc_relevance, 'confidence'):
            summary_data.append(["🔒 Confidence", f"{doc_relevance.confidence:.2f}"])
    
    # Display summary table
    if summary_data:
        df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
        st.table(df)
    
    # Show detailed evaluations in expandable section
    with st.expander("🔧 Detailed Evaluation Results"):
        
        # Document Evaluations Table
        if 'document_evaluations' in result and result['document_evaluations']:
            st.markdown("**📋 Document Evaluation Details:**")
            
            eval_data = []
            for i, eval in enumerate(result['document_evaluations']):
                row = [f"Document {i+1}", eval.score]
                
                if hasattr(eval, 'relevance_score'):
                    row.append(f"{eval.relevance_score:.2f}")
                else:
                    row.append("N/A")
                
                if hasattr(eval, 'coverage_assessment') and eval.coverage_assessment:
                    row.append(eval.coverage_assessment[:])
                else:
                    row.append("N/A")
                
                if hasattr(eval, 'missing_information') and eval.missing_information:
                    row.append(eval.missing_information[:])
                else:
                    row.append("N/A")
                
                eval_data.append(row)
            
            if eval_data:
                eval_df = pd.DataFrame(eval_data, columns=["Document", "Score", "Relevance", "Coverage", "Missing Info"])
                st.dataframe(eval_df, use_container_width=True)
        
        # Reasoning Table
        reasoning_data = []
        if 'question_relevance_score' in result and hasattr(result['question_relevance_score'], 'reasoning'):
            reasoning_data.append(["Question Relevance", result['question_relevance_score'].reasoning])
        
        if 'document_relevance_score' in result and hasattr(result['document_relevance_score'], 'reasoning'):
            reasoning_data.append(["Document Relevance", result['document_relevance_score'].reasoning])
        
        if reasoning_data:
            st.markdown("**🧠 Evaluation Reasoning:**")
            reasoning_df = pd.DataFrame(reasoning_data, columns=["Evaluation Type", "Reasoning"])
            st.dataframe(reasoning_df, use_container_width=True)


def handle_user_interaction(user_file):
    """Handle user interactions for Q&A"""
    # Always show the question section
    # If a retriever exists in session, use it; otherwise, try building from stored files
    retriever_available = "retriever" in st.session_state

    if not retriever_available:
        # Check if there are stored files in ChromaDB
        stored_files = list_stored_files()
        if stored_files:
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            chroma_db = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function,
                persist_directory=CHROMA_PERSIST_DIR
            )
            st.session_state.retriever = chroma_db.as_retriever()
            retriever_available = True

    # If no file uploaded AND no retriever available, show placeholder
    if not user_file and not retriever_available:
        render_upload_placeholder()

    # Render question input and button (now with chat history)
    question, ask_button = render_question_section(user_file)

    # Process question if submitted
    if ask_button and question.strip():
        handle_question_processing(question)
        # Clear the input by rerunning
        st.rerun()
    elif ask_button and not question.strip():
        st.warning("Please enter a question before clicking Ask.")


def init_retriever_from_chroma():
    """Initialize retriever from ChromaDB if not already in session_state"""
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        chroma_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        # Check if ChromaDB has any documents
        if chroma_db._collection.count() > 0:
            st.session_state.retriever = chroma_db.as_retriever()
            print(f"Retriever initialized from existing ChromaDB: {chroma_db._collection.count()} Chunks found")
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