"""
UI components for the Advanced RAG application with Chat History
"""
import streamlit as st
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE, 
     UPLOAD_PLACEHOLDER_TITLE, UPLOAD_PLACEHOLDER_TEXT
)
from utils import format_file_size, list_stored_files

def setup_page_config():
    """Sets up Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )


def render_header():
    """Shows the main header section"""
    st.title(f"{PAGE_ICON} HR_Onboarding Assistant")
    

def render_uploaded_files():
    """Sidebar UI with expandable stored file list"""
    with st.sidebar:

        # Expandable stored files section
        stored_files = list_stored_files()
        with st.expander(f"💾 Stored Files ({len(stored_files)})", expanded=False):
            if stored_files:
                st.markdown("""
                <div style="
                    max-height: 200px;
                    overflow-y: auto;
                    background: #f8f9fa;
                    padding: 0.5rem;
                    border-radius: 8px;
                    font-size: 0.85rem;
                    line-height: 1.4;
                ">
                """, unsafe_allow_html=True)

                for f in stored_files:
                    st.markdown(f"📄 {f}")

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="font-size: 0.85rem; color: gray;">No files currently stored.</p>',
                    unsafe_allow_html=True
                )


def render_upload_section(document_loader):
    """Shows the document upload section"""
    with st.sidebar:
        st.markdown("## 📤 Document Upload")
        
        # Upload area with simple styling
        st.info("📁 **Drag & Drop Your Document**\n\nSupported: PDF, png, jpg, jpeg, tiff, bmp")
        
        # File uploader
        user_file = st.file_uploader(
            "Choose a file", 
            type=document_loader.get_supported_extensions(),
            help="Upload any supported document type.",
            label_visibility="collapsed"
        )
        
        return user_file


def render_file_analysis(file_info):
    """Shows file analysis metrics"""
    st.markdown("### 📊 File Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📄 Filename**")
        st.write(file_info['filename'])
    
    with col2:
        st.markdown("**📏 Size**")
        size_display = format_file_size(file_info['size'])
        st.write(size_display)
    
    with col3:
        st.markdown("**🏷️ Type**")
        st.write(f".{file_info['extension'].upper()}")
    
    with col4:
        st.markdown("**📋 Status**")
        status_icon = "✅" if file_info['is_supported'] else "❌"
        status_text = "Supported" if file_info['is_supported'] else "Unsupported"
        st.write(f"{status_icon} {status_text}")


def render_upload_placeholder():
    """Shows placeholder when no file is uploaded"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem; background: #f8fafc; border-radius: 10px; margin: 2rem 0;">
        <h3>{UPLOAD_PLACEHOLDER_TITLE}</h3>
        <p>{UPLOAD_PLACEHOLDER_TEXT}</p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_history():
    """Renders the chat history"""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return
    
    st.markdown("### 💬 Chat History")
    
    # Display each Q&A pair
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            # User question
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🙋 You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # AI answer
            st.markdown(f"""
            <div style="background: #f1f8e9; padding: 1rem; border-radius: 10px; margin: 0.5rem 0 1.5rem 0;">
                <strong>🤖 Assistant:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)


def render_question_section(user_file=None):
    """Renders the question section with current document info"""
    st.markdown("### ❓ Ask a Question")
    
    # Show chat history first
    render_chat_history()
    
    # Show evaluation section (if available)
    #if 'latest_evaluation' in st.session_state and st.session_state.latest_evaluation:
    #    render_evaluation_section(st.session_state.latest_evaluation)
    
    # Question input at the bottom
    question = st.text_input(
        "Your question:",
        key="question_input",
        placeholder="Type your question here..."
    )
    
    col1, col2 = st.columns([6, 1])
    with col1:
        ask_button = st.button("Ask", use_container_width=True, type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    return question, ask_button

def render_evaluation_section(result):
    """Render the evaluation metrics section only evaluation nodes are present (fully expandable)."""
    if not result:
        return

    with st.expander("📊 Evaluation Results", expanded=False):

        st.markdown("### 🧩 Summary Metrics")

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

        # Divider before detailed results
        st.markdown("---")
        st.markdown("### 🔧 Detailed Evaluation Results")

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
                eval_df = pd.DataFrame(
                    eval_data,
                    columns=["Document", "Score", "Relevance", "Coverage", "Missing Info"]
                )
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
