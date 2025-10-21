"""
UI components for the Advanced RAG application
"""
import streamlit as st
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE, 
    FILE_CATEGORIES, UPLOAD_PLACEHOLDER_TITLE, UPLOAD_PLACEHOLDER_TEXT
)
from utils import format_file_size , list_stored_files

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
    st.subheader("Intelligent Document Search & Analysis")
    

def render_sidebar(document_loader):
    """Sidebar UI with expandable stored file list"""
    with st.sidebar:
        # App info
        st.markdown("""
        <div style="
            background: white;
            padding: 0.8rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <h4 style="margin-bottom: 0.5rem;">🔍 Advanced RAG System</h4>
            <p style="font-size: 0.9rem; margin: 0;">Upload company documents and ask intelligent questions using LLM-powered retrieval.</p>
        </div>
        """, unsafe_allow_html=True)

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
        
        # Show current supported extensions
        #with st.expander("ℹ️ View All Supported Formats", expanded=False):
        #    col1, col2 = st.columns(2)
        #    with col1:
        #        st.write(f"**Supported extensions:** {document_loader.get_supported_extensions_display()}")
        #    with col2:
        #        st.write(f"**Total formats:** {len(document_loader.get_supported_extensions())}")
        
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


def render_question_section(user_file):
    """Shows the question input section"""
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Display current file info
    file_display = f"📄 **Current Document:** {user_file.name}"
    if hasattr(user_file, 'type') and user_file.type:
        file_display += f" ({user_file.type})"
    st.markdown(file_display)
    
    # Question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            'Enter your question:', 
            placeholder="What is the main topic of this document?",
            disabled=not user_file,
            help="Ask any question about the content of your uploaded document"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        ask_button = st.button("Ask", use_container_width=True)
    
    return question, ask_button


def render_answer_section(result):
    """Shows the answer section"""
    st.markdown("### 📝 Answer")
    st.success(result['solution'])
    st.markdown("---")
