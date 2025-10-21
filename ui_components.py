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


def render_question_section(user_files):
    """Renders the question section with current document info"""
    
    if not user_files:
        return "", None  # nothing uploaded

    # If multiple files uploaded, take the first one for display
    if isinstance(user_files, list):
        first_file = user_files[0]
    else:
        first_file = user_files

    file_display = f"📄 **Current Document:** {first_file.name}"
    st.markdown(file_display)

    question = st.text_input("Ask a question about the document:")
    ask_button = st.button("Ask")
    
    return question, ask_button

def render_answer_section(result):
    """Shows the answer section"""
    st.markdown("### 📝 Answer")
    st.success(result['solution'])
    st.markdown("---")
