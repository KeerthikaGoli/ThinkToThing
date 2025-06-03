import streamlit as st
import os
from PIL import Image
import json
from datetime import datetime

from llm.llm_handler import LLMHandler
from memory.memory_handler import MemoryHandler
from pipeline.pipeline_handler import PipelineHandler
from openfabric_pysdk.utility import Stub


# Page config
st.set_page_config(
    page_title="Creative AI Partner",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .creation-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    .gallery-image {
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'reference_id' not in st.session_state:
    st.session_state.reference_id = None

# Initialize handlers
@st.cache_resource
def init_handlers():
    config = {
        "app_ids": [
            "f0997a01-d6d3-a5fe-53d8-561300318557",
            "69543f29-4d41-4afc-7f29-3d51591f11eb"
        ],
        "llm_model": "deepseek-ai/deepseek-coder-6.7b-base",
        "memory_type": "vector"
    }
    
    stub = Stub(config["app_ids"])
    llm_handler = LLMHandler(config["llm_model"])
    memory_handler = MemoryHandler(config["memory_type"])
    pipeline = PipelineHandler(stub, llm_handler, memory_handler, config)
    
    return pipeline

pipeline = init_handlers()

# Sidebar
with st.sidebar:
    st.title("🎨 Creative AI Partner")
    st.markdown("---")
    
    # Session info
    if st.session_state.session_id:
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    
    # Reference creation
    if st.session_state.reference_id:
        st.success(f"Reference Creation: {st.session_state.reference_id[:8]}...")
        if st.button("Clear Reference"):
            st.session_state.reference_id = None
            st.rerun()
    
    st.markdown("---")
    
    # Gallery
    st.subheader("📚 Your Creations")
    if st.session_state.history:
        for creation in reversed(st.session_state.history[-5:]):
            with st.expander(f"Creation {creation['creation_id'][:8]}..."):
                st.write(f"**Prompt:** {creation['prompt']}")
                if os.path.exists(creation['image_path']):
                    st.image(creation['image_path'], use_column_width=True)
                if st.button("Use as Reference", key=creation['creation_id']):
                    st.session_state.reference_id = creation['creation_id']
                    st.rerun()

# Main content
st.title("🎨 Create Something Magical")

# Input form
with st.form("creation_form"):
    prompt = st.text_area(
        "What would you like to create?",
        placeholder="e.g., A glowing dragon standing on a cliff at sunset..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.form_submit_button("🚀 Create Magic!")
    with col2:
        find_similar = st.form_submit_button("🔍 Find Similar")

# Process creation
if submit and prompt:
    with st.spinner("🎨 Creating your masterpiece..."):
        result = pipeline.process_creation(
            prompt=prompt,
            session_id=st.session_state.session_id,
            reference_id=st.session_state.reference_id
        )
        
        # Update session state
        st.session_state.session_id = result['session_id']
        st.session_state.history.append(result)
        
        # Display result
        st.success("✨ Creation complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼 Generated Image")
            st.image(result['image_path'], use_column_width=True)
        with col2:
            st.subheader("🎮 3D Model")
            st.markdown(f"Download the 3D model: [{result['creation_id']}.glb]({result['model_path']})")
            st.json(result['metadata'])

# Find similar creations
if find_similar and prompt:
    with st.spinner("🔍 Finding similar creations..."):
        similar = pipeline.find_similar_creations(prompt)
        
        if similar:
            st.subheader("🔍 Similar Creations")
            for item in similar:
                with st.container():
                    st.markdown(f"**Prompt:** {item['prompt']}")
                    if os.path.exists(item.get('image_path', '')):
                        st.image(item['image_path'], use_column_width=True)
                    if st.button("Use as Reference", key=f"similar_{item['id']}"):
                        st.session_state.reference_id = item['id']
                        st.rerun()
        else:
            st.info("No similar creations found.")
