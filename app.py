import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🔥 Interview Summary Generator")

uploaded_file = st.file_uploader("Upload (.txt)", type="txt")

if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
    st.success(f"📄 Loaded {len(content):,} chars")
    
    # Simple chunking
    chunks = [content[i:i+300] for i in range(0, len(content), 250)]
    
    with st.spinner("Finding topics..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks)
        
        n_clusters = min(6, max(3, len(chunks)//4))
        kmeans = KMeans(n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        topics = {}
        for i in range(n_clusters):
            topic_chunks = [chunks[j] for j, c in enumerate(clusters) if c == i]
            if topic_chunks:
                topics[f"Topic {i+1} ({len(topic_chunks)} mentions)"] = topic_chunks[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Topics Found", len(topics))
    with col2:
        st.metric("Total Chunks", len(chunks))
    
    for topic, example in topics.items():
        with st.expander(topic):
            st.write(example[:400] + "..." if len(example) > 400 else example)
    
    st.balloons()