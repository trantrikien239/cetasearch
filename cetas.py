# A search interface for ceta search using streamlit
# import pandas as pd 
# import numpy as np
import json
import os


import streamlit as st
from openai import OpenAI

from semantic_search_engine import SemanticSearchEngine
from utils import load_search_engine, load_data

# Wide mode
st.set_page_config(layout="wide")

# Add a sidebar
sb = st.sidebar
sb.title("Cetasearch")


# A search bar
st.title("Dolphinately the best search engine for marine life")
c1, c2 = st.columns([2, 3])
query = c1.text_input("Ask me whalever you want")

se = load_search_engine(use_local=False)
print("Search engine loaded")

if query:
    c1.markdown("*Hold on tight, I'm diving deep into the ocean to find the answer"
    " to your question*")
    anno_gen_text, df_top_paragraphs, logs = se.generate_answer(query, k=3, max_tokens=512)
    c1.write(anno_gen_text)
    c2.table(df_top_paragraphs[["paragraph"]])
    for step, duration in logs:
        sb.markdown(f"🟢 **[{duration:.2f} s] {step}**")
