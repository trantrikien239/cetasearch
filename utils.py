import os

import pandas as pd
import numpy as np
from openai import OpenAI

from semantic_search_engine import SemanticSearchEngine
import streamlit as st
from elasticsearch import Elasticsearch

@st.cache_data
def load_data(path, filetype="parquet"):
    if filetype == "parquet":
        output = pd.read_parquet(path)
    elif filetype == "npy":
        output = np.load(path)
    return output

@st.cache_resource(show_spinner="Hold on tight")
def load_search_engine(use_local=False, _es_client=None):
    df_titles = load_data("static/data_ocean/titles.parquet")
    df_paragraphs = load_data("static/data_ocean/paragraphs.parquet")
    emb_header = load_data("static/data_ocean/header_emb.npy", filetype="npy")
    emb_paragraph = load_data("static/data_ocean/paragraph_emb.npy", filetype="npy")
    
    try:
        # Include your OPENAI_API_KEY in the environment variables
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    except:
        print("OPENAI_API_KEY key not found")

    if use_local:
        # llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        llm_client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="lm-studio")
        model_id = "llama-3.2-1b-instruct"
    else:
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
        model_id = "gpt-4o-mini"

    se = SemanticSearchEngine(
        df_titles, df_paragraphs,
        emb_header=emb_header, emb_paragraph=emb_paragraph,
        llm_client=llm_client, model_id=model_id,
        _es_client=_es_client
        )
    return se

@st.cache_resource(show_spinner="Getting elasticsearch client")
def load_es_client():
    # password is mysecurepassword
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "mysecurepassword"),
        verify_certs=False
        )
    return es