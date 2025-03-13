from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import json
import os
from semantic_search_engine import SemanticSearchEngine
from openai import OpenAI

try:
    # Include your OPENAI_API_KEY in the environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
except:
    print("OPENAI_API_KEY key not found")

USE_LOCAL = False
if USE_LOCAL:
    # llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    llm_client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="llama.cpp")
    model_id = "llama-3.2-1b-instruct"
else:
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    model_id = "gpt-4o-mini"


app = Flask(__name__, static_folder="static")

TMP_QUERY_PATH = "static/tmp/curr_query.txt"
TMP_ANSWER_PATH = "static/tmp/curr_answer.json"

# Import wiki datasources (processed)
df_titles = pd.read_parquet("static/data_ocean/titles.parquet")
df_paragraphs = pd.read_parquet("static/data_ocean/paragraphs.parquet")
emb_header = np.load("static/data_ocean/header_emb.npy")
emb_paragraph = np.load("static/data_ocean/paragraph_emb.npy")

se = SemanticSearchEngine(
    df_titles, df_paragraphs,
    emb_header=emb_header, emb_paragraph=emb_paragraph,
    llm_client=llm_client, model_id=model_id
    )

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def collect_form_data():
    if os.path.exists(TMP_QUERY_PATH):
        os.remove(TMP_QUERY_PATH)

    query = request.form.get("Query")
    with open(TMP_QUERY_PATH, "w") as f:
        f.write(query)

    return render_template("search.html")

@app.route('/predict', methods=['POST'])
def generate_predictions():
    with open(TMP_QUERY_PATH, "r") as f:
        query = f.readlines()[0]
    anno_gen_text, df_top_paragraphs = se.generate_answer(
        query, k=3, max_tokens=256)

    curr_answer = {
        "query": query,
        "answer": anno_gen_text.split("\n"),
        "source": []
        }
    for idx in df_top_paragraphs.index:
        source_i = {
            "idx": idx,
            "title": str(df_top_paragraphs.loc[idx, "title"]),
            "paragraph": str(df_top_paragraphs.loc[idx, "paragraph"])}
        curr_answer["source"].append(source_i)
    
    with open(TMP_ANSWER_PATH, "w") as outfile:
        json.dump(curr_answer, outfile) 

    return ""


if __name__ == '__main__':
    app.run(debug=True)