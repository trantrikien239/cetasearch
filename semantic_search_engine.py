from striprtf.striprtf import rtf_to_text
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
pd.set_option('display.max_colwidth', None)
from time import time
from sentence_transformers import SentenceTransformer
from IPython.display import display

class SemanticSearchEngine(object):
    def __init__(self, df_header, df_paragraph) -> None:
        self.df_header = df_header
        self.df_paragraph = df_paragraph

        self.smodel = SentenceTransformer('multi-qa-distilbert-cos-v1')

        print("==== Loading header embeddings ====")
        start_ = time()
        self.header_emb = self.smodel.encode(self.df_header["header"].values, show_progress_bar=True)
        print(f"Loading header embeddings: {time() - start_:.2f} (s)")

        print("==== Loading paragraph embeddings ====")
        start_ = time()
        self.paragraph_emb = self.smodel.encode(self.df_paragraph["paragraph"].values, show_progress_bar=True)
        print(f"Loading paragraph embeddings: {time() - start_:.2f} (s)")
    
    def search_paragraph(self, query, k=3):
        query_emb = self.smodel.encode(query)
        scores = self.paragraph_emb @ query_emb
        arg_sort = np.argsort(scores)[::-1]
        top_idx = arg_sort[:k]
        top_score = scores[arg_sort][:k]
        tb_desc_tmp = self.paragraph_emb.loc[top_idx,:].copy()
        tb_desc_tmp["score"] = top_score
        return tb_desc_tmp
            
    # def search_schema(self, query, k=3):
    #     tb_desc_tmp = self.search_table(query=query, k=k)
    #     for tb_name in tb_desc_tmp["name"]:
    #         print(f"==== {tb_name} ====")
    #         display(self.schemas.get(tb_name))
            