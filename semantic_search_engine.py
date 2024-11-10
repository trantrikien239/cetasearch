import numpy as np
import pandas as pd
from time import time
from sentence_transformers import SentenceTransformer
import faiss

class SemanticSearchEngine(object):
    def __init__(self, 
                 df_header, df_paragraph,
                 llm_client, model_id="gpt-4o-mini",
                 emb_header=None, emb_paragraph=None,
                 ) -> None:
        self.df_header = df_header.reset_index(drop=True)[["title", "header"]]
        self.df_paragraph = df_paragraph.reset_index(drop=True)

        self.smodel = SentenceTransformer('multi-qa-distilbert-cos-v1')

        print("==== Loading header embeddings ====")
        start_ = time()
        if emb_header is None:
            self.emb_header = self.smodel.encode(self.df_header["header"].values, show_progress_bar=True)
        else:
            self.emb_header = emb_header
        print(f"Loading header embeddings: {time() - start_:.2f} (s)")

        print("==== Loading paragraph embeddings ====")
        start_ = time()
        if emb_paragraph is None:
            self.emb_paragraph = self.smodel.encode(self.df_paragraph["paragraph"].values, show_progress_bar=True)
        else:
            self.emb_paragraph = emb_paragraph
        print(f"Loading paragraph embeddings: {time() - start_:.2f} (s)")

        self.llm_client = llm_client
        self.model_id = model_id
        self.index = faiss.IndexFlatIP(self.emb_paragraph.shape[1])
        self.index.add(self.emb_paragraph)
    
    def search_paragraph(self, query, k=3):
        start_ = time()
        query_emb = self.smodel.encode(query)
        D, I = self.index.search(np.array([query_emb]), k)
        top_idx = I[0]
        top_score = D[0]
        tb_desc_tmp = self.df_paragraph.loc[top_idx,:].copy()
        tb_desc_tmp["score"] = top_score
        tb_desc_tmp = tb_desc_tmp.reset_index(drop=True)
        print(f"Retrieval time faiss: {time() - start_:.4f} (s)")
        
        return tb_desc_tmp
            
    def generate_answer(self, query, k=3, max_tokens=256):
        prompt_form = """Use the following context to answer my question:
{input_context}

My question is: {query}

Provide a concise answer."""

        df_top_paragraphs = self.search_paragraph(query, k=k)
        list_paragraph = list(df_top_paragraphs["paragraph"])
        context = "\n".join(list_paragraph)
        
        prompt = prompt_form.format(
            input_context=context,
            query=query)
        
        if len(prompt.split(" ")) > 1800:
            context_ = "\n".join(list_paragraph[:2])
            prompt = prompt_form.format(
                input_context=context_,
                query=query)

        start_ = time()
        response = self.llm_client.chat.completions.create(
                model=self.model_id, 
                messages=[
                    {"role": "system", "content": "You are a conversational search engine, you convert the search results to conversational answers. Provide informative but concise answers."},
                    {"role": "user", "content": prompt},
                ], 
                max_tokens=max_tokens)
        generated_text = response.choices[0].message.content
        print(f"Generation time: {time() - start_:.4f} (s)")
        
        anno_gen_text,  srces = self.annotation(generated_text, df_top_paragraphs)
        
        return anno_gen_text, df_top_paragraphs.loc[list(srces),:]

    def annotation(self, generated_text, df_top_paragraphs, sen_min_len=20):
        top_para_emb = self.smodel.encode(df_top_paragraphs["paragraph"].values)
        
        sentences = [x for x in generated_text.split(".")]
        sentences_clean =  [x.strip() for x in sentences]
        # sclean_matter = [x for x in sentences_clean if len(x) > 50]
        
        sclean_anno = []
        for sen in sentences_clean:
            if len(sen) < sen_min_len:
                sclean_anno.append(-1)
            else:
                sen_emb = self.smodel.encode(sen)
                sen_source = np.argsort(top_para_emb @ sen_emb)[::-1][0]
                sclean_anno.append(sen_source)
        
        for i in range(1,len(sclean_anno)):
            if sclean_anno[i-1] == sclean_anno[i]:
                sclean_anno[i-1] = -1
        

        sources = set()

        output_text = ""
        for i, text in enumerate(sentences):
            if sclean_anno[i] >= 0:
                sources.add(sclean_anno[i])
                output_text += text + f"[{sclean_anno[i]}]" + "."
            else:
                output_text += text + "."


        return output_text, sources
