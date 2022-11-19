#coding=UTF-8

import os, re, sys
import pandas as pd
import torch
# ====================================================
# CPC Data
# ====================================================
def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('./data/CPC/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)

    contexts = sorted(set(sum(contexts, [])))
    tts = pd.read_csv(f"./data/titles.csv")

    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        cpc_result = tts[tts.code==f'{cpc}']['title'].iloc[0]
        for ctxt in [c for c in contexts if c[0] == cpc]:
            ctxt_result = tts[tts.code==f'{ctxt}']['title'].iloc[0]

            file = open(f"./data/CPC_Subclass/{ctxt}.txt", "r")
            content = file.read()
            content = content.replace('\n',"").replace('\r',"").replace('\t',"")
            file.close()

            if len(content) > 0:
                results[ctxt] = cpc_result + ". " + ctxt_result + ". " + content
            else:
                results[ctxt] = cpc_result + ". " + ctxt_result

            #out = kw_model.extract_keywords(content, keyphrase_ngram_range=(3, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=5)
            #kw_model.extract_keywords(content, keyphrase_ngram_range=(3, 3), stop_words='english', use_mmr=True, diversity=0.7)
            one = 1
    return results

from keybert import KeyBERT
kw_model = KeyBERT()
cpc_texts = get_cpc_texts()
torch.save(cpc_texts, "./cpc_texts_ext.pth")

