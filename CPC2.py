#coding=UTF-8

import os, re, sys
import torch
import pandas as pd
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
    titles = pd.read_csv(f"./data/titles.csv")

    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        cpc_result = titles[titles.code==f'{cpc}']['title'].iloc[0]
        for ctxt in [c for c in contexts if c[0] == cpc]:
            ctxt_result = titles[titles.code==f'{ctxt}']['title'].iloc[0]
            results[ctxt] = cpc_result + ". " + ctxt_result
    return results


cpc_texts = get_cpc_texts()
torch.save(cpc_texts, "./cpc_texts.pth")
