
import os, re, gc, random
from ast import literal_eval
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast, RobertaModel

BASE_URL = "./data/"

###CFG
CFG = {
    "max_length": 512,
    "padding": "max_length",
    "return_offsets_mapping": False,
    "truncation": "only_second",
    #### "model_name": "../input/deberta-v3-large/deberta-v3-large",
    "model_name": "./model/huggingface-bert/microsoft/deberta-v3-large",  
    ##"model_name": "./model/huggingface-bert/microsoft/deberta-v2-xlarge",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 42,
    "batch_size": 32,
    "RTDM": False, ### replaced token detection, RTD model
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(CFG['seed'])

def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

# ====================================================
# Dataset
# ====================================================
def tokenize_text(tokenizer, text, config):
    out = tokenizer(
        text,
        #truncation=config['truncation'],
        add_special_tokens=True,
        max_length=config['max_length'],
        padding=config['padding'],
        return_offsets_mapping=config['return_offsets_mapping']
    )

    for k, v in out.items():
        out[k] = torch.tensor(v, dtype=torch.long)
    return out

class CustomDataset(Dataset):
    def __init__ (self, data, tokenizer, config):
        self.texts = data['text'].values
        #self.labels = data['score'].values
        self.tokenizer = tokenizer
        self.config = config

    def __len__ (self):
        return len(self.texts)

    def __getitem__ (self, item):
        inputs = tokenize_text(self.tokenizer, self.texts[item], self.config)
        #labels = torch.tensor(self.labels[item], dtype=torch.float)
        
        return inputs#, labels


# ====================================================
# Model
# ====================================================
class TransformerHead(nn.Module):
    def __init__(self, in_features, max_length, config, num_layers=1, nhead=8, num_targets=1):
        super().__init__()

        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=in_features,
                                                                                          nhead=nhead),
                                                 num_layers=num_layers)
        self.row_fc = nn.Linear(in_features, 1)
        self.cfg = config
        #self.out_features = max_length

    def forward(self, x):
        out = self.transformer(x)
        out = self.row_fc(out).squeeze(-1)
        return out

class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.modelCfg = AutoConfig.from_pretrained(config['model_name'], output_hidden_states=True)
        toInitializeWeights = True
        if self.config['RTDM']:
            self.fc = nn.Linear(512, 1)
        else:
            if "PatentSBERTa" in self.config['model_name']:
                self.fc = nn.Linear(768, 1)
            else:
                self.fc = nn.Linear(config['max_length'], 1)

        if "deberta" in config['model_name']:
            if self.config['RTDM']:
                self.model = ReplacedTokenDetectionModel.load_model(model_path=RTD_model_path, model_config=RTD_model_cfg)
            else:
                self.model = AutoModel.from_pretrained(config['model_name'], config=self.modelCfg)  # DeBERTa Model
                self._init_weights(self.fc)

        elif "roberta" in config['model_name']:
            self.model = RobertaModel.from_pretrained(config['model_name'])  # RoBERTa model
        else:
            self.model = AutoModel.from_pretrained(config['model_name'])  # BERT model
        self.current_epoch = -1

        if 'deberta-v2-xxlarge' in config['model_name']:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in config['model_name']:
            #self.model.embeddings.requires_grad_(False)
            #self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24
            pass

        """
        self.attention = nn.Sequential(
            nn.Linear(self.modelCfg.hidden_size, 512),
            nn.GELU(), #nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        """
        self.attention = TransformerHead(in_features=self.modelCfg.hidden_size, max_length=config['max_length'], config=config, num_layers=1, nhead=8, num_targets=1)
        if toInitializeWeights:
            self._init_weights(self.fc)
            self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.modelCfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.modelCfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:,0]

    def feature(self, inputs):
        outputs = self.model(**inputs)
        if "PatentSBERTa" in self.config['model_name']:
            feature = self.cls_pooling(outputs, inputs['attention_mask'])
        else:
            last_hidden_states = outputs[0]
            feature = self.attention(last_hidden_states)
        #weights = self.attention(last_hidden_states)
        #feature = torch.sum(weights * last_hidden_states, dim=1)
        
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        outputs = self.fc(self.fc_dropout(feature))
        return outputs


# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv(BASE_URL+'test.csv')
submission = pd.read_csv(BASE_URL+'sample_submission.csv')
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")

cpc_texts = torch.load("./cpc_texts.pth")
test['context_text'] = test['context'].map(cpc_texts)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
#test['text'] = test['text'].apply(str.lower)


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader), mininterval=5)
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


N_FOLDS = 1
DEVICE = "cuda"


# ====================================================
# model-1 inference
# ====================================================
from transformers.models.deberta_v2 import DebertaV2TokenizerFast
tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG['model_name'])
test_data = CustomDataset(test, tokenizer, CFG)
test_dataloader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

predictions1 = []
for fold in range(N_FOLDS):
    model = CustomModel(CFG)
    model.load_state_dict(torch.load(f'./pt_DeBerta-V3-Large_fold-{fold}-BS.pth', map_location = DEVICE))

    prediction = inference_fn(test_dataloader, model, DEVICE)
    predictions1.append(prediction)
    del model, prediction
    gc.collect()
    torch.cuda.empty_cache()
predictions1 = np.mean(predictions1, axis=0)

# ====================================================
# model-2 inference
# ====================================================
CFG["model_name"] = "./model/huggingface-bert/roberta-large"
tokenizer = RobertaTokenizerFast.from_pretrained(CFG["model_name"])
test_data = CustomDataset(test, tokenizer, CFG)
test_dataloader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

predictions2 = []
for fold in range(N_FOLDS):
    model = CustomModel(CFG)
    model.load_state_dict(torch.load(f'./pt_RoBerta-Large_fold-{fold}-BS.pth', map_location = DEVICE))

    prediction = inference_fn(test_dataloader, model, DEVICE)
    predictions2.append(prediction)
    del model, prediction
    gc.collect()
    torch.cuda.empty_cache()
predictions2 = np.mean(predictions2, axis=0)

"""
# ====================================================
# model-3 inference
# ====================================================
CFG["model_name"] = "./model/huggingface-bert/PatentSBERTa"
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
test_data = CustomDataset(test, tokenizer, CFG)
test_dataloader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

predictions3 = []
for fold in range(N_FOLDS):
    model = CustomModel(CFG)
    model.load_state_dict(torch.load(f'./pt_PatentSBERTa_fold-{fold}-BS.pth', map_location = DEVICE))

    prediction = inference_fn(test_dataloader, model, DEVICE)
    predictions3.append(prediction)
    del model, prediction
    gc.collect()
    torch.cuda.empty_cache()
predictions3 = np.mean(predictions3, axis=0)
"""


# ====================================================
# Multiple models ensemble
# ====================================================
from sklearn.preprocessing import MinMaxScaler
MMscaler = MinMaxScaler()
pred1_mm = MMscaler.fit_transform(predictions1.reshape(-1,1)).reshape(-1)
pred2_mm = MMscaler.fit_transform(predictions2.reshape(-1,1)).reshape(-1)

w1 = 0.66
w2 = 0.33

final_predictions =  pred1_mm * w1 + pred2_mm * w2

submission['score'] = final_predictions
submission[['id', 'score']].to_csv('submission.csv', index=False)





