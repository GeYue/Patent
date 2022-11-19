#coding=UTF-8

import ast, re, gc, copy, sys, random, os, math
from itertools import chain
import scipy as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from tqdm.notebook import tqdm
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast, RobertaModel, AutoModelForSeq2SeqLM, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import lr_scheduler as torch_lrs

os.environ['TORCH_HOME'] = "/home/xyb/Lab/TorchModels"

BASE_URL = "./data"

###Hyperparameters
CFG = {
    "wandb": False,
    "wandb_kernel": "US-PPPM", 
    "n_fold": 5, 
    "max_length": 512, ### Seems that the max lenth should be 466 not 416
    "padding": "max_length",
    "return_offsets_mapping": False,
    "truncation": True, #"only_second"
    "dropout": 0.2,
    "scheduler": "cosine", # ['linear', 'cosine', 'cosine_anneal_restart', polynomial']
    "batch_scheduler": True,
    "num_warmup_steps": 0, ##warmup epoch number, 0==>No Warmup.
    "num_cycles": 0.5,
    "lr": -1, ### good for all others :1e-5, except Dv2XL should use 2e-6,
    "min_lr": 1e-7,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "seed": 26628,
    "batch_size": -1, ### will update in parameter resolving , without [@torch.no_grad() + AMP], the value was 3. Now, 8!
    "gradient_accumulation_steps": 1,
    "epochs": 4,
    "model_name": "TBD",
    "model_prefix": "TBD",
    "enable_amp_half_precision": False,
    "RTDM": False, ### replaced token detection, RTD model
    "print_freq": 3,
    "AdvTrain": "None", # PGD / FGM / None
    "label_deNoise": False,
    "num_workers": 20,
}

model_list = {
    "deberta": "./model/huggingface-bert/microsoft/deberta-v3-large", #batchsize:: AMP 8 / NonAMP 3 / RTD 4
    #"deberta": "./deberta-v3-large-regular", #batchsize:: AMP 8 / NonAMP 3 / RTD 4
    "deberta_desc": "DeBerta-V3-Large",

    "dbtV2XL": "./model/huggingface-bert/microsoft/deberta-v2-xlarge", #batchsize:: after freeze[:12] AMP 4 / NonAMP 3
    #"dbtV2XL": "./deberta-v2-xlarge-regular/", #batchsize:: after freeze[:12] AMP 4 / NonAMP 3
    "dbtV2XL_desc": "DeBerta-V2-XLarge",                               # RTD 1 (no freeze)

    "dbtV2XXL": "./model/huggingface-bert/microsoft/deberta-v2-xxlarge", #batchsize:: after freeze[:24] AMP :( / NonAMP :(
    "dbtV2XXL_desc": "DeBerta-V2-XXLarge",

    "roberta": "./model/huggingface-bert/roberta-large",
    "roberta_desc": "RoBerta-Large",

    "bert": "./model/huggingface-bert/bert-base-uncased",
    "bert_desc": "Bert-Base-Uncased", 

    "bert-ptl": "./model/huggingface-bert/bert-for-patents",
    "bert-ptl-desc": "Bert-Patent-Large",

    "electra": "./model/huggingface-bert/electra-large-discriminator",
    "electra-desc": "Electra-Large-Discriminator",

    "gpt2": "./model/huggingface-bert/gpt2-medium",
    "gpt2-desc": "GPT2-Medium",
}


### Logger setting
import logging
logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. 💭BERTa model, KFold={CFG['n_fold']} 🔴🟡🟢 {sys.argv}")


if CFG['wandb']:
    import wandb
    try:
        wandb.login(key="67871c2e8f97fa74b52c18bcfccbee7fee0361d2")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    def readCfgDict(f):
        return dict((name, f[name]) for name in f.keys() if not name.startswith('__'))
    
    run = wandb.init(project=CFG['wandb_kernel'], 
                    #name="York_PPPM",
                    config=readCfgDict(CFG), #class2dict(CFG),
                    #group="DeBERTa-V3L",
                    #job_type="train",
                    )

"""Datasets Helper Function
need to merge features.csv, patient_notes.csv with train.csv
"""
def process_text_features(text):
    return text.replace(';', ' ').replace(',', ' ')

def prepare_datasets():
    train = pd.read_csv(f"{BASE_URL}/train.csv")
    cpc_texts = torch.load("./cpc_texts.pth")
    train['context_text'] = train['context'].map(cpc_texts)
    #train['context_text'] = [process_text_features(x) for x in train['context_text']]
    train['text'] = train['anchor'] + '[SEP]' + train['target'] + '[SEP]'  + train['context_text']
    #train['text'] = train['text'].apply(str.lower)

    # ====================================================
    # CV split
    # ====================================================
    train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})

    encoder = LabelEncoder()
    train['anchor_map'] = encoder.fit_transform(train['anchor'])

    kf = StratifiedGroupKFold(n_splits=CFG['n_fold'], shuffle=True, random_state=CFG['seed'])
    for n, (_, valid_index) in enumerate(kf.split(train, train['score_map'], groups=train['anchor_map'])):
        train.loc[valid_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    print (train['context'].apply(lambda x: x[0]).value_counts())
    print(train.groupby('fold').size())
    return train

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


### Dataset for DataLoader
class CustomDataset(Dataset):
    def __init__ (self, data, tokenizer, config):
        self.texts = data['text'].values
        self.labels = data['score'].values
        self.tokenizer = tokenizer
        self.config = config

    def __len__ (self):
        return len(self.texts)

    def __getitem__ (self, item):
        inputs = tokenize_text(self.tokenizer, self.texts[item], self.config)
        labels = torch.tensor(self.labels[item], dtype=torch.float)
        
        return inputs, labels

"""
Model
Lets use BERT base Architecture
Also Used 3 FC layers
Comments: 3 layers improve accuracy 2% on public score
"""
#from DeBERTa_RTD.DeBERTa.apps.models.replaced_token_detection_model import ReplacedTokenDetectionModel
RTD_model_path = "/home/xyb/Project/Kaggle/NBME/model/huggingface-bert/microsoft/deberta-v2-xlarge/pytorch_model.bin"
RTD_model_cfg = "/home/xyb/Project/Kaggle/NBME/DeBERTa_RTD/experiments/language_model/deberta_xlarge.json"

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
            self.fc = nn.Linear(config['max_length'], 1)

        if "deberta" in config['model_name']:
            if self.config['RTDM']:
                self.model = ReplacedTokenDetectionModel.load_model(model_path=RTD_model_path, model_config=RTD_model_cfg)
            else:
                self.model = AutoModel.from_pretrained(config['model_name'], config=self.modelCfg)  # DeBERTa Model
                self._init_weights(self.fc)

        elif "roberta" in config['model_name']:
            self.model = RobertaModel.from_pretrained(config['model_name'])  # RoBERTa model
        elif "gpt2" in config['model_name']:
            model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=config["model_name"], num_labels=1)
            self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config["model_name"], config=model_config)
            self.model.resize_token_embeddings(len(tokenizer))
            self.model.config.pad_token_id = self.model.config.eos_token_id
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

        if 'bart' in config['model_name']:
            toInitializeWeights = False

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

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.attention(last_hidden_states)
        #weights = self.attention(last_hidden_states)
        #feature = torch.sum(weights * last_hidden_states, dim=1)
        
        return feature

    def forward(self, inputs):
        if "gpt2" in self.config['model_name']:
            outputs = self.model(**inputs)
            return outputs[0]
        feature = self.feature(inputs)
        outputs = self.fc(self.fc_dropout(feature))
        return outputs

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(CFG['seed'])

# ====================================================
# Projected Gradient Descent（PGD）
# Reference:: https://github.com/Makaixin/similar-sentence-pairs-in-epidemic
# Fast Gradient Method（FGM）
# ====================================================
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FGM():
    """
    定义对抗训练方法FGM,对模型embedding参数进行扰动
    """
    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, embed_name='word_embeddings'):
        """
        得到对抗样本
        :param emb_name:模型中embedding的参数名
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, embed_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

def train_model(fold, model, dataloader, optimizer, criterion, scheduler):
    model.train()
    losses = AverageMeter()
    grad_trend = AverageMeter()
    total_steps = len(dataloader)
    optimizer.zero_grad()
    train_start = time.time()
    #scaler = torch.cuda.amp.GradScaler(enabled=CFG['enable_amp_half_precision'])

    pgd = PGD(model)
    K = 3
    fgm = FGM(model)

    for step, (inputs, labels) in enumerate(dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = labels.size(0)

        logits = model(inputs)
        loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

        if CFG["gradient_accumulation_steps"] > 1:
            loss = loss / CFG["gradient_accumulation_steps"]
        losses.update(loss.item(), batch_size)
        if (CFG['enable_amp_half_precision'] == True):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            #scaler.scale(loss).backward()
            loss.backward()

        if CFG['AdvTrain'] == "PGD":
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                y_pred = model(inputs)
                loss_adv = criterion(y_pred.view(-1, 1), labels.view(-1, 1))
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
        elif CFG['AdvTrain'] == "FGM":
            fgm.attack()
            y_pred = model(inputs)
            loss_adv = criterion(y_pred.view(-1, 1), labels.view(-1, 1))
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()
        else:
            pass

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # it's also improve f1 accuracy slightly
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        grad_trend.update(grad_norm, 1)
        if (step+1) % CFG["gradient_accumulation_steps"] == 0:
            #scaler.step(optimizer)
            #scaler.update()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if CFG['batch_scheduler']:
                if CFG['scheduler'] == "cosine_anneal_restart":
                    scheduler.step(model.current_epoch + step / totals_teps)
                else:
                    scheduler.step()

        ##lr = scheduler.get_last_lr()[0]
        ##print(f"LR: {lr:.8f}〽️")
        if step % CFG["print_freq"] == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                'Grad: {grad_norm:.4f}  '
                'LR: {lr:.8f}  '
                .format(model.current_epoch+1, CFG['epochs'], step, len(dataloader), 
                        remain=timeSince(train_start, float(step+1)/len(dataloader)),
                        loss=losses,
                        grad_norm=grad_norm,
                        lr=scheduler.get_last_lr()[0]), end='\r', flush=True)
        
        if CFG['wandb']:
            wandb.log({f"fold{fold}_train_loss": losses.avg,
                       f"fold{fold}_lr": scheduler.get_last_lr()[0],
                       f"fold{fold}_grad_norm": grad_trend.avg,
                       f"fold{fold}_epoch": model.current_epoch+1})
        
        del inputs, labels, logits

    lr = scheduler.get_last_lr()[0]
    print(f"LR: {lr:.8f}〽️")
    logger.info(f"LR: {lr:.8f}〽️")
    gc.collect()
    torch.cuda.empty_cache()

    return losses.avg

@torch.no_grad()
def eval_model(fold, model, dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    preds = []

    val_start = time.time()
    for step, (inputs, labels) in enumerate(dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = labels.size(0)

        logits = model(inputs)
        loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
        losses.update(loss.item(), batch_size)

        preds.append(logits.sigmoid().to('cpu').numpy())
        
        if step % CFG["print_freq"] == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(model.current_epoch+1, CFG['epochs'], step, len(dataloader), 
                        remain=timeSince(val_start, float(step+1)/len(dataloader)),
                        loss=losses), end='\r', flush=True)
        
        if CFG['wandb']:
            wandb.log({f"fold{fold}_validate_loss": losses.avg, 
                       f"fold{fold}_epoch": model.current_epoch+1})
        
        del inputs, labels, logits

    gc.collect()
    torch.cuda.empty_cache()

    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg['scheduler'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps
        )
    elif cfg['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps, num_cycles=cfg["num_cycles"]
        )
    elif cfg['scheduler'] == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps, lr_end=cfg["min_lr"], power=3
        )
    elif cfg['scheduler'] == 'cosine_anneal_restart':
        scheduler = torch_lrs.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=1, eta_min=cfg["min_lr"]
        )

    return scheduler

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

DEVICE = "cuda"
train_df = prepare_datasets()

print (f"parameter_number=={len(sys.argv)}")
if len(sys.argv) != 2:
        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} d' / '{sys.argv[0]} r' to use 'DeBERTa' or 'RoBERTa' model.")
        sys.exit()
#elif sys.argv[1] != "d" and sys.argv[1] != "r" and sys.argv[1] != 'b' and sys.argv[1] != 'x' and sys.argv[1] != 'xx':
elif sys.argv[1] not in ['d', 'r', 'b', 'bp', 'x', 'e' , 'g2', 'xx']:
        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} d' / '{sys.argv[0]} r' to use 'DeBERTa' or 'RoBERTa' model.")
        sys.exit()
else:
    if sys.argv[1] == "d":
        CFG["model_name"] = model_list["deberta"]
        CFG["model_prefix"] = model_list["deberta_desc"]
        CFG["batch_size"] = 5 ##4 for AdvTrain::PGD, 5 for no AdvTrain.
        CFG["lr"] = 1e-5
    elif sys.argv[1] == "r":
        CFG["model_name"] = model_list["roberta"]
        CFG["model_prefix"] = model_list["roberta_desc"]
        CFG["batch_size"] = 7
        CFG["lr"] = 1e-5
    elif sys.argv[1] == "b":
        CFG["model_name"] = model_list["bert"]
        CFG["model_prefix"] = model_list["bert_desc"]
        CFG["batch_size"] = 8
        CFG["lr"] = 1e-5
    elif sys.argv[1] == "bp":
        CFG["model_name"] = model_list["bert-ptl"]
        CFG["model_prefix"] = model_list["bert-ptl-desc"]
        CFG["batch_size"] = 7
        CFG["lr"] = 1e-5
    elif sys.argv[1] == "x":
        CFG["model_name"] = model_list["dbtV2XL"]
        CFG["model_prefix"] = model_list["dbtV2XL_desc"]
        CFG["batch_size"] = 1
        CFG["lr"] = 2e-6
    elif sys.argv[1] == "xx":
        CFG["model_name"] = model_list["dbtV2XXL"]
        CFG["model_prefix"] = model_list["dbtV2XXL_desc"]
        CFG["batch_size"] = 1
        CFG["lr"] = 2e-6
    elif sys.argv[1] == "e":
        CFG["model_name"] = model_list["electra"]
        CFG["model_prefix"] = model_list["electra-desc"]
        CFG["batch_size"] = 7
        CFG["lr"] = 2e-6
    elif sys.argv[1] == "g2":
        CFG["model_name"] = model_list["gpt2"]
        CFG["model_prefix"] = model_list["gpt2-desc"]
        CFG["batch_size"] = 5
        CFG["lr"] = 1e-5

from transformers.models.deberta_v2 import DebertaV2TokenizerFast
if "deberta" in CFG["model_name"]:
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG["model_name"])
elif "roberta" in CFG["model_name"]:
    tokenizer = RobertaTokenizerFast.from_pretrained(CFG["model_name"])
elif "gpt2" in CFG["model_name"]:
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=CFG["model_name"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])


##
## Lable Adjustments for training dataloader
##
def Label_DeNoise(model, raw_data, criterion, train_folder):
    predictions = []
    modelPath = f"./kaggle/finetuned/v1"
    training_data = CustomDataset(raw_data, tokenizer, CFG)
    train_dataloader = DataLoader(training_data, batch_size=CFG['batch_size']*16, shuffle=False, num_workers=CFG['num_workers'], pin_memory=False)
    for fold in range(CFG['n_fold']):
        model.load_state_dict(torch.load(f'{modelPath}/pt_DeBerta-V3-Large_fold-{fold}-BS.pth', map_location = DEVICE))
        fold_loss, fold_prediction = eval_model(fold, model, train_dataloader, criterion)
        print(f"")
        predictions.append(fold_prediction.reshape(-1))
 
        gc.collect()
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)
    fake_threshold = .5

    def update_label(pred):
        if -0.01 < pred <= .01:
            return .0
        elif .24 <= pred <= .26:
            return .25
        elif .49 <= pred <= .51:
            return .5
        elif .74 <= pred <= .76:
            return .75
        elif .99 <= pred <= 1.01:
            return 1.0
        
        return pred

    df_preds = pd.DataFrame({
        "score": raw_data['score'],
        "pred": predictions,})

    df_preds["abs_diff"] = (df_preds["score"] - df_preds["pred"]).abs()
    df_preds["is_higher"] = df_preds["abs_diff"] >= fake_threshold
    print(f"fold{train_folder}📉📈:Change ratio=={df_preds['is_higher'].mean()}")
    logger.info(f"fold{train_folder}📉📈:Change ratio=={df_preds['is_higher'].mean()}")
    updated_labels = df_preds.apply(
        lambda x: update_label(x["pred"]) if x["is_higher"] else x["score"], 
        axis=1
    ).values

    updated_tr_data = raw_data.copy()
    updated_tr_data["score"] = updated_labels
    return updated_tr_data

SINGLE_FOLD = -1
def train_K_Fold(fold, model, optimizer):
    """
    Prepare Datasets
    """
    if SINGLE_FOLD >= 0:
        print(f"☠️☠️☠️ Only run fold={SINGLE_FOLD}, other folds will skipped!!!🏮🏮🏮")
        if fold != SINGLE_FOLD:
            return
    print(f"*************************\nRuning fold==>{fold+1}/{CFG['n_fold']}\n*************************")
    logger.info(f"*************************\nRuning fold==>{fold+1}/{CFG['n_fold']} 🌱🌱🌱⚡️🍄🍄🍄")
    X_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    X_test = train_df[train_df["fold"] == fold].reset_index(drop=True)
    valid_labels = X_test['score'].values

    print ("TrainSet size=={}, TestSet size=={}\n*************************".format(len(X_train), len(X_test)))
    logger.info("TrainSet size=={}, TestSet size=={}".format(len(X_train), len(X_test)))

    criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
    deNoise = f""
    if CFG['label_deNoise']:
        updated_tr_data = Label_DeNoise(model, X_train, criterion, fold)
        training_data = CustomDataset(updated_tr_data, tokenizer, CFG)
        deNoise = f"🧪"
    else:
        training_data = CustomDataset(X_train, tokenizer, CFG)
    train_dataloader = DataLoader(training_data, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'], pin_memory=False)

    test_data = CustomDataset(X_test, tokenizer, CFG)
    test_dataloader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'], pin_memory=False)

    """
    Train
    Lets train the model with BCEWithLogitsLoss and AdamW as optimizer

    Notes: on BCEWithLogitsLoss, the default value for reductio
    n is mean (the sum of the output will be divided by the number of elements in the output). 
    If we use this default value, it will produce negative loss. 
    Because we have some negative labels. 
    To fix this negative loss issue, we can use none as parameter. 
    To calculate the mean, first, we have to filter out the negative values.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    """

    epochs = CFG['epochs']

    #model = CustomModel(CFG).to(DEVICE)
    checkpoint = torch.load("./model_initialized.bin")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    """
    optimizer_parameters = get_optimizer_params(model,
                                            encoder_lr=CFG['lr'], 
                                            decoder_lr=CFG['lr'],
                                            weight_decay=CFG['weight_decay'])
    """
    #optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'])
    ###optimizer = optim.AdamW(optimizer_parameters, lr=CFG['lr'])

    num_train_steps = int(len(X_train) / (CFG['batch_size'] * CFG["gradient_accumulation_steps"]) * epochs)
    num_warmup_steps = int(len(X_train) / (CFG['batch_size'] * CFG["gradient_accumulation_steps"]) * 0.4)
    #CFG['num_warmup_steps'] = num_warmup_steps
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    best_loss = np.inf
    best_score = 0

    for i in range(epochs):
        epoch_since = time.time()
        print("Epoch{}: {}/{}".format(deNoise,i + 1, epochs))
        logger.info("Epoch{}: {}/{}".format(deNoise, i + 1, epochs))

        # first train model
        model.current_epoch = i
        train_loss = train_model(fold, model, train_dataloader, optimizer, criterion, scheduler)
        print(f"Train loss: {train_loss}")
        logger.info(f"Train loss: {train_loss}")

        # then evaluate model
        valid_loss, predictions = eval_model(fold, model, test_dataloader, criterion)
        score = get_score(valid_labels, predictions)
        print(f"Valid loss: {valid_loss}")
        print(f"Valid score: {score}")
        
        if CFG['wandb']:
            wandb.log({f"fold{fold}_epoch": i+1, 
                       f"fold{fold}_avg_train_loss": train_loss, 
                       f"fold{fold}_avg_val_loss": valid_loss,
                       f"fold{fold}_score": score})

        if valid_loss < best_loss:
            best_loss = valid_loss
            #torch.save(model.state_dict(), f"pt_{CFG['model_prefix']}_fold-{fold}-BV.pth")
            logger.info(f"Valid loss: {valid_loss} --- !!!Fine-tuning UPDATED!!!🥰👽😽")
            #torch.save(model, "nbme_bert_model.model")
        else:
            logger.info(f"Valid loss: {valid_loss}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"pt_{CFG['model_prefix']}_fold-{fold}-BS.pth")
            logger.info(f"💯🆕Valid score: {score}")
        else:
            logger.info(f"Valid score: {score}")
            
        epoch_time_elapsed = time.time() - epoch_since
        logger.info('Epoch: {} completed in {:.0f}m {:.0f}s'.format(i+1, epoch_time_elapsed // 60, epoch_time_elapsed % 60))

   
"""
MAIN FUNC
"""
import time
if __name__=='__main__':

    since = time.time()

    model = CustomModel(CFG).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'])
    #optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG['lr'])

    if (CFG['enable_amp_half_precision'] == True):
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, "./model_initialized.bin")

    for fold in range(CFG['n_fold']):
        train_K_Fold(fold, model, optimizer)
    if CFG['wandb']:
        wandb.finish()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


