# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ANTIBODY SEQUENCE ENCODER
# Optuna Hyperparameter Optimisation  (paper §3.4.3)
#
# PAPER COMPLIANCE CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
# §3.3.2  Architecture        layer dims, num_heads, dropout, d_ff       ✓
# §3.4.2  Training spec       LR pretrain/finetune, grad_clip,            ✓
#                             weight_decay, batch_size, patience=10
# §3.4.3  Bayesian optimiser  Optuna TPE sampler, 50 trials               ✓
#         LR *schedules*      schedule type tunable per phase             ✓  GAP 2 fixed
#         Parallel execution  SQLite shared storage + n_jobs              ✓  GAP 3 fixed
# §3.4.4  Metric 1            token-level macro F1 (not raw accuracy)     ✓  GAP 1 fixed
#         Metric 2            silhouette score (V-gene clusters)           ✓
#         Metric 3            natural vs synthetic AUROC                   ✓
#         Weighted objective  composite = 0.50*F1 + 0.25*sil              ✓
#                                       + 0.25*auroc
#
# Parallel execution (paper: "2-3 days on multiple free GPU instances"):
#   Run this script simultaneously on N Colab/Kaggle workers all pointing
#   at the same SQLite file.  Set N_PARALLEL_JOBS below, or leave at 1
#   for single-machine runs.
# ══════════════════════════════════════════════════════════════════════════════

# ── Run-mode flags (mirror Module2Script exactly) ────────────────────────────
dev_mode    = True
sanity_mode = True      # True  -> tiny data + 1 epoch for smoke-testing
                        # False -> use dev_mode or full-scale settings

OAS_MAX_PER_FILE = 100   if sanity_mode else (5_000  if dev_mode else None)
CABREP_MAX       = 100   if sanity_mode else (5_000  if dev_mode else 50_000)

# Epochs per Optuna trial (short so many trials fit; final retraining uses
# the full epoch counts from Module2Script §3.4.2)
TRIAL_EPOCHS_PHASE1   = 1 if sanity_mode else (3  if dev_mode else 10)
TRIAL_EPOCHS_PHASE2   = 1 if sanity_mode else (5  if dev_mode else 20)
TRIAL_EPOCHS_FINETUNE = 1 if sanity_mode else (2  if dev_mode else 5)

# Full-retraining epochs (paper §3.4.2 values)
FULL_EPOCHS_PHASE1   = 1 if sanity_mode else (5  if dev_mode else 20)
FULL_EPOCHS_PHASE2   = 1 if sanity_mode else (10 if dev_mode else 80)
FULL_EPOCHS_FINETUNE = 1 if sanity_mode else (3  if dev_mode else 10)

# Optuna study settings (paper §3.4.3)
N_TRIALS        = 2  if sanity_mode else (10 if dev_mode else 50)  # paper: 50
STUDY_NAME      = 'module2_antibody_encoder'
# Parallel workers: set > 1 to match paper "parallel execution on multiple
# free GPU instances" (§3.4.3).  Each worker runs this same script and
# coordinates via the shared SQLite journal below.
N_PARALLEL_JOBS = 1                                   # number of concurrent workers
STORAGE_PATH    = f'sqlite:///{STUDY_NAME}.db'        # shared across workers

# Sequences used per trial for benchmark metric computation (capped for speed)
N_SIL_SEQS   = 100   # sequences fed into silhouette score
N_AUROC_SEQS = 100   # natural sequences; equal synthetics generated

# ── Objective weights (paper §3.4.3 + §3.4.4) ────────────────────────────────
# composite = W_F1 * token_macro_f1  +  W_SIL * sil_norm  +  W_AUROC * auroc
# All three metrics higher-is-better; negated for Optuna minimisation.
W_F1    = 0.50   # token-level macro F1  (§3.4.4 metric 1 — GAP 1 fixed)
W_SIL   = 0.25   # silhouette score      (§3.4.4 metric 2)
W_AUROC = 0.25   # nat vs syn AUROC      (§3.4.4 metric 3)

# ── Installs ──────────────────────────────────────────────────────────────────
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import subprocess, sys
for _pkg in ['torch', 'biopython', 'pandas', 'numpy', 'scikit-learn', 'optuna']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg, '-q'])

# ── Imports ───────────────────────────────────────────────────────────────────
import math, random, re, warnings, glob, json as _json
import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data     import Dataset, DataLoader
from sklearn.metrics      import roc_auc_score, silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from collections          import Counter
from Bio.PDB              import PDBParser, PPBuilder
from Bio                  import SeqIO
from datetime             import datetime
import optuna
from optuna.samplers      import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# Identical helpers to Module2Script so all results are directly comparable.
# ══════════════════════════════════════════════════════════════════════════════

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def translate_nt(nt_seq):
    nt_seq = nt_seq.upper().replace('-', '').replace(' ', '')
    aa = []
    for i in range(0, len(nt_seq) - 2, 3):
        codon = nt_seq[i:i+3]
        res   = CODON_TABLE.get(codon, 'X')
        if res == '*': break
        aa.append(res)
    return ''.join(aa)

def compute_shm_count(seq_aln, germ_aln):
    if not seq_aln or not germ_aln: return 0
    return sum(1 for o, g in zip(str(seq_aln), str(germ_aln))
               if o not in '-' and g not in '-N' and o != g)

def load_oas_csv(csv_path, max_seqs=None):
    sequences, cdr_data = [], []
    study_meta = {}
    try:
        with open(csv_path, 'r') as fh:
            first_line = fh.readline().strip().strip('"')
            try: study_meta = _json.loads(first_line)
            except: pass
    except: pass
    tissue  = study_meta.get('BSource', 'Unknown')
    btype   = study_meta.get('BType',   'Unknown')
    disease = study_meta.get('Disease', 'Unknown')
    subject = study_meta.get('Subject', 'Unknown')
    chain   = study_meta.get('Chain',   'Heavy')
    try:
        df     = pd.read_csv(csv_path, skiprows=1)
        aa_col = 'sequence_alignment_aa' if 'sequence_alignment_aa' in df.columns else None
        prod   = df[df['productive'] == 'T'] if 'productive' in df.columns else df
        for _, row in prod.iterrows():
            if max_seqs and len(sequences) >= max_seqs: break
            aa_seq = str(row.get(aa_col, '')).strip() if aa_col else ''
            valid  = sum(1 for a in aa_seq if a in 'ACDEFGHIKLMNPQRSTVWY')
            if valid < 20 or len(aa_seq) < 20 or 'X' in aa_seq[:10]: continue
            shm = compute_shm_count(row.get('v_sequence_alignment', ''),
                                    row.get('v_germline_alignment', ''))
            sequences.append(aa_seq)
            cdr_data.append({
                'full': aa_seq,
                'cdr1': str(row.get('cdr1_aa', '')),
                'cdr2': str(row.get('cdr2_aa', '')),
                'cdr3': str(row.get('cdr3_aa', '')),
                'v_call': str(row.get('v_call', '')),
                'j_call': str(row.get('j_call', '')),
                'shm_count': shm, 'tissue_source': tissue,
                'donor_status': btype, 'disease': disease,
                'subject': subject, 'chain': chain,
                'source_file': os.path.basename(csv_path),
            })
    except Exception as e:
        print(f'  OAS load ERROR ({csv_path}): {e}')
    return sequences, cdr_data

def load_fasta_as_proteins(fasta_path, max_seqs=50_000):
    proteins = []
    try:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            prot  = translate_nt(str(record.seq))
            valid = sum(1 for a in prot if a in 'ACDEFGHIKLMNPQRSTVWY')
            if valid >= 20 and len(prot) >= 20:
                proteins.append(prot)
            if len(proteins) >= max_seqs: break
    except FileNotFoundError:
        print(f'  WARNING: {fasta_path} not found — skipping.')
    return proteins


# ── CDR.pdb candidate antibody ───────────────────────────────────────────────
_pdb_parser  = PDBParser(QUIET=True)
_ppb         = PPBuilder()
structure    = _pdb_parser.get_structure('cdr', 'CDR.pdb')
CDR_FULL_SEQ = ''.join(str(pp.get_sequence())
                        for pp in _ppb.build_peptides(structure))
cdr3_match   = re.search(r'(CAR[A-Z]+?WGQGT)', CDR_FULL_SEQ)
CDR3_SEQ     = cdr3_match.group(1) if cdr3_match else CDR_FULL_SEQ[-25:]
CDR_REGIONS  = {
    'CDR1': CDR_FULL_SEQ[26:38], 'CDR2': CDR_FULL_SEQ[55:65],
    'CDR3': CDR3_SEQ,            'Full': CDR_FULL_SEQ,
}
print(f'CDR full seq ({len(CDR_FULL_SEQ)} aa): {CDR_FULL_SEQ}')
print(f'CDR3 : {CDR3_SEQ}')

vgene_df   = pd.read_csv('V-Gene.csv')
best_vgene = vgene_df.loc[vgene_df['Identity Percentage'].idxmax()]

# ── BCR corpus (OAS + cAb-Rep, heavy + light) ────────────────────────────────
print('\nLoading IGH (heavy chain) data...')
oas_seqs_400, cdr_data_400 = load_oas_csv('ERR220400_Heavy_Bulk.csv', OAS_MAX_PER_FILE)
oas_seqs_430, cdr_data_430 = load_oas_csv('ERR220430_Heavy_Bulk.csv', OAS_MAX_PER_FILE)
cabrep_heavy = load_fasta_as_proteins('cAb-Rep_heavy.nt.fasta', max_seqs=CABREP_MAX)

extra_oas_heavy, extra_cdr_heavy = [], []
for _f in glob.glob('*_Heavy_*.csv') + glob.glob('*_Heavy_Bulk.csv'):
    if 'ERR220400' in _f or 'ERR220430' in _f: continue
    _s, _c = load_oas_csv(_f, max_seqs=OAS_MAX_PER_FILE)
    extra_oas_heavy.extend(_s); extra_cdr_heavy.extend(_c)

print('\nLoading IGK/IGL (light chain) data...')
cabrep_light   = load_fasta_as_proteins('cAb-Rep_light.nt.fasta', max_seqs=CABREP_MAX)
extra_oas_light, light_cdr_data = [], []
for _f in (glob.glob('*_Light_*.csv') + glob.glob('*_Kappa_*.csv') +
           glob.glob('*_Lambda_*.csv') + glob.glob('*_Light_Bulk.csv')):
    _s, _c = load_oas_csv(_f, max_seqs=OAS_MAX_PER_FILE)
    extra_oas_light.extend(_s); light_cdr_data.extend(_c)

all_oas_seqs_heavy = oas_seqs_400 + oas_seqs_430 + extra_oas_heavy
all_cdr_data_heavy = cdr_data_400 + cdr_data_430 + extra_cdr_heavy
heavy_proteins     = all_oas_seqs_heavy + cabrep_heavy
all_oas_seqs_light = extra_oas_light
light_proteins     = cabrep_light + all_oas_seqs_light
all_cdr_data       = all_cdr_data_heavy + light_cdr_data
all_oas_seqs       = all_oas_seqs_heavy + all_oas_seqs_light
all_repertoire     = heavy_proteins + light_proteins
print(f'Total corpus : {len(all_repertoire)} seqs | CDR entries : {len(all_cdr_data)}')

# ── Donor-stratified train / val split (paper §3.4.1) ────────────────────────
print('\nBuilding donor-stratified train/val split...')
donor_to_seqs  = {}
annotated_seqs = set(d['full'] for d in all_cdr_data)
for d in all_cdr_data:
    donor = d['subject']
    if donor in ('Unknown', 'nan', '') or not donor:
        donor = d.get('source_file', 'Unknown')
    donor_to_seqs.setdefault(donor, []).append(d['full'])
unannotated = [s for s in all_repertoire if s not in annotated_seqs]
_chunk = max(1, len(unannotated) // 10)
for i, s in enumerate(unannotated):
    donor_to_seqs.setdefault(f'Unknown_{i // _chunk}', []).append(s)

all_donors   = list(donor_to_seqs.keys())
random.shuffle(all_donors)
n_val_donors = max(1, int(len(all_donors) * 0.15))
val_donors   = set(all_donors[:n_val_donors])
train_donors = set(all_donors[n_val_donors:])
train_seqs   = [s for d in train_donors for s in donor_to_seqs.get(d, [])]
val_seqs     = [s for d in val_donors   for s in donor_to_seqs.get(d, [])]
if not val_seqs:
    _all = list(all_repertoire); random.shuffle(_all)
    _sp  = int(0.9 * len(_all))
    train_seqs, val_seqs = _all[:_sp], _all[_sp:]

# PDB augmentation — training only, zero data leakage into val
train_seqs = train_seqs + [CDR_FULL_SEQ] * 100 + [CDR3_SEQ] * 100
print(f'Train : {len(train_seqs)} | Val : {len(val_seqs)}')

# ── CDR3-only corpus for Phase 1 curriculum ───────────────────────────────────
cdr_lookup = {d['full']: d for d in all_cdr_data}

def extract_cdr3(seq, cdr_row=None):
    if cdr_row and cdr_row.get('cdr3', '') and len(cdr_row['cdr3']) >= 5:
        return cdr_row['cdr3']
    m = re.search(r'(C[A-Z]{3,25}W[GQ][QG])', seq)
    return m.group(1) if m else seq[-20:]

cdr3_corpus, cdr3_val_list = [], []
for seq in train_seqs:
    cdr3 = extract_cdr3(seq, cdr_lookup.get(seq))
    if len(cdr3) >= 5: cdr3_corpus.append(cdr3)
cdr3_corpus += [CDR3_SEQ] * 50
for seq in val_seqs:
    cdr3 = extract_cdr3(seq, cdr_lookup.get(seq))
    if len(cdr3) >= 5: cdr3_val_list.append(cdr3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VOCABULARY  (identical to Module2Script)
# ══════════════════════════════════════════════════════════════════════════════

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
VOCAB = {
    '[PAD]': 0, '[UNK]': 1, '[MASK]': 2,
    '[CLS]': 3, '[SEP]': 4, '[CDR_START]': 5, '[CDR_END]': 6,
}
for _i, _aa in enumerate(AMINO_ACIDS):
    VOCAB[_aa] = _i + 7
ID2TOKEN  = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
MAX_LEN    = 128


def tokenize(sequence, cdr_start=None, cdr_end=None, max_len=MAX_LEN):
    tokens = [VOCAB['[CLS]']]
    for i, aa in enumerate(sequence.upper()):
        if cdr_start is not None and i == cdr_start:
            tokens.append(VOCAB['[CDR_START]'])
        if cdr_end is not None and i == cdr_end:
            tokens.append(VOCAB['[CDR_END]'])
        tokens.append(VOCAB.get(aa, VOCAB['[UNK]']))
    tokens.append(VOCAB['[SEP]'])
    tokens = tokens[:max_len]
    mask   = [1] * len(tokens)
    while len(tokens) < max_len:
        tokens.append(VOCAB['[PAD]']); mask.append(0)
    return tokens, mask


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASETS  (identical to Module2Script)
# ══════════════════════════════════════════════════════════════════════════════

class AntibodyMLMDataset(Dataset):
    SPECIAL = {VOCAB['[PAD]'], VOCAB['[CLS]'], VOCAB['[SEP]'],
               VOCAB['[CDR_START]'], VOCAB['[CDR_END]']}

    def __init__(self, sequences, mask_prob=0.15, max_len=MAX_LEN):
        self.sequences = sequences
        self.mask_prob = mask_prob
        self.max_len   = max_len

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq  = self.sequences[idx]
        m    = re.search(r'(CAR[A-Z]+?WGQGT)', seq)
        cs, ce = (m.start(), m.end()) if m else (None, None)
        token_ids, attn_mask = tokenize(seq, cdr_start=cs, cdr_end=ce,
                                         max_len=self.max_len)
        input_ids = token_ids.copy()
        labels    = [-100] * self.max_len
        for i in range(self.max_len):
            if attn_mask[i] == 0 or input_ids[i] in self.SPECIAL: continue
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]
                r = random.random()
                if   r < 0.80: input_ids[i] = VOCAB['[MASK]']
                elif r < 0.90: input_ids[i] = random.randint(7, VOCAB_SIZE - 1)
        return {
            'input_ids':      torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'labels':         torch.tensor(labels,    dtype=torch.long),
        }


class AntibodyNSPDataset(Dataset):
    def __init__(self, sequences, max_len=MAX_LEN):
        self.sequences = [s for s in sequences if len(s) >= 30]
        self.max_len   = max_len

    def _get_cdr_pair(self, seq):
        cdr1 = seq[26:38] if len(seq) > 38 else seq[:12]
        m    = re.search(r'(CAR[A-Z]+?WGQGT)', seq)
        cdr3 = m.group(1) if m else seq[-20:]
        return cdr1, cdr3

    def __len__(self): return len(self.sequences) * 2

    def __getitem__(self, idx):
        seq_idx     = idx % len(self.sequences)
        is_positive = idx < len(self.sequences)
        cdr1_a, cdr3_a = self._get_cdr_pair(self.sequences[seq_idx])
        if is_positive:
            cdr_b, label = cdr3_a, 1
        else:
            neg_idx  = random.randint(0, len(self.sequences) - 1)
            _, cdr_b = self._get_cdr_pair(self.sequences[neg_idx])
            label    = 0
        ids_a, mask_a = tokenize(cdr1_a, max_len=self.max_len // 2)
        ids_b, mask_b = tokenize(cdr_b,  max_len=self.max_len // 2)
        return {
            'input_ids_a':      torch.tensor(ids_a,  dtype=torch.long),
            'attention_mask_a': torch.tensor(mask_a, dtype=torch.long),
            'input_ids_b':      torch.tensor(ids_b,  dtype=torch.long),
            'attention_mask_b': torch.tensor(mask_b, dtype=torch.long),
            'nsp_label':        torch.tensor(label,  dtype=torch.long),
        }


# ── Safe fallback helpers (identical to Module2Script) ───────────────────────
def _safe_val_seqs(val_list, train_list, min_len=1):
    filtered = [s for s in val_list if len(s) >= min_len]
    if filtered: return filtered
    fallback = [s for s in train_list if len(s) >= min_len]
    return fallback[:max(1, len(fallback) // 10)]

def _safe_nsp(seq_list, min_len=30):
    filtered = [s for s in seq_list if len(s) >= min_len]
    if len(filtered) >= 2: return filtered
    padded = [s * (min_len // max(len(s), 1) + 1) for s in seq_list]
    padded = [s[:min_len] for s in padded if s]
    if len(padded) >= 2: return padded
    base = padded if padded else ['A' * min_len]
    return (base * 4)[:4]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL  (identical to Module2Script)
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class AntibodyEncoder(nn.Module):
    """
    Transformer encoder from Module2Script (paper §3.3.2).
    All architecture hyperparameters injected so Optuna can vary them.
    Paper spec: 6 layers, 8 heads, ~50M params (d_model=512).
    Smaller d_model values supported for free-GPU compatibility.
    d_model MUST be divisible by num_heads — enforced in the objective.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model,
                                      padding_idx=VOCAB['[PAD]'])
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                              enable_nested_tensor=False)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 2))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        key_pad = (attention_mask == 0) if attention_mask is not None else None
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=key_pad)

    def get_embedding(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)[:, 0, :]

    def forward_mlm(self, input_ids, attention_mask=None):
        return self.mlm_head(self.encode(input_ids, attention_mask))

    def forward_nsp(self, ids_a, mask_a, ids_b, mask_b):
        cls_a = self.get_embedding(ids_a, mask_a)
        cls_b = self.get_embedding(ids_b, mask_b)
        return self.nsp_head(torch.cat([cls_a, cls_b], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCHEDULER FACTORY
# Paper §3.4.3 explicitly lists "learning rate schedules" as a tunable
# hyperparameter.  This factory maps a string name (chosen by Optuna) to a
# PyTorch LR scheduler, fixing GAP 2 from the previous version.
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(name: str, optimizer,
                    total_steps: int, warmup_steps: int = 200):
    """
    Build a LR scheduler by name.

    Choices exposed to Optuna (paper §3.4.2 uses warmup_cosine for Phase 1
    and cosine for Phase 2/3 as defaults; all four are valid alternatives):

      'warmup_cosine' — linear warmup then cosine decay  (paper Phase 1 default)
      'cosine'        — cosine annealing, no warmup      (paper Phase 2/3 default)
      'linear'        — linear decay from LR to 0
      'constant'      — no schedule; constant LR

    Parameters
    ----------
    name         : scheduler name string
    optimizer    : AdamW optimiser to attach
    total_steps  : total optimiser steps across all epochs
    warmup_steps : steps for linear warmup (warmup_cosine only)
    """
    if name == 'warmup_cosine':
        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps))

    elif name == 'linear':
        def _linear(step):
            return max(0.0, 1.0 - step / max(1, total_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _linear)

    elif name == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f'Unknown scheduler: {name}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING LOOP  (mirrors Module2Script run_epoch exactly)
# Extended with collect_preds flag for per-token F1 computation.
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, mlm_loader, nsp_loader, optimizer, scheduler,
              mlm_loss_fn, nsp_loss_fn, nsp_weight, grad_clip,
              train=True, collect_preds=False):
    """
    One MLM + NSP epoch.

    Parameters
    ----------
    collect_preds : bool
        When True, accumulate (pred, label) pairs for every masked token so
        the caller can compute token-level macro F1 (paper §3.4.4 metric 1).
        Minor memory overhead; set True only on the final validation pass.

    Returns
    -------
    avg_mlm_loss : float
    all_preds    : list[int]  populated only when collect_preds=True
    all_labels   : list[int]  populated only when collect_preds=True
    """
    model.train() if train else model.eval()
    ep_mlm              = 0.0
    all_preds, all_labels = [], []
    nsp_iter = iter(nsp_loader) if (train and nsp_loader) else None

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mlm_batch in mlm_loader:
            ids  = mlm_batch['input_ids'].to(device)
            mask = mlm_batch['attention_mask'].to(device)
            lbl  = mlm_batch['labels'].to(device)

            logits   = model.forward_mlm(ids, mask)
            loss_mlm = mlm_loss_fn(logits.view(-1, VOCAB_SIZE), lbl.view(-1))
            active   = lbl.view(-1) != -100

            if collect_preds and active.sum() > 0:
                preds = logits.view(-1, VOCAB_SIZE).argmax(-1)
                all_preds.extend(preds[active].cpu().tolist())
                all_labels.extend(lbl.view(-1)[active].cpu().tolist())

            loss = loss_mlm
            if train and nsp_iter:
                try:
                    nsp_batch = next(nsp_iter)
                except StopIteration:
                    nsp_iter  = iter(nsp_loader)
                    nsp_batch = next(nsp_iter)
                ids_a    = nsp_batch['input_ids_a'].to(device)
                msk_a    = nsp_batch['attention_mask_a'].to(device)
                ids_b    = nsp_batch['input_ids_b'].to(device)
                msk_b    = nsp_batch['attention_mask_b'].to(device)
                nsp_lbl  = nsp_batch['nsp_label'].to(device)
                loss_nsp = nsp_loss_fn(
                    model.forward_nsp(ids_a, msk_a, ids_b, msk_b), nsp_lbl)
                loss = loss_mlm + nsp_weight * loss_nsp

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # paper §3.4.2
                optimizer.step()
                if scheduler: scheduler.step()

            ep_mlm += loss_mlm.item()

    avg_mlm = ep_mlm / max(1, len(mlm_loader))
    return avg_mlm, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BENCHMARK METRIC HELPERS  (mirror Module2Script §7.4 exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_batch(model, seqs, batch_size=64):
    """Encode sequences -> (N, d_model) numpy array.  Mirrors Module2Script §6.1."""
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            ids_list, mask_list = [], []
            for s in batch:
                ids, msk = tokenize(s)
                ids_list.append(ids); mask_list.append(msk)
            ids_t  = torch.tensor(ids_list,  dtype=torch.long).to(device)
            mask_t = torch.tensor(mask_list, dtype=torch.long).to(device)
            embs   = model.get_embedding(ids_t, mask_t)
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy()


def _compute_token_f1(all_preds, all_labels):
    """
    Token-level macro-averaged F1 for MLM reconstruction (paper §3.4.3 + §3.4.4).
    Paper §3.4.3: "F1-score for classification tasks."
    Only amino-acid tokens (vocab ids 7-26) appear in labels because special
    tokens are excluded by the -100 ignore_index in CrossEntropyLoss.

    Returns F1 in [0, 1].  Returns 0.0 if no predictions were collected.
    """
    if not all_preds:
        return 0.0
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)


def _compute_silhouette(model, all_cdr_data, n_seqs=N_SIL_SEQS):
    """
    Embedding space coherence — silhouette score over V-gene family clusters.
    Mirrors Module2Script §7.4 exactly.

    Labels = V-gene family  (e.g. IGHV1-2*02 -> 'IGHV1').
    Families with < 2 members excluded (silhouette requirement).

    Returns float in [-1, 1] (higher is better), or None if insufficient data.
    """
    sil_seqs, sil_labels = [], []
    for d in all_cdr_data[:n_seqs * 4]:
        v_call = d.get('v_call', '')
        if v_call and v_call not in ('nan', ''):
            family = v_call.split('*')[0].split('-')[0]
            sil_seqs.append(d['full'])
            sil_labels.append(family)

    counts   = Counter(sil_labels)
    filtered = [(s, l) for s, l in zip(sil_seqs, sil_labels) if counts[l] >= 2]
    if len(filtered) < 4 or len(set(l for _, l in filtered)) < 2:
        return None

    sil_seqs   = [x[0] for x in filtered[:n_seqs]]
    sil_labels = [x[1] for x in filtered[:n_seqs]]
    sil_embs   = _embed_batch(model, sil_seqs, batch_size=32)
    return silhouette_score(sil_embs, sil_labels)


def _compute_nat_syn_auroc(model, all_repertoire, n_seqs=N_AUROC_SEQS):
    """
    Discriminative AUROC — logistic classifier on embeddings separating real
    human BCR sequences from amino-acid-shuffled (synthetic) decoys.
    Mirrors Module2Script §7.4 exactly.

    Returns AUROC in [0, 1] (higher is better; > 0.5 means encoder learned
    sequence structure).  Returns 0.5 (chance) if data is insufficient.
    """
    if len(all_repertoire) < 4:
        return 0.5

    n        = min(n_seqs, len(all_repertoire))
    nat_seqs = random.sample(all_repertoire, n)
    syn_seqs = []
    for s in nat_seqs:
        shuffled = list(s); random.shuffle(shuffled)
        syn_seqs.append(''.join(shuffled))

    nat_embs = _embed_batch(model, nat_seqs, batch_size=32)
    syn_embs = _embed_batch(model, syn_seqs, batch_size=32)

    X   = np.concatenate([nat_embs, syn_embs])
    y   = np.array([1] * len(nat_embs) + [0] * len(syn_embs))
    idx = np.random.permutation(len(X))
    sp  = int(0.8 * len(X))
    X_tr, X_te = X[idx[:sp]], X[idx[sp:]]
    y_tr, y_te = y[idx[:sp]], y[idx[sp:]]

    if len(set(y_te)) < 2:
        return 0.5

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective for Module 2 — Antibody Sequence Encoder.

    SEARCH SPACE  (paper §3.3.2 + §3.4.2 + §3.4.3)
    ─────────────────────────────────────────────────────────────────────────
    Architecture  : d_model, num_heads, num_layers, d_ff, dropout
    Optimiser     : lr_pretrain, lr_finetune, weight_decay
    LR schedules  : sched_phase1, sched_phase2, sched_finetune  <- GAP 2 fixed
    Loss/training : nsp_weight, grad_clip, mlm_prob
    Batch sizes   : batch_phase1, batch_phase2

    OBJECTIVE  (paper §3.4.3 "weighted average of task-specific metrics")
    ─────────────────────────────────────────────────────────────────────────
    composite = W_F1    * token_macro_f1    (§3.4.4 metric 1)  <- GAP 1 fixed
              + W_SIL   * silhouette_norm   (§3.4.4 metric 2)
              + W_AUROC * nat_vs_syn_auroc  (§3.4.4 metric 3)

    All metrics higher-is-better; objective negated for Optuna minimisation.
    If silhouette cannot be computed, its weight is redistributed to the other
    two metrics proportionally.

    Returns
    -------
    float — lower is better (= -composite).
    """

    # ── 8.1  Architecture (paper §3.3.2) ─────────────────────────────────────
    num_heads  = trial.suggest_categorical('num_heads',  [4, 8])
    d_model    = trial.suggest_categorical('d_model',    [128, 256, 512])
    # Hard constraint: d_model must be divisible by num_heads.
    # 4 always divides 128, 256, 512 so is a safe fallback.
    if d_model % num_heads != 0:
        num_heads = 4

    num_layers = trial.suggest_int('num_layers', 2, 6)                    # paper: 6
    d_ff       = trial.suggest_categorical('d_ff', [256, 512, 1024, 2048])
    dropout    = trial.suggest_float('dropout', 0.05, 0.30, step=0.05)

    # ── 8.2  Optimiser (paper §3.4.2) ────────────────────────────────────────
    lr_pretrain  = trial.suggest_float('lr_pretrain',  1e-5, 1e-3, log=True)   # paper: 5e-5
    lr_finetune  = trial.suggest_float('lr_finetune',  1e-6, 1e-4, log=True)   # paper: 1e-5
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)   # Module2Script TODO

    # ── 8.3  LR schedules (paper §3.4.3 — schedules explicitly listed as      ─
    #         tunable; each phase gets an independent choice)                   ─
    SCHED_CHOICES = ['warmup_cosine', 'cosine', 'linear', 'constant']
    sched_p1 = trial.suggest_categorical('sched_phase1',   SCHED_CHOICES)
    sched_p2 = trial.suggest_categorical('sched_phase2',   SCHED_CHOICES)
    sched_ft = trial.suggest_categorical('sched_finetune', SCHED_CHOICES)

    # ── 8.4  Loss / training (paper §3.4.2 + Module2Script TODO comments) ────
    nsp_weight = trial.suggest_float('nsp_weight', 0.1, 1.0, step=0.1)   # TODO NSP_WEIGHT
    grad_clip  = trial.suggest_float('grad_clip',  0.5, 5.0, step=0.5)   # paper: 1.0
    mlm_prob   = trial.suggest_float('mlm_prob',   0.10, 0.25, step=0.05) # paper: 0.15

    # ── 8.5  Batch sizes (paper §3.4.2: "batch size 32") ─────────────────────
    batch_phase1 = trial.suggest_categorical('batch_phase1', [8,  16, 32])
    batch_phase2 = trial.suggest_categorical('batch_phase2', [16, 32, 64])

    # ── 8.6  Build model ──────────────────────────────────────────────────────
    try:
        model = AntibodyEncoder(
            vocab_size=VOCAB_SIZE, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, d_ff=d_ff, max_len=MAX_LEN, dropout=dropout,
        ).to(device)
    except Exception as e:
        raise optuna.exceptions.TrialPruned(f'Model build failed: {e}')

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    # ── 8.7  DataLoaders ──────────────────────────────────────────────────────
    _cdr3_val = _safe_val_seqs(cdr3_val_list, cdr3_corpus, min_len=5)
    _val_safe = _safe_val_seqs(val_seqs, train_seqs)

    mlm_cdr3_tr = DataLoader(AntibodyMLMDataset(cdr3_corpus, mask_prob=mlm_prob),
                              batch_size=batch_phase1, shuffle=True,  num_workers=0)
    mlm_cdr3_vl = DataLoader(AntibodyMLMDataset(_cdr3_val, mask_prob=mlm_prob),
                              batch_size=batch_phase1, shuffle=False, num_workers=0)
    nsp_cdr3_tr = DataLoader(AntibodyNSPDataset(_safe_nsp(cdr3_corpus)),
                              batch_size=batch_phase1, shuffle=True,
                              drop_last=True, num_workers=0)
    mlm_tr = DataLoader(AntibodyMLMDataset(train_seqs, mask_prob=mlm_prob),
                         batch_size=batch_phase2, shuffle=True,  num_workers=0)
    mlm_vl = DataLoader(AntibodyMLMDataset(_val_safe, mask_prob=mlm_prob),
                         batch_size=batch_phase2, shuffle=False, num_workers=0)
    nsp_tr = DataLoader(AntibodyNSPDataset(_safe_nsp(train_seqs)),
                         batch_size=max(4, batch_phase2 // 4), shuffle=True,
                         drop_last=True, num_workers=0)

    # ── 8.8  Phase 1 — CDR3-only pre-training (curriculum, paper §3.4.2) ─────
    opt1   = torch.optim.AdamW(model.parameters(),
                                lr=lr_pretrain, weight_decay=weight_decay)
    steps1 = TRIAL_EPOCHS_PHASE1 * max(1, len(mlm_cdr3_tr))
    sc1    = build_scheduler(sched_p1, opt1, steps1)

    best_p1 = float('inf'); pat_p1 = 0
    for _ in range(TRIAL_EPOCHS_PHASE1):
        run_epoch(model, mlm_cdr3_tr, nsp_cdr3_tr,
                  opt1, sc1, mlm_loss_fn, nsp_loss_fn,
                  nsp_weight, grad_clip, train=True)
        vl, _, _ = run_epoch(model, mlm_cdr3_vl, None, None, None,
                              mlm_loss_fn, nsp_loss_fn,
                              nsp_weight, grad_clip, train=False)
        if vl < best_p1: best_p1, pat_p1 = vl, 0
        else:
            pat_p1 += 1
            if pat_p1 >= 10: break          # paper §3.4.2 patience = 10

    # ── 8.9  Phase 2 — Full variable region pre-training ─────────────────────
    opt2   = torch.optim.AdamW(model.parameters(),
                                lr=lr_pretrain, weight_decay=weight_decay)
    steps2 = TRIAL_EPOCHS_PHASE2 * max(1, len(mlm_tr))
    sc2    = build_scheduler(sched_p2, opt2, steps2)

    best_p2 = float('inf'); pat_p2 = 0
    for _ in range(TRIAL_EPOCHS_PHASE2):
        run_epoch(model, mlm_tr, nsp_tr,
                  opt2, sc2, mlm_loss_fn, nsp_loss_fn,
                  nsp_weight, grad_clip, train=True)
        vl, _, _ = run_epoch(model, mlm_vl, None, None, None,
                              mlm_loss_fn, nsp_loss_fn,
                              nsp_weight, grad_clip, train=False)
        if vl < best_p2: best_p2, pat_p2 = vl, 0
        else:
            pat_p2 += 1
            if pat_p2 >= 10: break

    # ── 8.10 Phase 3 — Fine-tuning at lower LR (paper §3.4.2) ────────────────
    opt3   = torch.optim.AdamW(model.parameters(),
                                lr=lr_finetune, weight_decay=weight_decay)
    steps3 = TRIAL_EPOCHS_FINETUNE * max(1, len(mlm_tr))
    sc3    = build_scheduler(sched_ft, opt3, steps3)

    best_p3 = float('inf'); pat_p3 = 0
    best_preds, best_labels = [], []

    for ep in range(TRIAL_EPOCHS_FINETUNE):
        run_epoch(model, mlm_tr, nsp_tr,
                  opt3, sc3, mlm_loss_fn, nsp_loss_fn,
                  nsp_weight, grad_clip, train=True)
        # collect_preds=True so we have token predictions for F1
        vl, ep_preds, ep_labels = run_epoch(
            model, mlm_vl, None, None, None,
            mlm_loss_fn, nsp_loss_fn,
            nsp_weight, grad_clip, train=False, collect_preds=True)

        # Report val loss to Optuna for MedianPruner
        trial.report(vl, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if vl < best_p3:
            best_p3     = vl
            best_preds  = ep_preds    # keep preds from lowest-loss val epoch
            best_labels = ep_labels
            pat_p3      = 0
        else:
            pat_p3 += 1
            if pat_p3 >= 10: break

    # ── 8.11 Benchmark metrics (paper §3.4.4) ─────────────────────────────────

    # Metric 1 — token-level macro F1 for MLM reconstruction  (GAP 1 fixed)
    # Paper §3.4.3: "F1-score for classification".
    # Macro averaging treats each amino-acid token class equally, handling
    # natural class imbalance in the 20-aa vocabulary better than raw accuracy.
    mlm_f1 = _compute_token_f1(best_preds, best_labels)

    # Metric 2 — silhouette score (embedding space coherence)
    # Clustered by V-gene family; normalised [-1,1] -> [0,1]
    sil = _compute_silhouette(model, all_cdr_data, n_seqs=N_SIL_SEQS)
    sil_norm = (sil + 1.0) / 2.0 if sil is not None else None

    # Metric 3 — natural vs synthetic AUROC (discriminative ability)
    auroc = _compute_nat_syn_auroc(model, all_repertoire, n_seqs=N_AUROC_SEQS)

    # ── 8.12 Weighted composite (paper §3.4.3) ────────────────────────────────
    if sil_norm is not None:
        composite    = W_F1 * mlm_f1 + W_SIL * sil_norm + W_AUROC * auroc
        _w_f1, _w_sil, _w_auroc = W_F1, W_SIL, W_AUROC
    else:
        # Silhouette unavailable — redistribute its weight proportionally
        _total   = W_F1 + W_AUROC
        _w_f1    = W_F1    / _total
        _w_auroc = W_AUROC / _total
        _w_sil   = 0.0
        composite = _w_f1 * mlm_f1 + _w_auroc * auroc
        sil_norm  = float('nan')

    # Store individual metrics as user attributes for post-hoc analysis
    trial.set_user_attr('mlm_f1',    round(mlm_f1,    4))
    trial.set_user_attr('sil_score', round(sil_norm,  4)
                        if not math.isnan(sil_norm) else None)
    trial.set_user_attr('auroc',     round(auroc,     4))
    trial.set_user_attr('composite', round(composite, 4))
    trial.set_user_attr('weights',   {'W_F1': _w_f1,
                                      'W_SIL': _w_sil,
                                      'W_AUROC': _w_auroc})

    # Negate: Optuna minimises; all metrics are higher-is-better
    return -composite


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RUN THE STUDY
# Paper §3.4.3: Bayesian TPE, 50 trials, parallel on multiple GPU instances.
# Parallelism uses a shared SQLite backend so any number of workers running
# this same script coordinate automatically without extra configuration.
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n{"="*70}')
print(f'  MODULE 2 OPTUNA STUDY')
print(f'  Trials        : {N_TRIALS}  (paper: 50)')
print(f'  Sampler       : Bayesian TPE        (paper §3.4.3)')
print(f'  Pruner        : MedianPruner')
print(f'  Parallel jobs : {N_PARALLEL_JOBS}                   (paper §3.4.3)')
print(f'  Storage       : {STORAGE_PATH}')
print(f'  Objective     : -(W_F1={W_F1}*macro_f1 '
      f'+ W_SIL={W_SIL}*sil + W_AUROC={W_AUROC}*auroc)')
print(f'{"="*70}\n')

sampler = TPESampler(seed=42)                   # Bayesian TPE as in §3.4.3
pruner  = optuna.pruners.MedianPruner(
    n_startup_trials=3,                         # complete >= 3 trials before pruning
    n_warmup_steps=1,                           # wait >= 1 fine-tune epoch
)

# load_if_exists=True lets additional workers join an in-progress study
study = optuna.create_study(
    study_name     = STUDY_NAME,
    direction      = 'minimize',
    sampler        = sampler,
    pruner         = pruner,
    storage        = STORAGE_PATH,              # shared SQLite — enables parallelism
    load_if_exists = True,
)

study.optimize(
    objective,
    n_trials          = N_TRIALS,
    n_jobs            = N_PARALLEL_JOBS,        # paper: parallel GPU instances
    timeout           = None,
    show_progress_bar = False,
    callbacks         = [
        lambda study, trial: print(
            f'  Trial {trial.number:>3d} | '
            f'composite={(-trial.value):.4f} | '
            f'f1={trial.user_attrs.get("mlm_f1",    "n/a")} | '
            f'sil={trial.user_attrs.get("sil_score","n/a")} | '
            f'auroc={trial.user_attrs.get("auroc",  "n/a")} | '
            f'state={trial.state.name[:4]} | '
            f'best={(-study.best_value):.4f}'
        ) if trial.value is not None else None
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RESULTS & JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

best      = study.best_trial
completed = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials
             if t.state == optuna.trial.TrialState.PRUNED]

print(f'\n{"="*70}')
print('  OPTUNA STUDY COMPLETE')
print(f'  Best trial     : #{best.number}')
print(f'  Best composite : {-best.value:.6f}')
print(f'\n  Best trial metrics (paper §3.4.4):')
print(f'    Token macro F1 (MLM)         : {best.user_attrs.get("mlm_f1",    "n/a")}')
print(f'    Silhouette score (normalised) : {best.user_attrs.get("sil_score", "n/a")}')
print(f'    Natural vs Synthetic AUROC   : {best.user_attrs.get("auroc",     "n/a")}')
print(f'\n  Best hyperparameters:')
for k, v in best.params.items():
    print(f'    {k:<26s}: {v}')
print(f'\n  Trials completed : {len(completed)} / {N_TRIALS}')
print(f'  Trials pruned    : {len(pruned)}')

results_path = f'optuna_module2_{RUN_ID}.json'
with open(results_path, 'w') as _fh:
    _json.dump({
        'run_id'           : RUN_ID,
        'n_trials'         : N_TRIALS,
        'n_completed'      : len(completed),
        'n_pruned'         : len(pruned),
        'best_trial'       : best.number,
        'best_composite'   : -best.value,
        'best_metrics'     : {
            'mlm_f1'   : best.user_attrs.get('mlm_f1'),
            'sil_score': best.user_attrs.get('sil_score'),
            'auroc'    : best.user_attrs.get('auroc'),
        },
        'objective_weights': {'W_F1': W_F1, 'W_SIL': W_SIL, 'W_AUROC': W_AUROC},
        'best_params'      : best.params,
        'all_trials'       : [
            {
                'number'   : t.number,
                'composite': -t.value if t.value is not None else None,
                'state'    : t.state.name,
                'params'   : t.params,
                'mlm_f1'   : t.user_attrs.get('mlm_f1'),
                'sil_score': t.user_attrs.get('sil_score'),
                'auroc'    : t.user_attrs.get('auroc'),
            }
            for t in study.trials
        ],
    }, _fh, indent=2)
print(f'\n  Results saved -> {results_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RETRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# Uses full epoch counts from Module2Script §3.4.2, not the short trial counts.
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n{"="*70}')
print('  RETRAINING FINAL MODEL WITH BEST HYPERPARAMETERS')
print(f'  Full epoch counts: P1={FULL_EPOCHS_PHASE1}, '
      f'P2={FULL_EPOCHS_PHASE2}, FT={FULL_EPOCHS_FINETUNE}')
print(f'{"="*70}')

bp  = best.params
_nh = bp['num_heads']
_dm = bp['d_model']
if _dm % _nh != 0: _nh = 4

final_model = AntibodyEncoder(
    vocab_size=VOCAB_SIZE, d_model=_dm, num_heads=_nh,
    num_layers=bp['num_layers'], d_ff=bp['d_ff'],
    max_len=MAX_LEN, dropout=bp['dropout'],
).to(device)

n_params = sum(p.numel() for p in final_model.parameters())
print(f'  Parameters : {n_params:,}  (~{n_params/1e6:.2f}M)')
print(f'  d_model={_dm} | layers={bp["num_layers"]} | '
      f'heads={_nh} | d_ff={bp["d_ff"]}')
print(f'  Schedules  : P1={bp["sched_phase1"]} | '
      f'P2={bp["sched_phase2"]} | FT={bp["sched_finetune"]}')

mlm_lf = nn.CrossEntropyLoss(ignore_index=-100)
nsp_lf = nn.CrossEntropyLoss()

_cv = _safe_val_seqs(cdr3_val_list, cdr3_corpus, min_len=5)
_vs = _safe_val_seqs(val_seqs, train_seqs)
mp  = bp['mlm_prob']
b1  = bp['batch_phase1']
b2  = bp['batch_phase2']

f_mlm_c_tr = DataLoader(AntibodyMLMDataset(cdr3_corpus, mask_prob=mp),
                         batch_size=b1, shuffle=True,  num_workers=0)
f_mlm_c_vl = DataLoader(AntibodyMLMDataset(_cv, mask_prob=mp),
                         batch_size=b1, shuffle=False, num_workers=0)
f_nsp_c_tr = DataLoader(AntibodyNSPDataset(_safe_nsp(cdr3_corpus)),
                         batch_size=b1, shuffle=True, drop_last=True, num_workers=0)
f_mlm_tr   = DataLoader(AntibodyMLMDataset(train_seqs, mask_prob=mp),
                         batch_size=b2, shuffle=True,  num_workers=0)
f_mlm_vl   = DataLoader(AntibodyMLMDataset(_vs, mask_prob=mp),
                         batch_size=b2, shuffle=False, num_workers=0)
f_nsp_tr   = DataLoader(AntibodyNSPDataset(_safe_nsp(train_seqs)),
                         batch_size=max(4, b2 // 4), shuffle=True,
                         drop_last=True, num_workers=0)

all_val_losses, all_val_f1s, all_phase_labels = [], [], []

# ── Final Phase 1 ─────────────────────────────────────────────────────────────
fo1  = torch.optim.AdamW(final_model.parameters(),
                          lr=bp['lr_pretrain'], weight_decay=bp['weight_decay'])
fsc1 = build_scheduler(bp['sched_phase1'], fo1,
                        FULL_EPOCHS_PHASE1 * max(1, len(f_mlm_c_tr)))
bvp1 = float('inf'); pp1 = 0
for ep in range(1, FULL_EPOCHS_PHASE1 + 1):
    run_epoch(final_model, f_mlm_c_tr, f_nsp_c_tr,
              fo1, fsc1, mlm_lf, nsp_lf,
              bp['nsp_weight'], bp['grad_clip'], train=True)
    vl, preds, labels = run_epoch(
        final_model, f_mlm_c_vl, None, None, None,
        mlm_lf, nsp_lf, bp['nsp_weight'], bp['grad_clip'],
        train=False, collect_preds=True)
    vf1 = _compute_token_f1(preds, labels)
    all_val_losses.append(vl); all_val_f1s.append(vf1)
    all_phase_labels.append('P1-CDR3')
    print(f'  [P1] Epoch {ep}/{FULL_EPOCHS_PHASE1} | '
          f'val_loss={vl:.4f}  macro_f1={vf1:.4f}')
    if vl < bvp1: bvp1, pp1 = vl, 0
    else:
        pp1 += 1
        if pp1 >= 10: break

# ── Final Phase 2 ─────────────────────────────────────────────────────────────
fo2  = torch.optim.AdamW(final_model.parameters(),
                          lr=bp['lr_pretrain'], weight_decay=bp['weight_decay'])
fsc2 = build_scheduler(bp['sched_phase2'], fo2,
                        FULL_EPOCHS_PHASE2 * max(1, len(f_mlm_tr)))
bvp2 = float('inf'); pp2 = 0
for ep in range(1, FULL_EPOCHS_PHASE2 + 1):
    run_epoch(final_model, f_mlm_tr, f_nsp_tr,
              fo2, fsc2, mlm_lf, nsp_lf,
              bp['nsp_weight'], bp['grad_clip'], train=True)
    vl, preds, labels = run_epoch(
        final_model, f_mlm_vl, None, None, None,
        mlm_lf, nsp_lf, bp['nsp_weight'], bp['grad_clip'],
        train=False, collect_preds=True)
    vf1 = _compute_token_f1(preds, labels)
    all_val_losses.append(vl); all_val_f1s.append(vf1)
    all_phase_labels.append('P2-Full')
    print(f'  [P2] Epoch {ep}/{FULL_EPOCHS_PHASE2} | '
          f'val_loss={vl:.4f}  macro_f1={vf1:.4f}')
    if vl < bvp2: bvp2, pp2 = vl, 0
    else:
        pp2 += 1
        if pp2 >= 10: break

torch.save({
    'model_state_dict': final_model.state_dict(),
    'phase'           : 'pretrain',
    'val_mlm_loss'    : all_val_losses[-1],
    'val_macro_f1'    : all_val_f1s[-1],
    'best_params'     : bp,
    'vocab'           : VOCAB,
    'd_model'         : final_model.d_model,
}, f'antibody_encoder_pretrained_optuna_{RUN_ID}.pt')
print(f'\n  Pre-trained weights saved -> '
      f'antibody_encoder_pretrained_optuna_{RUN_ID}.pt')

# ── Final Phase 3 — fine-tuning ───────────────────────────────────────────────
fo3  = torch.optim.AdamW(final_model.parameters(),
                          lr=bp['lr_finetune'], weight_decay=bp['weight_decay'])
fsc3 = build_scheduler(bp['sched_finetune'], fo3,
                        FULL_EPOCHS_FINETUNE * max(1, len(f_mlm_tr)))
bvp3 = float('inf'); pp3 = 0
for ep in range(1, FULL_EPOCHS_FINETUNE + 1):
    run_epoch(final_model, f_mlm_tr, f_nsp_tr,
              fo3, fsc3, mlm_lf, nsp_lf,
              bp['nsp_weight'], bp['grad_clip'], train=True)
    vl, preds, labels = run_epoch(
        final_model, f_mlm_vl, None, None, None,
        mlm_lf, nsp_lf, bp['nsp_weight'], bp['grad_clip'],
        train=False, collect_preds=True)
    vf1 = _compute_token_f1(preds, labels)
    all_val_losses.append(vl); all_val_f1s.append(vf1)
    all_phase_labels.append('P3-Finetune')
    print(f'  [FT] Epoch {ep}/{FULL_EPOCHS_FINETUNE} | '
          f'val_loss={vl:.4f}  macro_f1={vf1:.4f}')
    if vl < bvp3: bvp3, pp3 = vl, 0
    else:
        pp3 += 1
        if pp3 >= 10: break

torch.save({
    'model_state_dict': final_model.state_dict(),
    'phase'           : 'finetuned',
    'val_mlm_loss'    : all_val_losses[-1],
    'val_macro_f1'    : all_val_f1s[-1],
    'best_params'     : bp,
    'n_params'        : n_params,
    'vocab'           : VOCAB,
    'd_model'         : final_model.d_model,
}, f'antibody_encoder_module2_optuna_{RUN_ID}.pt')
print(f'  Fine-tuned weights saved    -> '
      f'antibody_encoder_module2_optuna_{RUN_ID}.pt  <- pass to Module 3')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n{"="*70}')
print('  MODULE 2 OPTUNA — FINAL SUMMARY')
print(f'{"="*70}')
print(f'  Study name           : {STUDY_NAME}')
print(f'  Storage              : {STORAGE_PATH}')
print(f'  Sampler              : Bayesian TPE         (paper §3.4.3)')
print(f'  Pruner               : MedianPruner')
print(f'  Parallel jobs        : {N_PARALLEL_JOBS}              (paper §3.4.3)')
print(f'  Trials requested     : {N_TRIALS}              (paper: 50)')
print(f'  Trials completed     : {len(completed)}')
print(f'  Trials pruned        : {len(pruned)}')
print()
print(f'  OBJECTIVE  (paper §3.4.3 + §3.4.4)')
print(f'    composite = W_F1*token_macro_f1 + W_SIL*sil_norm + W_AUROC*auroc')
print(f'    W_F1={W_F1}  W_SIL={W_SIL}  W_AUROC={W_AUROC}')
print()
print(f'  BEST HYPERPARAMETERS')
for k, v in best.params.items():
    print(f'    {k:<26s}: {v}')
print()
print(f'  FINAL MODEL AFTER FULL RETRAINING')
print(f'    Parameters           : {n_params:,}  (~{n_params/1e6:.2f}M)')
print(f'    Final val MLM loss   : {all_val_losses[-1]:.4f}')
print(f'    Final val macro F1   : {all_val_f1s[-1]:.4f}')
print()
print(f'  SAVED FILES')
print(f'    {results_path}')
print(f'    antibody_encoder_pretrained_optuna_{RUN_ID}.pt')
print(f'    antibody_encoder_module2_optuna_{RUN_ID}.pt  <- Module 3 input')
print(f'\n  Run ID : {RUN_ID}')
