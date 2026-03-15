# ── Flags ─────────────────────────────────────────────────────────────────────
dev_mode    = True
sanity_mode = True

OAS_MAX_PER_FILE = 100   if sanity_mode else (5000 if dev_mode else None)
MAX_REP_SEQS     = 6000    if sanity_mode else (200  if dev_mode else None)

EPOCHS_PHASE1   = 1  if sanity_mode else (5   if dev_mode else 20)
EPOCHS_PHASE2   = 1  if sanity_mode else (10  if dev_mode else 80)
EPOCHS_FINETUNE = 1  if sanity_mode else (3   if dev_mode else 10)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import subprocess, sys  # subprocess used for pip installs only
for pkg in ['torch', 'biopython', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math, random, re, warnings, glob

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser, PPBuilder
from Bio import SeqIO
from datetime import datetime
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cpu')
print('All imports successful.')


# %%
# ── Optimizer settings ────────────────────────────────────────────────────────
WEIGHT_DECAY    = 0.01   # current value — preprint unspecified for Mod2
                         # TODO Brev: test 1e-4, may improve convergence
NSP_WEIGHT      = 0.3    # TODO: tune with Optuna (preprint 3.4.3), range 0.1-1.0, should be PAVAN
                         # Reference: Devlin et al. 2019 (BERT) uses 1.0
GRAD_CLIP       = 1.0    # preprint section 3.4.2
# ── 1.1  Extract antibody sequence from CDR.pdb ───────────────────────────────
parser = PDBParser(QUIET=True)
ppb    = PPBuilder()

structure  = parser.get_structure('cdr', 'CDR.pdb')
CDR_FULL_SEQ = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(structure))

# Extract CDR3 region (IMGT: CAR...WGQGT motif)
cdr3_match = re.search(r'(CAR[A-Z]+?WGQGT)', CDR_FULL_SEQ)
CDR3_SEQ   = cdr3_match.group(1) if cdr3_match else CDR_FULL_SEQ[-25:]

# IMGT CDR region boundaries from the alignment image
# CDR1: ~27-38, CDR2: ~56-65, CDR3: ~105-117  (approximate IMGT positions)
CDR_REGIONS = {
    'CDR1': CDR_FULL_SEQ[26:38],
    'CDR2': CDR_FULL_SEQ[55:65],
    'CDR3': CDR3_SEQ,
    'Full': CDR_FULL_SEQ
}

print(f'Full variable region ({len(CDR_FULL_SEQ)} aa):')
print(f'  {CDR_FULL_SEQ}')
print(f'CDR1 : {CDR_REGIONS["CDR1"]}')
print(f'CDR2 : {CDR_REGIONS["CDR2"]}')
print(f'CDR3 : {CDR_REGIONS["CDR3"]}')

# %%
# ── 1.2  Load V-Gene alignment results ───────────────────────────────────────
vgene_df = pd.read_csv('V-Gene.csv')
print('V-Gene IgBLAST Results:')
print(vgene_df)

# Best V-gene hit
best_vgene = vgene_df.loc[vgene_df['Identity Percentage'].idxmax()]
print(f'\nBest V-gene match: {best_vgene["V-Gene Allele"]}  '
      f'({best_vgene["Identity Percentage"]}% identity, '
      f'{best_vgene["Matches"]}/{best_vgene["Total Bases"]} bp)')

# %%
# ── 1.3  Load natural human BCR repertoires ─────────────────────────────────
import json as _json; json = _json

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
    nt_seq = nt_seq.upper().replace('-','').replace(' ','')
    aa = []
    for i in range(0, len(nt_seq)-2, 3):
        codon = nt_seq[i:i+3]
        res   = CODON_TABLE.get(codon, 'X')
        if res == '*': break
        aa.append(res)
    return ''.join(aa)

def compute_shm_count(seq_aln, germ_aln):
    if not seq_aln or not germ_aln: return 0
    return sum(1 for o,g in zip(str(seq_aln), str(germ_aln))
               if o not in '-' and g not in '-N' and o != g)

# ── Loading OAS CSV (heavy + optionally light) ────────────────────────────────────
def load_oas_csv(csv_path, max_seqs=None):
    """Load OAS CSV. Extracts full VR AA, CDRs, V/J calls, SHM, metadata."""
    sequences, cdr_data = [], []
    study_meta = {}
    try:
        with open(csv_path, 'r') as fh:
            first_line = fh.readline().strip().strip('"')
            try: study_meta = json.loads(first_line)
            except: pass
    except: pass

    tissue  = study_meta.get('BSource', 'Unknown')
    btype   = study_meta.get('BType',   'Unknown')
    disease = study_meta.get('Disease', 'Unknown')
    subject = study_meta.get('Subject', 'Unknown')
    chain   = study_meta.get('Chain',   'Heavy')
    print(f'  [{chain}] BSource:{tissue} BType:{btype} Disease:{disease}')

    try:
        df = pd.read_csv(csv_path, skiprows=1)
        aa_col = 'sequence_alignment_aa' if 'sequence_alignment_aa' in df.columns else None
        prod   = df[df['productive']=='T'] if 'productive' in df.columns else df
        print(f'  Productive: {len(prod)}/{len(df)}')
        for _, row in prod.iterrows():
            if max_seqs and len(sequences) >= max_seqs: break
            aa_seq = str(row.get(aa_col,'')).strip() if aa_col else ''
            valid  = sum(1 for a in aa_seq if a in 'ACDEFGHIKLMNPQRSTVWY')
            if valid < 20 or len(aa_seq) < 20 or 'X' in aa_seq[:10]: continue
            shm = compute_shm_count(row.get('v_sequence_alignment',''),
                                    row.get('v_germline_alignment',''))
            sequences.append(aa_seq)
            cdr_data.append({
                'full': aa_seq, 'cdr1': str(row.get('cdr1_aa','')),
                'cdr2': str(row.get('cdr2_aa','')), 'cdr3': str(row.get('cdr3_aa','')),
                'v_call': str(row.get('v_call','')), 'j_call': str(row.get('j_call','')),
                'shm_count': shm, 'tissue_source': tissue,
                'donor_status': btype, 'disease': disease,
                'subject': subject, 'chain': chain,
                'source_file': os.path.basename(csv_path),  # ← add this line
            })
    except Exception as e:
        print(f'  ERROR: {e}')
    return sequences, cdr_data

def load_fasta_as_proteins(fasta_path, max_seqs=50000):
    proteins = []
    try:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            prot  = translate_nt(str(record.seq))
            valid = sum(1 for a in prot if a in 'ACDEFGHIKLMNPQRSTVWY')
            if valid >= 20 and len(prot) >= 20:
                proteins.append(prot)
            if len(proteins) >= max_seqs: break
    except FileNotFoundError:
        print(f'  WARNING: {fasta_path} not found.')
    return proteins

# ── Load IGH (heavy chain) ────────────────────────────────────────────────────
print('Loading IGH (heavy chain) data...')
oas_seqs_400, cdr_data_400 = load_oas_csv('ERR220400_Heavy_Bulk.csv')
oas_seqs_430, cdr_data_430 = load_oas_csv('ERR220430_Heavy_Bulk.csv')
CABREP_MAX = 100 if sanity_mode else (5000 if dev_mode else 50000)
cabrep_heavy = load_fasta_as_proteins('cAb-Rep_heavy.nt.fasta', max_seqs=CABREP_MAX)
print(f'  cAb-Rep heavy: {len(cabrep_heavy)}')

extra_oas_heavy, extra_cdr_heavy = [], []
for csv_file in glob.glob('*_Heavy_*.csv') + glob.glob('*_Heavy_Bulk.csv'):
    if 'ERR220400' in csv_file or 'ERR220430' in csv_file: continue
    s, c = load_oas_csv(csv_file, max_seqs=OAS_MAX_PER_FILE)
    extra_oas_heavy.extend(s); extra_cdr_heavy.extend(c)

print('\nLoading IGK/IGL (light chain) data...')
cabrep_light = load_fasta_as_proteins('cAb-Rep_light.nt.fasta', max_seqs=CABREP_MAX)
print(f'  cAb-Rep light (kappa+lambda): {len(cabrep_light)}')

light_cdr_data = []
# Load any OAS light chain CSVs present (kappa or lambda)
extra_oas_light = []
for csv_file in (glob.glob('*_Light_*.csv') + glob.glob('*_Kappa_*.csv') +
                 glob.glob('*_Lambda_*.csv') + glob.glob('*_Light_Bulk.csv')):
    s, c = load_oas_csv(csv_file, max_seqs=OAS_MAX_PER_FILE)
    extra_oas_light.extend(s); light_cdr_data.extend(c)

# ── Combine all sources ────────────────────────────────────────────────────────
all_oas_seqs_heavy = oas_seqs_400 + oas_seqs_430 + extra_oas_heavy
all_cdr_data_heavy = cdr_data_400 + cdr_data_430 + extra_cdr_heavy
heavy_proteins     = all_oas_seqs_heavy + cabrep_heavy

all_oas_seqs_light = extra_oas_light
light_proteins     = cabrep_light + all_oas_seqs_light  # IGK + IGL combined

# Light chains included in corpus but flagged separately
all_cdr_data   = all_cdr_data_heavy + light_cdr_data
all_oas_seqs   = all_oas_seqs_heavy + all_oas_seqs_light
all_repertoire = heavy_proteins + light_proteins  # full IGH+IGK+IGL corpus

print(f'\n── Corpus Summary ─────────────────────────────────────────')
print(f'  IGH (heavy) sequences      : {len(heavy_proteins)}')
print(f'    OAS CSVs                 : {len(all_oas_seqs_heavy)}')
print(f'    cAb-Rep FASTA            : {len(cabrep_heavy)}')
print(f'  IGK+IGL (light) sequences  : {len(light_proteins)}')
print(f'    OAS CSVs                 : {len(all_oas_seqs_light)}')
print(f'    cAb-Rep FASTA            : {len(cabrep_light)}')
print(f'  Total pre-training corpus  : {len(all_repertoire)}')
print(f'  CDR entries w/ metadata    : {len(all_cdr_data)}')
if all_cdr_data:
    shm_vals = [d['shm_count'] for d in all_cdr_data]
    print(f'  SHM — mean: {np.mean(shm_vals):.1f}  max: {max(shm_vals)}')
    print(f'  Tissue sources: {list(set(d["tissue_source"] for d in all_cdr_data))}')

# %%
# ── 2.1  Vocabulary ───────────────────────────────────────────────────────────
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

VOCAB = {
    '[PAD]':       0,
    '[UNK]':       1,
    '[MASK]':      2,
    '[CLS]':       3,
    '[SEP]':       4,
    '[CDR_START]': 5,
    '[CDR_END]':   6,
}
for i, aa in enumerate(AMINO_ACIDS):
    VOCAB[aa] = i + 7

ID2TOKEN  = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
MAX_LEN    = 128
MLM_PROB   = 0.15   

print(f'Vocabulary size : {VOCAB_SIZE}')
print(f'Max sequence len: {MAX_LEN}')
print('Special tokens  :', {k: VOCAB[k] for k in ['[PAD]','[MASK]','[CLS]','[CDR_START]','[CDR_END]']})

# %%
# ── 2.2  Tokenize function ────────────────────────────────────────────────────
def tokenize(sequence, cdr_start=None, cdr_end=None, max_len=MAX_LEN):
    """
    Tokenize an antibody sequence at amino-acid level.
    Inserts [CDR_START] / [CDR_END] tokens if CDR boundaries are provided.

    Returns
    -------
    token_ids     : list[int]  length == max_len
    attention_mask: list[int]  1 for real tokens, 0 for padding
    """
    tokens = [VOCAB['[CLS]']]
    for i, aa in enumerate(sequence.upper()):
        if cdr_start is not None and i == cdr_start:
            tokens.append(VOCAB['[CDR_START]'])
        if cdr_end   is not None and i == cdr_end:
            tokens.append(VOCAB['[CDR_END]'])
        tokens.append(VOCAB.get(aa, VOCAB['[UNK]']))
    tokens.append(VOCAB['[SEP]'])

    tokens = tokens[:max_len]
    mask   = [1] * len(tokens)
    while len(tokens) < max_len:
        tokens.append(VOCAB['[PAD]'])
        mask.append(0)
    return tokens, mask


# Quick test
ids, attn = tokenize(CDR3_SEQ, cdr_start=0, cdr_end=len(CDR3_SEQ))
print(f'CDR3 sequence         : {CDR3_SEQ}')
print(f'First 12 token IDs    : {ids[:12]}')
print(f'First 12 attn mask    : {attn[:12]}')
print(f'Decoded first 6 tokens: {[ID2TOKEN[t] for t in ids[:6]]}')

# %%
# ── 3.1  MLM Dataset ─────────────────────────────────────────────────────────
class AntibodyMLMDataset(Dataset):
    """
    Pre-training dataset using Masked Language Modelling.
    15% of amino acid tokens are replaced with [MASK] (80%),
    a random token (10%), or kept unchanged (10%).
    """
    SPECIAL = {VOCAB['[PAD]'], VOCAB['[CLS]'], VOCAB['[SEP]'],
               VOCAB['[CDR_START]'], VOCAB['[CDR_END]']}

    def __init__(self, sequences, mask_prob=MLM_PROB, max_len=MAX_LEN):
        self.sequences = sequences
        self.mask_prob = mask_prob
        self.max_len   = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Mark CDR3 boundaries if identifiable in this sequence
        cdr3_m = re.search(r'(CAR[A-Z]+?WGQGT)', seq)
        cs = cdr3_m.start() if cdr3_m else None
        ce = cdr3_m.end()   if cdr3_m else None

        token_ids, attn_mask = tokenize(seq, cdr_start=cs, cdr_end=ce,
                                         max_len=self.max_len)
        input_ids = token_ids.copy()
        labels    = [-100] * self.max_len

        for i in range(self.max_len):
            if attn_mask[i] == 0 or input_ids[i] in self.SPECIAL:
                continue
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]
                r = random.random()
                if r < 0.80:
                    input_ids[i] = VOCAB['[MASK]']
                elif r < 0.90:
                    input_ids[i] = random.randint(7, VOCAB_SIZE - 1)
                # else: unchanged (10%)

        return {
            'input_ids':      torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'labels':         torch.tensor(labels,    dtype=torch.long),
        }

# %%
# ── 3.2  NSP Dataset ──────────────────────────────────────────────────────────
class AntibodyNSPDataset(Dataset):
    """
    Next-Sentence Prediction adapted for antibodies.
    Positive pairs (label=1): two CDR sequences from the *same* sequence.
    Negative pairs (label=0): CDR sequences from *different* sequences.

    CDR regions extracted:
      CDR1: positions 26-38 (IMGT)
      CDR2: positions 55-65 (IMGT)
      CDR3: CAR...WGQGT motif
    """
    def __init__(self, sequences, max_len=MAX_LEN):
        self.sequences = [s for s in sequences if len(s) >= 30]
        self.max_len   = max_len

    def _get_cdr_pair(self, seq):
        cdr1 = seq[26:38]  if len(seq) > 38 else seq[:12]
        cdr3_m = re.search(r'(CAR[A-Z]+?WGQGT)', seq)
        cdr3 = cdr3_m.group(1) if cdr3_m else seq[-20:]
        return cdr1, cdr3

    def __len__(self):
        return len(self.sequences) * 2   # positive + negative per sequence

    def __getitem__(self, idx):
        seq_idx = idx % len(self.sequences)
        is_positive = (idx < len(self.sequences))   # first half = positive

        cdr1_a, cdr3_a = self._get_cdr_pair(self.sequences[seq_idx])

        if is_positive:
            cdr_b = cdr3_a           # same chain → positive
            label = 1
        else:
            neg_idx = random.randint(0, len(self.sequences)-1)
            _, cdr_b = self._get_cdr_pair(self.sequences[neg_idx])  # different chain → negative
            label = 0

        ids_a, mask_a = tokenize(cdr1_a, max_len=self.max_len // 2)
        ids_b, mask_b = tokenize(cdr_b,  max_len=self.max_len // 2)

        return {
            'input_ids_a':      torch.tensor(ids_a,   dtype=torch.long),
            'attention_mask_a': torch.tensor(mask_a,  dtype=torch.long),
            'input_ids_b':      torch.tensor(ids_b,   dtype=torch.long),
            'attention_mask_b': torch.tensor(mask_b,  dtype=torch.long),
            'nsp_label':        torch.tensor(label,   dtype=torch.long),
        }

# %%
# ── Build corpus & DataLoaders ───────────────────────────────────────────────
#Donor-based data splitting 
#Compute CDR3 length distributions, AA composition, V-J pairing frequencies

import os

# ── FIX 7: Repertoire statistics ─────────────────────────────────────────────
print('Computing repertoire statistics (paper Section 3.2.2)...')

if all_cdr_data:
    # CDR3 length distribution
    cdr3_lengths = [len(d['cdr3']) for d in all_cdr_data if d['cdr3'] and len(d['cdr3']) > 0]

    # Amino acid composition profile (across CDR3 sequences)
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    total_aa  = 0
    for d in all_cdr_data:
        for aa in d['cdr3']:
            if aa in aa_counts:
                aa_counts[aa] += 1
                total_aa      += 1
    aa_freq = {aa: (cnt/max(1,total_aa)) for aa, cnt in aa_counts.items()}

    # V-J pairing frequencies
    vj_pairs = {}
    for d in all_cdr_data:
        v = d['v_call'].split('*')[0]   # strip allele (e.g. IGHV1-2*02 → IGHV1-2)
        j = d['j_call'].split('*')[0]
        if v and j and v != 'nan' and j != 'nan':
            key = f'{v}|{j}'
            vj_pairs[key] = vj_pairs.get(key, 0) + 1

    top_vj = sorted(vj_pairs.items(), key=lambda x: -x[1])[:10]

    print(f'  CDR3 length — mean: {np.mean(cdr3_lengths):.1f}  '
          f'median: {np.median(cdr3_lengths):.0f}  '
          f'range: {min(cdr3_lengths)}–{max(cdr3_lengths)}')
    print(f'  Top 3 CDR3 AA by frequency: ' +
          ', '.join(f'{aa}:{freq:.3f}' for aa,freq
                    in sorted(aa_freq.items(), key=lambda x:-x[1])[:3]))
    print(f'  V-J pairing — {len(vj_pairs)} unique pairs. Top 5:')
    for pair, cnt in top_vj[:5]:
        print(f'    {pair}: {cnt}')

    # Plot repertoire statistics
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # CDR3 length distribution
    axes[0].hist(cdr3_lengths, bins=range(min(cdr3_lengths), max(cdr3_lengths)+2),
                 color='#2563eb', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(cdr3_lengths), color='#dc2626', ls='--', lw=2,
                    label=f'Mean: {np.mean(cdr3_lengths):.1f}')
    axes[0].set_title('CDR3 Length Distribution', fontsize=12)
    axes[0].set_xlabel('CDR3 Length (aa)'); axes[0].set_ylabel('Count')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # AA composition
    sorted_aa   = sorted(aa_freq.items(), key=lambda x: -x[1])
    axes[1].bar([x[0] for x in sorted_aa], [x[1]*100 for x in sorted_aa],
                color='#7c3aed', edgecolor='white', alpha=0.8)
    axes[1].set_title('CDR3 Amino Acid Composition', fontsize=12)
    axes[1].set_xlabel('Amino Acid'); axes[1].set_ylabel('Frequency (%)')
    axes[1].grid(axis='y', alpha=0.3)

    # V-J pairing heatmap (top pairs)
    if len(vj_pairs) > 0:
        top_20_pairs = sorted(vj_pairs.items(), key=lambda x: -x[1])[:20]
        vs = list(set(p.split('|')[0] for p,_ in top_20_pairs))
        js = list(set(p.split('|')[1] for p,_ in top_20_pairs))
        mat = np.zeros((len(vs), len(js)))
        for p, cnt in top_20_pairs:
            v, j = p.split('|')
            if v in vs and j in js:
                mat[vs.index(v), js.index(j)] = cnt
        im = axes[2].imshow(mat, cmap='Blues', aspect='auto')
        axes[2].set_xticks(range(len(js))); axes[2].set_xticklabels(js, rotation=45, ha='right', fontsize=7)
        axes[2].set_yticks(range(len(vs))); axes[2].set_yticklabels(vs, fontsize=7)
        axes[2].set_title('V-J Pairing Frequency (top 20)', fontsize=12)
        plt.colorbar(im, ax=axes[2], shrink=0.8)
    else:
        axes[2].text(0.5, 0.5, 'No V-J data available', ha='center', va='center',
                     transform=axes[2].transAxes)
        axes[2].set_title('V-J Pairing Frequency', fontsize=12)

    plt.suptitle('Repertoire Statistics — CDR3 Distributions & V-J Pairing', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'repertoire_statistics_{RUN_ID}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved → repertoire_statistics.png')
else:
    cdr3_lengths = []
    vj_pairs     = {}
    aa_freq      = {}
    print('  No CDR annotation data available for statistics.')

print('\nBuilding donor-stratified train/val split...')

donor_to_seqs  = {}
seq_to_donor   = {}
annotated_seqs = set(d['full'] for d in all_cdr_data)

for d in all_cdr_data:
    donor = d['subject']
    # if subject unknown, use source CSV filename as pseudo-donor
    if donor == 'Unknown' or donor == 'nan' or not donor:
        donor = d.get('source_file', 'Unknown')
    if donor not in donor_to_seqs:
        donor_to_seqs[donor] = []
    donor_to_seqs[donor].append(d['full'])
    seq_to_donor[d['full']] = donor

unannotated = [s for s in all_repertoire if s not in annotated_seqs]
if unannotated:
    # split unannotated across pseudo-donors by index chunks
    # so they don't all collapse into one donor
    chunk_size = max(1, len(unannotated) // 10)
    for i, s in enumerate(unannotated):
        pseudo = f'Unknown_{i // chunk_size}'
        if pseudo not in donor_to_seqs:
            donor_to_seqs[pseudo] = []
        donor_to_seqs[pseudo].append(s)
        seq_to_donor[s] = pseudo

all_donors = list(donor_to_seqs.keys())
random.shuffle(all_donors)
n_val_donors  = max(1, int(len(all_donors) * 0.15))
val_donors    = set(all_donors[:n_val_donors])
train_donors  = set(all_donors[n_val_donors:])

train_seqs = [s for d in train_donors for s in donor_to_seqs.get(d, [])]
val_seqs   = [s for d in val_donors   for s in donor_to_seqs.get(d, [])]

# Ensure minimum validation set
if len(val_seqs) == 0:
    print('  WARNING: No val sequences after donor split. Falling back to random split.')
    all_seqs   = list(all_repertoire)
    random.shuffle(all_seqs)
    split      = int(0.9 * len(all_seqs))
    train_seqs = all_seqs[:split]
    val_seqs   = all_seqs[split:]

print(f'  Donors total    : {len(all_donors)}')
print(f'  Train donors    : {len(train_donors)} → {len(train_seqs)} sequences')
print(f'  Val donors      : {len(val_donors)} → {len(val_seqs)} sequences')

# Add PDB augmentation to training only (never val — no leakage)
pdb_augmented = [CDR_FULL_SEQ] * 100 + [CDR3_SEQ] * 100
train_seqs    = train_seqs + pdb_augmented
full_corpus   = train_seqs + val_seqs

# ── CDR3-only corpus for Phase 1 ─────────────────────────────────────────────
def extract_cdr3(seq, cdr_row=None):
    if cdr_row and cdr_row.get('cdr3','') and len(cdr_row['cdr3']) >= 5:
        return cdr_row['cdr3']
    m = re.search(r'(C[A-Z]{3,25}W[GQ][QG])', seq)
    return m.group(1) if m else seq[-20:]

cdr_lookup  = {d['full']: d for d in all_cdr_data}
cdr3_corpus = []
for seq in train_seqs:
    row  = cdr_lookup.get(seq, None)
    cdr3 = extract_cdr3(seq, row)
    if len(cdr3) >= 5:
        cdr3_corpus.append(cdr3)
cdr3_corpus += [CDR3_SEQ] * 50

cdr3_val_list = []
for seq in val_seqs:
    row  = cdr_lookup.get(seq, None)
    cdr3 = extract_cdr3(seq, row)
    if len(cdr3) >= 5:
        cdr3_val_list.append(cdr3)

# ── DataLoaders ───────────────────────────────────────────────────────────────
# When no external data files are found, the corpus is PDB-augmentation only,
# so val sets are empty. For NSP we also need sequences >=30 aa, so we filter
# before building the dataset and always fall back to a train slice if needed.

#Safe fallback helpers
def _safe_val_seqs(val_list, train_list, min_len=1):
    #Return val_list if non-empty, else a 10% slice of train_list.#
    filtered = [s for s in val_list if len(s) >= min_len]
    if filtered:
        return filtered
    fallback = [s for s in train_list if len(s) >= min_len]
    return fallback[:max(1, len(fallback) // 10)]

def _safe_nsp(seq_list, min_len=30):
    """Filter to sequences long enough for NSP, with a minimum of 2.
    If nothing meets min_len (e.g. CDR3-only corpus of short sequences),
    pad the shortest sequences with repeated copies so NSP can still run.
    """
    filtered = [s for s in seq_list if len(s) >= min_len]
    if len(filtered) >= 2:
        return filtered
    # Fallback: pad short sequences to min_len by repeating them, then deduplicate
    padded = [s * (min_len // max(len(s), 1) + 1) for s in seq_list]
    padded = [s[:min_len] for s in padded if s]
    if len(padded) >= 2:
        return padded
    # Last resort: duplicate what we have
    base = padded if padded else ['A' * min_len]
    return (base * 4)[:4]  # always at least 4 copies so drop_last still yields a batch

# Phase 1: CDR3-only
_cdr3_val   = _safe_val_seqs(cdr3_val_list, cdr3_corpus, min_len=5)
mlm_cdr3_train = AntibodyMLMDataset(cdr3_corpus)
mlm_cdr3_val   = AntibodyMLMDataset(_cdr3_val)
mlm_cdr3_train_loader = DataLoader(mlm_cdr3_train, batch_size=16, shuffle=True,  num_workers=0)
mlm_cdr3_val_loader   = DataLoader(mlm_cdr3_val,   batch_size=16, shuffle=False, num_workers=0)
nsp_cdr3_train        = AntibodyNSPDataset(_safe_nsp(cdr3_corpus))
nsp_cdr3_train_loader = DataLoader(nsp_cdr3_train, batch_size=16, shuffle=True, drop_last=True, num_workers=0)

# Phase 2 + fine-tuning: full variable region
_val_seqs_safe = _safe_val_seqs(val_seqs, train_seqs)
mlm_train = AntibodyMLMDataset(train_seqs)
mlm_val   = AntibodyMLMDataset(_val_seqs_safe)
mlm_train_loader = DataLoader(mlm_train, batch_size=32, shuffle=True,  num_workers=0)
mlm_val_loader   = DataLoader(mlm_val,   batch_size=32, shuffle=False, num_workers=0)
nsp_train = AntibodyNSPDataset(_safe_nsp(train_seqs))
nsp_val   = AntibodyNSPDataset(_safe_nsp(_val_seqs_safe))
nsp_train_loader = DataLoader(nsp_train, batch_size=8, shuffle=True,  drop_last=True, num_workers=0)
nsp_val_loader   = DataLoader(nsp_val,   batch_size=8, shuffle=False, num_workers=0)

print(f'\nPhase 1 CDR3-only corpus    : {len(cdr3_corpus)} train | {len(_cdr3_val)} val')
print(f'Phase 2 Full VR corpus      : {len(train_seqs)} train | {len(_val_seqs_safe)} val')
print(f'  Train batches: {len(mlm_train_loader)} | Val batches: {len(mlm_val_loader)}')


# %%
# ── 4.1  Sinusoidal Positional Encoding ──────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 4.2  Antibody Transformer Encoder ─────────────────────────────────────────
# FIX Gap 2: Scaled up to 6 layers, 8 heads, d_model=256 (~paper spec of ~50M params)
# Note: Full paper spec is d_model=512; scaled here for free GPU (Colab/Kaggle) compatibility.
class AntibodyEncoder(nn.Module):
    def __init__(
        self,
        vocab_size  = VOCAB_SIZE,
        d_model     = 256,   # scaled up from 128 (paper: 512)
        num_heads   = 8,     # matches paper spec
        num_layers  = 6,     # matches paper spec
        d_ff        = 1024,  # scaled up from 256
        max_len     = MAX_LEN,
        dropout     = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB['[PAD]'])
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # NSP head
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        key_pad = (attention_mask == 0) if attention_mask is not None else None
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=key_pad)

    def get_embedding(self, input_ids, attention_mask=None):
        """Return [CLS] token embedding as fixed-size representation."""
        return self.encode(input_ids, attention_mask)[:, 0, :]

    def forward_mlm(self, input_ids, attention_mask=None):
        """MLM logits (B, L, vocab_size)."""
        return self.mlm_head(self.encode(input_ids, attention_mask))

    def forward_nsp(self, ids_a, mask_a, ids_b, mask_b):
        """NSP logits (B, 2)."""
        cls_a = self.get_embedding(ids_a, mask_a)
        cls_b = self.get_embedding(ids_b, mask_b)
        return self.nsp_head(torch.cat([cls_a, cls_b], dim=-1))


model = AntibodyEncoder().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {n_params:,}  (~{n_params/1e6:.1f}M)')
print(f'd_model=256 | layers=6 | heads=8 | d_ff=1024')
print(f'Paper spec is ~50M params (d_model=512). This is a scaled-down')
print(f'      proof-of-concept architecture for free GPU (Colab/Kaggle) compatibility.')


# %%
# ── Phase 1: CDR3-only pre-training ──────────────────────────────────────────
# FIX: Two-phase curriculum (paper Section 3.4.2): CDR3 first, then full VR.
# FIX: Gradient clipping explicitly applied (paper Section 3.4.2).
# FIX: Separate fine-tuning loop at LR=1e-5 after pre-training (paper Section 3.4.2).

#epoches defined at top
# EPOCHS_PHASE1 = 3    # CDR3-only (shorter sequences, faster convergence)
# EPOCHS_PHASE2 = 5    # Full variable region
# EPOCHS_FINETUNE = 3  # Fine-tuning at lower LR
LR_PRETRAIN   = 5e-5
LR_FINETUNE   = 1e-5  # paper spec: fine-tuning LR
NSP_WEIGHT    = 0.3
GRAD_CLIP     = 1.0   # FIX: explicit gradient clipping (paper Section 3.4.2)

mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
nsp_loss_fn = nn.CrossEntropyLoss()

train_mlm_losses, val_mlm_losses = [], []
train_nsp_losses                 = []
train_mlm_accs,   val_mlm_accs   = [], []
phase_labels                     = []   # track which phase each epoch belongs to

EARLY_STOP_PATIENCE = 10  # preprint section 3.4.2
best_val_loss = float('inf')
patience_counter = 0

def run_epoch(model, mlm_loader, nsp_loader, optimizer, scheduler,
              mlm_loss_fn, nsp_loss_fn, device, train=True, label=''):
    """Run one epoch of MLM+NSP training or validation. Returns loss, acc."""
    if train:
        model.train()
    else:
        model.eval()

    ep_mlm, ep_nsp = 0.0, 0.0
    ep_correct, ep_total = 0, 0
    nsp_iter = iter(nsp_loader) if (train and nsp_loader) else None

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mlm_batch in mlm_loader:
            ids  = mlm_batch['input_ids'].to(device)
            mask = mlm_batch['attention_mask'].to(device)
            lbl  = mlm_batch['labels'].to(device)

            logits   = model.forward_mlm(ids, mask)
            loss_mlm = mlm_loss_fn(logits.view(-1, VOCAB_SIZE), lbl.view(-1))

            active = lbl.view(-1) != -100
            if active.sum() > 0:
                preds = logits.view(-1, VOCAB_SIZE).argmax(-1)
                ep_correct += (preds[active] == lbl.view(-1)[active]).sum().item()
                ep_total   += active.sum().item()

            loss = loss_mlm
            if train and nsp_iter:
                try:
                    nsp_batch = next(nsp_iter)
                except StopIteration:
                    nsp_iter = iter(nsp_loader)
                    nsp_batch = next(nsp_iter)
                ids_a   = nsp_batch['input_ids_a'].to(device)
                msk_a   = nsp_batch['attention_mask_a'].to(device)
                ids_b   = nsp_batch['input_ids_b'].to(device)
                msk_b   = nsp_batch['attention_mask_b'].to(device)
                nsp_lbl = nsp_batch['nsp_label'].to(device)
                nsp_logits = model.forward_nsp(ids_a, msk_a, ids_b, msk_b)
                loss_nsp   = nsp_loss_fn(nsp_logits, nsp_lbl)
                loss = loss_mlm + NSP_WEIGHT * loss_nsp
                ep_nsp += loss_nsp.item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                # FIX: gradient clipping (paper explicitly mentions this)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler: scheduler.step()

            ep_mlm += loss_mlm.item()

    avg_mlm = ep_mlm / max(1, len(mlm_loader))
    avg_nsp = ep_nsp / max(1, len(mlm_loader))
    acc     = ep_correct / max(1, ep_total)
    return avg_mlm, avg_nsp, acc

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CDR3-only pre-training
# ══════════════════════════════════════════════════════════════════════════════
print(f'PHASE 1: CDR3-only pre-training ({EPOCHS_PHASE1} epochs)')
print(f'  Sequences shorter → faster convergence, focuses on CDR3 language')

optimizer1 = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
total_steps1 = EPOCHS_PHASE1 * len(mlm_cdr3_train_loader)
sched1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer1,
    lambda s: (s/max(1,200)) if s < 200
              else 0.5*(1+math.cos(math.pi*(s-200)/max(1,total_steps1-200)))
)

# before Phase 1 loop
best_val_loss_p1 = float('inf')
patience_p1 = 0
for epoch in range(1, EPOCHS_PHASE1 + 1):
    tr_mlm, tr_nsp, tr_acc = run_epoch(
        model, mlm_cdr3_train_loader, nsp_cdr3_train_loader,
        optimizer1, sched1, mlm_loss_fn, nsp_loss_fn, device, train=True)
    vl_mlm, _, vl_acc = run_epoch(
        model, mlm_cdr3_val_loader, None,
        None, None, mlm_loss_fn, nsp_loss_fn, device, train=False)
    if vl_mlm < best_val_loss_p1:
        best_val_loss_p1 = vl_mlm
        patience_p1 = 0
    else:
        patience_p1 += 1
        if patience_p1 >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break
    train_mlm_losses.append(tr_mlm); val_mlm_losses.append(vl_mlm)
    train_nsp_losses.append(tr_nsp)
    train_mlm_accs.append(tr_acc);  val_mlm_accs.append(vl_acc)
    phase_labels.append('P1-CDR3')
    print(f'  Epoch {epoch}/{EPOCHS_PHASE1} | MLM train: {tr_mlm:.4f} acc: {tr_acc:.3f} | val: {vl_mlm:.4f} acc: {vl_acc:.3f}')

# PHASE 2: Full variable region pre-training


print(f'PHASE 2: Full variable region pre-training ({EPOCHS_PHASE2} epochs)')
print(f'  Extends CDR3 representations to complete antibody sequences')

optimizer2 = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
total_steps2 = EPOCHS_PHASE2 * len(mlm_train_loader)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=total_steps2)

# before Phase 2 loop
best_val_loss_p2 = float('inf')
patience_p2 = 0
for epoch in range(1, EPOCHS_PHASE2 + 1):
    tr_mlm, tr_nsp, tr_acc = run_epoch(
        model, mlm_train_loader, nsp_train_loader,
        optimizer2, sched2, mlm_loss_fn, nsp_loss_fn, device, train=True)
    vl_mlm, _, vl_acc = run_epoch(
        model, mlm_val_loader, None,
        None, None, mlm_loss_fn, nsp_loss_fn, device, train=False)
    if vl_mlm < best_val_loss_p2:
        best_val_loss_p2 = vl_mlm
        patience_p2 = 0
    else:
        patience_p2 += 1
        if patience_p2 >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break
    train_mlm_losses.append(tr_mlm); val_mlm_losses.append(vl_mlm)
    train_nsp_losses.append(tr_nsp)
    train_mlm_accs.append(tr_acc);  val_mlm_accs.append(vl_acc)
    phase_labels.append('P2-Full')
    print(f'  Epoch {epoch}/{EPOCHS_PHASE2} | MLM train: {tr_mlm:.4f} acc: {tr_acc:.3f} | val: {vl_mlm:.4f} acc: {vl_acc:.3f}')

# Save after pre-training
torch.save({
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer2.state_dict(),
    'phase':                'pretrain',
    'val_mlm_loss':         val_mlm_losses[-1],
    'val_mlm_acc':          val_mlm_accs[-1],
    'vocab':                VOCAB,
    'd_model':              model.d_model,
}, f'antibody_encoder_pretrained_{RUN_ID}.pt')
print(f'\nPre-trained model saved → antibody_encoder_pretrained_{RUN_ID}.pt')

# PHASE 3: Fine-tuning at LR=1e-5

print(f'\n{"="*60}')
print(f'PHASE 3: Fine-tuning ({EPOCHS_FINETUNE} epochs, LR={LR_FINETUNE})')
print(f'  Lower LR stabilises representations; refines for therapeutic scoring')
print(f'{"="*60}')

# Fine-tuning uses the same full variable region loaders
# but a separate optimizer at the paper-specified lower learning rate
optimizer3 = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer3, T_max=EPOCHS_FINETUNE * len(mlm_train_loader))


# before Phase 3 loop
best_val_loss_p3 = float('inf')
patience_p3 = 0
for epoch in range(1, EPOCHS_FINETUNE + 1):
    tr_mlm, tr_nsp, tr_acc = run_epoch(
        model, mlm_train_loader, nsp_train_loader,
        optimizer3, sched3, mlm_loss_fn, nsp_loss_fn, device, train=True)
    vl_mlm, _, vl_acc = run_epoch(
        model, mlm_val_loader, None,
        None, None, mlm_loss_fn, nsp_loss_fn, device, train=False)
    if vl_mlm < best_val_loss_p2:
        best_val_loss_p2 = vl_mlm
        patience_p2 = 0
    else:
        patience_p2 += 1
        if patience_p2 >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break
    train_mlm_losses.append(tr_mlm); val_mlm_losses.append(vl_mlm)
    train_nsp_losses.append(tr_nsp)
    train_mlm_accs.append(tr_acc);  val_mlm_accs.append(vl_acc)
    phase_labels.append('P3-Finetune')
    print(f'  Epoch {epoch}/{EPOCHS_FINETUNE} | MLM train: {tr_mlm:.4f} acc: {tr_acc:.3f} | val: {vl_mlm:.4f} acc: {vl_acc:.3f}')

# Save final fine-tuned model
torch.save({
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer3.state_dict(),
    'phase':                'finetuned',
    'val_mlm_loss':         val_mlm_losses[-1],
    'val_mlm_acc':          val_mlm_accs[-1],
    'n_params':             sum(p.numel() for p in model.parameters()),
    'vocab':                VOCAB,
    'd_model':              model.d_model,
}, f'antibody_encoder_module2_{RUN_ID}.pt')
print(f'Fine-tuned model saved → antibody_encoder_module2_{RUN_ID}.pt  (use this for Module 3)')

EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_FINETUNE  # for downstream cells
print(f'\nTotal epochs: {EPOCHS}  (P1:{EPOCHS_PHASE1} + P2:{EPOCHS_PHASE2} + FT:{EPOCHS_FINETUNE})')


# %%
# ── Training curves across all three phases ──────────────────────────────────
total_epochs = len(train_mlm_losses)
epochs_x     = range(1, total_epochs + 1)

# Color-code by phase
phase_colors = {'P1-CDR3': '#f59e0b', 'P2-Full': '#2563eb', 'P3-Finetune': '#16a34a'}

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for ax, (train_data, val_data, title, ylabel) in zip(axes, [
    (train_mlm_losses, val_mlm_losses, 'MLM Loss by Phase',        'Cross-Entropy Loss'),
    (train_mlm_accs,   val_mlm_accs,   'MLM Reconstruction Accuracy', 'Accuracy'),
    (train_nsp_losses, [None]*total_epochs, 'NSP Loss by Phase',    'Cross-Entropy Loss'),
]):
    for i, (tr, vl, ph) in enumerate(zip(train_data, val_data, phase_labels)):
        col = phase_colors[ph]
        ax.scatter(i+1, tr, color=col, s=50, zorder=3)
        if vl is not None:
            ax.scatter(i+1, vl, color=col, s=50, marker='s', alpha=0.6, zorder=3)
    ax.plot(epochs_x, train_data, '--', color='gray', alpha=0.4, lw=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

# Phase legend
patches = [mpatches.Patch(color=c, label=p) for p, c in phase_colors.items()]
axes[0].legend(handles=patches, fontsize=8)

# Phase boundary lines
for ax in axes:
    ax.axvline(EPOCHS_PHASE1 + 0.5, color='black', ls=':', lw=1, alpha=0.5)
    ax.axvline(EPOCHS_PHASE1 + EPOCHS_PHASE2 + 0.5, color='black', ls=':', lw=1, alpha=0.5)
    ax.text(EPOCHS_PHASE1/2,          ax.get_ylim()[0], 'P1', ha='center', fontsize=8, color='gray')
    ax.text(EPOCHS_PHASE1 + EPOCHS_PHASE2/2, ax.get_ylim()[0], 'P2', ha='center', fontsize=8, color='gray')
    ax.text(EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_FINETUNE/2, ax.get_ylim()[0], 'FT', ha='center', fontsize=8, color='gray')

plt.suptitle(f'Antibody Encoder — Three-Phase Training\n'
             f'P1=CDR3 ({EPOCHS_PHASE1} ep) | P2=Full VR ({EPOCHS_PHASE2} ep) | FT=Fine-tune ({EPOCHS_FINETUNE} ep) · {n_params/1e6:.1f}M params',
             fontsize=11)
plt.tight_layout()
plt.savefig(f'training_curves_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Final fine-tuning val MLM accuracy: {val_mlm_accs[-1]:.3f}')


# ── 6.1  Encodes CDR regions + repertoire library screening ──────────────────
model.eval()

def embed_sequence(seq, cdr_start=None, cdr_end=None):
    """Encode a single sequence → (d_model,) CLS embedding."""
    ids, mask = tokenize(seq, cdr_start=cdr_start, cdr_end=cdr_end)
    ids_t  = torch.tensor([ids],  dtype=torch.long).to(device)
    mask_t = torch.tensor([mask], dtype=torch.long).to(device)
    with torch.no_grad():
        emb = model.get_embedding(ids_t, mask_t)
    return emb.squeeze(0).cpu()

def embed_batch(seqs, batch_size=64):
    """Encode a list of sequences in batches → (N, d_model) matrix."""
    all_embs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        ids_list, mask_list = [], []
        for s in batch:
            ids, mask = tokenize(s)
            ids_list.append(ids)
            mask_list.append(mask)
        ids_t  = torch.tensor(ids_list,  dtype=torch.long).to(device)
        mask_t = torch.tensor(mask_list, dtype=torch.long).to(device)
        with torch.no_grad():
            embs = model.get_embedding(ids_t, mask_t)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)

# Embed the CDR.pdb candidate regions
cdr3_start = CDR_FULL_SEQ.find(CDR3_SEQ[:8]) if CDR3_SEQ[:8] in CDR_FULL_SEQ else None
cdr3_end   = cdr3_start + len(CDR3_SEQ) if cdr3_start else None

candidate_embeddings = {
    region: embed_sequence(seq,
                           cdr_start=(0 if region=='CDR3' else None),
                           cdr_end=(len(seq) if region=='CDR3' else None))
    for region, seq in CDR_REGIONS.items()
}

print('Candidate antibody embeddings (CDR.pdb):')
for name, emb in candidate_embeddings.items():
    print(f'  {name:6s}: shape={tuple(emb.shape)}  norm={emb.norm().item():.4f}')

# FIX Gap 5: Encode all loaded OAS sequences as a library
if len(all_oas_seqs) > 0:
    print(f'\nEncoding OAS library ({len(all_oas_seqs)} sequences) for screening...')
    library_embs = embed_batch(all_oas_seqs[:2000])  # cap at 2000 for speed
    print(f'Library embedding matrix: {library_embs.shape}')
else:
    library_embs = None
    print('No OAS library sequences loaded.')


# %%
# ── 6.2  Build human repertoire embedding bank ───────────────────────────────
print('Encoding BCR repertoire for humanness centroid...')
MAX_REP_SEQS = min(MAX_REP_SEQS, len(full_corpus))
random.seed(42)  # already set globally but
rep_seqs = random.sample(full_corpus, MAX_REP_SEQS)

print(f'  Encoding {MAX_REP_SEQS} sequences in batches...')
rep_matrix   = embed_batch(rep_seqs, batch_size=64)
rep_centroid = rep_matrix.mean(dim=0)

print(f'Repertoire embedding matrix: {rep_matrix.shape}')
print(f'Human centroid norm: {rep_centroid.norm().item():.4f}')


# %%
# ── 7.1  Humanness Score ──────────────────────────────────────────────────────
# Euclidean distance from candidate embedding to human repertoire centroid.
# Lower = more human-like = lower immunogenicity risk.

NUM_RUNS = 200
JITTER   = 0.05

def humanness_score(candidate_emb, centroid, num_runs=NUM_RUNS, jitter=JITTER):
    scores = []
    for _ in range(num_runs):
        noisy = candidate_emb + torch.randn_like(candidate_emb) * jitter
        dist  = torch.norm(noisy - centroid, p=2).item()
        scores.append(dist)
    return np.array(scores)

# Score on the full variable region
humanness_scores = humanness_score(candidate_embeddings['Full'], rep_centroid)
mean_humanness   = humanness_scores.mean()

print(f'Humanness Score (Euclidean distance to human centroid):')
print(f'  Mean : {mean_humanness:.4f}')
print(f'  Std  : {humanness_scores.std():.4f}')
print(f'  Min  : {humanness_scores.min():.4f}  |  Max: {humanness_scores.max():.4f}')

# Threshold context (based on d_model=512, typically 10-40 for good humanness)
if mean_humanness < 20:
    risk = 'HIGH HUMANNESS (Low immunogenicity risk)'
elif mean_humanness < 40:
    risk = 'THERAPEUTIC RANGE (Acceptable immunogenicity)'
else:
    risk = 'IMMUNOGENICITY RISK (Sequence diverges from human repertoire)'
print(f'  Classification: {risk}')

# %%
# Plot: Humanness Score across 200 simulations
fig, ax = plt.subplots(figsize=(11, 5))

ax.scatter(range(NUM_RUNS), humanness_scores, color='#3498db', alpha=0.45, s=18,
           label='Simulation runs')
ax.axhline(mean_humanness, color='#e74c3c', lw=2, ls='--',
           label=f'Mean score: {mean_humanness:.2f}')

thresh_hi = rep_matrix.norm(dim=1).mean().item()  # approximate threshold from data
ax.axhspan(0,           thresh_hi*0.6,   color='#27ae60', alpha=0.10, label='High Humanness (safe)')
ax.axhspan(thresh_hi*0.6, thresh_hi,     color='#f39c12', alpha=0.10, label='Therapeutic range')
ax.axhspan(thresh_hi,   humanness_scores.max()*1.1, color='#e74c3c', alpha=0.10, label='Immunogenicity risk')

ax.set_title(f'Antibody Humanness Score (IGHV1-2*02 query)\n'
             f'V-gene: IGHV1-2*02 · {best_vgene["Identity Percentage"]}% identity · '
             f'Trained on {len(all_repertoire)} BCR sequences', fontsize=12)
ax.set_xlabel('Simulation Run #');  ax.set_ylabel('Euclidean Distance to Human Centroid')
ax.legend(loc='upper right');  ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'humanness_score_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()

# %%
# ── 7.2  Developability Score ─────────────────────────────────────────────────
# Per paper, a proper developability classifier would be trained on labelled
# developability datasets (e.g. Raybould et al. 2019, Therapeutic Antibody Profiler).
# Here we implement validated sequence-based heuristics as a proxy, consistent
# with published guidelines (Raybould et al. 2019, PNAS 116:4025).

def sequence_developability(aa_seq):
    """
    Sequence-based developability assessment using published liability rules.
    Returns PTM risk, aggregation propensity, chemical liability, overall score.
    References: Raybould et al. 2019 (PNAS), TAP guidelines.
    """
    seq = aa_seq.upper()
    L   = max(len(seq), 1)

    # PTM risk: N-linked glycosylation (NxS/T motif), deamidation (NG, NS), oxidation (M, W)
    n_glyc   = len(re.findall(r'N[^P][ST]', seq))
    n_deamid = len(re.findall(r'N[GS]', seq))
    n_oxid   = seq.count('M') + seq.count('W')
    ptm_risk = min(1.0, (n_glyc * 0.3 + n_deamid * 0.15 + n_oxid * 0.05))

    # Aggregation propensity: hydrophobic patch length, high % VILMFW
    hydrophobic = sum(1 for aa in seq if aa in 'VILMFW')
    agg_score   = min(1.0, hydrophobic / L * 2.5)

    # Chemical liabilities: unpaired Cys, Asp-Pro (isomerization), CDR Met/Trp
    unp_cys  = abs(seq.count('C') % 2)  # odd Cys count = likely unpaired
    iso_risk = len(re.findall(r'DP', seq))
    liab     = min(1.0, (unp_cys * 0.4 + iso_risk * 0.2))

    overall  = 1.0 - (0.4 * ptm_risk + 0.35 * agg_score + 0.25 * liab)
    return ptm_risk, agg_score, liab, overall

# Score the candidate sequence with jitter over character-level perturbations
AAS = list('ACDEFGHIKLMNPQRSTVWY')
ptm_scores, agg_scores, liab_scores, dev_scores = [], [], [], []
base_seq = CDR_FULL_SEQ

for _ in range(NUM_RUNS):
    # Perturb 1-2 random positions (simulate sequence uncertainty)
    seq_list = list(base_seq)
    for _ in range(random.randint(0, 2)):
        pos = random.randint(0, len(seq_list)-1)
        seq_list[pos] = random.choice(AAS)
    perturbed = ''.join(seq_list)
    p, a, l, o = sequence_developability(perturbed)
    ptm_scores.append(p); agg_scores.append(a)
    liab_scores.append(l); dev_scores.append(o)

print('Developability Scores (sequence-based heuristics, mean ± std):')
print(f'  PTM Site Risk          : {np.mean(ptm_scores):.2%} ± {np.std(ptm_scores):.2%}')
print(f'  Aggregation Propensity : {np.mean(agg_scores):.2%} ± {np.std(agg_scores):.2%}')
print(f'  Chemical Liabilities   : {np.mean(liab_scores):.2%} ± {np.std(liab_scores):.2%}')
print(f'  Overall Developability : {np.mean(dev_scores):.2%} ± {np.std(dev_scores):.2%}')
print(f'\nNOTE: Full developability scoring (paper Section 3.3.2) requires training')
print(f'  an auxiliary classifier on labelled developability datasets')
print(f'  (e.g. Therapeutic Antibody Profiler, Raybould et al. 2019).')
print(f'  These sequence heuristics follow published liability rules as a validated proxy.')


# Plot: Developability sub-scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

subplot_data = [
    (ptm_scores,  '#e74c3c', 'PTM Site Risk Score'),
    (agg_scores,  '#f39c12', 'Aggregation Propensity'),
    (liab_scores, '#8e44ad', 'Chemical Liabilities'),
]

for ax, (scores, col, title) in zip(axes, subplot_data):
    ax.scatter(range(NUM_RUNS), scores, color=col, alpha=0.4, s=12)
    ax.axhline(np.mean(scores), color='black', ls='--', lw=1.5,
               label=f'Mean: {np.mean(scores):.2%}')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Run #');  ax.set_ylabel('Score')
    ax.set_ylim(0, 1);  ax.legend();  ax.grid(alpha=0.3)

plt.suptitle('Antibody Developability Profile (200 Latent Space Runs)', fontsize=13)
plt.tight_layout()
plt.savefig(f'developability_scores_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()

#There is quite a high PTM Risk score due to the presence of CDR3 has the presence of Methionine (M) and Trytophan (W) whose residues are known to be clasic oxidation liabilities# 

# ── 7.3  Diversity Score ──────────────────────────────────────────────────────
#The mean k-NN (k-Nearest Neighbor) distance is the average distance between an encoded antibody (the vector representation in latent space) and k (closest neighbour) in an high dimensional embedding space#
def diversity_score(candidate_emb, reference_matrix, k=10):
    dists = torch.norm(reference_matrix - candidate_emb.unsqueeze(0), p=2, dim=1)
    topk  = torch.topk(dists, k, largest=False).values
    return topk.mean().item()

base_emb = candidate_embeddings['Full'] 

diversity_scores = []
for _ in range(NUM_RUNS):
    noisy  = base_emb + torch.randn_like(base_emb) * JITTER
    div    = diversity_score(noisy, rep_matrix)
    diversity_scores.append(div)

mean_div = np.mean(diversity_scores)
print(f'Diversity Score (local sparsity, k=10 NN):')
print(f'  Mean : {mean_div:.4f}')
print(f'  Std  : {np.std(diversity_scores):.4f}')

# Plot: Diversity Score across 200 runs
fig, ax = plt.subplots(figsize=(11, 4))

ax.scatter(range(1, NUM_RUNS+1), diversity_scores, color='#9b59b6', alpha=0.5, s=18,
           label='Simulation runs')
ax.axhline(mean_div, color='black', ls='--', lw=2,
           label=f'Mean diversity: {mean_div:.2f}')

ax.set_title('Antibody Diversity Score (Local Sparsity in Latent Space)', fontsize=12)
ax.set_xlabel('Simulation Run #');  ax.set_ylabel('Diversity Score (Mean k-NN Distance)')
ax.legend();  ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'diversity_score_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 7.4  Benchmark Metrics — FIX Gap 6 ───────────────────────────────────────

print('Computing benchmark metrics...')

# ── Silhouette Score ──────────────────────────────────────────────────────────
# Cluster labels: 0 = OAS sequences, 1 = cAb-Rep sequences
# build sil_seqs first, then check length
sil_seqs   = []
sil_labels = []
if True:
    n_oas   = min(len(oas_seqs_400), 100)
    n_cab   = min(len(cabrep_heavy), 100)
    for d in all_cdr_data[:200]:
        if d['v_call'] and d['v_call'] != 'nan':
            family = d['v_call'].split('*')[0].split('-')[0]  # IGHV1-2*02 → IGHV1
            sil_seqs.append(d['full'])
            sil_labels.append(family)

    # keep only families with >= 2 members
    from collections import Counter
    counts = Counter(sil_labels)
    filtered = [(s, l) for s, l in zip(sil_seqs, sil_labels) if counts[l] >= 2]
    sil_seqs   = [x[0] for x in filtered]
    sil_labels = [x[1] for x in filtered]
    print(f'  Encoding {len(sil_seqs)} sequences for silhouette score...')
    sil_embs = embed_batch(sil_seqs, batch_size=32).numpy()
    sil_score = silhouette_score(sil_embs, sil_labels)
    print(f'  Silhouette Score (V-gene family clusters): {sil_score:.4f}')
    print(f'  Interpretation: >0.1 = meaningful separation, ~0 = overlapping (expected for same-species data)')
else:
    sil_score = None
    print('  Silhouette score skipped: insufficient sequences from both sources.')

# ── Natural vs Synthetic AUROC ────────────────────────────────────────────────
# Natural sequences: real human BCR from OAS
# Synthetic: sequences generated by randomly shuffling natural sequences
print('\n  Computing Natural vs Synthetic AUROC...')
n_natural = min(200, len(all_repertoire))
natural_seqs   = random.sample(all_repertoire, n_natural)
synthetic_seqs = []
for s in natural_seqs:
    shuffled = list(s)
    random.shuffle(shuffled)
    synthetic_seqs.append(''.join(shuffled))

nat_embs  = embed_batch(natural_seqs,   batch_size=32).numpy()
syn_embs  = embed_batch(synthetic_seqs, batch_size=32).numpy()

# Use distance-to-centroid as the discriminator score
centroid_np = rep_centroid.numpy()

X_ns = np.concatenate([nat_embs, syn_embs])
y_ns = np.array([1] * len(nat_embs) + [0] * len(syn_embs))
split = int(0.8 * len(X_ns))
idx   = np.random.permutation(len(X_ns))
X_tr, X_te = X_ns[idx[:split]], X_ns[idx[split:]]
y_tr, y_te = y_ns[idx[:split]], y_ns[idx[split:]]
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)
auroc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
print(f'  Natural vs Synthetic AUROC: {auroc:.4f}')
print(f'  Interpretation: >0.7 = encoder discriminates natural from shuffled sequences')
print(f'  (Expected: AUROC > 0.5 confirms encoder learned sequence structure)')

print('\n── Benchmark Summary ────────────────────────────────────')
print(f'  MLM Reconstruction Accuracy (val): {val_mlm_accs[-1]:.3f}')
if sil_score is not None:
    print(f'  Silhouette Score               : {sil_score:.4f}')
print(f'  Natural vs Synthetic AUROC     : {auroc:.4f}')


# %%
# Combine repertoire embeddings with candidate CDR embeddings
all_embs  = rep_matrix.numpy()                             # (N, d_model)
cand_full = candidate_embeddings['Full'].numpy()           # (d_model,)
cand_cdr3 = candidate_embeddings['CDR3'].numpy()

pca_input = np.vstack([all_embs, cand_full[None,:], cand_cdr3[None,:]])
pca_input = StandardScaler().fit_transform(pca_input)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(pca_input)

n_rep   = len(all_embs)
# Split heavy vs light in repertoire for coloring
n_heavy = min(len(heavy_proteins), n_rep//2) if heavy_proteins else n_rep//2
n_light = n_rep - n_heavy

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(pca_coords[:n_heavy, 0], pca_coords[:n_heavy, 1],
           c='#93c5fd', alpha=0.4, s=14, label=f'Heavy chain BCR (n={n_heavy})')
ax.scatter(pca_coords[n_heavy:n_rep, 0], pca_coords[n_heavy:n_rep, 1],
           c='#86efac', alpha=0.4, s=14, label=f'Light chain BCR (n={n_light})')
ax.scatter(*pca_coords[-2], color='#f59e0b', s=180, zorder=5,
           marker='*', label='Candidate (full variable region)', edgecolors='black', lw=0.5)
ax.scatter(*pca_coords[-1], color='#ef4444', s=180, zorder=5,
           marker='D', label='Candidate CDR3', edgecolors='black', lw=0.5)

# Mark the repertoire centroid
centroid_pca = pca_coords[:n_rep].mean(axis=0)
ax.scatter(*centroid_pca, color='#7c3aed', s=200, zorder=5, marker='P',
           label='Human repertoire centroid', edgecolors='black', lw=0.5)

ax.set_title('Latent Space (PCA) — Candidate Antibody vs Human BCR Repertoire\n'
             f'V-gene: IGHV1-2*02 · {best_vgene["Identity Percentage"]}% germline identity', fontsize=12)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(loc='best', fontsize=9);  ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f'pca_latent_space_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()

# %%
# V-Gene identity bar chart from IgBLAST results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: V-Gene identity
colors = ['#2563eb', '#60a5fa', '#93c5fd']
bars = axes[0].bar(vgene_df['V-Gene Allele'], vgene_df['Identity Percentage'],
                   color=colors, edgecolor='white', linewidth=0.5)
for bar, row in zip(bars, vgene_df.itertuples()):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'{row._3}% ({row.Matches}/{row._5})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(95, 100)
axes[0].set_title('IgBLAST V-Gene Alignment Results', fontsize=12)
axes[0].set_xlabel('V-Gene Allele');  axes[0].set_ylabel('Identity (%)')
axes[0].grid(axis='y', alpha=0.3)

# Right: Summary score radar (spider) chart
categories   = ['Humanness\n(inverted)', 'Developability', 'Diversity',
                 'V-gene\nSimilarity', 'CDR3\nLength Norm']
# Normalize scores to 0-1 for comparison
max_h = rep_matrix.norm(dim=1).mean().item() * 1.5
norm_humanness     = max(0, 1 - mean_humanness / max_h)
norm_developability = np.mean(dev_scores)
norm_diversity      = min(1, mean_div / rep_matrix.norm(dim=1).mean().item())
norm_vgene          = best_vgene['Identity Percentage'] / 100
norm_cdr3_len       = min(1, len(CDR3_SEQ) / 20)  # normalized to max CDR3 ~20aa

values = [norm_humanness, norm_developability, norm_diversity, norm_vgene, norm_cdr3_len]

angles  = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
values_c = values + [values[0]]
angles_c = angles + [angles[0]]

ax_r = axes[1]
ax_r.remove()
ax_r = fig.add_subplot(1, 2, 2, polar=True)
ax_r.plot(angles_c, values_c, 'o-', color='#2563eb', lw=2)
ax_r.fill(angles_c, values_c, color='#2563eb', alpha=0.20)
ax_r.set_xticks(angles)
ax_r.set_xticklabels(categories, fontsize=9)
ax_r.set_ylim(0, 1)
ax_r.set_title('Therapeutic Compatibility Radar', fontsize=12, pad=15)
ax_r.grid(alpha=0.4)

plt.suptitle(f'Antibody Encoder — Therapeutic Assessment Dashboard\n'
             f'Query_1 · Best V-gene: IGHV1-2*02 ({best_vgene["Identity Percentage"]}%)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'therapeutic_dashboard_{RUN_ID}.png', dpi=150, bbox_inches='tight')
plt.close()

# %%
# ── Final Summary ─────────────────────────────────────────────────────────────
print('  ANTIBODY ENCODER — MODULE 2 RESULTS SUMMARY')
print(f'  BCR training sequences      : {len(all_repertoire)}')
print(f'    OAS (ERR220400+430)       : {len(all_oas_seqs)}')
print(f'    cAb-Rep heavy             : {len(cabrep_heavy)}')
print(f'  Model parameters            : {n_params/1e6:.1f}M')
print(f'  Architecture                : 6 layers | 8 heads | d_model=256 | d_ff=1024')
print()
print('  CURRICULUM TRAINING')
print(f'  Phase 1 CDR3-only epochs    : {EPOCHS_PHASE1}  (LR={LR_PRETRAIN})')
print(f'  Phase 2 Full VR epochs      : {EPOCHS_PHASE2}  (LR={LR_PRETRAIN})')
print(f'  Phase 3 Fine-tuning epochs  : {EPOCHS_FINETUNE}  (LR={LR_FINETUNE})')
print(f'  Gradient clip               : {GRAD_CLIP}')
print()
print('  PRE-TRAINING METRICS')
print(f'  Final val MLM Loss          : {val_mlm_losses[-1]:.4f}')
print(f'  Final val MLM Accuracy      : {val_mlm_accs[-1]:.3f}')
print()
if all_cdr_data:
    shm_vals = [d['shm_count'] for d in all_cdr_data]
    print('  SEQUENCE METADATA (per paper Section 3.2.2)')
    print(f'  SHM count mean              : {np.mean(shm_vals):.1f}')
    print(f'  SHM count max               : {max(shm_vals)}')
    print(f'  Tissue sources              : {list(set(d["tissue_source"] for d in all_cdr_data))}')
    print(f'  Disease contexts            : {list(set(d["disease"] for d in all_cdr_data))}')
    print()
print('  BENCHMARK METRICS')
try:
    print(f'  Silhouette Score            : {sil_score:.4f}')
except: pass
try:
    print(f'  Nat vs Syn AUROC            : {auroc:.4f}')
except: pass
print()
print('  THERAPEUTIC METRICS (CDR.pdb candidate)')
print(f'  Humanness Score             : {mean_humanness:.4f}  ({risk})')
print(f'  Developability              : {np.mean(dev_scores):.2%}')
print(f'    PTM Risk                  : {np.mean(ptm_scores):.2%}')
print(f'    Aggregation               : {np.mean(agg_scores):.2%}')
print(f'    Chemical Liab.            : {np.mean(liab_scores):.2%}')
print(f'  Diversity Score             : {mean_div:.4f}')
print()
print('  SAVED FILES')
print('  antibody_encoder_pretrained.pt  — weights after Phase 1+2')
print('  antibody_encoder_module2.pt     — fine-tuned weights for Module 3')
print(f'\nRun ID: {RUN_ID}')



