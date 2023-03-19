import sacrebleu
import torch
import torch.nn as nn
import torchtext
import tqdm.auto
import wandb

import dataclasses
import math
import os
import pathlib


if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    import kaggle_secrets
    DATA_DIR = pathlib.Path('../input/dl1-bhw2-data')
    WANDB_KEY = kaggle_secrets.UserSecretsClient().get_secret('wandb_key')
else:
    DATA_DIR = pathlib.Path('data')
    WANDB_KEY = os.environ.get('WANDB_KEY')
OUTPUT_DIR = pathlib.Path('.')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    NUM_WORKERS = len(os.sched_getaffinity(0))
except AttributeError:
    NUM_WORKERS = os.cpu_count()

SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = range(4)

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
LANGUAGES = [SRC_LANGUAGE, TGT_LANGUAGE]


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the token embedding to introduce a notion of word order."""
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """Converts a tensor of input indices into the corresponding tensor of token embeddings."""
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    """A seq2seq transformer model."""
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def make_subsequent_mask(size):
    """
    Returns a tensor (on `DEVICE`) of shape `(size, size)`
    where the `(i,j)`-th element is `0.0` if `j <= i` and `-inf` otherwise.
    """
    return torch.full((size, size), float('-inf'), device=DEVICE).triu(1)


def make_masks(src, tgt):
    """Returns masks for the source and target tokens to be fed into a transformer model."""
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = make_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool, device=DEVICE)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    """Combines transforms sequentially."""
    def func(arg):
        for transform in transforms:
            arg = transform(arg)
        return arg
    return func


def token_transform(line):
    """Tokenizes a string."""
    return line.strip().split()


def tensor_transform(token_ids):
    """Creates a tensor for a sequence of token indices, adding <bos> and <eos>."""
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long)


def collate_fn(batch):
    """Collates data samples into batch tensors."""
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


class TextDataset(torch.utils.data.Dataset):
    """Stores lines from a text file."""
    def __init__(self, file_path, transform, portion=1):
        """
        Reads lines from `file_path`, applies `transform` and saves both the raw and transformed lines to memory.
        If a `portion` between 0 and 1 is specified, only stores that portion of the lines in the file
        (from the beginning of the file).
        """
        super().__init__()
        assert 0 <= portion <= 1
        self.raw_lines = file_path.read_text().split('\n')
        self.raw_lines = self.raw_lines[:int(len(self.raw_lines) * portion)]
        self.data = [transform(line) for line in self.raw_lines]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class ZipDataset(torch.utils.data.Dataset):
    """Zips multiple datasets of the same length analogously to the built-in zip() function."""
    def __init__(self, *datasets):
        super().__init__()
        assert datasets and all(len(ds) == len(datasets[0]) for ds in datasets[1:])
        self.datasets = datasets
    
    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.datasets)
    
    def __len__(self):
        return len(self.datasets[0])


def train_epoch(model, optimizer, loader):
    """Trains `model` on data from `loader` using `optimizer`. Returns the mean loss."""
    model.train()
    loss_sum = 0

    for src, tgt in tqdm.auto.tqdm(loader, desc='training'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = make_masks(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sum += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_sum / len(loader)


@torch.inference_mode()
def test(model, loader):
    """Tests `model` on data from `loader`. Returns the mean loss."""
    model.eval()
    loss_sum = 0
    
    for src, tgt in tqdm.auto.tqdm(loader, desc='testing'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = make_masks(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sum += loss.item()

    return loss_sum / len(loader)


@dataclasses.dataclass
class Ray:
    """Represents a single sequence of tokens during beam search."""
    tokens: torch.Tensor
    log_prob: torch.float32
    done: bool

    def __len__(self):
        return self.tokens.shape[0]


@torch.inference_mode()
def beam_search(model, src, beam_size, alpha, max_added_tokens=15):
    """
    Generates an output sequence with beam search.
    Implements length normalization with parameter `alpha`,
    as in section 7 of https://arxiv.org/pdf/1609.08144.pdf.
    The source sequence `src` must have the shape `(num_src_tokens, 1)`.
    The returned sequence will have the shape `(num_out_tokens, 1)`.
    """
    model.eval()
    src = src.to(DEVICE)
    src_mask = torch.zeros((src.shape[0], src.shape[0]), dtype=torch.bool, device=DEVICE)
    memory = model.encode(src, src_mask)

    rays = [Ray(torch.full((1, 1), BOS_IDX, dtype=src.dtype), 0, False)]
    for _ in range(src.shape[0] + max_added_tokens - 1):
        candidate_rays = [ray for ray in rays if ray.done]
        active_rays = [ray for ray in rays if not ray.done]
        if not active_rays:
            break

        batch_tgts = torch.cat([ray.tokens for ray in active_rays], dim=1).to(DEVICE)
        batch_memory = memory.repeat((1, len(active_rays), 1)).to(DEVICE)
        tgt_mask = make_subsequent_mask(batch_tgts.shape[0]).bool().to(DEVICE)

        out = model.decode(batch_tgts, batch_memory, tgt_mask).transpose(0, 1)
        batch_logits = model.generator(out[:, -1])
        batch_log_probs = nn.functional.log_softmax(batch_logits, dim=1)

        topk = batch_log_probs.topk(k=beam_size, dim=1)
        for ray, log_probs, tokens in zip(active_rays, topk.values, topk.indices):
            for log_prob, token in zip(log_probs, tokens):
                candidate_rays.append(Ray(
                    torch.cat([ray.tokens, torch.full((1, 1), token.item(), dtype=src.dtype)], dim=0),
                    ray.log_prob + log_prob.item(),
                    token == EOS_IDX
                ))

        candidate_rays.sort(key=lambda ray: ray.log_prob / (((5 + len(ray)) / 6) ** alpha), reverse=True)
        rays = candidate_rays[:beam_size]

    return rays[0].tokens


def post_processing(tokens):
    """Performs post-processing on a translated sequence of tokens. Returns a string."""
    return ' '.join(token for token in tokens if token not in SPECIALS)


def translate(model, src_sentence, beam_size, alpha):
    """Translates a string. Returns the result as a string."""
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    tgt_token_ids = beam_search(model, src, beam_size, alpha).flatten()
    tgt_tokens = vocab_transform[TGT_LANGUAGE].lookup_tokens(tgt_token_ids.tolist())
    return post_processing(tgt_tokens)


def translate_file(model, input_path, output_path, beam_size, alpha):
    """Translates an entire text file, line-by-line."""
    print(f'Translating {input_path}...')
    lines = []
    for line in tqdm.auto.tqdm(input_path.read_text().split('\n'), desc='translating'):  
        lines.append(translate(model, line, beam_size, alpha) + '\n')
    output_path.write_text(''.join(lines))
    print(f'Translations saved to {output_path}')


def compute_bleu(predictions, targets):
    """Computes the BLEU score from a list of model predictions and their corresponding true translations."""
    return sacrebleu.metrics.BLEU(tokenize='none').corpus_score(predictions, [targets]).score


print(f'\ndevice: {DEVICE}')

cfg = {
    # Preprocessing parameters
    'vocab_min_freq': 3,

    # Model parameters
    'model': 'transformer',
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'emb_size': 512,
    'nhead': 8,
    'dim_feedforward': 512,
    'dropout': 0.2,

    # Loss parameters
    'label_smoothing': 0.1,

    # Optimizer parameters
    'optimizer': 'adam',
    'init_lr': 2e-4,
    'beta_1': 0.9,
    'beta_2': 0.98,
    'eps': 1e-9,

    # Scheduler parameters
    'scheduler': 'rop',
    'patience': 1,
    'factor': 0.5,

    # Inference parameters
    'val_beam_size': 1,
    'test_beam_size': 10,
    'beam_alpha': 3.6,

    # Misc
    'batch_size': 128,
    'n_epochs': 27,
    'samples_portion': 1
}

vocab_transform = {}
text_transform = {}
for lang in LANGUAGES:
    tokens = map(token_transform, (DATA_DIR / f'train.de-en.{lang}').read_text().split('\n'))
    vocab_transform[lang] = torchtext.vocab.build_vocab_from_iterator(
        tokens,
        min_freq=cfg['vocab_min_freq'],
        specials=SPECIALS,
        special_first=True
    )
    vocab_transform[lang].set_default_index(UNK_IDX)
    text_transform[lang] = sequential_transforms(token_transform, vocab_transform[lang], tensor_transform)

train_set_mono = {}
val_set_mono = {}
for lang in LANGUAGES:
    train_set_mono[lang] = TextDataset(
        DATA_DIR / f'train.de-en.{lang}',
        transform=text_transform[lang],
        portion=cfg['samples_portion']
    )
    val_set_mono[lang] = TextDataset(
        DATA_DIR / f'val.de-en.{lang}',
        transform=text_transform[lang]
    )
train_set = ZipDataset(train_set_mono[SRC_LANGUAGE], train_set_mono[TGT_LANGUAGE])
val_set = ZipDataset(val_set_mono[SRC_LANGUAGE], val_set_mono[TGT_LANGUAGE])

train_loader = torch.utils.data.DataLoader(
    train_set,
    collate_fn=collate_fn,
    batch_size=cfg['batch_size'],
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    collate_fn=collate_fn,
    batch_size=cfg['batch_size'],
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = Seq2SeqTransformer(
    num_encoder_layers=cfg['num_encoder_layers'],
    num_decoder_layers=cfg['num_decoder_layers'],
    emb_size=cfg['emb_size'],
    nhead=cfg['nhead'],
    dim_feedforward=cfg['dim_feedforward'],
    src_vocab_size=len(vocab_transform[SRC_LANGUAGE]),
    tgt_vocab_size=len(vocab_transform[TGT_LANGUAGE]),
    dropout=cfg['dropout']
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg['label_smoothing'])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'], betas=(cfg['beta_1'], cfg['beta_2']), eps=cfg['eps'])
schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg['patience'], factor=cfg['factor'])]

log_wandb = False
resume_wandb_run_id = None
resume_checkpoint_path = None

if resume_checkpoint_path is not None:
    checkpoint = torch.load(resume_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for sk, state in zip(schedulers, checkpoint['schedulers']):
        sk.load_state_dict(state)
    first_epoch = checkpoint['epoch'] + 1
    min_val_loss = checkpoint['min_val_loss']
    max_val_bleu = checkpoint['max_val_bleu']
else:
    first_epoch = 1
    min_val_loss = 1e9
    max_val_bleu = -1

if log_wandb:
    wandb.login(anonymous='never', key=WANDB_KEY)
    if resume_wandb_run_id is None:
        wandb.init(project='dl1-bhw2', config=cfg, save_code=True)
    else:
        wandb.init(project='dl1-bhw2', config=cfg, save_code=True, id=resume_wandb_run_id, resume='must')
    run_name, run_number = wandb.run.name.rsplit('-', maxsplit=1)
    run_id = f'{run_number.zfill(2)}-{run_name}'
else:
    run_id = 'translator'

log = []
for epoch in range(first_epoch, cfg['n_epochs'] + 1):
    print(f'Epoch {epoch}')

    train_loss = train_epoch(model, optimizer, train_loader)
    print(f'train loss: {train_loss:.3f}')

    val_loss = test(model, val_loader)
    print(f'val loss: {val_loss:.3f}')

    val_translations = [
        translate(model, line, cfg['val_beam_size'], cfg['beam_alpha'])
        for line in tqdm.auto.tqdm(val_set_mono[SRC_LANGUAGE].raw_lines, desc='translating for bleu')
    ]
    val_bleu = compute_bleu(val_translations, val_set_mono[TGT_LANGUAGE].raw_lines)
    print(f'val bleu: {val_bleu:.3f}\n')

    log_entry = {
        'train/loss': train_loss,
        'val/loss': val_loss,
        'val/bleu': val_bleu,
        'lr': optimizer.param_groups[0]['lr']
    }
    log.append(log_entry)
    if log_wandb:
        wandb.log(log_entry)

    for scheduler in schedulers:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'schedulers': [sk.state_dict() for sk in schedulers],
        'epoch': epoch,
        'min_val_loss': min_val_loss,
        'max_val_bleu': max_val_bleu
    }

    suffixes = ['last']
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        suffixes.append('minloss')
    if val_bleu > max_val_bleu:
        max_val_bleu = val_bleu
        suffixes.append('maxbleu')

    for suffix in suffixes:
        path = OUTPUT_DIR / f'ckpt-{run_id}-{suffix}.pt'
        torch.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')
    print()

if log_wandb:
    wandb.finish()

if val_bleu != max_val_bleu:
    model.load_state_dict(torch.load(OUTPUT_DIR / f'ckpt-{run_id}-maxbleu.pt')['model'])

translate_file(
    model,
    DATA_DIR / 'test1.de-en.de',
    OUTPUT_DIR / f'out-{run_id}-maxbleu.en',
    cfg['test_beam_size'],
    cfg['beam_alpha']
)
