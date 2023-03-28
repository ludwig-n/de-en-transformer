import torch
import torch.nn as nn
import tqdm.auto
import wandb

import os
import pathlib

import datasets
import inference
import metrics
import models
import preprocessing


if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    import kaggle_secrets
    DATA_DIR = pathlib.Path('../input/de-en-transformer-data')
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

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'


def train_epoch(model, loss_fn, optimizer, loader):
    """Trains `model` on data from `loader` with `loss_fn` and `optimizer`. Returns the mean loss."""
    model.train()
    loss_sum = 0

    for src, tgt in tqdm.auto.tqdm(loader, desc='training'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        masks = preprocessing.make_masks(src, tgt_input)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (mask.to(DEVICE) for mask in masks)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sum += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_sum / len(loader)


@torch.inference_mode()
def test(model, loss_fn, loader):
    """Tests `model` on data from `loader` with `loss_fn`. Returns the mean loss."""
    model.eval()
    loss_sum = 0
    
    for src, tgt in tqdm.auto.tqdm(loader, desc='testing'):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        masks = preprocessing.make_masks(src, tgt_input)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (mask.to(DEVICE) for mask in masks)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sum += loss.item()

    return loss_sum / len(loader)


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

text_transform = {
    lang: preprocessing.TextTransform(DATA_DIR / f'train.de-en.{lang}', cfg['vocab_min_freq'])
    for lang in [SRC_LANGUAGE, TGT_LANGUAGE]
}

train_set_mono = {}
val_set_mono = {}
for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_set_mono[lang] = datasets.TextDataset(
        DATA_DIR / f'train.de-en.{lang}',
        transform=text_transform[lang],
        portion=cfg['samples_portion']
    )
    val_set_mono[lang] = datasets.TextDataset(
        DATA_DIR / f'val.de-en.{lang}',
        transform=text_transform[lang]
    )
train_set = datasets.ZipDataset(train_set_mono[SRC_LANGUAGE], train_set_mono[TGT_LANGUAGE])
val_set = datasets.ZipDataset(val_set_mono[SRC_LANGUAGE], val_set_mono[TGT_LANGUAGE])

train_loader = torch.utils.data.DataLoader(
    train_set,
    collate_fn=preprocessing.collate_fn,
    batch_size=cfg['batch_size'],
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    collate_fn=preprocessing.collate_fn,
    batch_size=cfg['batch_size'],
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = models.Seq2SeqTransformer(
    num_encoder_layers=cfg['num_encoder_layers'],
    num_decoder_layers=cfg['num_decoder_layers'],
    emb_size=cfg['emb_size'],
    nhead=cfg['nhead'],
    dim_feedforward=cfg['dim_feedforward'],
    src_vocab_size=len(text_transform[SRC_LANGUAGE].vocab),
    tgt_vocab_size=len(text_transform[TGT_LANGUAGE].vocab),
    dropout=cfg['dropout']
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=preprocessing.PAD_IDX, label_smoothing=cfg['label_smoothing'])
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

    train_loss = train_epoch(model, loss_fn, optimizer, train_loader)
    print(f'train loss: {train_loss:.3f}')

    val_loss = test(model, loss_fn, val_loader)
    print(f'val loss: {val_loss:.3f}')

    val_translations = [
        inference.translate(
            model,
            line,
            text_transform[SRC_LANGUAGE],
            text_transform[TGT_LANGUAGE],
            cfg['val_beam_size'],
            cfg['beam_alpha']
        )
        for line in tqdm.auto.tqdm(val_set_mono[SRC_LANGUAGE].raw_lines, desc='translating for bleu')
    ]
    val_bleu = metrics.compute_bleu(val_translations, val_set_mono[TGT_LANGUAGE].raw_lines)
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

inference.translate_file(
    model,
    DATA_DIR / 'test1.de-en.de',
    OUTPUT_DIR / f'out-{run_id}-maxbleu.en',
    text_transform[SRC_LANGUAGE],
    text_transform[TGT_LANGUAGE],
    cfg['test_beam_size'],
    cfg['beam_alpha']
)
