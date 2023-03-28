import dataclasses
import pathlib

import torch
import torch.nn as nn
import tqdm.auto

import preprocessing


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
    device = next(model.parameters()).device
    src = src.to(device)
    src_mask = torch.zeros((src.shape[0], src.shape[0]), dtype=torch.bool, device=device)
    memory = model.encode(src, src_mask)

    rays = [Ray(torch.full((1, 1), preprocessing.BOS_IDX, dtype=src.dtype), 0, False)]
    for _ in range(src.shape[0] + max_added_tokens - 1):
        candidate_rays = [ray for ray in rays if ray.done]
        active_rays = [ray for ray in rays if not ray.done]
        if not active_rays:
            break

        batch_tgts = torch.cat([ray.tokens for ray in active_rays], dim=1).to(device)
        batch_memory = memory.repeat((1, len(active_rays), 1)).to(device)
        tgt_mask = preprocessing.make_subsequent_mask(batch_tgts.shape[0]).bool().to(device)

        out = model.decode(batch_tgts, batch_memory, tgt_mask).transpose(0, 1)
        batch_logits = model.generator(out[:, -1])
        batch_log_probs = nn.functional.log_softmax(batch_logits, dim=1)

        topk = batch_log_probs.topk(k=beam_size, dim=1)
        for ray, log_probs, tokens in zip(active_rays, topk.values, topk.indices):
            for log_prob, token in zip(log_probs, tokens):
                candidate_rays.append(Ray(
                    torch.cat([ray.tokens, torch.full((1, 1), token.item(), dtype=src.dtype)], dim=0),
                    ray.log_prob + log_prob.item(),
                    token == preprocessing.EOS_IDX
                ))

        candidate_rays.sort(key=lambda ray: ray.log_prob / (((5 + len(ray)) / 6) ** alpha), reverse=True)
        rays = candidate_rays[:beam_size]

    return rays[0].tokens


def post_processing(tokens):
    """Performs post-processing on a translated sequence of tokens. Returns a string."""
    return ' '.join(token for token in tokens if token not in preprocessing.SPECIALS)


def translate(model, src_sentence, src_text_transform, tgt_text_transform, beam_size, alpha):
    """Translates a string. Returns the result as a string."""
    src = src_text_transform(src_sentence).view(-1, 1)
    tgt_token_ids = beam_search(model, src, beam_size, alpha).flatten()
    tgt_tokens = tgt_text_transform.vocab.lookup_tokens(tgt_token_ids.tolist())
    return post_processing(tgt_tokens)


def translate_file(model, input_path, output_path, src_text_transform, tgt_text_transform, beam_size, alpha):
    """Translates an entire text file, line-by-line."""
    print(f'Translating {input_path}...')
    lines = []
    for line in tqdm.auto.tqdm(pathlib.Path(input_path).read_text('utf-8').split('\n'), desc='translating'):  
        lines.append(translate(model, line, src_text_transform, tgt_text_transform, beam_size, alpha) + '\n')
    output_path.write_text(''.join(lines))
    print(f'Translations saved to {output_path}')
