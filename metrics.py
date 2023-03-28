import sacrebleu


def compute_bleu(predictions, targets):
    """Computes the BLEU score from a list of model predictions and their corresponding true translations."""
    return sacrebleu.metrics.BLEU(tokenize='none').corpus_score(predictions, [targets]).score
