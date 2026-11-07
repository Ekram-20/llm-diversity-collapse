"""Helpers for epistemic_diversity_experiment.ipynb.

Keeps the notebook short: file I/O, prompt formatting, condition names, the
OpenAI-compatible model wrapper for claim decomposition, and the exact
(analytic) coverage/diversity functions used as a robustness check.
"""
import gc, json, random, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# One prompt format for training, generation, and the untuned base model
# ---------------------------------------------------------------------------
PROMPT_NO_CTX = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_CTX = "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
# The base model never learned to emit EOS, so all models are cut at the next turn marker.
STOP_STRINGS = ["### Instruction:", "### Response:", "### Context:"]


def format_prompt(instruction, context=""):
    context = (context or "").strip()
    if context:
        return PROMPT_CTX.format(instruction=instruction.strip(), context=context)
    return PROMPT_NO_CTX.format(instruction=instruction.strip())


def cut_at_stop(text):
    hits = [text.find(s) for s in STOP_STRINGS if s in text]
    return (text[:min(hits)] if hits else text).strip()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def set_all_seeds(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------------------------------------------------------------------------
# Condition names: "base" or "n<size>_s<seed>"
# ---------------------------------------------------------------------------
def cond_name(size, seed):
    return f"n{size}_s{seed}"


def size_label(n):
    return "base" if n == 0 else (f"{n // 1000}k" if n >= 1000 else str(n))


def parse_cond(c):
    if c == "base":
        return 0, "base"
    n, s = re.match(r"n(\d+)_s(\d+)", c).groups()
    return int(n), s


# ---------------------------------------------------------------------------
# Claim decomposition model (any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
class APIModel:
    """Stand-in for llm_knowledge's model object so `extract_claims_bulk` can call
    any OpenAI-compatible endpoint (a vLLM server or a hosted provider).

    `tokenizer = None` makes llm_knowledge chunk responses by sentences only.
    Failed calls return "" (no claims) and are counted in `self.failures`.
    """

    def __init__(self, model_id, base_url, api_key, workers=16):
        from openai import OpenAI
        self.model_id, self.tokenizer, self.workers = model_id, None, workers
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=6, timeout=300)
        self.failures = 0

    def _one(self, msgs, max_new_tokens, temperature, top_p):
        try:
            r = self.client.chat.completions.create(
                model=self.model_id, messages=msgs, max_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p)
            return r.choices[0].message.content or ""
        except Exception as e:
            self.failures += 1
            print("decomposition call failed:", type(e).__name__, str(e)[:150])
            return ""

    def generate(self, messages, max_new_tokens=700, do_sample=True, top_p=0.9,
                 temperature=1.0, **kwargs):
        from tqdm.auto import tqdm
        temperature = temperature if do_sample else 0.0
        with ThreadPoolExecutor(self.workers) as ex:
            texts = list(tqdm(ex.map(lambda m: self._one(m, max_new_tokens, temperature, top_p), messages),
                              total=len(messages), leave=False))
        return {"text": texts}


# ---------------------------------------------------------------------------
# Exact coverage-based rarefaction (Chao & Jost 2012; Chao et al. 2014, as in iNEXT)
# Used as a deterministic robustness check next to llm_knowledge's resampling.
# ---------------------------------------------------------------------------
def _lc(n, k):
    """log C(n, k), -inf where k > n."""
    n, k = np.asarray(n, float), np.asarray(k, float)
    with np.errstate(invalid="ignore"):
        out = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return np.where((k >= 0) & (k <= n), out, -np.inf)


def coverage_observed(x):
    x = np.asarray(x); x = x[x > 0]; n = x.sum()
    f1, f2 = (x == 1).sum(), (x == 2).sum()
    if n == 0:
        return np.nan
    if f1 == 0:
        return 1.0
    a = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2) if f2 > 0 else (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)
    return float(1 - f1 / n * a)


def coverage_at(x, m):
    x = np.asarray(x); x = x[x > 0]; n = x.sum()
    if m >= n:
        return coverage_observed(x)
    return float(1 - np.sum(x / n * np.exp(_lc(n - x, m) - _lc(n - 1, m))))


def hill_shannon_at(x, m):
    x = np.asarray(x); x = x[x > 0]; n = int(x.sum()); m = int(m)
    if m >= n:
        p = x / n
        return float(np.exp(-np.sum(p * np.log(p))))
    vals, mult = np.unique(x, return_counts=True)
    fk = np.zeros(m + 1)
    for xi, c in zip(vals, mult):
        k = np.arange(1, min(xi, m) + 1)
        fk[k] += c * np.exp(_lc(xi, k) + _lc(n - xi, m - k) - _lc(n, m))
    k = np.arange(1, m + 1)
    return float(np.exp(-np.sum(k / m * np.log(k / m) * fk[1:])))


def size_for_coverage(x, target):
    """Smallest sample size m whose interpolated coverage reaches the target (m = n if needed)."""
    n = int(np.sum(x))
    if n <= 2 or coverage_at(x, n - 1) < target:
        return n
    lo, hi = 1, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if coverage_at(x, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def exact_rarefied_hsd(x, target):
    """Hill-Shannon diversity of abundance vector x, interpolated to coverage `target`."""
    return hill_shannon_at(x, size_for_coverage(x, target))


if __name__ == "__main__":
    # Sanity checks: even community -> D = S at full size; coverage behaves sensibly
    even = np.array([10] * 20)
    assert abs(hill_shannon_at(even, even.sum()) - 20) < 1e-9
    assert coverage_observed(even) == 1.0
    skew = np.array([50, 20, 10, 5, 3, 2, 1, 1, 1, 1])
    assert hill_shannon_at(skew, 20) <= hill_shannon_at(skew, skew.sum()) + 1e-9
    assert coverage_at(skew, size_for_coverage(skew, 0.8)) >= 0.8
    print("utils self-test passed")
