"""
Tokenize a public text dataset with the GPT-2 BPE and write flat uint16 token files
(train.bin / val.bin) that the benchmark memory-maps.

Adapted from nanoGPT's data/shakespeare/prepare.py and data/openwebtext/prepare.py
(https://github.com/karpathy/nanoGPT, MIT License, Copyright (c) 2022 Andrej Karpathy).

Datasets:
  shakespeare  ~1 MB tiny-shakespeare. Used for CPU smoke tests.
  openwebtext  A subset of OpenWebText: the first N parquet shards (of 80, ~300 MB each)
               of the Hugging Face parquet conversion of Skylion007/openwebtext.
               Used for the GPU runs. The full dataset (~9B tokens) is not needed:
               the benchmark measures throughput over a few hundred steps, not convergence.

Usage:
  python scripts/prepare_data.py --dataset shakespeare
  python scripts/prepare_data.py --dataset openwebtext --owt_shards 1
  python scripts/prepare_data.py --dataset openwebtext --owt_shards 1 --max_docs 20000

Every run writes data/<dataset>/meta.json with the source URLs, SHA-256 of each downloaded
file, and exact token counts, so the data behind any result can be identified later.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import requests
import tiktoken

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Hugging Face's auto-converted parquet files for OpenWebText. "refs/convert/parquet" is a
# branch HF can regenerate, so we record the SHA-256 of what we actually downloaded.
OWT_SHARD_URL = ("https://huggingface.co/datasets/Skylion007/openwebtext/resolve/"
                 "refs%2Fconvert%2Fparquet/plain_text/train/{:04d}.parquet")
OWT_NUM_SHARDS = 80

# Every 200th document goes to val (0.5%), close to nanoGPT's 0.05% but large enough to be
# a usable val set on a single-shard subset. Deterministic, so reruns give identical files.
OWT_VAL_EVERY = 200


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    """Stream url to path. Skips the download if the file already exists."""
    if os.path.exists(path):
        print(f"  already downloaded: {path}")
        return
    tmp = path + ".part"  # rename only on success, so a crash never leaves a truncated file
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        # With gzip transfer encoding, content-length is the compressed size but iter_content
        # yields decompressed bytes, so a percentage would be wrong. Only show it when exact.
        compressed = r.headers.get("content-encoding") not in (None, "identity")
        total = 0 if compressed else int(r.headers.get("content-length", 0))
        done, last_print = 0, 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last_print > 5:
                    pct = f"{100 * done / total:.0f}%" if total else "?"
                    print(f"  {os.path.basename(path)}: {done / 1e6:.0f} MB ({pct})")
                    last_print = time.time()
    os.replace(tmp, path)


def prepare_shakespeare(out_dir, enc):
    raw = os.path.join(out_dir, "input.txt")
    download(SHAKESPEARE_URL, raw)
    with open(raw, "r", encoding="utf-8") as f:
        data = f.read()

    # Same 90/10 character split as nanoGPT, so token counts are comparable to its readme.
    n = len(data)
    train_ids = enc.encode_ordinary(data[: int(n * 0.9)])
    val_ids = enc.encode_ordinary(data[int(n * 0.9):])
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "val.bin"))

    return {
        "sources": [{"url": SHAKESPEARE_URL, "sha256": sha256_of(raw)}],
        "num_docs": 1,
        "split": "first 90% of characters -> train, last 10% -> val",
        "eot_appended": False,
    }


def prepare_openwebtext(out_dir, enc, num_shards, max_docs):
    import pyarrow.parquet as pq  # only needed for this dataset

    assert 1 <= num_shards <= OWT_NUM_SHARDS, f"--owt_shards must be in [1, {OWT_NUM_SHARDS}]"
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    sources = []
    doc_idx = 0
    # Write tokens incrementally so memory stays flat no matter how many shards are used.
    with open(os.path.join(out_dir, "train.bin"), "wb") as f_train, \
         open(os.path.join(out_dir, "val.bin"), "wb") as f_val:
        for shard in range(num_shards):
            url = OWT_SHARD_URL.format(shard)
            path = os.path.join(raw_dir, f"{shard:04d}.parquet")
            print(f"shard {shard}: {url}")
            download(url, path)
            sources.append({"url": url, "sha256": sha256_of(path)})

            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
                texts = batch.column("text").to_pylist()
                if max_docs is not None:
                    texts = texts[: max_docs - doc_idx]
                # encode_ordinary ignores special tokens; we append <|endoftext|> (50256) as a
                # document separator, as nanoGPT does.
                for ids in enc.encode_ordinary_batch(texts, num_threads=os.cpu_count() or 1):
                    ids.append(enc.eot_token)
                    out = f_val if doc_idx % OWT_VAL_EVERY == 0 else f_train
                    np.array(ids, dtype=np.uint16).tofile(out)
                    doc_idx += 1
                if max_docs is not None and doc_idx >= max_docs:
                    break
            print(f"  documents so far: {doc_idx:,}")
            if max_docs is not None and doc_idx >= max_docs:
                break

    return {
        "sources": sources,
        "num_docs": doc_idx,
        "split": f"document i -> val if i % {OWT_VAL_EVERY} == 0 else train",
        "eot_appended": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["shakespeare", "openwebtext"], required=True)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data"),
                   help="files go to <out_dir>/<dataset>/ (default: ./data)")
    p.add_argument("--owt_shards", type=int, default=1,
                   help="openwebtext only: number of ~300 MB parquet shards to use (1-80)")
    p.add_argument("--max_docs", type=int, default=None,
                   help="openwebtext only: stop after this many documents (for quick tests)")
    args = p.parse_args()

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    # uint16 storage is only valid because every GPT-2 token id (max 50256) fits in 16 bits.
    assert enc.max_token_value < 2**16

    t0 = time.time()
    if args.dataset == "shakespeare":
        info = prepare_shakespeare(out_dir, enc)
    else:
        info = prepare_openwebtext(out_dir, enc, args.owt_shards, args.max_docs)

    # Token counts are read back from the files actually written, not tracked separately.
    itemsize = np.dtype(np.uint16).itemsize
    train_tokens = os.path.getsize(os.path.join(out_dir, "train.bin")) // itemsize
    val_tokens = os.path.getsize(os.path.join(out_dir, "val.bin")) // itemsize

    meta = {
        "dataset": args.dataset,
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": enc.n_vocab,
        "dtype": "uint16",
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "args": vars(args),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **info,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"train.bin: {train_tokens:,} tokens")
    print(f"val.bin:   {val_tokens:,} tokens")
    print(f"wrote {out_dir} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
