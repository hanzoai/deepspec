#!/usr/bin/env python3
"""Extract token_embd (F16) + output head (Q8_0 -> dequant f16) from a V4 GGUF
as the frozen draft init (DeepSpec copies target embed/lm_head into the draft)."""
import argparse, struct
import numpy as np
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

f = open(args.gguf, "rb")
f.read(8)
n_t = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
def rd_str():
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)
sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
align = 32
for _ in range(n_kv):
    k = rd_str().decode(); t = struct.unpack("<I", f.read(4))[0]
    if t == 8: rd_str()
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n): rd_str()
        else: f.seek(sizes[et]*n, 1)
    else:
        v = f.read(sizes[t])
        if k == "general.alignment": align = struct.unpack("<I", v)[0]
infos = {}
for _ in range(n_t):
    name = rd_str().decode()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
    ttype = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    if name in ("token_embd.weight", "output.weight"):
        infos[name] = (dims, ttype, off)
data_start = (f.tell() + align - 1) // align * align
out = {}
for name, (dims, ttype, off) in infos.items():
    ne = 1
    for d in dims: ne *= d
    f.seek(data_start + off)
    if ttype == 1:  # F16
        w = np.frombuffer(f.read(ne*2), dtype=np.float16).reshape(dims[::-1])
    elif ttype == 8:  # Q8_0: 32-blocks of f16 scale + 32 i8
        nb = ne // 32
        blk = np.frombuffer(f.read(nb*34), dtype=np.uint8).reshape(nb, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].copy().view(np.int8).astype(np.float32)
        w = (q * d).astype(np.float16).reshape(dims[::-1])
    else:
        raise SystemExit(f"unexpected ggml type {ttype} for {name}")
    out[name] = w
    print(name, w.shape, w.dtype)
save_file({"embed_tokens.weight": out["token_embd.weight"],
           "lm_head.weight": out["output.weight"]}, args.out)
print("saved ->", args.out)
