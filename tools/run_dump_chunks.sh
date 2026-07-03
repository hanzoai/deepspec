#!/bin/bash
# Chunked V4 training-cache dump: 5000-sample chunks, resumable, nohup-friendly.
# Usage: run_dump_chunks.sh <corpus.jsonl> <out_root> <start_chunk> <n_chunks> <engine_dir>
set -u
CORPUS=$1; ROOT=$2; START=${3:-0}; N=${4:-4}; ENG=${5:-/home/z/work/hanzo/engine-v4bench}
CH=5000
for ((i=START; i<START+N; i++)); do
  DIR=$ROOT/chunk_$(printf %03d $i)
  [ -f "$DIR/.done" ] && echo "chunk $i done, skip" && continue
  mkdir -p "$DIR"
  FROM=$((i*CH+1)); TO=$(((i+1)*CH))
  sed -n "${FROM},${TO}p" "$CORPUS" > "$DIR/input.jsonl"
  echo "════ chunk $i: lines $FROM-$TO ════"
  (cd "$ENG" && IGPU_MEMORY_FRACTION=0.92 RUST_LOG=warn CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    ./target/release/examples/v4_cache_dump "$DIR/input.jsonl" "$DIR" $CH 2048) \
    >> "$ROOT/dump.log" 2>&1
  # exit 139 teardown-after-write is benign; judge by the index
  LINES=$(wc -l < "$DIR/index.jsonl" 2>/dev/null || echo 0)
  echo "chunk $i: $LINES samples indexed"
  [ "$LINES" -gt $((CH*9/10)) ] && touch "$DIR/.done" || { echo "chunk $i INCOMPLETE, stopping"; exit 1; }
done
echo ALL_CHUNKS_DONE
