
corpus_file=/root/autodl-tmp/wiki2023/corpus.jsonl
save_dir=/root/autodl-tmp/wiki2023/index
retriever_name=e5
retriever_model=/root/autodl-tmp/models/e5-base-v2
num_gpus=5
faiss_type="IVF4096,PQ96"

mkdir -p $save_dir

# ── Step 1: Dense index (IVF4096,PQ96) ───────────────────────────────────────
# Encode each shard on its own GPU in parallel, saving raw embeddings only.
pids=()
for i in $(seq 0 $((num_gpus - 1))); do
    CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python index_builder.py \
        --retrieval_method $retriever_name \
        --model_path $retriever_model \
        --corpus_path $corpus_file \
        --save_dir $save_dir \
        --use_fp16 \
        --max_length 256 \
        --batch_size 4096 \
        --pooling_method mean \
        --faiss_type "$faiss_type" \
        --save_embedding \
        --shard_id $i \
        --num_shards $num_gpus \
        > $save_dir/shard${i}.log 2>&1 &
    pids+=($!)
    echo "Launched shard $i (PID ${pids[-1]})"
done

failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]}
    status=$?
    if [ $status -ne 0 ]; then
        echo "Shard $i failed (exit $status). Check $save_dir/shard${i}.log"
        failed=1
    else
        echo "Shard $i done."
    fi
done

if [ $failed -ne 0 ]; then
    echo "One or more shards failed. Aborting merge."
    exit 1
fi

# Train IVF centroids on a sample then add all shards incrementally (CPU).
echo "Merging shards and building $faiss_type index..."
python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 4096 \
    --pooling_method mean \
    --faiss_type "$faiss_type" \
    --num_shards $num_gpus \
    --merge_shards

echo "Dense index done: $save_dir/${retriever_name}_${faiss_type}.index"

# ── Step 2: BM25 index ────────────────────────────────────────────────────────
echo "Building BM25 index..."
python index_builder.py \
    --retrieval_method bm25 \
    --corpus_path $corpus_file \
    --save_dir $save_dir

echo "BM25 index done: $save_dir/bm25/"
echo "All done."
