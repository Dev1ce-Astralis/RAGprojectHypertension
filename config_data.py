

md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", ",", "，", "。", "！", "？", "；", ";"]
max_split_char_number = 1000

# retriever
similarity_threshold = 1            # 向量检索返回文档数
bm25_k = 3                          # BM25 检索返回文档数
rrf_k = 60                          # RRF 融合常数（默认 60）              

# model
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# session
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}