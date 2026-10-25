"""加载已持久化的 LlamaIndex 索引并执行相似性搜索。"""

import argparse
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


DEFAULT_PERSIST_DIR = Path(__file__).parent / "llamaindex_index_store"
DEFAULT_QUERY = "LlamaIndex是什么？"


def load_index(persist_dir: Path):
    """从本地目录加载已持久化的索引。"""
    if not persist_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 LlamaIndex 索引目录: {persist_dir}\n"
            "请先运行 03_llamaindex_vector.py 创建索引。"
        )

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context)


def search(index, query: str, top_k: int):
    """返回与查询最相似的节点及其相似度分数。"""
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 LlamaIndex 持久化索引并执行相似性搜索"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"搜索文本（默认：{DEFAULT_QUERY}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回最相似的节点数量（默认：3）",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help=f"索引存储目录（默认：{DEFAULT_PERSIST_DIR}）",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须是大于 0 的整数")
    return args


def main() -> None:
    args = parse_args()

    # 查询时需要使用与创建索引时相同的嵌入模型。
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    index = load_index(args.persist_dir)
    results = search(index, args.query, args.top_k)

    print(f"查询: {args.query}")
    print(f"返回结果数: {len(results)}\n")
    for rank, result in enumerate(results, start=1):
        score = "未知" if result.score is None else f"{result.score:.4f}"
        print(f"[{rank}] 相似度: {score}")
        print(result.node.get_content())
        print()


if __name__ == "__main__":
    main()
