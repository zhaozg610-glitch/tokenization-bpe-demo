"""GPT-2 tokenizer"""
from hashlib import sha1, sha256
from importlib.metadata import version
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent
FILES = {
    "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
PREFIX = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/"
SAMPLES = [
    "Hello, world!", "中文", "🎵", "playing", "港口机器人",
    "Akwirw ier", "Hello  world!\n", "def f():\n    return 1\n", "",
]


def load_encoding():
    cache = BASE / "tokenizer_cache"
    for name, digest in FILES.items():
        file = cache / sha1((PREFIX + name).encode("utf-8")).hexdigest()
        if not file.is_file() or sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"离线词表缺失或校验失败：{name}。请使用完整演示包。")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def run_demo():
    enc = load_encoding()
    rows = []
    for text in SAMPLES:
        ids = enc.encode(text)
        restored = enc.decode(ids)
        if restored != text:
            raise AssertionError(f"原文重建失败：{text!r}")
        if enc.encode(text) != ids:
            raise AssertionError("同一输入编码不确定")
        raw_pieces = [enc.decode_single_token_bytes(i) for i in ids]
        if b"".join(raw_pieces) != text.encode("utf-8"):
            raise AssertionError("Token 字节拼接与原文 UTF-8 不一致")
        rows.append({
            "text": text, "codepoints": len(text), "utf8_bytes": len(text.encode("utf-8")),
            "token_count": len(ids), "ids": ids, "restored": restored,
            "roundtrip": restored == text,
            "token_bytes_hex": [piece.hex() for piece in raw_pieces],
        })
    # 这两个固定参考值检验加载的编码身份，其他样本以实际输出为准。
    if enc.n_vocab != 50257 or enc.encode("Hello, world!") != [15496, 11, 995, 0]:
        raise AssertionError("加载的 tokenizer 与 GPT-2 固定参考不一致")
    emoji = next(row for row in rows if row["text"] == "🎵")
    partial = enc.decode_single_token_bytes(emoji["ids"][0])
    try:
        partial.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        partial_invalid = True
    else:
        partial_invalid = False
    if not partial_invalid:
        raise AssertionError("本例的 emoji 首 Token 应为非完整 UTF-8 片段")
    return {
        "library": "tiktoken", "library_version": version("tiktoken"),
        "encoding": enc.name, "vocabulary_size": enc.n_vocab,
        "rows": rows,
        "emoji_partial": {
            "first_token_id": emoji["ids"][0], "bytes_hex": partial.hex(),
            "standalone_utf8_valid": False,
            "whole_sequence_roundtrip": emoji["roundtrip"],
        },
        "checks": {
            "cache_sha256_valid": True, "gpt2_reference_ids_match": True,
            "nine_roundtrips_pass": True, "encoding_deterministic": True,
            "all_token_bytes_reconstruct_utf8": True,
            "emoji_partial_byte_behavior_confirmed": True,
        },
    }


def print_demo(result):
    print("\n=== 演示二：真实 GPT-2 tokenizer ===")
    print(f'tiktoken={result["library_version"]}, encoding={result["encoding"]}, vocab={result["vocabulary_size"]}')
    for row in result["rows"]:
        print(f'text={row["text"]!r}')
        print(f'  chars={row["codepoints"]}, bytes={row["utf8_bytes"]}, tokens={row["token_count"]}')
        print(f'  ids={row["ids"]}, roundtrip={row["roundtrip"]}')
    partial = result["emoji_partial"]
    print(f'emoji 首 Token 字节={partial["bytes_hex"]}，单独 UTF-8 合法=False，整段还原=True')
    print("验证：9 组整段还原、字节重建、确定性与 GPT-2 参考值通过。")


if __name__ == "__main__":
    print_demo(run_demo())
