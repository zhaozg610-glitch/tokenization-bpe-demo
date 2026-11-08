"""PPT 上展示的短代码与其实际输出；完整逻辑在两份 demo 文件中。"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import json

BPE_SNIPPET = '''from bpe_demo import CORPUS, train_bpe, encode_word
rules, trace = train_bpe(CORPUS, 4)
for row in trace:
    print(row["round"], row["pair"], row["count"])
tokens = encode_word("bug", rules)
print("bug =", tokens)
assert tokens == ["b", "ug"]'''

TOKENIZER_SNIPPET = '''from tokenizer_demo import load_encoding
enc = load_encoding()  # GPT-2，使用离线词表
for text in ["Hello, world!", "中文", "🎵"]:
    ids = enc.encode(text)
    restored = enc.decode(ids)
    print(repr(text), ids)
    print(len(text), len(ids), restored == text)
    assert restored == text'''


def main():
    results = {}
    for name, code in [("bpe", BPE_SNIPPET), ("tokenizer", TOKENIZER_SNIPPET)]:
        buffer = StringIO()
        with patch("socket.socket.connect", side_effect=RuntimeError("")):
            with redirect_stdout(buffer):
                exec(compile(code, f"<classroom-{name}>", "exec"), {})
        results[name] = {"code": code, "stdout": buffer.getvalue()}
        print(f"=== {name} 课堂片段 ===")
        print(buffer.getvalue(), end="")
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    (output / "课堂片段实测.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
