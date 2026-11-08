"""运行两组验证并保存可追溯的 JSON、控制台文本。"""
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import json
import platform
import sys

import bpe_demo
import tokenizer_demo


def main():
    folder = Path(__file__).resolve().parent
    captured = StringIO()
    # 成功运行即同时验证：两组演示不会依赖课堂网络。
    with patch("socket.socket.connect", side_effect=RuntimeError("离线验证禁用网络连接")):
        with redirect_stdout(captured):
            toy = bpe_demo.run_demo()
            real = tokenizer_demo.run_demo()
            bpe_demo.print_demo(toy)
            tokenizer_demo.print_demo(real)
            print("\nALL CHECKS PASSED：所有列出的检查均通过，运行期间禁止网络连接。")
    text = captured.getvalue()
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    report = {
        "run_time_utc8": now,
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: version(name) for name in [
            "tiktoken", "regex", "requests", "charset-normalizer", "idna", "urllib3", "certifi"
        ]},
        "source_sha256": {name: sha256((folder / name).read_bytes()).hexdigest()
                          for name in ["bpe_demo.py", "tokenizer_demo.py", "run_all.py"]},
        "network_disabled_during_validation": True,
        "toy_bpe": toy, "real_tokenizer": real,
        "scope": "局部算法与编码验证；未运行语言模型，未评估模型语义能力或多语言质量。",
    }
    output = folder / "outputs"
    output.mkdir(exist_ok=True)
    (output / "实测结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "实测输出.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print("结果已保存到：", output)


if __name__ == "__main__":
    main()
