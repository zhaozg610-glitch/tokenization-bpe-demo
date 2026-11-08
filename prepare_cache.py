"""下载官方 GPT-2 编码数据并校验 SHA-256；准备完成后演示可离线运行。"""
from hashlib import sha1, sha256
from pathlib import Path
from urllib.request import urlopen
import os

from tokenizer_demo import FILES, PREFIX


def main():
    cache = Path(__file__).resolve().parent / "tokenizer_cache"
    cache.mkdir(exist_ok=True)
    for name, expected_digest in FILES.items():
        url = PREFIX + name
        target = cache / sha1(url.encode("utf-8")).hexdigest()
        if target.is_file() and sha256(target.read_bytes()).hexdigest() == expected_digest:
            print(f"已校验本地缓存：{name}")
            continue
        print(f"下载官方编码数据：{url}")
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected_digest:
            raise RuntimeError(f"SHA-256 不匹配：{name}，未写入缓存。")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        print(f"下载及 SHA-256 校验通过：{name}")
    print("GPT-2 缓存准备完成。现在可以离线运行 run_all.py。")


if __name__ == "__main__":
    main()
