# Tokenization 与子词编码：课程代码演示

本仓库配套“Tokenization 与子词编码”课程汇报，包含两组可运行演示：

1. **教学 BPE**：从带频次的语料学习四轮合并规则，与手算结果核对，再编码新词 `bug`。
2. **真实 GPT-2 tokenizer**：使用 `tiktoken` 将文本编码为 Token ID，再解码并验证原文还原。

代码不调用大模型 API，不需要账号、API Key 或模型权重。安装依赖、准备好官方编码数据后，演示与验证均可离线执行。

## 快速运行

推荐 **Python 3.12**。本项目固定使用 `tiktoken==0.7.0`，其余依赖见 [requirements.txt](requirements.txt)。建议使用独立虚拟环境，避免与系统其他项目混用。

在本仓库目录执行：

```bash
python --version
python -m venv .venv
```

Windows PowerShell 7：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_cache.py
.\.venv\Scripts\python.exe run_all.py
.\.venv\Scripts\python.exe classroom_snippets.py
```

macOS / Linux：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_cache.py
.venv/bin/python run_all.py
.venv/bin/python classroom_snippets.py
```

`prepare_cache.py` 首次运行需要网络，从官方地址下载 `vocab.bpe` 和 `encoder.json`，逐项检查固定的 SHA-256，然后保存到 `tokenizer_cache/`。后续发现正确缓存时直接使用本地文件。依赖安装和缓存准备完成后，`run_all.py`、`classroom_snippets.py` 在执行验证时禁止建立网络连接。

Windows 也可在安装依赖后，用下面的入口执行缓存准备及两组验证：

```powershell
.\run_demo.ps1 -Python .\.venv\Scripts\python.exe
```

仅运行教学 BPE 不需要任何第三方包：

```bash
python bpe_demo.py
```

## 演示一：教学 BPE

语料为 `hug×10、pug×5、pun×12、bun×4、hugs×5`。程序按词预切分，从字符开始，每轮统计**当前单位**的相邻对加权频数，合并最高频对后重新计数。并列时按候选对字典序选择。

与汇报对应的短代码：

```python
from bpe_demo import CORPUS, train_bpe, encode_word

rules, trace = train_bpe(CORPUS, 4)
for row in trace:
    print(row["round"], row["pair"], row["count"])
tokens = encode_word("bug", rules)
print("bug =", tokens)
assert tokens == ["b", "ug"]
```

实际运行输出：

```text
1 ('u', 'g') 20
2 ('u', 'n') 16
3 ('h', 'ug') 15
4 ('p', 'un') 12
bug = ['b', 'ug']
```

第一轮合并后，`pug` 已切成 `[p, ug]`，原来的 `(p,u)` 计数从 17 降到 12。因此每轮必须重新计数，不能沿用最初的排序。

这是字符起步、按词预切分的教学变体，不代替完整 GPT-2 tokenizer。基础字符表缺少 `m`，因此编码 `mug` 会显式报错。词表没有整个新词时可尝试复用已有片段，但能否覆盖输入仍取决于基础单位。

完整检查包括：四轮规则与频数、最终五个词的切分、各轮加权单位数、`bug` 切分、6 组 Token-ID-文本往返、未知字符拒绝、语料输入顺序改变后的结果一致性。

## 演示二：真实 GPT-2 编解码

`tokenizer_demo.load_encoding()` 检查官方缓存后，调用真实的 `tiktoken.get_encoding("gpt2")`。

```python
from tokenizer_demo import load_encoding

enc = load_encoding()
for text in ["Hello, world!", "中文", "🎵"]:
    ids = enc.encode(text)
    restored = enc.decode(ids)
    print(repr(text), ids)
    print(len(text), len(ids), restored == text)
    assert restored == text
```

对应的实测值：

| 输入 | Python `len(text)` | Token 数 | Token ID | 整段还原 |
|---|---:|---:|---|---|
| `Hello, world!` | 13 | 4 | `[15496, 11, 995, 0]` | True |
| `中文` | 2 | 3 | `[40792, 23877, 229]` | True |
| `🎵` | 1 | 3 | `[8582, 236, 113]` | True |

Python `len(text)` 按 Unicode 码点计数，不对所有文本都等同于视觉字素数。Token 数与码点数没有固定比例。

表情的首 Token ID `8582` 对应字节 `f09f`，单独不是完整的 UTF-8 字符；整个 Token ID 序列仍能正确还原原文。因此应对完整 ID 序列解码，不能假设每个 Token 都对应完整字符。

完整脚本验证 9 组输入：英文、中文、表情、`playing`、`港口机器人`、陌生拼写、连续空格和换行、带缩进的 Python 代码、空字符串。逐组检查整段还原、编码确定性、Token 字节拼接等于原文 UTF-8，并核对 GPT-2 词表大小及固定参考 ID。

## 文件与实测结果

| 文件 | 作用 |
|---|---|
| [bpe_demo.py](bpe_demo.py) | 教学 BPE 训练、编码与断言 |
| [tokenizer_demo.py](tokenizer_demo.py) | 真实 GPT-2 tokenizer 和 9 组输入验证 |
| [prepare_cache.py](prepare_cache.py) | 官方编码数据下载与 SHA-256 校验 |
| [run_all.py](run_all.py) | 禁用网络连接，执行完整验证并保存结果 |
| [classroom_snippets.py](classroom_snippets.py) | 执行汇报所用短代码，保存准确输出 |
| [run_demo.ps1](run_demo.ps1) | Windows PowerShell 运行入口 |
| [outputs/实测结果.json](outputs/实测结果.json) | 运行环境、源文件哈希、样本与检查结果 |
| [outputs/实测输出.txt](outputs/实测输出.txt) | 完整控制台输出 |
| [outputs/课堂片段实测.json](outputs/课堂片段实测.json) | 课堂短代码及其真实 stdout |

已在 Windows x64、Python 3.12.14、tiktoken 0.7.0 下运行。所有列出的检查通过时，控制台显示 `ALL CHECKS PASSED`。重新运行会更新 `outputs/` 中的运行时间、环境与结果。

本仓库中的检查验证算法示例与编解码行为，没有运行语言模型，也没有评估理解能力、多语言质量或推理速度。固定 ID 适用于明确指定的 `gpt2` 编码，不代表所有模型。更换编码、依赖版本或样本后应重新验证。

## 来源

- 教学语料：[Hugging Face LLM Course，Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)。教学实现为本课程作业编写。
- 真实编码库：[OpenAI tiktoken](https://github.com/openai/tiktoken)，固定使用 0.7.0。
- 官方 GPT-2 编码数据：[vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe)、[encoder.json](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json)。程序记录并校验 SHA-256。

仓库仅包含代码、说明及实测结果。依赖包、虚拟环境、编码缓存、课程参考书 PDF 和个人电脑路径均不随仓库提交。
