"""字符起步的教学 BPE；语料来源见 README，非 GPT-2 tokenizer。"""
from collections import Counter
import json

CORPUS = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
ALPHABET = frozenset("".join(CORPUS))


def merge_pair(tokens, pair):
    """在当前词内，从左到右进行非重叠替换。"""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tuple(tokens[i:i + 2]) == pair:
            out.append("".join(pair))
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(corpus, n_merges=4):
    """按词预切分，加权计数；并列时按候选对字典序选择。"""
    words = {word: list(word) for word in corpus}
    rules, trace = [], []
    for step in range(1, n_merges + 1):
        counts = Counter()
        for word, tokens in words.items():
            for pair in zip(tokens, tokens[1:]):
                counts[pair] += corpus[word]
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        words = {w: merge_pair(tokens, pair) for w, tokens in words.items()}
        rules.append(pair)
        trace.append({
            "round": step, "pair": pair, "count": counts[pair],
            "splits": {w: tokens[:] for w, tokens in words.items()},
            "weighted_token_count": sum(len(words[w]) * f for w, f in corpus.items()),
        })
    return rules, trace


def encode_word(word, rules, alphabet=ALPHABET):
    """仅应用已学规则；基础字符未覆盖时显式报告，避免伪造 UNK 回退。"""
    missing = set(word) - set(alphabet)
    if missing:
        raise ValueError(f"未覆盖基础字符：{sorted(missing)}")
    tokens = list(word)
    ranks = {pair: rank for rank, pair in enumerate(rules)}
    while True:
        candidates = [pair for pair in zip(tokens, tokens[1:]) if pair in ranks]
        if not candidates:
            return tokens
        tokens = merge_pair(tokens, min(candidates, key=ranks.__getitem__))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_demo():
    rules, trace = train_bpe(CORPUS, 4)
    expected_rules = [("u", "g"), ("u", "n"), ("h", "ug"), ("p", "un")]
    require(rules == expected_rules, "四轮规则与手算不一致")
    require([r["count"] for r in trace] == [20, 16, 15, 12], "频数与手算不一致")
    require([r["weighted_token_count"] for r in trace] == [93, 77, 62, 50], "长度核验失败")
    expected_splits = {
        "hug": ["hug"], "pug": ["p", "ug"], "pun": ["pun"],
        "bun": ["b", "un"], "hugs": ["hug", "s"],
    }
    require(trace[-1]["splits"] == expected_splits, "最终切分与 PPT 不一致")
    tokens = encode_word("bug", rules)
    require(tokens == ["b", "ug"], "新词 bug 的切分错误")
    vocab = sorted(set(ALPHABET) | {"".join(pair) for pair in rules})
    to_id = {piece: i for i, piece in enumerate(vocab)}
    roundtrips = []
    for word in [*CORPUS, "bug"]:
        pieces = encode_word(word, rules)
        ids = [to_id[piece] for piece in pieces]
        restored = "".join(vocab[i] for i in ids)
        require(restored == word, f"编解码不一致：{word}")
        roundtrips.append({"text": word, "pieces": pieces, "ids": ids, "restored": restored})
    try:
        encode_word("mug", rules)
    except ValueError as error:
        coverage_error = str(error)
    else:
        raise AssertionError("m 未收录，应显式报告覆盖错误")
    # 语料顺序改变不应改变本例没有最高频并列时的四轮结果。
    reversed_rules, _ = train_bpe(dict(reversed(list(CORPUS.items()))), 4)
    require(reversed_rules == rules, "语料顺序影响了结果")
    return {
        "algorithm": "toy character BPE, word-pretokenized",
        "source": "https://huggingface.co/learn/llm-course/chapter6/5",
        "corpus": CORPUS, "rules": rules, "trace": trace,
        "bug_tokens": tokens, "roundtrips": roundtrips,
        "coverage_error_mug": coverage_error,
        "checks": {
            "rules_match_manual": True, "counts_match_manual": True,
            "lengths_match_manual": True, "final_splits_match_ppt": True,
            "bug_split_correct": True, "six_roundtrips_pass": True,
            "unknown_character_rejected": True, "input_order_invariant": True,
        },
    }


def print_demo(result):
    print("=== 演示一：教学 BPE ===")
    for row in result["trace"]:
        a, b = row["pair"]
        print(f'round {row["round"]}: {a} + {b} -> {a+b}, count={row["count"]}')
    print("bug =", result["bug_tokens"])
    print("mug =", result["coverage_error_mug"])
    print("验证：规则、频数、最终状态、6 组编解码及覆盖边界通过。")


if __name__ == "__main__":
    print_demo(run_demo())
