import json
import regex as re
from collections.abc import Iterable, Iterator
import os

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # 构造函数，接收以下参数创建分词器：
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.encoded_tokens = {}
        self.merge_rks = {merge: idx for idx, merge in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # 类方法，从文件读取vocab和merges，返回tokenizer的cls。其中vocab是.json，merges是.txt
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_str = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_str.items()}
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                for i in range(1, len(line) - 1):
                    if line[i] == ' ':
                        pair = (line[:i].encode('utf-8'), line[i + 1:].encode('utf-8'))
                        merges.append(pair)
                        break
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode_clean_text(self, text: str) -> list[int]:
        # 将输入文本编码为token ID序列
        text_ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for token_match in re.finditer(PAT, text):
            token = token_match.group(0)
            token_ids = []
            bytes_seq = tuple(bytes([i]) for i in token.encode(encoding="utf-8"))
            if bytes_seq in self.encoded_tokens.keys():
                text_ids.extend(self.encoded_tokens[bytes_seq])
                continue

            while True:
                pairs_dict = {i:(bytes_seq[i], bytes_seq[i + 1]) for i in range(len(bytes_seq) - 1)}
                pairs = pairs_dict.values()
                min_pair = min((pair for pair in pairs if pair in self.merge_rks), key=self.merge_rks.get, default=None) 
                if not min_pair:
                    break
                min_idx = min((idx for idx, pair in pairs_dict.items() if pair == min_pair), default=None)
                bytes_seq = bytes_seq[:min_idx] + (min_pair[0] + min_pair[1],) + bytes_seq[min_idx + 2:]
            
            token_ids = [self.inv_vocab[vocab_bytes] for vocab_bytes in bytes_seq]
            self.encoded_tokens[bytes_seq] = token_ids
            text_ids.extend(token_ids)

        return text_ids

    def encode(self, text: str) -> list[int]:
        # 将输入文本编码为token ID序列
        token_ids = []
        if self.special_tokens is not None:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(map(re.escape, special_tokens_sorted)) + ")"
            chunks = re.split(pattern, text)
            chunks = [chunk for chunk in chunks if chunk != ""]
            for chunk in chunks:
                if chunk in self.special_tokens:
                    token_ids.append(self.inv_vocab[chunk.encode(encoding="utf-8")])
                else:
                    token_ids.extend(self.encode_clean_text(chunk))
        else:
            token_ids = self.encode_clean_text(text)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        # 将token ID序列解码为文本
        token_list = [self.vocab[i] for i in ids]
        text = b''
        for token in token_list:
            text += token
        text = text.decode("utf-8", errors="replace")
        return text
    

def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)


if __name__ == "__main__":
    VOCAB_PATH = "assignment1/tests/fixtures/gpt2_vocab.json"
    MERGES_PATH = "assignment1/tests/fixtures/gpt2_merges.txt"
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    pattern = "(" + "|".join(map(re.escape, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])) + ")"
    chunks = re.split(pattern, test_string)
    chunks = [chunk for chunk in chunks if chunk != ""]
    print("Chunks:", chunks)
    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
