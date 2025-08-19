import multiprocessing
import os
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    local_counts = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    rm_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk)
    for rm_chunk in rm_chunks:
        for word_match in re.finditer(PAT, rm_chunk):
            word = word_match.group(0)
            bytes_seq = tuple(bytes([i]) for i in word.encode(encoding="utf-8"))
            if bytes_seq in local_counts:
                local_counts[bytes_seq] += 1
            else:
                local_counts[bytes_seq] = 1
    return local_counts

def count_byte_pairs(word_counter: dict, update_list: list = None, pair_counter: dict[tuple[bytes, bytes], int] = None) -> dict[tuple[bytes, bytes], int]:
    if pair_counter is None:
        pair_counter = {}
        for bytes_seq, count in word_counter.items():
            # 遍历每个预分词的相邻字节对
            for i in range(len(bytes_seq) - 1):
                pair = (bytes_seq[i], bytes_seq[i + 1])
                if pair not in pair_counter:
                    pair_counter[pair] = count
                else:
                    pair_counter[pair] += count
    else:
        for bytes_seq, new_bytes_seq, count in update_list:
            for i in range(len(bytes_seq) - 1):
                pair = (bytes_seq[i], bytes_seq[i + 1])
                if pair_counter[pair] == count:
                    del pair_counter[pair]
                else:
                    pair_counter[pair] -= count
            for i in range(len(new_bytes_seq) - 1):
                pair = (new_bytes_seq[i], new_bytes_seq[i + 1])
                if pair not in pair_counter:
                    pair_counter[pair] = count
                else:
                    pair_counter[pair] += count
    return pair_counter

def get_max_pair(pairs_counter: dict) -> tuple[bytes, bytes]:
    max_freq = max(pairs_counter.values())
    candidates = [pair for pair, cnt in pairs_counter.items() if cnt == max_freq]
    return max(candidates)

def merge_pair(word_counter: dict, pair: tuple[bytes, bytes]) -> dict:
    p1, p2 = pair
    new_token = p1 + p2
    update_list = []
    for bytes_seq, count in word_counter.items():
        update_index = [i for i in range(len(bytes_seq) - 1) if (bytes_seq[i], bytes_seq[i + 1]) == pair]
        if update_index != []:
            new_bytes_seq = []
            i = 0
            while i < len(bytes_seq):
                if i in update_index:
                    new_bytes_seq.append(new_token)
                    i += 2
                else:
                    new_bytes_seq.append(bytes_seq[i])
                    i += 1
            update_list.append((bytes_seq, new_bytes_seq, count))
    for bytes_seq, new_bytes_seq, count in update_list:
        word_counter[tuple(new_bytes_seq)] = count
        del word_counter[bytes_seq]
    return word_counter, update_list

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if os.name == "posix":  # 仅Unix系统需要处理
        multiprocessing.set_start_method("spawn", force=True)
    
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        b_token = token.encode(encoding="utf-8")
        if b_token not in vocab.values():
            vocab[len(vocab)] = b_token
    merges = [] 
    word_counter = {} # 用于计数words

    # Pretokenization
    with open(input_path, "rb") as f: # 对输入数据分块
        chunks_boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
    chunks = [(chunks_boundaries[i], chunks_boundaries[i + 1]) for i in range(len(chunks_boundaries) - 1)]
 
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, input_path, start, end, special_tokens) 
                  for start, end in chunks] 
        for future in futures:
            # 输出是局部words的计数
            chunk_counts = future.result()
            # 合并分块结果到全局字典
            for bytes_seq, count in chunk_counts.items():
                if bytes_seq in word_counter:
                    word_counter[bytes_seq] += count
                else:
                    word_counter[bytes_seq] = count

    num_iter = vocab_size - len(vocab)
    for i in range(num_iter):
        # 根据word_counter计数所有 pair
        if i == 0:
            pairs_counter = count_byte_pairs(word_counter)
        else:
            pairs_counter = count_byte_pairs(word_counter, update_list, pairs_counter)
        if not pairs_counter:
            break  # 没有可以合并的 pair 了

        # 找最多的 pair
        max_pair = get_max_pair(pairs_counter)
        # 执行合并，更新 word_counter
        word_counter, update_list = merge_pair(word_counter, max_pair)

        # 把合并后的 token 加入 vocab 和 merges 
        merged_token = b"".join(max_pair)
        vocab[len(vocab)] = merged_token
        merges.append(max_pair)

    return vocab, merges 

def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    import json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(i): token.decode("utf-8", errors="ignore") for i, token in vocab.items()}, f, ensure_ascii=False, indent=2)
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('utf-8', errors='ignore')} {merge[1].decode('utf-8', errors='ignore')}\n")


if __name__ == "__main__":
    input_path = "assignment1/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab_and_merges(vocab, merges, "myoutput/TinyStoriesVocab.json", "myoutput/TinyStoriesMerges.txt")
    key, value = max(vocab.items(), key=lambda kv: len(kv[1]))
    print("Key:", key)
    print("Value:", value)
    print("Value_decode:", value.decode("utf-8", errors="ignore"))
    print("Length:", len(value))
    
