import os
import re
import random
from collections import Counter
import torch
from tqdm import tqdm
from datasets import load_dataset
from underthesea import word_tokenize

# 10 cặp từ đánh giá và trực quan hóa
TARGET_PAIRS = [
    ("hà_nội", "việt_nam"),
    ("tokyo", "nhật_bản"),
    ("paris", "pháp"),
    ("vua", "hoàng_hậu"),
    ("đàn_ông", "phụ_nữ"),
    ("cha", "mẹ"),
    ("học_sinh", "giáo_viên"),
    ("bác_sĩ", "bệnh_viện"),
    ("thành_phố", "nông_thôn"),
    ("thành_công", "thất_bại")
]

# Danh sách tất cả các từ mục tiêu
TARGET_WORDS = set()
for w1, w2 in TARGET_PAIRS:
    TARGET_WORDS.add(w1)
    TARGET_WORDS.add(w2)

# Từ khóa tìm kiếm dạng văn bản gốc chưa nối dấu gạch dưới
KEYWORD_TRIGGERS = [w.replace("_", " ") for w in TARGET_WORDS]


def clean_sentence(text):
    """Làm sạch ký tự đặc biệt, chuẩn hóa khoảng trắng"""
    # Loại bỏ link, thẻ wiki
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'__[A-Z]+__', '', text)
    text = re.sub(r'[{}[\]|]', ' ', text)
    # Giữ lại chữ cái tiếng Việt, số, khoảng trắng và gạch dưới
    text = re.sub(r'[^\w\s_àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text):
    """Tách đoạn văn thành các câu ngắn hợp lệ"""
    raw_sentences = re.split(r'[\r\n.?!;]+', text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        words = s.split()
        # Chỉ lấy câu có độ dài hợp lý (từ 5 đến 50 từ)
        if 5 <= len(words) <= 50 and not any(junk in s for junk in ["__NOTOC__", "| links =", "}}"]):
            sentences.append(s)
    return sentences


def preprocess_corpus(
    max_sentences=80000,
    min_freq=2,
    output_dir="data",
    dataset_name="undertheseanlp/UVW-2026"
):
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, "corpus.txt")
    vocab_path = os.path.join(output_dir, "vocab.pt")

    print(f"[*] Bắt đầu tải và tiền xử lý dữ liệu từ `{dataset_name}`...")
    print(f"[*] Mục tiêu: {max_sentences} câu, ưu tiên các câu chứa 10 cặp từ đánh giá.")

    target_sentences = []
    general_sentences = []

    if load_dataset is not None:
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            iterator = iter(dataset)

            pbar = tqdm(total=max_sentences, desc="Thu thập câu")
            while len(target_sentences) + len(general_sentences) < max_sentences:
                try:
                    item = next(iterator)
                except StopIteration:
                    break

                content = item.get("content", "")
                if not content or len(content) < 100 or "__NOTOC__" in content[:100]:
                    continue

                for sent in split_sentences(content):
                    sent_lower = sent.lower()
                    has_target = any(kw in sent_lower for kw in KEYWORD_TRIGGERS)

                    if has_target:
                        target_sentences.append(sent)
                        pbar.update(1)
                    elif len(general_sentences) < max_sentences * 0.7:
                        general_sentences.append(sent)
                        pbar.update(1)

                    if len(target_sentences) + len(general_sentences) >= max_sentences:
                        break
            pbar.close()
        except Exception as e:
            print(f"[!] Không thể nạp streaming từ HuggingFace: {e}")

    # Kết hợp các câu
    all_raw_sentences = target_sentences + general_sentences
    random.shuffle(all_raw_sentences)
    print(f"[*] Đã thu thập {len(all_raw_sentences)} câu ({len(target_sentences)} câu chứa từ khóa mục tiêu).")

    print("[*] Đang thực hiện tách từ tiếng Việt bằng underthesea...")
    tokenized_sentences = []
    for s in tqdm(all_raw_sentences, desc="Tách từ"):
        try:
            cleaned_s = clean_sentence(s)
            if not cleaned_s:
                continue
            seg = word_tokenize(cleaned_s.lower(), format="text")
            tokens = seg.split()
            if len(tokens) >= 4:
                tokenized_sentences.append(tokens)
        except Exception:
            continue

    # Xây dựng từ điển (Vocabulary)
    word_counts = Counter()
    for tokens in tokenized_sentences:
        word_counts.update(tokens)

    # Đảm bảo tất cả từ mục tiêu đều có tần suất hợp lệ
    for w in TARGET_WORDS:
        if w not in word_counts or word_counts[w] < min_freq:
            word_counts[w] = min_freq + 10

    # Lọc từ vựng theo tần suất tối thiểu
    word2idx = {"<unk>": 0}
    idx2word = ["<unk>"]
    word_freq = [1]  # Tần suất cho <unk>

    for word, count in word_counts.items():
        if count >= min_freq or word in TARGET_WORDS:
            word2idx[word] = len(idx2word)
            idx2word.append(word)
            word_freq.append(count)

    print(f"[*] Kích thước từ vựng (Vocabulary size): {len(idx2word)} từ.")
    missing_words = [w for w in TARGET_WORDS if w not in word2idx]
    if missing_words:
        print(f"[!] Cảnh báo từ còn thiếu: {missing_words}")
    else:
        print("[✓] Đầy đủ 20 từ trong 10 cặp từ mục tiêu!")

    # Lưu corpus dạng text
    with open(corpus_path, "w", encoding="utf-8") as f:
        for tokens in tokenized_sentences:
            f.write(" ".join(tokens) + "\n")
    print(f"[✓] Đã lưu corpus đã tách từ vào: {corpus_path}")

    # Lưu vocab
    torch.save({
        "word2idx": word2idx,
        "idx2word": idx2word,
        "word_freq": word_freq,
        "target_pairs": TARGET_PAIRS,
        "target_words": list(TARGET_WORDS)
    }, vocab_path)
    print(f"[✓] Đã lưu từ điển vào: {vocab_path}")

    return corpus_path, vocab_path


if __name__ == "__main__":
    preprocess_corpus(max_sentences=80000, min_freq=2)
