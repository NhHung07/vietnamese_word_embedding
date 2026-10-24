import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_vocab(vocab_path="data/vocab.pt"):
    """Nạp từ điển vocab.pt"""
    data = torch.load(vocab_path, map_location="cpu", weights_only=False)
    return data["word2idx"], data["idx2word"], data["word_freq"]


def get_unigram_distribution(word_freq, power=0.75):
    """Tính phân phối unigram mũ 0.75 cho Negative Sampling"""
    weights = np.array(word_freq, dtype=np.float32) ** power
    weights[0] = 0.0  # Không lấy mẫu negative là token <unk>
    weights = weights / np.sum(weights)
    return torch.tensor(weights, dtype=torch.float)


class SkipgramDataset(Dataset):
    """Dataset cho mô hình Skip-gram: (center_word, context_word)"""
    def __init__(self, corpus_path="data/corpus.txt", vocab_path="data/vocab.pt", window_size=4):
        self.word2idx, self.idx2word, self.word_freq = load_vocab(vocab_path)
        self.window_size = window_size
        self.pairs = []

        unk_idx = self.word2idx.get("<unk>", 0)

        # Đọc corpus và tạo các cặp (center, context)
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                indices = [self.word2idx.get(t, unk_idx) for t in tokens]
                
                n = len(indices)
                for i in range(n):
                    center = indices[i]
                    start = max(0, i - window_size)
                    end = min(n, i + window_size + 1)
                    for j in range(start, end):
                        if i != j:
                            context = indices[j]
                            self.pairs.append((center, context))

        print(f"[*] Tổng số cặp huấn luyện Skip-gram: {len(self.pairs):,}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class CBOWDataset(Dataset):
    """Dataset cho mô hình CBOW: (context_words, target_word)"""
    def __init__(self, corpus_path="data/corpus.txt", vocab_path="data/vocab.pt", window_size=4):
        self.word2idx, self.idx2word, self.word_freq = load_vocab(vocab_path)
        self.window_size = window_size
        self.samples = []

        unk_idx = self.word2idx.get("<unk>", 0)

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                indices = [self.word2idx.get(t, unk_idx) for t in tokens]
                
                n = len(indices)
                for i in range(n):
                    target = indices[i]
                    start = max(0, i - window_size)
                    end = min(n, i + window_size + 1)
                    contexts = [indices[j] for j in range(start, end) if i != j]
                    if contexts:
                        self.samples.append((contexts, target))

        print(f"[*] Tổng số mẫu huấn luyện CBOW: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_skipgram_collate_fn(unigram_dist, num_negatives=5):
    """Collate function sinh mẫu âm (Negative Sampling) theo batch cho Skip-gram"""
    def collate_fn(batch):
        centers = torch.tensor([item[0] for item in batch], dtype=torch.long)
        contexts = torch.tensor([item[1] for item in batch], dtype=torch.long)
        batch_size = len(batch)

        # Lấy mẫu ngẫu nhiên num_negatives từ cho mỗi mẫu trong batch
        neg_samples = torch.multinomial(
            unigram_dist,
            batch_size * num_negatives,
            replacement=True
        ).view(batch_size, num_negatives)

        return centers, contexts, neg_samples
    return collate_fn


def get_cbow_collate_fn(unigram_dist, num_negatives=5):
    """Collate function đệm ngữ cảnh và sinh mẫu âm cho CBOW"""
    def collate_fn(batch):
        batch_size = len(batch)
        max_context_len = max(len(item[0]) for item in batch)

        contexts_tensor = torch.zeros(batch_size, max_context_len, dtype=torch.long)
        masks_tensor = torch.zeros(batch_size, max_context_len, dtype=torch.float)
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)

        for i, (ctx, _) in enumerate(batch):
            contexts_tensor[i, :len(ctx)] = torch.tensor(ctx, dtype=torch.long)
            masks_tensor[i, :len(ctx)] = 1.0

        # Lấy mẫu âm cho từ đích (target)
        neg_samples = torch.multinomial(
            unigram_dist,
            batch_size * num_negatives,
            replacement=True
        ).view(batch_size, num_negatives)

        return contexts_tensor, masks_tensor, targets, neg_samples
    return collate_fn

