import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CBOWDataset, load_vocab, get_unigram_distribution, get_cbow_collate_fn


class CBOWNegSampling(nn.Module):
    """Mô hình CBOW (Continuous Bag of Words) với hàm mất mát Negative Sampling"""
    def __init__(self, vocab_size, embed_dim):
        super(CBOWNegSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Ma trận embedding đầu vào (ngữ cảnh) và đầu ra (từ đích)
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Khởi tạo trọng số ngẫu nhiên nhỏ
        init_range = 0.5 / embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, contexts, masks, targets, neg_words):
        """
        contexts: (B, max_len)
        masks: (B, max_len) - 1.0 cho từ hợp lệ, 0.0 cho padding
        targets: (B,)
        neg_words: (B, num_negatives)
        """
        # Vector nhúng của ngữ cảnh: (B, max_len, embed_dim)
        ctx_embed = self.in_embeddings(contexts)
        masked_ctx = ctx_embed * masks.unsqueeze(2)
        
        # Tính trung bình cộng vector của các từ ngữ cảnh
        ctx_lens = torch.clamp(masks.sum(dim=1, keepdim=True), min=1.0)
        v_ctx_mean = masked_ctx.sum(dim=1) / ctx_lens  # (B, embed_dim)
        
        # Vector của từ đích thực tế (mẫu dương): (B, embed_dim)
        u_target = self.out_embeddings(targets)
        
        # Vector của các từ âm (mẫu âm): (B, num_negatives, embed_dim)
        u_neg = self.out_embeddings(neg_words)
        
        # Tích vô hướng cho mẫu dương: (B,)
        pos_dot = torch.sum(v_ctx_mean * u_target, dim=1)
        pos_loss = -F.logsigmoid(pos_dot)
        
        # Tích vô hướng cho mẫu âm: (B, num_negatives)
        neg_dot = torch.bmm(u_neg, v_ctx_mean.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(F.logsigmoid(-neg_dot), dim=1)
        
        loss = torch.mean(pos_loss + neg_loss)
        return loss

    def get_embeddings(self):
        """Lấy ma trận embedding chuẩn hóa để suy luận / trực quan hóa"""
        return self.in_embeddings.weight.data.cpu().numpy()


def train_cbow(
    corpus_path="data/corpus.txt",
    vocab_path="data/vocab.pt",
    model_dir="models",
    embed_dim=100,
    window_size=4,
    num_negatives=5,
    batch_size=512,
    epochs=10,
    lr=0.003
):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Huấn luyện CBOW trên thiết bị: {device}")

    # Nạp từ vựng và phân phối unigram
    word2idx, idx2word, word_freq = load_vocab(vocab_path)
    vocab_size = len(idx2word)
    unigram_dist = get_unigram_distribution(word_freq)

    # Chuẩn bị Dataset và DataLoader
    dataset = CBOWDataset(corpus_path, vocab_path, window_size=window_size)
    collate_fn = get_cbow_collate_fn(unigram_dist, num_negatives=num_negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Khởi tạo mô hình và bộ tối ưu hóa
    model = CBOWNegSampling(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[*] Bắt đầu huấn luyện {epochs} epochs (vocab_size={vocab_size}, embed_dim={embed_dim})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for contexts, masks, targets, neg_words in pbar:
            contexts = contexts.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            neg_words = neg_words.to(device)

            optimizer.zero_grad()
            loss = model(contexts, masks, targets, neg_words)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"[*] Epoch {epoch} hoàn thành - Average Loss: {avg_loss:.4f}")

    # Xuất và lưu ma trận embedding
    embeddings = model.get_embeddings()
    save_data = {
        "embeddings": embeddings,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "embed_dim": embed_dim
    }
    
    pt_path = os.path.join(model_dir, "cbow.pt")
    torch.save(save_data, pt_path)
    print(f"[✓] Đã lưu mô hình CBOW vào: {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Word2Vec CBOW")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượt huấn luyện (epochs)")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều vector embedding")
    parser.add_argument("--window", type=int, default=4, help="Kích thước cửa sổ ngữ cảnh")
    parser.add_argument("--batch_size", type=int, default=512, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.003, help="Tốc độ học")
    args = parser.parse_args()

    train_cbow(
        embed_dim=args.dim,
        window_size=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

