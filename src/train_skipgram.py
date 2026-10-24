import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SkipgramDataset, load_vocab, get_unigram_distribution, get_skipgram_collate_fn


class SkipgramNegSampling(nn.Module):
    """Mô hình Skip-gram với hàm mất mát Negative Sampling"""
    def __init__(self, vocab_size, embed_dim):
        super(SkipgramNegSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Ma trận embedding đầu vào (từ trung tâm) và đầu ra (từ ngữ cảnh)
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Khởi tạo trọng số ngẫu nhiên nhỏ
        init_range = 0.5 / embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, centers, contexts, neg_words):
        """
        centers: (B,)
        contexts: (B,)
        neg_words: (B, num_negatives)
        """
        # Vector của từ trung tâm: (B, embed_dim)
        v_center = self.in_embeddings(centers)
        
        # Vector của từ ngữ cảnh thực tế (mẫu dương): (B, embed_dim)
        u_context = self.out_embeddings(contexts)
        
        # Vector của các từ âm (mẫu âm): (B, num_negatives, embed_dim)
        u_neg = self.out_embeddings(neg_words)
        
        # Tích vô hướng cho mẫu dương: (B,)
        pos_dot = torch.sum(v_center * u_context, dim=1)
        pos_loss = -F.logsigmoid(pos_dot)
        
        # Tích vô hướng cho mẫu âm: (B, num_negatives)
        # u_neg * v_center: (B, num_negatives, embed_dim) x (B, embed_dim, 1) -> (B, num_negatives)
        neg_dot = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(F.logsigmoid(-neg_dot), dim=1)
        
        loss = torch.mean(pos_loss + neg_loss)
        return loss

    def get_embeddings(self):
        """Lấy ma trận embedding chuẩn hóa để suy luận / trực quan hóa"""
        return self.in_embeddings.weight.data.cpu().numpy()


def train_skipgram(
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
    print(f"[*] Huấn luyện Skip-gram trên thiết bị: {device}")

    # Nạp từ vựng và phân phối unigram
    word2idx, idx2word, word_freq = load_vocab(vocab_path)
    vocab_size = len(idx2word)
    unigram_dist = get_unigram_distribution(word_freq)

    # Chuẩn bị Dataset và DataLoader
    dataset = SkipgramDataset(corpus_path, vocab_path, window_size=window_size)
    collate_fn = get_skipgram_collate_fn(unigram_dist, num_negatives=num_negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Khởi tạo mô hình và bộ tối ưu hóa
    model = SkipgramNegSampling(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[*] Bắt đầu huấn luyện {epochs} epochs (vocab_size={vocab_size}, embed_dim={embed_dim})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for centers, contexts, neg_words in pbar:
            centers = centers.to(device)
            contexts = contexts.to(device)
            neg_words = neg_words.to(device)

            optimizer.zero_grad()
            loss = model(centers, contexts, neg_words)
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
    
    pt_path = os.path.join(model_dir, "skipgram.pt")
    torch.save(save_data, pt_path)
    print(f"[✓] Đã lưu mô hình Skip-gram vào: {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Word2Vec Skip-gram")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượt huấn luyện (epochs)")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều vector embedding")
    parser.add_argument("--window", type=int, default=4, help="Kích thước cửa sổ ngữ cảnh")
    parser.add_argument("--batch_size", type=int, default=512, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.003, help="Tốc độ học")
    args = parser.parse_args()

    train_skipgram(
        embed_dim=args.dim,
        window_size=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

