import os
import numpy as np
import pandas as pd
import torch

WORD_PAIRS = [
    ("hà_nội", "việt_nam", "Thủ đô - Quốc gia"),
    ("tokyo", "nhật_bản", "Thủ đô - Quốc gia"),
    ("paris", "pháp", "Thủ đô - Quốc gia"),
    ("vua", "hoàng_hậu", "Hoàng gia / Giới tính"),
    ("đàn_ông", "phụ_nữ", "Giới tính"),
    ("cha", "mẹ", "Gia đình"),
    ("học_sinh", "giáo_viên", "Giáo dục"),
    ("bác_sĩ", "bệnh_viện", "Y tế"),
    ("thành_phố", "nông_thôn", "Địa lý"),
    ("thành_công", "thất_bại", "Trái nghĩa")
]


def load_model_embeddings(model_path):
    """Nạp file checkpoint mô hình (.pt)"""
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    return embeddings, data["word2idx"], data["idx2word"]


def cosine_similarity(v1, v2):
    """Tính độ tương đồng cosine giữa 2 vector"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_nearest_neighbors(word, embeddings, word2idx, idx2word, top_k=5):
    """Tìm top_k từ có độ tương đồng cosine cao nhất với word"""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]
    
    # Chuẩn hóa vector ma trận
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_embeds = embeddings / norms
    
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    sims = np.dot(norm_embeds, vec_norm)
    
    # Sắp xếp giảm dần
    top_indices = np.argsort(sims)[::-1]
    
    neighbors = []
    for i in top_indices:
        w = idx2word[i]
        if w != word and w != "<unk>":
            neighbors.append((w, float(sims[i])))
            if len(neighbors) == top_k:
                break
    return neighbors


def evaluate_models(
    skipgram_path="models/skipgram.pt",
    cbow_path="models/cbow.pt",
    output_csv="results/similarity.csv"
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print("[*] Nạp mô hình Skip-gram và CBOW...")
    sg_embeds, sg_w2i, sg_i2w = load_model_embeddings(skipgram_path)
    cbow_embeds, cbow_w2i, cbow_i2w = load_model_embeddings(cbow_path)

    results = []
    print("\n" + "=" * 80)
    print(f"{'STT':<4} | {'Từ 1':<12} | {'Từ 2':<12} | {'Skip-gram Cosine':<18} | {'CBOW Cosine':<14} | {'Nhóm quan hệ'}")
    print("-" * 80)

    for idx, (w1, w2, cat) in enumerate(WORD_PAIRS, 1):
        # Tính cho Skip-gram
        if w1 in sg_w2i and w2 in sg_w2i:
            sg_sim = cosine_similarity(sg_embeds[sg_w2i[w1]], sg_embeds[sg_w2i[w2]])
        else:
            sg_sim = np.nan

        # Tính cho CBOW
        if w1 in cbow_w2i and w2 in cbow_w2i:
            cbow_sim = cosine_similarity(cbow_embeds[cbow_w2i[w1]], cbow_embeds[cbow_w2i[w2]])
        else:
            cbow_sim = np.nan

        results.append({
            "STT": idx,
            "Tu_1": w1,
            "Tu_2": w2,
            "Nhom_Quan_He": cat,
            "Skipgram_Cosine_Similarity": round(sg_sim, 4),
            "CBOW_Cosine_Similarity": round(cbow_sim, 4)
        })

        sg_str = f"{sg_sim:.4f}" if not np.isnan(sg_sim) else "N/A"
        cbow_str = f"{cbow_sim:.4f}" if not np.isnan(cbow_sim) else "N/A"
        print(f"{idx:<4} | {w1:<12} | {w2:<12} | {sg_str:<18} | {cbow_str:<14} | {cat}")

    print("=" * 80)

    # Lưu kết quả CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Đã lưu bảng độ tương đồng vào: {output_csv}")

    # Hiển thị Top từ lân cận cho một số từ tiêu biểu
    sample_words = ["hà_nội", "vua", "bác_sĩ", "thành_phố"]
    print("\n[*] Top 3 từ lân cận (Nearest Neighbors) của các từ tiêu biểu:")
    for w in sample_words:
        print(f"\n-> Từ: '{w}'")
        sg_nbrs = get_nearest_neighbors(w, sg_embeds, sg_w2i, sg_i2w, top_k=3)
        cbow_nbrs = get_nearest_neighbors(w, cbow_embeds, cbow_w2i, cbow_i2w, top_k=3)
        sg_text = ", ".join([f"{n[0]} ({n[1]:.3f})" for n in sg_nbrs]) if sg_nbrs else "N/A"
        cbow_text = ", ".join([f"{n[0]} ({n[1]:.3f})" for n in cbow_nbrs]) if cbow_nbrs else "N/A"
        print(f"   [Skip-gram] : {sg_text}")
        print(f"   [CBOW]      : {cbow_text}")

    return df


if __name__ == "__main__":
    evaluate_models()

