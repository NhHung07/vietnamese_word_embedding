# Mô hình Word Embedding tiếng Việt: Skip-gram & CBOW

Dự án triển khai mô hình Word2Vec (Skip-gram và CBOW) từ đầu (from-scratch) bằng PyTorch cho tiếng Việt, huấn luyện trên tập dữ liệu `undertheseanlp/UVW-2026` (Wikipedia tiếng Việt) và trực quan hóa 10 cặp từ mục tiêu trong không gian 2D và 3D.

---

## 1. Cấu trúc thư mục

```
vietnamese_word_embedding/
├── data/                       # Dữ liệu sau khi tiền xử lý (corpus.txt, vocab.pt)
├── models/                     # Checkpoint mô hình đã huấn luyện (skipgram.pt, cbow.pt)
├── results/                    # Kết quả đánh giá
│   └── similarity.csv          # Bảng độ tương đồng cosine của 10 cặp từ
├── src/
│   ├── preprocess.py           # Tiền xử lý, lọc mẫu UVW-2026 và tách từ tiếng Việt
│   ├── dataset.py              # PyTorch Dataset, DataLoader và Negative Sampling
│   ├── train_skipgram.py       # Huấn luyện mô hình Skip-gram (Negative Sampling)
│   ├── train_cbow.py           # Huấn luyện mô hình CBOW (Negative Sampling)
│   └── evaluate.py             # Đánh giá độ tương đồng cosine và tìm từ lân cận
├── main.py                     # Script thực thi toàn bộ quy trình huấn luyện & đánh giá
├── visualize.ipynb             # Jupyter Notebook trực quan hóa 2D và 3D bằng PCA
└── requirements.txt            # Danh sách các thư viện cần thiết
```

---

## 2. 10 Cặp từ đánh giá và trực quan hóa

1. `hà_nội` - `việt_nam` *(Thủ đô - Quốc gia)*
2. `tokyo` - `nhật_bản` *(Thủ đô - Quốc gia)*
3. `paris` - `pháp` *(Thủ đô - Quốc gia)*
4. `vua` - `hoàng_hậu` *(Hoàng gia / Giới tính)*
5. `đàn_ông` - `phụ_nữ` *(Giới tính)*
6. `cha` - `mẹ` *(Gia đình)*
7. `học_sinh` - `giáo_viên` *(Giáo dục)*
8. `bác_sĩ` - `bệnh_viện` *(Y tế)*
9. `thành_phố` - `nông_thôn` *(Địa lý)*
10. `thành_công` - `thất_bại` *(Trái nghĩa)*

---

## 3. Hướng dẫn chạy

### Chạy quy trình huấn luyện và đánh giá:
```bash
python main.py
```

Tuỳ chỉnh các tham số nếu cần:
```bash
python main.py --sentences 80000 --epochs 10 --dim 100 --window 4 --batch_size 512
```

### Trực quan hóa kết quả:
Mở file `visualize.ipynb` để trực quan hóa không gian véc-tơ 2D và 3D (bằng PCA) cho 10 cặp từ.
