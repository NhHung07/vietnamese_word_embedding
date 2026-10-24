import os
import sys
import argparse
import time

# Thêm thư mục src vào đường dẫn tìm kiếm module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import preprocess_corpus
from train_skipgram import train_skipgram
from train_cbow import train_cbow
from evaluate import evaluate_models


def main():
    parser = argparse.ArgumentParser(description="Pipeline huấn luyện và đánh giá Word Embedding tiếng Việt (Skip-gram & CBOW)")
    parser.add_argument("--sentences", type=int, default=80000, help="Số câu trích xuất từ undertheseanlp/UVW-2026 (mặc định: 80000)")
    parser.add_argument("--min_freq", type=int, default=2, help="Tần suất xuất hiện tối thiểu của từ trong từ điển")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều vector embedding (mặc định: 100)")
    parser.add_argument("--window", type=int, default=4, help="Kích thước cửa sổ ngữ cảnh (mặc định: 4)")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch huấn luyện mỗi mô hình (mặc định: 10)")
    parser.add_argument("--batch_size", type=int, default=512, help="Kích thước batch (mặc định: 512)")
    parser.add_argument("--lr", type=float, default=0.003, help="Tốc độ học (learning rate)")
    parser.add_argument("--skip_preprocess", action="store_true", help="Bỏ qua bước tiền xử lý nếu đã có data/corpus.txt")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print("      WORD EMBEDDING TIẾNG VIỆT VỚI PYTORCH: SKIP-GRAM & CBOW      ")
    print("=" * 80)

    # 1. TIỀN XỬ LÝ DỮ LIỆU
    print("\n>>> BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU TỪ `undertheseanlp/UVW-2026`")
    corpus_file = "data/corpus.txt"
    vocab_file = "data/vocab.pt"
    if args.skip_preprocess and os.path.exists(corpus_file) and os.path.exists(vocab_file):
        print(f"[*] Đã tìm thấy {corpus_file} và {vocab_file}, bỏ qua bước tiền xử lý.")
    else:
        preprocess_corpus(
            max_sentences=args.sentences,
            min_freq=args.min_freq,
            output_dir="data"
        )

    # 2. HUẤN LUYỆN SKIP-GRAM
    print("\n>>> BƯỚC 2: HUẤN LUYỆN MÔ HÌNH SKIP-GRAM (NEGATIVE SAMPLING)")
    train_skipgram(
        corpus_path=corpus_file,
        vocab_path=vocab_file,
        model_dir="models",
        embed_dim=args.dim,
        window_size=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

    # 3. HUẤN LUYỆN CBOW
    print("\n>>> BƯỚC 3: HUẤN LUYỆN MÔ HÌNH CBOW (NEGATIVE SAMPLING)")
    train_cbow(
        corpus_path=corpus_file,
        vocab_path=vocab_file,
        model_dir="models",
        embed_dim=args.dim,
        window_size=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

    # 4. ĐÁNH GIÁ 10 CẶP TỪ
    print("\n>>> BƯỚC 4: ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG COSINE CỦA 10 CẶP TỪ")
    evaluate_models(
        skipgram_path="models/skipgram.pt",
        cbow_path="models/cbow.pt",
        output_csv="results/similarity.csv"
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT HUẤN LUYỆN & ĐÁNH GIÁ TRONG: {elapsed:.2f} giây ({elapsed / 60:.2f} phút)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
