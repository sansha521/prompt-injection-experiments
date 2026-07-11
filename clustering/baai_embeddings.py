import argparse
import numpy as np

from sentence_transformers import SentenceTransformer

from load_malicious import load_malicious


model = SentenceTransformer("BAAI/bge-base-en-v1.5")
BATCH_SIZE=32


def embed_baai(sentences, model, batch_size=BATCH_SIZE):
    print("Embedding sentences...")
    baai_embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print(baai_embeddings.shape)
    return baai_embeddings

def main(split):

    print("Preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_malicious()
    # X_train = pd.concat([X_train, X_val], ignore_index=True)
    # y_train = pd.concat([y_train, y_val], ignore_index=True)
    splits = {"train": X_train, "val": X_val, "test": X_test}
    X = splits[split]
    print("Data ready.")

    # Sentences we want embeddings for
    print("Converting sentences to list...")
    sentences = X.tolist()

    print(f"Embedding {len(sentences)} sentence(s)...")
    sentence_embeddings = embed_baai(sentences, model, BATCH_SIZE)

    print("Sentence embeddings:")
    print(sentence_embeddings)
    print(sentence_embeddings.shape)

    output_file = f"{split}_baai_embeddings.npy"
    print(f"Saving embeddings to {output_file}")
    np.save(output_file, sentence_embeddings)
    print("Saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed a malicious prompt-dataset with BAAI"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which dataset split to embed (default: train).",
    )
    args = parser.parse_args()

    main(args.split)

### run : python baai_embeddings.py --split train/val/test