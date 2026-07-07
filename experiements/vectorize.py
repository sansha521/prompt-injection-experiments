# from sentence_transformers import SentenceTransformer

from load_malicious import load_malicious

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE=32


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_model(model_name=MODEL_NAME):
    # Load model from HuggingFace Hub
    print("Loading model from HF...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    print("Model loaded.")
    return tokenizer, model

def embed_sentences(sentences, tokenizer, model, batch_size):
    print("Embedding sentences...")
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            print(f"Embedding {i} to {i + batch_size}")
            batch = sentences[i:i + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            output = model(**encoded)

            embeddings = mean_pooling(output, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)
            print("Embeddings done.")

        print("Concatenating embeddings...")
        sentence_embeddings = torch.cat(all_embeddings, dim=0)
        print("Done.")

        return sentence_embeddings
        

print("Preparing data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_malicious()
# X_train = pd.concat([X_train, X_val], ignore_index=True)
# y_train = pd.concat([y_train, y_val], ignore_index=True)
print("Data ready.")

# Sentences we want sentence embeddings for
print("Converting sentences to list...")
sentences = X_train.tolist()
print(len(sentences))

tokenizer, model = load_model()

sentence_embeddings = embed_sentences(sentences, tokenizer, model, BATCH_SIZE)

print("Sentence embeddings:")
print(sentence_embeddings)
print(sentence_embeddings.shape)

print("Saving embeddings")
torch.save(sentence_embeddings, "train_embeddings.pt")
print("Saved successfully.")

sentence_embeddings = torch.load("train_embeddings.pt")
print(sentence_embeddings.shape)
