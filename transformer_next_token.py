import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
SEQ_LEN = 8
D_MODEL = 64
FF_DIM = 128
NUM_BLOCKS = 2
DROPOUT = 0.1
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
CORPUS_FILE = "corpus.txt"
OUT_DIR = "transformer_outputs"

np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_TEXT = """
Machine learning is a branch of artificial intelligence that learns patterns from data.
A neural network is made of layers of simple units called neurons.
Each neuron takes some inputs, multiplies them by weights, and passes the result through an activation function.
Deep learning uses many layers to learn complex features from images, text and sound.
A convolutional neural network is very good at understanding images.
A recurrent neural network reads a sentence one word at a time and keeps a memory of what it has seen.
The transformer is a newer model that reads all the words at the same time.
The transformer uses self attention to decide which words are important for each word.
Self attention compares every word with every other word in the sentence.
Each word creates a query, a key and a value.
The query of one word is compared with the keys of all the other words to produce scores.
The scores are scaled and passed through a softmax function to become attention weights.
The attention weights are used to take a weighted sum of the values.
In a language model the next word is predicted using only the words that came before it.
A causal mask hides the future words so the model cannot cheat while it is learning.
Positional embeddings tell the model the order of the words in the sentence.
A transformer block has self attention, a feed forward network, residual connections and layer normalization.
Many blocks are stacked on top of each other to build a deeper model.
The model is trained by comparing its predicted word with the real next word.
The loss is the cross entropy between the predicted probabilities and the correct word.
A lower loss means the model is predicting the next word more accurately.
Perplexity is the exponent of the loss and tells how surprised the model is by the text.
After training, the model can generate new text by predicting one word and adding it to the input again and again.
A higher temperature makes the generated text more random and a lower temperature makes it more careful.
Large language models are transformers trained on huge amounts of text from the internet.
They learn grammar, facts and reasoning patterns only by predicting the next word.
Students can build a small transformer to understand how these large models work.
Machine learning is a branch of artificial intelligence that learns patterns from data.
The transformer uses self attention to decide which words are important for each word.
A causal mask hides the future words so the model cannot cheat while it is learning.
"""

if os.path.exists(CORPUS_FILE):
    with open(CORPUS_FILE, encoding="utf-8") as f:
        text = f.read()
    print(f"Using your own corpus from {CORPUS_FILE}")
else:
    text = SAMPLE_TEXT
    print("Using the built-in sample corpus")

TOKEN_PATTERN = r"[a-z]+(?:[-'][a-z]+)*|[.,]"
words = re.findall(TOKEN_PATTERN, text.lower())
vocab = ["<pad>", "<unk>"] + sorted(set(words))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}
VOCAB_SIZE = len(vocab)
print(f"Total tokens: {len(words)} | Vocabulary size: {VOCAB_SIZE}")

with open(f"{OUT_DIR}/vocab.json", "w") as f:
    json.dump(vocab, f)

def encode(text_in):
    return [word_to_id.get(w, 1) for w in re.findall(TOKEN_PATTERN, text_in.lower())]

def decode(ids):
    out = " ".join(id_to_word[i] for i in ids if i != 0)
    return re.sub(r"\s+([.,])", r"\1", out)

ids = np.array([word_to_id[w] for w in words], dtype="int32")
n_windows = len(ids) - SEQ_LEN
x_all = np.stack([ids[i:i + SEQ_LEN] for i in range(n_windows)])
y_all = np.stack([ids[i + 1:i + SEQ_LEN + 1] for i in range(n_windows)])

perm = np.random.permutation(n_windows)
x_all, y_all = x_all[perm], y_all[perm]
split = int(0.9 * n_windows)
x_train, y_train = x_all[:split], y_all[:split]
x_val, y_val = x_all[split:], y_all[split:]
print(f"Train windows: {len(x_train)} | Validation windows: {len(x_val)}")
print("Example input :", decode(x_train[0]))
print("Example target:", decode(y_train[0]))

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(seq_len, d_model)

    def call(self, x):
        positions = tf.range(tf.shape(x)[1])
        return self.token_emb(x) + self.pos_emb(positions)

class CausalSelfAttention(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.wq = layers.Dense(d_model, use_bias=False)
        self.wk = layers.Dense(d_model, use_bias=False)
        self.wv = layers.Dense(d_model, use_bias=False)
        self.wo = layers.Dense(d_model, use_bias=False)

    def attention_weights(self, x):
        q = self.wq(x)
        k = self.wk(x)
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        length = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((length, length)), -1, 0)
        scores = tf.where(mask > 0, scores, -1e9)
        return tf.nn.softmax(scores, axis=-1)

    def call(self, x):
        weights = self.attention_weights(x)
        v = self.wv(x)
        return self.wo(tf.matmul(weights, v))

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.attn = CausalSelfAttention(d_model)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.drop1(self.attn(x), training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + ffn_out)

def build_model():
    inputs = keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = TokenAndPositionEmbedding(SEQ_LEN, VOCAB_SIZE, D_MODEL, name="embedding")(inputs)
    for i in range(NUM_BLOCKS):
        x = TransformerBlock(D_MODEL, FF_DIM, DROPOUT, name=f"block_{i + 1}")(x)
    logits = layers.Dense(VOCAB_SIZE, name="lm_head")(x)
    return keras.Model(inputs, logits, name="mini_transformer")

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=8, min_lr=1e-5, verbose=1),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()

ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Next-token accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()

ax[2].plot(np.exp(history.history["loss"]), label="train")
ax[2].plot(np.exp(history.history["val_loss"]), label="validation")
ax[2].set_title("Perplexity"); ax[2].set_xlabel("Epoch"); ax[2].set_yscale("log")
ax[2].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\nValidation loss      : {val_loss:.4f}")
print(f"Validation perplexity: {np.exp(val_loss):.2f}")
print(f"Validation accuracy  : {val_acc:.4f}")

def pad_context(token_ids):
    ctx = token_ids[-SEQ_LEN:]
    return [0] * (SEQ_LEN - len(ctx)) + ctx

def next_token_probs(token_ids):
    ctx = np.array([pad_context(token_ids)], dtype="int32")
    logits = model(ctx, training=False)[0, -1].numpy()
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()

def predict_next(prompt, k=5):
    probs = next_token_probs(encode(prompt))
    top = np.argsort(probs)[::-1][:k]
    print(f'\nPrompt: "{prompt}"')
    for rank, t in enumerate(top, 1):
        print(f"  {rank}. {id_to_word[t]:<15s} {probs[t]:.1%}")

def generate(prompt, n_words=20, temperature=0.8, top_k=10):
    token_ids = encode(prompt)
    for _ in range(n_words):
        probs = next_token_probs(token_ids)
        probs[:2] = 0.0
        if temperature <= 0:
            next_id = int(np.argmax(probs))
        else:
            logits = np.log(probs + 1e-12) / temperature
            if top_k:
                cutoff = np.sort(logits)[-top_k]
                logits[logits < cutoff] = -1e9
            p = np.exp(logits - logits.max())
            p /= p.sum()
            next_id = int(np.random.choice(len(p), p=p))
        token_ids.append(next_id)
    return decode(token_ids)

print("\n" + "=" * 60)
print("TOP-5 NEXT-WORD PREDICTIONS")
print("=" * 60)
predict_next("the transformer uses self")
predict_next("a causal mask hides the future")
predict_next("each word creates a query , a key and a")

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)
for temp in (0.0, 0.7, 1.2):
    label = "greedy" if temp == 0 else f"temperature {temp}"
    print(f"\n[{label}]")
    print(generate("the transformer", n_words=25, temperature=temp))

sample_ids = x_val[0].tolist()
sample_input = tf.constant([sample_ids], dtype="int32")
embedded = model.get_layer("embedding")(sample_input)
labels = [id_to_word[i] for i in sample_ids]

n_blocks = NUM_BLOCKS
fig, axes = plt.subplots(1, n_blocks, figsize=(6.5 * n_blocks, 5.5))
axes = np.atleast_1d(axes)
x_state = embedded
for i in range(n_blocks):
    block = model.get_layer(f"block_{i + 1}")
    weights = block.attn.attention_weights(x_state)[0].numpy()
    im = axes[i].imshow(weights, cmap="viridis", vmin=0, vmax=1)
    axes[i].set_xticks(range(SEQ_LEN)); axes[i].set_xticklabels(labels, rotation=45, ha="right")
    axes[i].set_yticks(range(SEQ_LEN)); axes[i].set_yticklabels(labels)
    axes[i].set_xlabel("Key (words being looked at)")
    axes[i].set_ylabel("Query (current word)")
    axes[i].set_title(f"Block {i + 1} attention weights")
    plt.colorbar(im, ax=axes[i], fraction=0.046)
    x_state = block(x_state, training=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/attention_heatmap.png", dpi=120)
plt.close()

model.save_weights(f"{OUT_DIR}/transformer.weights.h5")
print(f"\nAll outputs saved in: {OUT_DIR}/")
