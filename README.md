# Transformer - Next-Token Prediction Using Basic Self-Attention

A small Transformer language model built with TensorFlow and Keras that predicts the next word in a sentence. The self-attention mechanism is written from scratch, without using a built-in attention layer.

## About the Project

A language model learns to predict the next token from the tokens that came before it. This project trains a mini Transformer on a short text corpus. At every position in a window of words, the model predicts the word that follows. After training, it can suggest the most likely next words and generate new text.

## Features

- Word-level tokenizer with a vocabulary built from the corpus
- Token embeddings and learned positional embeddings
- Single-head causal self-attention built from scratch (query, key, value, scaled dot-product, causal mask, softmax)
- Transformer blocks with feed-forward network, residual connections and Layer Normalization
- Training with EarlyStopping and ReduceLROnPlateau callbacks
- Evaluation with loss, perplexity and next-token accuracy
- Top-5 next-word predictions for any prompt
- Text generation with greedy decoding and temperature sampling
- Attention heatmap for every Transformer block
- Saves the trained weights and vocabulary

## How Self-Attention Works

1. Each word is turned into a query, a key and a value using three learned weight matrices.
2. The query of a word is compared with the keys of all words using a dot product, then scaled by the square root of the dimension.
3. A causal mask hides the future words, so a word can only look at itself and the words before it.
4. A softmax converts the scores into attention weights that add up to 1.
5. The weights are used to take a weighted sum of the values, which becomes the output for that word.

## Model Architecture

| Stage | Details |
|-------|---------|
| Input | Window of 8 word IDs |
| Embedding | Token embedding + positional embedding (64 dimensions) |
| Blocks | 2 Transformer blocks |
| Each block | Causal self-attention, residual + LayerNorm, feed-forward (128 units), residual + LayerNorm, Dropout 0.1 |
| Output | Dense layer over the vocabulary, one prediction per position |
| Loss | Sparse categorical cross-entropy |

Settings such as `SEQ_LEN`, `D_MODEL`, `NUM_BLOCKS` and `EPOCHS` are at the top of the script and can be changed easily.

## Requirements

- Python 3.9 or higher
- TensorFlow
- NumPy
- Matplotlib

Install everything with:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

```bash
python transformer_next_token.py
```

The script trains on a short built-in text about machine learning and Transformers. To use your own data, save any plain text file as `corpus.txt` in the same folder as the script. A larger corpus gives better and more varied results.

## Output

All results are saved in the `transformer_outputs/` folder:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss, next-token accuracy and perplexity over epochs |
| `attention_heatmap.png` | Attention weights for each Transformer block |
| `transformer.weights.h5` | Trained model weights |
| `vocab.json` | Vocabulary used by the model |

## Example Usage

After training, the script prints predictions like this:

```
Prompt: "the transformer uses self"
  1. attention       ...
```

and generates text from a starting prompt using greedy decoding and different temperatures.

## Results

Add your validation loss, perplexity and accuracy here after running the script.


