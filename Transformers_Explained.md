# Transformers: The Architecture Behind Modern AI

## What Is a Transformer?

A transformer is a neural network architecture introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al. It was originally designed for language translation but has since become the foundation of nearly all large language models (GPT, Claude, Gemini, LLaMA) and many vision and audio models as well. The key breakthrough is a mechanism called **self-attention**, which lets the model weigh the importance of every word relative to every other word in a sequence — all at once, rather than one word at a time.

## Why Transformers Replaced Earlier Models

Before transformers, recurrent neural networks (RNNs) and LSTMs processed text sequentially — word by word, left to right. This created two problems: training was slow because each step depended on the previous one, and the model struggled to remember information from far back in a long sequence. Transformers solve both problems. Because self-attention looks at all positions simultaneously, the entire sequence can be processed in parallel during training. And because attention scores directly connect distant words, a transformer can easily learn that "bank" in sentence position 3 relates to "river" in position 47.

## How It Works (The Core Ideas)

**Tokenization and Embedding.** Input text is split into tokens (roughly word-pieces). Each token is converted into a numeric vector — its embedding — that captures meaning in a high-dimensional space.

**Positional Encoding.** Since transformers process all tokens in parallel, they have no built-in sense of order. Positional encodings — patterns of sine and cosine waves at different frequencies — are added to embeddings so the model knows that "the" came before "cat."

**Self-Attention.** Each token generates three vectors: a Query (what am I looking for?), a Key (what do I contain?), and a Value (what information do I carry?). Attention scores are computed by taking the dot product of every Query with every Key, scaling, applying softmax to get weights, then multiplying by the Values. The result is a new representation of each token that blends in context from the entire sequence. Multi-head attention repeats this process several times in parallel with different learned projections, letting the model attend to different types of relationships simultaneously.

**Feed-Forward Network.** After attention, each token's representation passes through a small two-layer neural network (applied identically to every position). This adds non-linear transformation capacity — essentially letting the model "think" about what the attention step gathered.

**Layer Stacking, Residuals, and Normalization.** A transformer stacks many identical layers (GPT-3 uses 96). Each layer has a residual connection (adding the input back to the output) and layer normalization, which stabilize training and allow information to flow through many layers without degrading.

**The Encoder-Decoder Split.** The original transformer has two halves: an encoder (which reads the full input with bidirectional attention) and a decoder (which generates output left-to-right, masking future tokens). Many modern LLMs use only the decoder half, trained to predict the next token.

## The Attention Formula at a Glance

The math of self-attention fits in one line: **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**, where dₖ is the dimension of the key vectors. The division by √dₖ prevents dot products from growing too large, which would push softmax into regions with tiny gradients.

## Why It Matters

Transformers scale remarkably well. Doubling the data and parameters tends to predictably improve performance (known as "scaling laws"). This property, combined with parallelizable training, enabled the leap from models with millions of parameters to models with hundreds of billions — and with that leap came the emergent capabilities (reasoning, code generation, summarization) that define today's AI landscape.
