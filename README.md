# Creating-a-Large-Language-Model-from-Scratch


# GPT Architecture Overview

This repository contains an implementation of a GPT architecture with a focus on understanding the components and flow of a Transformer-based model. The core concept revolves around attention mechanisms, particularly the scaled dot-product attention, which is essential to GPT's functionality.

---

## Key Components

### 1. **Input Processing**
   - **Tokenized Inputs**: The raw input text is tokenized into numerical representations.
   - **Embedding + Positional Encodings**: Tokens are embedded into dense vectors, with positional encodings added to retain sequence information.

---

### 2. **Decoder Blocks**
   The architecture uses a stack of decoders (`n_layers`), where each decoder block comprises:
   - **Multihead Attention**:
     - Queries (Q), Keys (K), and Values (V) are computed.
     - Scaled Dot Product Attention is applied:
       - Dot product between Q and K is scaled.
       - Optional masking with `torch.tril` for causal attention.
       - Softmax normalization followed by matrix multiplication with V.
       - Outputs represent a blend of input vectors weighted by attention.
     - Multihead attention combines multiple attention heads, enabling the model to capture diverse semantic information.
   - **Residual Connection and Normalization**:
     - Results are passed through a residual connection and normalized.
   - **Feed Forward Layer**:
     - A two-layer network with:
       - Linear transformation.
       - ReLU activation.
       - Final linear transformation.

---

### 3. **Output Processing**
   - **Linear Layer**: Maps the outputs of the decoder stack to logits.
   - **Softmax**: Converts logits into probabilities for token generation.
   - **Sampling & Backpropagation**:
     - Probabilities are used for token sampling and generation.
     - Target outputs are compared to generated outputs, and backpropagation is performed to minimize the loss.

---

## Attention Mechanism Breakdown

### **Scaled Dot-Product Attention**
   - Formula: 
     ```
     Attention(Q, K, V) = Softmax((QK^T) / sqrt(d_k)) * V
     ```
   - Steps:
     1. Dot product between Q and K.
     2. Scale by `1/sqrt(length of Q/K rows)`.
     3. Apply optional masking (e.g., `torch.tril`).
     4. Normalize with Softmax.
     5. Multiply with V.

### **Multihead Attention**
   - Combines several attention heads.
   - Each head focuses on a unique semantic aspect.
   - Outputs from heads are concatenated and passed through a linear layer.

---

## Training Workflow

1. Tokenized inputs are passed through the embedding and positional encodings layer.
2. Inputs go through multiple decoder blocks, each refining the representation.
3. Final decoder outputs are passed to a linear layer.
4. Softmax is applied for token generation.
5. Loss is computed by comparing generated tokens with target tokens.
6. Backpropagation optimizes the model.

---

## Diagram

Refer to the diagram (`gpt_architecture.png`) for a visual representation of the workflow.

---

## Notes

- **Post-Normalization Technique**: Residual connections are normalized after each layer.
- **Flexible Decoder Stack**: The number of decoder blocks (`n_layers`) can be adjusted based on the model size.
- **Causal Masking**: Ensures that future tokens are not attended to during training.

---

## Usage

This repository is a learning-focused implementation of GPT. For full-scale deployment or optimization, consider additional techniques like mixed-precision training, parallelism, and dataset-specific fine-tuning.
