# Character-Level Text Generation: LSTM vs. Transformer from Scratch in PyTorch

An in-depth, CPU-friendly deep learning project that builds, trains, and evaluates two influential sequence models—a Long Short-Term Memory (LSTM) network and an autoregressive Transformer—completely from scratch using PyTorch for character-level text generation. 

Both models are trained on the classic **TinyShakespeare** corpus to learn the structure, spelling, spelling flow, and styling of dramatic plays one individual character at a time. The project is fully containerized via Docker and Docker Compose for easy, reproducible execution.

---

## 🚀 Key Features

*   **Custom LSTM Module**: Built with a dense embedding layer, stacked recurrent LSTM units, and a fully connected projection head.
*   **From-Scratch Autoregressive Transformer**: Implemented from basic mathematical layers (embedding, custom positional encodings, feed-forward, layer normalizations, and custom multi-head self-attention with a causal triangular mask).
*   **Temperature Gated Sampling**: A customizable generation pipeline that scales raw logit outputs at different temperature scales ($0.5$, $1.0$, $1.5$) to balance deterministic structure against creative variation.
*   **Fully Containerized Execution**: Single-command builds and isolated service orchestration via Docker and Docker Compose.
*   **Automated Verification**: End-to-end logging, metrics tracking, and automatic dual-curve visualization saving to `results/loss_curves.png`.

---

## 📂 Project Repository Structure

```
/
├── Dockerfile                  # Builds isolated PyTorch CPU execution container
├── docker-compose.yml          # Services orchestration and volume mapping
├── .env.example                # Example environment configuration template
├── .env                        # Active environment hyperparameters and paths
├── requirements.txt            # Python dependencies (torch, numpy, matplotlib, etc.)
├── README.md                   # Complete project documentation and portfolio report
├── input/
│   └── shakespeare.txt        # TinyShakespeare dataset (automatically downloaded)
├── results/
│   ├── loss_curves.png         # Combined training and validation loss curves
│   ├── generated_samples.json  # Multi-temperature generated text samples
│   └── comparison_report.md    # Detailed quantitative and qualitative analysis report
├── src/
│   ├── __init__.py             # Python package initializer
│   ├── prepare_data.py         # Data download, mappings, and loader setup
│   ├── model_lstm.py           # PyTorch LSTM character generator
│   ├── model_transformer.py    # Custom from-scratch causal Transformer generator
│   ├── train.py                # Training runner and plotting orchestration
│   └── generate.py             # Single and batch autoregressive text generator
```

---

## 🛠️ Quick Start & Setup

### Prerequisites
*   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your host system.

### 1. Initialize Configuration
Make a copy of the environment variables template to create your `.env` configuration file:
```bash
cp .env.example .env
```

### 2. Build the Docker Container
Build the isolated environment image containing Python, PyTorch (CPU version), Matplotlib, and other utilities:
```bash
docker-compose build
```

---

## ⚙️ How to Run the Pipelines

All commands run inside the Docker container to ensure reproducible results across different machines.

### 1. Data Preparation
To download the Shakespeare text corpus and build vocabulary character-to-integer mappings:
```bash
docker-compose run --rm app python src/prepare_data.py
```

### 2. Train the LSTM Model
To start the training loop and save weights for the LSTM architecture:
```bash
docker-compose run --rm app python src/train.py --model lstm
```
*Note: To run a quick validation, you can limit the batches processed per epoch (e.g. `--max_batches 300`):*
```bash
docker-compose run --rm app python src/train.py --model lstm --epochs 5 --max_batches 300
```

### 3. Train the Transformer Model
To train the custom Mini-Transformer architecture:
```bash
docker-compose run --rm app python src/train.py --model transformer
```
*Quick CPU validation run:*
```bash
docker-compose run --rm app python src/train.py --model transformer --epochs 5 --max_batches 300
```

### 4. Text Generation (Single Inference)
Generate text starting with a custom seed text and a specific temperature scale:
```bash
docker-compose run --rm app python src/generate.py --model lstm --seed_text "ROMEO:\n" --temperature 0.8 --length 300
```

### 5. Run Comparison Batch Generation
To automatically load both models, run text generations at multiple temperatures ($0.5$, $1.0$, $1.5$), and write them to `results/generated_samples.json`:
```bash
docker-compose run --rm app python src/generate.py --generate_all
```

---

## 📊 Summary of Findings and Results

Below is a summary of our model comparison report. For a comprehensive review, see [results/comparison_report.md](file:///e:/Partnr_Tasks/Build_and_Compare_Character-Level_Text_Generation_Models_with_PyTorch/results/comparison_report.md).

### 1. Quantitative Comparison (Perplexity on Validation Set)

Perplexity represents the uncertainty of predicting the next character (lower is better):

| Model Architecture | Total Parameters | Final Val Loss | Final Perplexity | Training Epoch Time (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | **946,625** | **1.3590** | **3.8924** | **~51.5 seconds** |
| **Mini-Transformer** | 825,409 | 2.3623 | 10.6150 | ~79.8 seconds |

### 2. Qualitative Observations
*   **The Inductive Bias of Recurrence**: The LSTM model converges much faster on character sequences because recurrent layers have a natural linear inductive bias that tracks token offsets sequentially. Within 5 epochs, the LSTM generates grammatically cohesive sentences and correct character dialogue structures.
*   **Transformers from Scratch**: Because Transformers do not have a built-in recurrent bias, they must learn token positioning entirely from the ground up using positional encodings and self-attention weight projections. This explains its higher initial perplexity under low epoch bounds, as it requires more training epochs or larger datasets to match recurrent convergence.
*   **Effect of Temperature Scaling**:
    *   **Low Temperature ($0.5$)**: Output is highly confident and spelling is 100% correct, but patterns are repetitive.
    *   **Balanced Temperature ($1.0$)**: Optimal blend of creativity and structure. The LSTM generates realistic dramatic scripts with correct formatting and vocabulary.
    *   **High Temperature ($1.5$)**: The probability distribution flattens completely, leading to highly chaotic gibberish and structural breakdown.

---

## 🛠️ Custom Transformer Implementation Highlights

To understand the core mechanisms of sequence models, the **Mini-Transformer Model** is written entirely from scratch using primitive tensor operations.

1.  **Causal Self-Attention**:
    Autoregressive generators must not attend to future tokens. We implemented a causal mask using `torch.tril` (lower triangular mask of 1s). The self-attention projection scores are filled with `-1e9` at future index positions, effectively zeroing out future attention weights during softmax:
    ```python
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    ```

2.  **Sinusoidal Positional Encoding**:
    Unlike recurrent networks, attention layers process sequences in parallel, discarding order. Encodings are injected by adding a fixed matrix of multi-frequency sine and cosine waves:
    ```python
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    ```

3.  **Pre-LN Residual Connections**:
    Applying layer normalization *before* each block's self-attention and feed-forward layers (Pre-LN) guarantees stable gradient flows and prevents exploding/vanishing gradients during backpropagation:
    ```python
    x = x + self.attn(self.norm1(x), mask=mask)
    x = x + self.ff(self.norm2(x))
    ```
