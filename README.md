# Character-Level Text Generation: LSTM vs. Transformer from Scratch in PyTorch

An in-depth, CPU-friendly deep learning project that builds, trains, and evaluates two influential sequence models—a Long Short-Term Memory (LSTM) network and an autoregressive Transformer—completely from scratch using PyTorch for character-level text generation. 

Both models are trained on the classic **TinyShakespeare** corpus to learn the structure, spelling, spelling flow, and styling of dramatic plays one individual character at a time. The project is fully containerized via Docker and Docker Compose for easy, reproducible execution.

---

## Key Features

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
