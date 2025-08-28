# MoLER

<p align="center">
  <img src="./figure/framework.pdf"/>
</p>

MoLER is a domain-aware Retrieval-Augmented Generation (RAG) framework that integrates MoL-Enhanced Reinforcement Learning to optimize retrieval efficiency and scalability in low-knowledge domains. By addressing the critical limitation of existing query augmentation methods that fail to align with large language models' (LLMs) contextual understanding, MoLER bridges the knowledge gap in RAG systems through end-to-end optimization of query and passage generation.

## Key Features

- **Mixture of Losses (MoL) Continual Pre-training**: Balances domain-specific knowledge acquisition with general language capabilities through dual-loss architecture
- **Group Relative Policy Optimization (GRPO)**: End-to-end optimization for maximizing document recall in retrieval tasks
- **Multi-query Single-passage Late Fusion (MSLF)**: Reduces computational overhead during RL training while maintaining inference effectiveness
- **Multi-query Multi-passage Late Fusion (MMLF)**: Enhances final recall performance through diverse query-document interactions
- **Domain-aware Training Pipeline**: Two-stage approach combining continual pre-training and reinforcement learning phases

## Architecture

MoLER operates through a comprehensive two-stage pipeline:

1. **Continual Pre-training (CPT) Phase**: Employs MoL approach to balance domain-specific (CE loss) and general knowledge (KL divergence) corpora
2. **Reinforcement Learning Phase**: Leverages GRPO to optimize Multi-Query Retriever (MQR) and Combined Query Expansion (CQE) for enhanced document recall

The framework introduces an innovative training-inference strategy where MSLF is used during RL training for computational efficiency, while MMLF is deployed during inference for optimal retrieval performance.

## Performance

MoLER achieves state-of-the-art performance on benchmark datasets:

- **NFCORPUS**: 61.42% Recall@1k, 25.44% nDCG@10
- **SCIFACT**: 79.69% Recall@10, 62.59% nDCG@10

Notably, a 1.7B parameter model with MoLER significantly outperforms a 32B parameter baseline, demonstrating exceptional parameter efficiency.

## Building

After cloning the repository, run the following commands:

```bash
pip install -r requirements.txt
```

## Model Requirements

MoLER supports various model scales:
- **Base Models**: Qwen3-0.6B, Qwen3-1.7B (or compatible architectures)
- **Embedding Model**: OpenAI text-embedding-ada-002
- **Framework**: PEFT with LoRA (rank 64)

## Training Configuration

### MoL Continual Pre-training
- **Context Window**: 8192 tokens
- **Corpus Ratio**: 1:1 domain-specific to general corpora
- **Training Epochs**: 4 epochs for optimal performance
- **Loss Functions**: CE loss for domain data, KL divergence for general data

### GRPO Post-training
- **Reward Signal**: Document recall rate via RRF fusion
- **Query Expansions**: 3 sub-queries for balanced efficiency/effectiveness
- **Fusion Strategy**: MSLF during training, MMLF during inference

## Usage

The framework implements a three-step retrieval process:

1. **Instruction Expansion**: Generate semantically diverse sub-queries using MQR prompting
2. **Pre-Answer Guidance**: Create contextual pseudo-documents via CQE prompting
3. **Reciprocal Rank Fusion**: Combine retrieval results for enhanced recall

## Datasets

Evaluation conducted on BEIR benchmark datasets:
- **NFCORPUS**: Biomedical QA (3,633 docs, 323 test queries)
- **SCIFACT**: Scientific fact-checking (5,183 abstracts, 300 test queries)

## Citation

```bibtex
@article{lin2024moler,
  title={Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval},
  author={Lin, Hao},
  journal={arXiv preprint},
  year={2024},
  institution={Southeast University}
}
```

## Contributing

We welcome contributions to improve MoLER's capabilities. Please submit issues and pull requests following standard academic collaboration practices.