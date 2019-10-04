# Graph Neural Networks for Natural Language Processing

[![Conference](http://img.shields.io/badge/EMNLP-2019-4b44ce.svg)](https://www.emnlp-ijcnlp2019.org/program/tutorials/)
[![Slides](http://img.shields.io/badge/slides-pdf-red.svg)]()
[![Supplementary](http://img.shields.io/badge/supplementary-arxiv.xxxx.xxxx-B31B1B.svg)]()

The repository contains code examples for [GNN-for-NLP](https://www.emnlp-ijcnlp2019.org/program/tutorials/) tutorial at [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/).

<img align="right"  src="./graph.jpeg">

### Dependencies

- Compatible with PyTorch 1.x, TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### TensorFlow Examples:

* `kipf_gcn.py` contains simplified implementation of first-order approximation of GCN model proposed by [Kipf et. al. (2016)](https://arxiv.org/abs/1609.02907)
* Extensions of the same implementation for different problems:
  * Relation Extraction: [RESIDE](https://github.com/malllabiisc/RESIDE)
  * GCNs for Word Embeddings: [WordGCN](https://github.com/malllabiisc/WordGCN)
  * Document Time-stamping: [NeuralDater](https://github.com/malllabiisc/NeuralDater)

### PyTorch Examples:

