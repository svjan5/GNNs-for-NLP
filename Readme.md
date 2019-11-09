# Graph Neural Networks for Natural Language Processing

[![Conference](http://img.shields.io/badge/EMNLP-2019-4b44ce.svg)](https://www.emnlp-ijcnlp2019.org/program/tutorials/)
[![Slides](http://img.shields.io/badge/slides-pdf-red.svg)](https://shikhar-vashishth.github.io/assets/pdf/emnlp19_tutorial.pdf)
[![Colab](http://img.shields.io/badge/colab-run-yellow.svg)](https://drive.google.com/drive/u/0/folders/1ljM-k34uYyI3Sp3IYGiZp_i5X1TgaYl1)

The repository contains code examples for [GNN-for-NLP](https://www.emnlp-ijcnlp2019.org/program/tutorials/) tutorial at [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/). 

Slides can be downloaded from [here](https://shikhar-vashishth.github.io/assets/pdf/emnlp19_tutorial.pdf). 

<img align="right"  src="./graph.jpeg">

### Dependencies

- Compatible with PyTorch 1.x, TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### TensorFlow Examples:

* `tf_gcn.py` contains simplified implementation of first-order approximation of GCN model proposed by [Kipf et. al. (2016)](https://arxiv.org/abs/1609.02907)
* Extensions of the same implementation for different problems:
  * Relation Extraction: [RESIDE](https://github.com/malllabiisc/RESIDE)
  * GCNs for Word Embeddings: [WordGCN](https://github.com/malllabiisc/WordGCN)
  * Document Time-stamping: [NeuralDater](https://github.com/malllabiisc/NeuralDater)

### PyTorch Examples:

* `pytorch_gcn.py` is pytorch equivalent of `tf_gcn.py` implemented using [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric). 
* Several other examples are available [here](https://github.com/rusty1s/pytorch_geometric/tree/master/examples). 

### Additional Resources:

* Short writeup on theory behind Graph Convolutional Networks [[Pdf]](https://shikhar-vashishth.github.io/assets/pdf/phd_thesis.pdf) (refer Chapter-2).
* [GNN recent papers](https://github.com/naganandy/graph-based-deep-learning-literature).
