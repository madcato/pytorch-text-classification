# PyTroch Text Classification

This is a projecto to play with pytorch. It's based in this PyTorch Tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

## Requirements

- torch
- torchtext

## Doc
- [Text classification with the torchtext library](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [PyTorch documenttion: EmbeddinBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)

## Tasks

- [ ] Optimize GPU utilisation (was 10%)
- [ ] Use GloVe embeddings

## Run

    $ python3 run.py

## Conclusions

- By incrementing the batch size, the model trains faster (even using less GPU), but model accuracy drops
- Making batch size lower, the model is trained with more accuracy, but trainning is slower.
