---
title: "Single GPU"
date: 2023-02-15T15:52:41+13:00
slug: ""
description: ""
keywords: []
draft: true
tags: []
math: false
toc: false
---

One of the most surprising findings in NLP over the previous five years is that
the *scale* of a model is generally one of the more reliable predictors of its performance.  

We define ***scale*** abstractly as a **quantity** that generalizes the **number of parameters**, **training examples**, and **floating point operations (FLOPS)** experienced during the training process. 

Scaling up the model is a tricky process, the mechanics of which remain as fertile grounds for research exploration. As an example, researchers have recently discovered that many large language models with hundreds of billions of parameters are massively [undertrained](https://arxiv.org/abs/2203.15556). Surprisingly, sheer scaling have also seems to have given models [emergent behaviors](https://arxiv.org/abs/2206.07682), which are abilities defined to only be observed once the model has reached some threshold scale. Looking at some of the evluation benchmarks, performance suddenly jumps once the model reaches some threshold scale. 

## Dev logs of me trying to get any version of GPT running on a single gpu. 

## Specs
GPU: NVIDIA GeForce GTX 1660 Super Ti. 
- Memory: 6GB
- Theoretical max TFLOPS: 10.05

## Purpose
Basically just get a model running on about 6GB of memory. 
Basic napkin math with float16 precision: (6 * 2e30 / 2) ~~ 3 Billion parameters. 
I estimate that less than half of these are free for use... so lets try to hit 1 Billion parameter model. 

## Cramming

"We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings.
Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture. "

if the only thing that matters is # of params, just pick the architecture that maximizes throughput. 


Honestly I think that if you can just get RLHF working on a local gpu that is already half the battle. 


### Stuffs
They use FlashAttention, apparently its a faster kernel for the attention mechanism. Look into this?
nvFuser: some sort of DL Compiler for NVIDIA GPUs, I need a tldr on compilers for DL workflows. 
it is becoming standard to run studies with a setup of uatomated mixed precision for standard 16- and
32-bit floating point precision. 


## huggingface/peft
Basically a way toplay with large scale language models on commodity hardware.

side note: catastrophic forgetting is a phenomenon where, when trained on one task, then trained on a second
task, many ML models "forget" how to perform the first task. Best minimized with Dropout (ensemble learning).


### LoRA
Low-Rank Adaptation of matrices. The key is: "Freezes the pretrained model weights and injects trainable rank decomposition matrices
into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. "

# Training != Inference
you can even use one-bit encodings of neural networks.


## Cool blog post could be about how to achieve 100% utilization on a gpu.


## another idea could be [analog neural networks](https://spectrum.ieee.org/analog-ai)
digital logic could be a little inefficient when it comes to inference and
training.

why is logic digital when it could be analog (same amount of compute at a far
greater energy efficiency and speed.) A basic implementation takes us to about
50% state of the art compute on the highest end gpu today. According to the
national shortage, each A100 NVIDIA gpu costs $32,097, which is prohibitively
expensive for many reasons. 

Most analog accelerators were developed for deployment on the edge, but there is
also significant value in training with it as well.




