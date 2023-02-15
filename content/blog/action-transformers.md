---
title: "Action Transformers"
date: 2023-02-15T15:52:49+13:00
slug: ""
description: ""
keywords: []
draft: true
tags: []
math: false
toc: false
---


# Brainstorming how to write differentiable software 

For ChatGPT, OpenAI trained a reward model and then chose to align the language
model using RLHF to this reward model. This reward model was trained based on
data that they themselves collected using humans, and that is something
difficult to entirely replicate. 

So with GPT3.5, what you basically get is a really good completer. 

We can't fine tune a model of that size, so maybe we can RLHF a smaller model. 

Something that is interesting is the reward model is what we are *aligning* our
model to. What if we can define a reward model to be a reinforcement learning
task? e.g. any task in the mujuco or

Can we define a reward model for using a particular software?

- Need to research how reward environments work... and how reinforcement
  learning works at all. 
- Do we need to engineer a clear signal in here?

Dialogue systems can benefit from incorporating the sequential processing
techniques. Sequential RL methods go maximizing deisred outcomes will really
bring out the power of these methods. 

side note: off-policy algorithms are those where we train a new policy using
rollouts collected from some other policy. Imitation learning is a good example
of a off-policy algorithm. 

## new generation of apis - differentiable
- I think that the action space has to be an API, something that clearly defines
  the action space and steps that the model can take. 

- environment can include loading commands to the api into some format and
  sending some http requests over. 


## InstructGPT basically just trained a reward model. 

first was supervised fine tuning, where they trained for 16 epochs, using a
cosine learning rate decay, and residual dropout of 0.2. 
despite overfitting on validation set, training for more epochs helps both the
RM scores and human preference ratings, despite this overfitting. 

They found that 175B RM training for a value function was unstable, so they only
use 6B sized models. 

Reward modeling was inspired by Stiennon et al (2020) where the difference in
rewards represents the log odds that one response will be preferred to the other
by a human labeler. See that paper 

Not sure what this means, but it seems very important:
"Since comparisons are very correlated within each labeling task, we found that
if we simply shuffle the comparisons into one dataset, a single pass over the
dataset caused the reward model to overfit. Instead, we train on all K c 2
comparisons from each prompt as a single batch element."

They also add a per-token KL pentalty from the SFT model at each token to
mitigate over-optimization of the reward model.

reward model, supervised fine tunine and ppo datasets are about 30k prompts in
order to achieve the same results?  This is somewhat acceptable i think. 


### oh but wait, what if we don't need to use RLHF?

- key: reformulate alignment as a goal-conditioned RL problem. This allows us to
  apply previous methods for fine-tuning such as hindsight experience replay.
- Under this formulation, the model is both the policy *and* the 
on second thought, this doesn't really work. 

## Lol Toolformer is here.

"Our approach for achieving these goals is based on the recent idea of using
large LMs with in-context learning to generate entire datasets from scratch."

Basically querying the model to get the training data corupus, then fine tuning
on that using an autoaggressive approach. I think this is becoming a lot more
popular nowadays in terms of generating our training data. 

Emergent behavior occurs after about 700 million parameters, and everything
before that is pretty bs. 

So to reproduce we would need to do this while fitting the entire model on a
commodity gpu.

achieves good score on math benchmarks, but falls short of gpt-3 on others. One
limitation is the inability of Toolformer to use tools ina chain (i.e., using
the output of one tool as an input for another tool). This due to the fact that
API calls for each tool are generated independently. 

Does not allow the LM to use a tool in an interactive way - especailly for tools
such as search engines, that could potential return hundreds of different
results

Models trained with Toolformer to often be sensitive to the exact wording of
their input when deciding whether or not to call an API; this is perhaps
unsurprising given that LMs are known to be very sensitive to the prompt they
are provided with in both zero and few-shot settings. 


- tested it out quickly on chatgpt, and i do agree that a training set could be
  generated in this fashion. 
- its quite interesting that we didn't need rl in order for the model to
  understand exactly what format to output.

idea: one way to use a tool in an interactive way and use tool in chain is to provide a whiteboarding area
where the lm can save an output to. This whiteboarding area is fed as context into the LM every single time
we run a forward pass. 

provide a whiteboard and a set of commands that they can type in order to manipulate this program state. Teach
the LLM how to manage this state effectively. 

This paper taught by trying to predict where to insert an API call. However, the training data was also generated
from the same model, but ultimately it was the LOSS criterion by which they filtered api calls that provided
the signal for when API calls are more useful. 

critique: the loss criteria simply says that: we want you to make an API call that provides information which further
enforces the model's beliefs in the first place.

they fine tuned on specific datasets, so the objective that they are
autoregressing on is guaranteed to be correct. 

They use up to 25k examples per API. max sequence length 1024. Effective batch
size of 128. All models were trained using DeepSpeed's ZeRO-3. Not really sure what that is. 
All models used 8 NVIDIA A100 40GB GPUs with BF16. 
Training up to 2k steps where we evaluate PPL on a small development set from CCNet containing
1000 examples every 500 steps. 


## webgpt
prediction: text is the new interface. 
fine-tuned gpt-3 to answer long-form questions using a text-based web-browsing
environment, which allows the model to search and navigate the web.
Uses imitation learning and then rejection sampling against a reward model
trained to predict human preferences.  
- Generate a text-based web-browsing environment that a fine-tuned language
  moidel can interact with. This allows us to improve both retrieval and
  synthesis in an end-to-end fashion using general methods such as imitation
  learning and reinforcement learning. 
- jeez, it does seem like text is the new interface. 
- geenrate answers with references: passages extracted by the model from web
  pages while browsing. This is crucial for allowing labelers to judge the
  factual accuracy of answers, without engaging in a difficult and subjective
  process of independent research.  <- used in microsoft bing, but is still
  hallucinating answers. I guess you can have a punishment objective for
  hallucination if the semantic meaning differs from that of the api call. can
  either optimize the model around the api, or the api around the model. 
- i think initially less impactful because of results.  
- rl did not provide much benefit against the benchmarks.RL was used for RLHF
  fine tuning for alignment. 

## toolformer, but with it learning how to extend its own memory. 

Give it a directory and teach it how to write down / retrieve information
from it. This could be done using a few simple actions substituted out in the
toolformer paper. 

teaching it how to use a computer, or a command line is pretty important. 




## Emergent behavior in large langugage models

- measured based on evaluation benchmarks.
- usually only achieved from 10-1000B parameters. 
- scale is more reliable indicator for performance than model architecture
- however, there are more factors than scale. "high-quality training data,
  architectural differences, different pretraining objective" could all
  separately unlock different emergent behaviors. 

"Once an ability is discovered, further research may make the ability available
for smaller scale models. Consider the nascent direction of enabling language
models to follow natural language instructions describing a task. Although they
initially found that instruction-based finetuning only worked for 68B parameter
or larger decoder-only models, they induced similar behavior in a 11B model with
an encoder-decoder architecture, which typically has higher performance after
finetuning than decoderr-only architectures. InstructGPT set of models has
emergent behaviors that are available in a 1.B parameter regime."

   
