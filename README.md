# DA6401 Assignment-3
#### Detailed Weights & Biases Report for My Project: [Click Here](https://api.wandb.ai/links/ma24m025-indian-institute-of-technology-madras/9y9edofe)
#### Github Link: [Click Here](https://github.com/dassrikrishna/da6401_assignment3)
#### Seq2Seq Vanila: [Click Here](https://github.com/dassrikrishna/da6401_assignment3/blob/main/da6401-assignment3-vanila.ipynb)
#### Seq2Seq Attention: [Click Here](https://github.com/dassrikrishna/da6401_assignment3/blob/main/da6401-assignment3-attention.ipynb)
#### predictions_vanilla: [Click Here](https://github.com/dassrikrishna/da6401_assignment3/tree/main/predictions_vanilla)
#### predictions_attention [Click Here](https://github.com/dassrikrishna/da6401_assignment3/tree/main/predictions_attention)

## DEEP LEARNING
#### ```SRIKRISHNA DAS (MA24M025)```
#### `M.Tech (Industrial Mathematics and Scientific Computing) IIT Madras`
 

## [Problem Statement](https://wandb.ai/sivasankar1234/DA6401/reports/Assignment-3--VmlldzoxMjM4MjYzMg)

## Goal of the Assignment
The primary objectives of this assignment are as follows:

### 1.Model Sequence-to-Sequence Learning Problems
Understand how to model sequence-to-sequence (seq2seq) learning tasks using Recurrent Neural Networks (RNNs).

### 2.Compare Different RNN Cells
Explore and compare the performance of different RNN cell architectures including:
- Vanilla RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
  
### 3.Understand Attention Mechanisms
Learn how attention mechanisms help overcome the limitations of traditional seq2seq models by improving context representation and performance.

### 4.Visualize RNN Components
Gain insights into the internal workings of RNN-based models by visualizing the interactions between various components during the learning process.

## Dataset
For this assignment, I worked with the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina) released by Google Research.

I specifically used the Bengali lexicon available at:
`dakshina_dataset_v1.0/bn/lexicons/`

The dataset contains pairs of words, where each pair consists of:

A word written in Romanized form (Latin script)

Its corresponding Bengali form in Devanagari script

### Task Objective
The goal is to train a model $\hat{f}(x) = y$ that takes a Romanized Bengali word as input (e.g., `onoitik`) and predicts its equivalent in Devanagari script (e.g, `অনৈতিক`).

This is framed as a character-level sequence-to-sequence learning task. It simplifies the machine translation problem by focusing on transliteration — converting sequences of characters from one script to another.

### Dataset Statistics
- Training samples: 94,546
- Validation samples: 9,279
- Test samples: 9,228
- Input vocabulary size (Romanized): 29
- Target vocabulary size (Bengali): 63
- Maximum input sequence length: 22 characters
- Maximum target sequence length: 24 characters

### Sample Word Pairs
Here are a few examples from the dataset:
| Romanized (Input) | Bengali (Target) |
| ----------------- | ---------------- |
| hindusthan        | হিন্দুস্থান      |
| hindu             | হিন্দু           |
| oksygen           | অক্সিজেন         |
| anumoti           | অনুমতি           |
| anushashon        | অনুশাসন          |
| gantabya          | গন্তব্য          |
| tatkalin          | তৎকালীন          |
| dokkho            | দক্ষ             |
| pratijogita       | প্রতিযোগিতা      |
| madern            | মডার্ন           |
| sangho            | সংঘ              |

## Model Architecture
I implemented a flexible Recurrent Neural Network (RNN) based sequence-to-sequence (seq2seq) model for character-level transliteration from Romanized Bengali to native Bengali script. The architecture consists of the following components:

### 1.Input Embedding Layer
The input Romanized character sequence is passed through an embedding layer to learn dense vector representations of characters.

### 2.Encoder RNN
The embedded input is processed by an encoder RNN (configurable as RNN, GRU, or LSTM) which captures the sequential context. The final hidden state(s) from the encoder summarize the entire input sequence.

### 3.Decoder RNN
The decoder is initialized with the encoder’s final hidden state and generates one output character at a time. It also uses an embedding layer for its input characters and ends with a linear projection to the output vocabulary.

### 4.Beam Search (Inference)
For inference, a beam search decoding strategy is implemented to improve prediction quality by exploring multiple candidate output sequences.

### 5.Hidden State Adjustment
A utility function handles cases where the number of encoder and decoder layers do not match by appropriately padding or trimming the hidden states.

## Model Flexibility
The model is fully configurable through hyperparameters, allowing easy experimentation. You can control:

- Input embedding dimension
- Hidden state size
- RNN cell type: rnn, lstm, or gru
- Number of layers in both encoder and decoder
- Dropout rate in multi-layer setups

