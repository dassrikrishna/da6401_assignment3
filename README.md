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
- Beam width for inference

## Training and Evaluation
To train the model, I implemented standard training and evaluation loops using PyTorch. The training process includes **gradient clipping** for stability and uses **teacher forcing** for sequence prediction.

### Training Loop
Function: `train_epoch(model, dataloader, optimizer, criterion, device)`

Applies teacher forcing (`default ratio = 0.5`)

Performs backpropagation and updates model weights

Uses gradient clipping to prevent exploding gradients

### Evaluation Loop
Function: `evaluate(model, dataloader, criterion, device)`

Runs the model in evaluation mode

No teacher forcing (`teacher_forcing_ratio = 0`)

Returns average loss over the validation/test set

### Sequence-Level Accuracy
Function: `sequence_accuracy(model, dataloader, target_idx2char, device)`

Computes accuracy based on exact match between predicted and ground truth character sequences

Uses beam search decoding (`beam_width = 3`) for higher-quality predictions
## Main Training Loop (with W&B)
I integrated Weights & Biases (W&B) to manage experiments, perform hyperparameter sweeps, and track metrics like accuracy, loss, and model configuration.
```python
wandb.log({
    "epoch": epoch+1,
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc
})
```
## Hyperparameter Tuning (W&B Sweeps)
### Sweep Strategy
- Type: `Bayesian Optimization`
- Early Termination: Hyperband (`min_iter = 3`)
### Swept Parameters:
| Hyperparameter   | Values                  |
| ---------------- | ----------------------- |
| `emb_dim`        | 16, 32, 64, 128, 256 |
| `hidden_dim`     | 16, 32, 64, 128, 256 |
| `encoder_layers` | 1, 2, 3              |
| `decoder_layers` | 1, 2, 3             |
| `dropout`        | 0.2, 0.3]            |
| `cell`           | 'RNN', 'GRU', 'LSTM' |
| `beam_size`      | 1, 3, 5             |

## **Smart Search Strategies for Fewer Runs**
To avoid running all 1,350 combinations, I used the following strategies to make the sweep more efficient:

-  **Bayesian Optimization:**

Focused the search on promising hyperparameter regions using past performance. More effective than random search for large spaces.

- **Early Stopping with Hyperband:**

Automatically stopped poorly performing runs early - after at least 3 iterations, saving time and compute.

- **Conditional Dropout:**

Applied dropout only when the encoder or decoder had more than one layer, avoiding unnecessary regularization on shallow models.

## W&B Visualizations
To visualize the hyperparameter sweep results using Weights & Biases (W&B), we generated the following plots:

- Accuracy vs. Created Time - shows how accuracy evolved across different experiment runs.
- Parallel Coordinates Plot - displays relationships between hyperparameters and model performance.
- Correlation Summary Table - highlights correlations between hyperparameters and metrics like accuracy and loss.

## Evaluation on Test Data:

After completing hyperparameter tuning and validation on the training and validation sets, I evaluated the best model on the test set.

**(a) Test Accuracy:**
Using the best configuration from the sweep (`Run ID: vb1yhddm`):
```bash
Best Validation Accuracy (Exact Match): 0.3448
```
**Best Hyperparameters:**
```python
cell: LSTM
dropout: 0.2
emb_dim: 64
hidden_dim: 256
encoder_layers: 3
decoder_layers: 2
beam_size: 3
```
**Test Set Result:**
```bash
Test Accuracy (Exact Match): 0.3447
```
### Predictions:
All predictions from the best model on the test set have been saved and uploaded to my GitHub repository under the folder: [**predictions_vanilla**](https://github.com/dassrikrishna/da6401_assignment3/blob/main/predictions_vanilla/predictions_vanilla.csv)


## Model with Attention
To enhance the baseline sequence-to-sequence architecture, I incorporated an attention mechanism, which allows the decoder to focus on relevant parts of the input sequence dynamically during decoding. This architectural modification improves the model's ability to handle longer sequences and complex alignments.

I reused the same hyperparameter sweep strategy (Bayesian optimization with early stopping via Hyperband) to tune the model effectively with minimal runs while maintaining strong performance.

### Evaluate Best Model with Attention
After completing hyperparameter tuning and validating only on the training and validation sets, I evaluated my best attention-based model on the test set.

``` bash
Best Validation Accuracy (Exact Match):0.3915
```
**Best Hyperparameters:**
``` python
cell: LSTM
dropout: 0.2
emb_dim: 128
hidden_dim: 256
encoder_layers: 3
decoder_layers: 2
beam_size: 3
```
**Test Set Result:**
```bash
Test Accuracy (Exact Match): 0.3804
```
### Predictions:
All predictions on the test set using the best attention-based model have been saved and uploaded to my GitHub repository: [predictions_attention](https://github.com/dassrikrishna/da6401_assignment3/blob/main/predictions_attention/predictions_attention.csv)

The attention-based model performs better but requires more computation and memory, leading to longer training times.

## Attention Heatmaps (3×3 Grid)
To visualize the attention mechanism of the best model, I randomly selected 9 test examples and plotted their attention heatmaps in a 3×3 grid. Each heatmap shows how the model aligns source (Latin) characters to target (Bengali) characters during decoding.

Bengali font rendering was handled using [Lohit-Bengali](https://github.com/dassrikrishna/da6401_assignment3/blob/main/Lohit-Bengali.ttf).

Attention weights were extracted during inference from the decoder.

Heatmaps were created using seaborn, and Bengali characters were displayed using proper font support.

The final grid was saved as [heatmap_attension.png](https://github.com/dassrikrishna/da6401_assignment3/blob/main/predictions_attention/heatmap_attension.png) in the working directory.

## Connectivity Visualization
The `visualize_connectivity` function generates heatmaps that show the attention weights between input and output characters. Each row in the heatmap corresponds to a character in the output (decoded sequence), and each column corresponds to a character in the input.

The color intensity in each cell indicates how much attention the model pays to a specific input character while generating a specific output character. Cells are also annotated with input characters to make it easier to interpret.

By looking at $i$-th row, you can see which input character(s) the model focused on while generating the i-th output character. This helps us understand how the model is learning to align the input and output sequences during decoding.

The final visualizations are saved to the file [visualize_connectivity.png](https://github.com/dassrikrishna/da6401_assignment3/blob/main/visualize_connectivity.png).

**Location:** Inside `da6401-assignment3-attention.ipynb` notebook, under `#11. Visualization Connectivity`.


