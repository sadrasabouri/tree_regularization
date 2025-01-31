# Sneaking Syntax into Transformer Language Models with Tree Regularization
[<a href="https://arxiv.org/abs/2411.18885">Paper</a>] 


<div align="center">
<img src="assets/treereg.jpg" alt="TreeReg overview" title="Overview of TreeReg" width="600">
</div>

This codebase can be used to pretrain language models with TreeReg on parsed corpora such as [BLLIP-LG](https://github.com/cpllab/syntactic-generalization). Our implementation of TreeReg is also very easily transferable to any codebase, as shown in [Using TreeReg with your Code](#using-treereg-with-your-code).

## TODOs
- [ ] Optimize support for multi-GPU training.
- [ ] Add evaluation scripts for SyntaxGym and Parsevals.

## Table of Contents
 
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Processing](#dataset-processing)
- [Pretraining from Scratch](#pretraining-from-scratch)
- [Continued Pretraining of HuggingFace models](#continued-pretraining-of-huggingface-models)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Citation](#citation)

---
## Project Structure

```plaintext
tree_regularization/
├── src/
│   ├── callbacks/                      # Callback functions for training-time evaluation
│   ├── data/                           # Dataset processing scripts
│   ├── interfaces/                     # Wrapper scripts to return hidden states and LM loss
│   ├── layers/                         # Implementation of Relative and Rotary Transformer layers
│   ├── models/                         # Implementation of autoregressive Transformer models
│   ├── regularizer/                    # Code for TreeReg
│   ├── trainer/                        # Training loop code
│   ├── argument_parser.py                            
│   ├── dataset_util.py
│   ├── model_util.py
│   ├── train.py                        # Entry point into the training code
│   ├── util.py     
│   └── vocabulary.py                   
├── eval_scripts/                       # Different executable evaluation scripts
│   ├── eval_ptb.py                     # Parsing Accuracies on datasets like Penn TreeBank
│   ├── eval_sg.py                      # Accuracies on SyntaxGym
│   └── ...
│   
├── setup.py
├── requirements.txt             # Dependencies
└── README.md
```

## Using TreeReg with your Code

Using TreeReg during finetuning or pretraining in any codebase only requires the following three steps:
1. Copy the `regularizer` folder over to your codebase.
2. Initialize the `TreeRegularizer` class:
```python
from regularizer.regularizer_main import TreeRegularizer

regularizer = TreeRegularizer()
```
3. Compute the TreeReg loss on a batch of parsed sentences:
```python
scin_charts = regularizer.build_chart(hidden_states, word_boundaries, parses)
treereg_scores, _ = regularizer.get_score(scin_charts, word_boundaries, parses, device)
treereg_loss = -torch.mean(torch.stack(treereg_scores))
```

`hidden_states`: The hidden states on which TreeReg is to be computed, obtained during the forward pass of the model over the batch of sentences. These will typically come from a subset of attention heads at an intermediate layer of the model. Also note that if a start of sentence token is used, the corresponding hidden state should be omitted. 

`word_boundaries`: A list of boolean masks indicating the tokens at which each word in the sentence starts for each sentence. For example, if "To kill a mockingbird" is tokenized as ["To", "kill", "a", "mocking", "bird"], the mask is [1,1,1,0].

`parses`: A list of bracketed representation of constituency parses of input sentences. The parse is represented as a mapping from spans of the sentence to the index at which they are split in the parse tree. For example, for the tree:
```
                     S
      _______________|_________________
     |                           S|<,-NP-VP-.>
     |                _________________|________________
     |               |                             S|<NP-VP-.>
     |               |                           _______|_______________________________________
     |               |                          NP                                              |
     |               |         _________________|_______                                        |
     PP              |        |                         PP                                   S|<VP-.>
  ___|___            |        |                  _______|_______                           _____|______
 |       NP          |        NP                |               NP                        VP           |
 |    ___|_____      |    ____|________         |        _______|________            _____|_____       |
 IN  DT        NN    ,   DT           NNS       IN      DT               NN        VBP          JJ     .
 |   |         |     |   |             |        |       |                |          |           |      |
 As  a       result   ,  the        specifics   of      the         announcement  remain       vague   .
```
the representation is: `{"0 3": 1, "6 9": 7, "4 9": 6, "9 12": 11, "4 12": 9, "3 12": 4, "0 12": 3}`.

## Environment Setup

```bash
conda create -n treereg python=3.8.10;
pip install -r requirements.txt
pip install -e .
```

## Dataset Processing

To train a model using TreeReg, a preprocessing script has to be written to turn the training data into a HFDataset. The HFDataset should have the following keys:
```
"in": tokenized input sentences
"in_lens": lengths of the tokenized sentence
"parses": constituency parses on input sentences, in the format mentioned above. The dictionary can be dumped to a string using JSON, to make it compatible with HFDataset
"word_boundaries": A boolean mask indicating the tokens at which each word in the sentence start.
```

Preprocessing scripts for BLLIP-LG can be found inside `data`. If new datasets are added, a corresponding entry can be added to `dataset_helper` inside `dataset_util.py`.

## Pretraining from Scratch

To run pretraining from scratch, the following command can be used:

```
torchrun --nproc_per_node=1 --standalone src/train.py --dataset=DATASET_NAME --save_dir SAVE_DIRECTORY --encoder_n_layers NUMBER_OF_LAYERS_IN_MODEL --seed 10 --callback --max_train_steps NUMBER_OF_TRAINING_STEPS --eval_every NUMBER_OF_STEPS_BETWEEN_EVALUATION_CALLBACKS --save_interval NUMBER_OF_TRAINING_STEPS_BETWEEN_MODEL_SAVES --batch_size BATCH_SIZE --accum_steps GRADIENT_ACCUMULATION_STEPS --start_lr LEARNING_RATE_AT_START_OF_TRAINING --end_lr LEARNING_RATE_AT_END_OF_TRAINING --relative True --regularize --regularizer_steps NUMBER_OF_TRAINING_STEPS_BETWEEN_EACH_TREEREG_CALL --embedding_dropout 0.1 --output_dropout 0.1 --orth_bidir --layer_id LAYER_AT_WHICH_TREEREG_IS_APPLIED --sci_heads FRACTION_OF_ATTENTION_HEADS_FOR_TREEREG --wandb_entity WANDB_USER_NAME
```

If the same data is to be used for language modelling and TreeReg, an additional flag `--treereg_same_data` can be set as well. We also provide the command used for training on BLLIP-LG:

```
torchrun --nproc_per_node=1 --standalone src/train.py --dataset=bllip-lg --save_dir SAVE_DIR --encoder_n_layers 16 --seed 10 --callback --max_train_steps 60000 --eval_every 1000 --save_interval 30000 --batch_size 32 --accum_steps 5 --start_lr 1e-4 --end_lr 6e-5 --relative True --regularize --regularizer_steps 10 --embedding_dropout 0.1 --output_dropout 0.1 --orth_bidir --layer_id 12 --sci_heads 0.25 --wandb_entity WANDB_USER_NAME --treereg_same_data
```

## Continued Pretraining of HuggingFace models

To continue pretraining of a pretrained model available through HuggingFace, the following command can be used:

```
torchrun --nproc_per_node=1 --standalone src/train.py --dataset=DATASET_NAME --save_dir SAVE_DIRECTORY --seed 10 --callback --max_train_steps NUMBER_OF_TRAINING_STEPS --eval_every NUMBER_OF_STEPS_BETWEEN_EVALUATION_CALLBACKS --save_interval NUMBER_OF_TRAINING_STEPS_BETWEEN_MODEL_SAVES --batch_size BATCH_SIZE --accum_steps GRADIENT_ACCUMULATION_STEPS --start_lr LEARNING_RATE_AT_START_OF_TRAINING --end_lr LEARNING_RATE_AT_END_OF_TRAINING --relative True --regularize --regularizer_steps NUMBER_OF_TRAINING_STEPS_BETWEEN_EACH_TREEREG_CALL --orth_bidir --layer_id LAYER_AT_WHICH_TREEREG_IS_APPLIED --sci_heads FRACTION_OF_ATTENTION_HEADS_FOR_TREEREG --wandb_entity WANDB_USER_NAME --hf --hf_model_name NAME_OF_HF_MODEL
```

The command used for training Sheared LLama-1.3B on BLLIP-LG is:

```
torchrun --nproc_per_node=1 --standalone src/train.py --dataset=bllip-lg --save_dir SAVE_DIR --wandb_entity WANDB_USER_NAME --seed 10 --callback  --max_train_steps 10000 --save_interval 5000 --eval_every 200 --pack --max_seq_len 512 --batch_size 4 --accum_steps 8 --start_lr 2e-5 --end_lr 4e-6 --regularize --regularizer_steps 2 --orth_bidir --layer_id 12 --sci_heads 0.25 --ce --hf --hf_model_name princeton-nlp/Sheared-LLaMA-1.3B
```

## Evaluation

Evaluation callbacks that are periodically run during training can be defined inside `callbacks`, and corresponding entries added to `get_callback_fn` inside `dataset_util.py`. Other evaluation scripts can be found inside `eval_scripts`.

## Contributing

Contributions are welcome! If you encounter any issues, feel free to open a GitHub issue or submit a pull request.

## Citation

If you use TreeReg in your work, please cite us using the following BibTeX entry:

```bibtex
@article{nandi2024sneakingsyntaxtransformerlanguage,
      title={Sneaking Syntax into Transformer Language Models with Tree Regularization}, 
      author={Ananjan Nandi and Christopher D. Manning and Shikhar Murty},
      year={2024},
      journal={arXiv preprint arXiv:2411.18885}
}
```
