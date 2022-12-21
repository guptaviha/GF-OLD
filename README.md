# CS-GY 6953 / ECE-GY 7123 Final Project

## Detecting Offensive Language in Tweets with Attention-fused Text and User Embeddings

Transformer-based models have come to dominate natural language processing tasks including difficult problems in computational linguistics such as semantic and sentiment analysis. One application for these models is automated content moderation for social media platforms where they can be deployed to flag and remove abusive or harmful content. In the context of offensive language detection, recent research demonstrates that the performance of such models can be improved by incorporating information about the communities where that content is produced. For our project, we experiment with a novel technique for fusing community structure features and text features to classify offensive language in an end-to-end model via attention mechanisms. We incorporate a "corrected" procedure for computing attention on graph-structured data to produce our user embeddings and a better performing bidirectional encoder for our text embeddings into an existing architecture. Our model achieves a mean F1 score of 0.8936 barely outperforming our baseline model by 0.0009. Further, our ablation analysis demonstrates that attention fusion tends towards diminishing returns as the underlying embeddings become more expressive.

You can find notebooks that for training and evaluating the models using this repository under `examples/`. The
notebook `model_training_colab.ipynb` can be used to reproduce our final results on Google Colab and the
notebook `plots.ipynb` can be used to generate the plots used in our report.

The scripts here are a modified fork of [Miao 2022](https://github.com/mzx4936/GF-OLD) that incorporates different bidirectional encoder and graph attention modules for the purpose and a modified training loop that allows for multiple model iterations and metrics logging.


## Instructions

All required packages can be found in `requirements.txt`. For system specific instructions for installing DGL, please see the [DGL install guide](https://www.dgl.ai/pages/start.html). 

The scripts can be run from the command line. The example below will train and evaluate one iteration of our final model.

```bash
git clone https://github.com/guptaviha/GF-OLD.git && cd GF-OLD
python train_joint.py -bs=32 -lr_other=1e-5 -lr_gat=1e-2 -ep=20 -dr=0.5 -ad=0.1 -hs=768 --model=jointv2_twitter_roberta \
--clip --cuda=1 --num-trials=1
```

The various different models available to evaluate are: ```gat```, ```gatv2```,```bert```, ```roberta```, ```twitter_roberta```, ```joint```, ```joint_roberta```, ```joint_twitter_roberta```, ```jointv2```,```jointv2_roberta```, ```jointv2_twitter_roberta```

## Outputs

The training script logs information to standard output and saves training and test metrics to a `.json` file. The format of that file is `{trial_num: {train_loss, test_loss, train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1, best_train_f1, best_test_f1}}`
