# Facebook Hateful Memes Competition

This repository contains the solution of my team for the Facebook Hateful Memes Competition hosted on DrivenData.

Competition paper: https://arxiv.org/abs/2005.04790?fbclid=IwAR3hdA_-nPAM7DdZ5oBW-B48NYD-pA2aRkxDm43ljfwLrRXNkF7re_bzBaQ

DrivenData: https://www.drivendata.org/competitions/70/hateful-memes-phase-2/page/266/

Facebook AI: https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/

Our system achieves:
- Cross-Validation score over 10-fold: **0.882 AUC** and **0.813 acc**.
- Test unseen score: **0.788 AUC** and **0.745 acc**.

Leaderboard position: (X)

Team name: MemeLords

Team members:
- C.S Bahusruth
- Victor Callejas Fuentes

## Overview

The aim of this competition is to **create an algorithm that identifies multimodal hate speech in memes**.

The dataset is constructed in a way that the algorithm **must be multimodal** to excel. The dataset consists of pairs of **text-images**. 

![](https://drivendata-public-assets.s3.amazonaws.com/memes-overview.png)

The competition is a **binary classification task**, not hateful or hateful.

The metric to optimize is the **Area Under the ROC Curve**.

## Methodology
Our full methodology and findings will be released on a paper soon (maybe)

## Requirements
Python 3.7.6

### Enviroment

1. (Recommended) Create a virtual environment:
```
python -m virtualenv env
```
Enter the virtual environment.
```
source env/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

### Data

1. Download raw data

Go to the [competition page](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/data/), download and extract raw data into:

```
data/raw/
```

2. You can generate the processed data by the methods described below or download it from here:

- [processed-data](https://dd-mmhm-ml.s3.eu-west-2.amazonaws.com/processed-data.zip)

and extract it into:

```
data/
```

### Pretrained models
You can train the models or download our pretrained ones provided here:
- [ensemble](https://dd-mmhm-ml.s3.eu-west-2.amazonaws.com/ensemble.zip) (154 MB)
- [level-0](https://dd-mmhm-ml.s3.eu-west-2.amazonaws.com/level-0.zip) (22 GB)
- [VQA Multimodal checkpoints](https://dd-mmhm-ml.s3.eu-west-2.amazonaws.com/vqa-checkpoints.zip) (2 GB)

and extract into:
```
artifacts/
```
## Quick Tour

### Repository structure
#### Top level
>
    .
    ├── artifacts               # Model checkpoints with logs and predictions
    ├── data                   
    ├── src                     # Source files
    ├── notebooks               # Source files on IPython format
    ├── submissions             # Submissions to competition leaderboard
    ├── LICENSE
    ├── README.md
    └── requiremets.txt         # pip env frozen

#### Artifacts
>
    .             
    ├── ensemble                # Metaclassfier model checkpoints 
    ├── level-0                 # Level-0 models
    |   ├── DistilBert          
    |   |    ├── model          # Model and optimizer checkpoint for each fold
    |   |    ├── preds          # Test and validation predictions for each fold
    |   |    └── logs...        # Log and results for each fold
    |   └── Roberta...               
    └── MM                      # Multimodal pre-trained checkpoints from VQA repository

#### Data
>
    .
    ├── BUTD_features           # Image features extracted from Faster R-CNN 
    ├── external                # External data   
    ├── folds                   # K-Folds created
    ├── interim                 # Half-processed data
    ├── processed               # Processed data
    └── raw                     # Raw competition data
    
#### Src
>
    .
    ├── config                  # Run and model configurations
    ├── data                    # Data preprocessing to Data loaders
    ├── models                  # Model definitions
    ├── utils                   # Training and model auxiliary files
    └── train.py                # Training entry point

### Data
#### Train, validation and testing
We used a cross-validation strategy in order to choose our best level-0 models and parameters.

The folds can be generated with the script provided at:

```
notebooks/folds.ipynb
```

Once the best parameters are chosen, we perform a final training on all data [train + dev]. [Stochastic Weight Averaging (SWA)](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/) is key to avoid overfitting.

This behavior is chosen in the run configuration file.

The cross-validation strategy it's as follows:

Level 1
- Divide train data into K-folds
- Train some models
- For each model, we save validation and test features when validation AUC is higher

Level 2
- Validation saved features from each model are now train data, we use the **same folds as in level 1**
- For each K-fold we train a meta-classifier

Level 3
- Final predictions are the median of the probabilities of each meta-classifier

#### From captions
We use them as they are provided

#### From images

Image features: are features extracted from this repository [Bottom-up Attention](https://github.com/airsplay/py-bottom-up-attention), which uses a **Faster R-CNN trained on Visual Genome dataset**.
We run this on Google Colab: [notebook](https://colab.research.google.com/drive/1JgRzK2sjIsZDgAYwN7IkqG_9lMqaAxhr?usp=sharing)

Objects and attributes: image features **converted to words** using Visual Genome dictionary

Web entities: we extract these with the script:
```
cd src
python utils/web_entities.py
```

Topic: from web entities we get **internet knowledge** about them using [Duck Duck Go API](https://duckduckgo.com/api)

**All of these are combined** to generate inputs for the models in these notebooks:
```
notebooks/prepro.ipynb
notebooks/generate_vision.ipynb
```

### Models
We developed two types of base models, one that directly uses features from a pre-trained Faster R-CNN network as Uniter, Visualbert... (**Pure Multimodal**) and another type of model where we remove the lineal projection from Faster R-CNN extracted features to transformer embedding space by using directly Visual Genome Objects and Attributes dictionary(**Multimodal text**).

#### Pure Multimodal

This are models like [UNITER](https://arxiv.org/abs/1909.11740), [LXMERT](https://arxiv.org/abs/1908.07490), [VisualBert](https://arxiv.org/abs/1908.03557)...

We use this repository [Transformers-VQA](https://github.com/YIKUAN8/Transformers-VQA).

Best results are achieved with **Uniter**.

For these models, you need to provide **text and image features**. 

Text: combinations from text generated data. For example:
- captions
- captions + web entities

Image features: are features extracted from a Faster R-CNN, see section Data.

#### Multimodal Text

These are models derivated from [Bert](https://arxiv.org/abs/1810.04805).

The input of these models is just text, combinations from text generated data. For example:
- captions + frcnn objects + frcnn attributes + web entities + topic...
- captions + frcnn objects

We used the [transformers library](https://huggingface.co/transformers/) to try different transformers models, we achieved best results with:
- [Ernie](nghuyong/ernie-2.0-en)
- [Distilbert](https://huggingface.co/transformers/model_doc/distilbert.html)
- [Distilroberta](https://huggingface.co/distilroberta-base)

#### Metaclassifier
The meta classifier consists of a simple dense linear layer over the features extracted from the level-0 models.

In the case of K-fold, there will be K meta-classifiers and the final probability will be a simple median over their probabilities, this helps predictions to be robust.

Best cross-validation and test unseen scores stacking these 5 models:

- Pure Multimodal Uniter (captions and image features)
- Pure Multimodal Uniter (extended captions and image features)
- Multimodal Text Ernie (extended captions)
- Multimodal Text Distilbert (extended captions)
- Multimodal Text Distilroberta (extended captions)


### Training

#### Training level-0
You can use these models already for inference or later in an ensemble.

1. Modify configuration files
    - Modify **cfg.py**
    - Modify model configuration files **BertConfig.py** and **MMConfig.py**

2. Execute training
    ```
    cd src # Must be inside this directory 
    python train.py
    ```

Logs will be output in the terminal and saved in the artifacts folder for further inspection

#### Training ensemble or stacking

Run notebook
```
notebooks/Stacking.ipynb
```

## Competition Findings
Here we expose some learnings and tips.
#### Using directly words
Using Visual Genome labels (words) instead of features (2048 dimensional vector) allows us to **reduce the model complexity**. 

These labels are already in the same embedding space as captions, so you do not depend on the projection of the features.

There is a loss of information by doing this, but **helps with overfitting**.

#### Web entities - Transformers lack of historic or internet knowledge
Some of the examples proposed in this dataset are very difficult to classify with current approaches as **further context is needed** to make a good classification out of it.

For example, for the image with id = 16395:

- Faster R-CNN trained on Visual Genome: chin long hair face nose eyebrow hair hairstyle facial expression blond

- Web entities: Bethany hamilton

- Topic: Shark attack victims

Just with the features extracted from the Faster R-CNN, some examples can't make good predictions due to lack of context.

Most of the time the Faster R-CNN features are the best and web entities and topics fail, but the combination of this works best.

These web entities are constructed by searching for similar images on the internet.

We assert the **necessity of creating a new architecture that it's able to retain information that could be scrapped from internet knowledge**.

#### Stochastic Weight Averaging
**[SWA](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/) allow us to achieve the best results during multiple epochs**, that way we can perform final training without validation more confidently.

#### Using different models
Different models have different tokenizers and pretraining methods, so **each one of them can extract information that the others can not and vice versa**, so the **combination of them achieves best results**.

#### FP16 training
[FP16 training](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) should be the norm and not the exception. This allows us to achieve the same **results as with FP32 training but twice faster**.

#### Transformers classification head
We tried multiple classification heads for the transformers (MLPs, Convolutional layers...) but always got **best results with just one linear dense layer**, this **could be because of overfitting**.

#### Faster R-CNN dataset
Models trained on **Visual Genome works better for our task than the one trained on COCO**.


## Further work

- [ ] **Image objects relationships detection via Scene Graph**

    The current SOTA multimodal model for Image-Text Classification is [ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph](https://arxiv.org/abs/2006.16934)
    It is like the Uniter model we use but also taken into account the relationships between the objects apart from objects and attributes.

    ERNIE-ViL is only available for the PaddlePaddle Framework but a similar model could be created from Uniter or our Multimodal Text models and Scene Graph reconstruction from [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949)

- [ ] **Conterfactual Training**

    As proposed here [Counterfactual VQA: A Cause-Effect Look at
Language Bias](https://arxiv.org/abs/2006.04315)

- [ ] **Multimodal Blending loss**

    As proposed here [What Makes Training Multi-modal Classification Networks Hard?](https://arxiv.org/pdf/1905.12681.pdf)

- [ ] **Policy Based Learning**
 
    During the competition, we found that hate speech depends on the definition of this, and usually is not just about the hate sentiment expressed, it's about this **hate sentiment expressed to a protected entity**, which varies on the definition, and therefore a preprocessing to detect the protected entities on text and image could help the model to improve performance and diminish overfitting.

    We proposed a text and object detection based on a protected entities database and add them as additional tokens for the transformer.

## Related papers
Our work has been based on these papers and we recommend them to learn more about Multimodal Classification problems.

- [ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data](https://arxiv.org/abs/2001.07966)

- [What Makes Training Multi-Modal Classification Networks Hard?](https://arxiv.org/abs/1905.12681)

- [Hate Speech in Pixels: Detection of Offensive Memes towards Automatic Moderation](https://arxiv.org/abs/1910.02334)

- [Exploring Deep Multimodal Fusion of Text and Photo for Hate Speech Classification](https://research.fb.com/wp-content/uploads/2019/07/Exploring-Deep-Multimodal-Fusion-of-Text-and-Photo-for-Hate-Speech-Classification.pdf)

- [Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

- [Image–text sentiment analysis via deep multimodal attentive fusion](https://www.researchgate.net/publication/330428583_Image-text_sentiment_analysis_via_deep_multimodal_attentive_fusion)

- [Exploring Hate Speech Detection in Multimodal Publications](https://arxiv.org/abs/1910.03814)

- [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950)

## License
MIT
