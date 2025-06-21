# Attention-Based BiLSTM-CNN (Att-BiLSTM-CNN) for Relation Extraction on SemEval-2010 Task 8

In this project, we provide an improved algorithm (**Att_BLSTM_CNN**), a baseline model (**BLSTM**), and several comparative algorithms for evaluation, including **Att_BLSTM**, **BLSTM_CNN**, and **Multi_Att_BLSTM**. 

1. [**Att_BLSTM_CNN (Our Improved Model)**](./model/att_blstm_cnn.py): Attention-based Bidirectional LSTM with Convolutional Neural Networks

2. [**BLSTM (Baseline Model)**](./model/blstm.py): Bidirectional LSTM model (Baseline model)

3. [**Att_BLSTM**](./model/att_blstm.py): Attention-based Bidirectional LSTM model 
Implementation of [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034.pdf).

4. [**BLSTM_CNN**](./model/blstm_cnn.py): Bidirectional LSTM with Convolutional Neural Networks

5. [**Multi_Att_BLSTM**](./model/multi_att_blstm.py): Multi-Attention Bidirectional LSTM model

More details can be seen by `python run.py  --model_name='model_name' -h`.

[`main.ipynb`](./main.ipynb) provides:
* Data Exploration: Visualization of the input data, including statistical summaries and graphical representations to gain insights into the dataset.
* Model Training: The code to train the model, including setting up the architecture, training loop, loss functions, and optimization strategies.
* Model Testing: Evaluation of the trained model using test data, showcasing how the model performs on previously unseen data.
* Interactive Demo: A section where users can input a new sentence to test the model’s prediction in real-time. This part helps demonstrate the model's effectiveness and performance on new, live inputs.

## Environment Requirements
* python 3.8
* pytorch 2.1.2
- Requirements for this project:
  - nltk
  - scikit-learn
  - datasets
  - pandas
  - torch


## Data
* [SemEval2010 Task8](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) [[paper](https://www.aclweb.org/anthology/S10-1006.pdf)\]
* [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/)

We program and develop our baseline model, improved model and comparative algorithms based on https://github.com/onehaitao/Att-BLSTM-relation-extraction
## Usage
1. Download the embedding [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) and decompress it into the `embedding` folder.
    * You can click [Download glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) to download the zip file and find the **glove.6B.100d.txt** file.
2. After completing the above operations, you can run data visualization, model training, testing, and demo execution in [`main.ipynb`](./main.ipynb).
    * **Note:** [`main.ipynb`](./main.ipynb)provides a detailed overview of data visualization and operations, including instructions on how to call different models, train and test the model, and use the Prediction Demo to input a new sentence for testing.\

    <span style="color:red">Before using [`main.ipynb`](./main.ipynb), you need to perform the following operations. </span>You have the following two options:

    1. **Using a Pretrained Model**: You can download all trained model data and experimental results from our SemEval2010 Task 8 dataset via [output.zip](https://drive.google.com/file/d/1cPhTx1i0uwbxMFFd2YxnIkNQbC_ik461/view?usp=sharing) from Google Drive. After unzip the file, place the `output` folder in the `ATT_BLSTM_CNN_relation_extraction` directory, replacing the existing [`output`](./output) folder if it already exists. This allows you to directly use the pretrained models for inference and testing in the `3. Model Testing` and `4. Prediction Demo` sections of [`main.ipynb`](./main.ipynb).  

    2. **Training a New Model**: If you prefer to train a new model from scratch, ensure that the [`output`](./output) folder already exists in the `ATT_BLSTM_CNN_relation_extraction` directory. Then, run the training module `2. Model Training` in `main.ipynb`, where examples are already provided.

### [output.zip](https://drive.google.com/file/d/1cPhTx1i0uwbxMFFd2YxnIkNQbC_ik461/view?usp=sharing) is composed of:
The `output.zip` file contains results organized by model directories. Each model's directory includes the following files:  

- **`model.pkl`**: The pre-trained model file.  
- **`Config.json`**: A file storing the model's training parameter settings.  
- **Training Performance Metrics**: A file with performance metrics recorded during training.  
- **Test Performance Metrics**: A file containing evaluation results on the test set.  
- **`predicted_result.txt`**: A text file with the predicted labels.  

Each model has its own subdirectory in `output.zip`, containing these respective files.



## Code Structure
The following is a summary of the content and function of each file and folder in the `Att-BLSTM-relation-extraction` project:
### Code



#### [**att_blstm_cnn.py (Our Improved Model)**](./model/att_blstm_cnn.py): Attention-based Bidirectional LSTM with Convolutional Neural Networks

#### [**blstm.py (Baseline Model)**](./model/blstm.py): Bidirectional LSTM model (Baseline model)

#### [**att_blstm.py**](./model/att_blstm.py): Attention-based Bidirectional LSTM model  

#### [**blstm_cnn.py**](./model/blstm_cnn.py): Bidirectional LSTM with Convolutional Neural Networks

#### [**multi_att_blstm.py**](./model/multi_att_blstm.py): Multi-Attention Bidirectional LSTM model

Here is a markdown for `run.py` and `evaluate.py` with descriptions similar to the one you provided for `config.py` and `utils.py`.

---

#### run.py
- **Content**: Contains the main script for training, optimizing, and testing the model. It initializes the necessary components, including data loaders, model architecture, and optimization strategy. This file handles the training process, evaluates performance on validation datasets, and saves the best model weights during training.


#### evaluate.py
- **Content**: Evaluates the performance of the trained model using various metrics such as accuracy, precision, recall, F1-score, and any other domain-specific performance measures. 

#### `config.py`
- **Content**: Defines the `Config` class for configuring various parameters of the model, including data directory, output directory, word embedding path, training settings (such as learning rate, batch size, number of iterations, etc.), random seed, etc. It also provides command line parameter parsing and configuration backup functions.


#### `utils.py`
- **Content**: Contains some auxiliary functions and classes, such as the `__load_data` function for loading data, the `__init__` function for initializing relevant parameters, and the `__symbolize_sentence` function which may be used for symbolizing sentences.

#### [`main.ipynb`](./main.ipynb) provides:
* Data Exploration: Visualization of the input data, including statistical summaries and graphical representations to gain insights into the dataset.
* Model Training: The code to train the model, including setting up the architecture, training loop, loss functions, and optimization strategies.
* Model Testing: Evaluation of the trained model using test data, showcasing how the model performs on previously unseen data.
* Interactive Demo: A section where users can input a new sentence to test the model’s prediction in real-time. This part helps demonstrate the model's effectiveness and performance on new, live inputs.

### Files

#### `predicted_result.txt` in ./output/model_name
- **Content**: Stores the relation extraction results predicted by the model for use by the evaluation script.


#### `train.log`
- **Content**: Records the log information during the model training process, such as training loss and accuracy.

### Folders

#### `data`
Stores and processes the data sets required by the project to provide data support for model training and testing
- **Content**: Contains files related to data processing, such as `README.md` which introduces the usage method of the data, [`Data_Preprocess.ipynb`](./data/Data_Preprocess.ipynb) which may be used for data preprocessing, [`relation2id.txt`](./data/relation2id.txt) which stores the mapping from relations to IDs, and [`test.json`](./data/test.json), [`validation.json`](./data/validation.json) and [`train.json`](./data/train.json) which store the test data and training data respectively.


#### `embedding`
- **Content**: Stores pre-trained word embedding files to provide word vector representations for the model and improve its performance. (such as `glove.6B.100d.txt`).


## Result

| Model                  | Macro F1  |
|------------------------|----------|
| Att_BiLSTM_CNN (our improved model) | **0.830207** |
| BiLSTM (Baseline)       | 0.818777  |
| Att_BiLSTM            | 0.822220  |
| BiLSTM_CNN           | 0.828734  |
| Multi_Att_BiLSTM     | 0.818851  |

The training log of our improved model can be seen in `train.log`.

*Note*:
* Some settings may be different from those mentioned in the paper.
* No validation set used during training.


## Reference Link
[1] O. Haitao, "Att-BLSTM-relation-extraction," GitHub repository, 2025. [Online]. Available: https://github.com/onehaitao/Att-BLSTM-relation-extraction.

[2] Zhang S, Zheng D, Hu X, et al. Bidirectional long short-term memory networks for relation classification[C]//Proceedings of the 29th Pacific Asia conference on language, information and computation. 2015: 73-78.

[3] Zhou P, Shi W, Tian J, et al. Attention-based bidirectional long short-term memory networks for relation classification[C]//Proceedings of the 54th annual meeting of the association for computational linguistics (volume 2: Short papers). 2016: 207-212.

[4] Hendrickx I, Kim S N, Kozareva Z, et al. Semeval-2010 task 8: Multi-way classification of semantic relations between pairs of nominals[J]. arXiv preprint arXiv:1911.10422, 2019.
