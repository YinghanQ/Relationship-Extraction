import re
import os
import torch
import nltk
import numpy as np
from config import Config
from nltk.tokenize import word_tokenize
from utils import WordEmbeddingLoader, RelationLoader
from model.att_blstm import Att_BLSTM
from model.blstm import BLSTM
from model.blstm_cnn import BLSTM_CNN
from model.multi_att_blstm import Multi_Att_BLSTM
from model.att_blstm_cnn import Att_BLSTM_CNN
nltk.download('punkt_tab')
# Function to preprocess a sentence by extracting and labeling entities
def sentence_preprocess(sentence):
    """
    This function extracts the two marked entities from the input sentence
    and processes the sentence for tokenization while preserving entity labels.

    Steps:
    1. Extract the words inside the <e1>...</e1> and <e2>...</e2> tags.
    2. Add spaces around the entity markers to prevent tokenization errors.
    3. Tokenize the sentence into words.
    4. Restore the original entity tag format.
    5. Return the processed sentence along with the extracted entity words.
    """
    e1 = re.findall(r'<e1>(.*?)</e1>', sentence)[0]  # Extract the first entity
    e2 = re.findall(r'<e2>(.*?)</e2>', sentence)[0]  # Extract the second entity
    
    # Add spaces around entity markers to prevent tokenization from breaking them
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)

    # Tokenize the sentence
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)

    # Restore the original entity tag format after tokenization
    sentence = sentence.replace('< e1 >', '<e1>').replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< e2 >', '<e2>').replace('< /e2 >', '</e2>')

    # Return the tokenized sentence along with the extracted entities
    return sentence, e1, e2

# Function to encode a sentence into numerical format, ensuring compatibility with training data
def encode_sentence(sentence, word2id, max_len):
    """
    Converts a sentence into a sequence of word indices based on a given vocabulary.

    Steps:
    1. Tokenize the sentence into words (convert to lowercase for consistency with training data).
    2. Replace each word with its corresponding index in the word2id dictionary.
       - If a word is not found, use the 'UNK' (unknown) token.
    3. Create a mask to indicate valid tokens (1 for valid words, 0 for padding).
    4. If the sentence is shorter than max_len, pad it with 'PAD' tokens.
    5. If the sentence is longer than max_len, truncate it.
    6. Convert the tokenized and padded sentence into a PyTorch tensor.
    
    Returns:
    - A tensor of shape (1, 2, max_len), where:
      - First row represents token indices.
      - Second row represents the mask indicating valid tokens.
    """
    tokens = [word2id.get(word.lower(), word2id['UNK']) for word in sentence.split()]
    mask = [1] * len(tokens)

    # Padding or truncation to match max_len
    if len(tokens) < max_len:
        # If the sentence is shorter than max_len, add padding tokens
        padding = [word2id['PAD']] * (max_len - len(tokens))
        tokens += padding
        mask += [0] * (max_len - len(mask))
    else:
        # If the sentence is longer than max_len, truncate it
        tokens = tokens[:max_len]
        mask = mask[:max_len]

    # Convert to NumPy array and reshape to match training data format
    data = np.asarray([tokens, mask], dtype=np.int64)
    data = np.reshape(data, newshape=(1, 2, max_len))

    # Convert NumPy array to PyTorch tensor
    return torch.from_numpy(data)

# Function to predict the relation between two entities in a given sentence
def predict_relation(sentence, model, relation_map, config):
    """
    Predicts the relation between two entities in a given sentence.

    Steps:
    1. Load the pre-trained model's weights.
    2. Move the model to the configured device (CPU or GPU).
    3. Encode the input sentence to match the model's expected format.
    4. Perform inference (disable gradient computation for efficiency).
    5. Obtain the predicted class label and map it to a relation name.

    Returns:
    - The predicted relation label (or "Unknown" if the label is not in relation_map).
    """
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'model.pkl')))  # Load model weights
    model.to(config.device)  # Move model to CPU/GPU
    model.eval()  # Set the model to evaluation mode

    # Encode the sentence and move it to the appropriate device
    sentence = encode_sentence(sentence, word2id, config.max_len).to(config.device)

    # Perform inference without gradient computation
    with torch.no_grad():
        logits = model(sentence)  # Get model output (logits)
        predicted_label = torch.argmax(logits, dim=1).item()  # Get the class with the highest probability

    # Return the corresponding relation name
    return relation_map.get(predicted_label, "Unknown")


if __name__ == '__main__':
    # Load model and relation mappings
    print('--------------------------------------')
    print('Starting data loading process...')
    config = Config()

    # Load word embeddings and relation mappings
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    relation_file = os.path.join(config.data_dir, 'relation2id.txt')

    # Load the appropriate model based on configuration
    if config.model_name == 'BLSTM':
        model = BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Att_BLSTM':
        model = Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'BLSTM_CNN':
        model = BLSTM_CNN(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Multi_Att_BLSTM':
        model = Multi_Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    elif config.model_name == 'Att_BLSTM_CNN':
        model = Att_BLSTM_CNN(word_vec=word_vec, class_num=class_num, config=config)
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")  # Raise an error if the model name is invalid

    model.eval()  # Set model to evaluation mode
    print('Data loading complete!')
    print('--------------------------------------')

    while True:
        # Prompt user for input sentence
        input_sentence = input(
            "********** Please enter a sentence containing <e1> and <e2> tags (type 'break' to exit) **********\n"
            "Example: The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.\n\n"
            "Test sentence: "
        )
        if input_sentence.lower() == 'break':
            print("Exiting the program. Goodbye!")
            break  # Exit the program if the user types 'break'

        # Parse the input sentence
        try:
            sentence, e1, e2 = sentence_preprocess(input_sentence)
            print(f"Extracted entities: e1 = {e1}, e2 = {e2}")  # Display extracted entities
        except IndexError:
            print("Input format error. Please ensure the sentence contains both <e1> and <e2> tags.")
            continue  # Skip to the next iteration if there is an error in the input format

        # Predict the relation and display the result
        relation = predict_relation(sentence, model, id2rel, config)
        print(f"Predicted relation: {relation}\n\n")
