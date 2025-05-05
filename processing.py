from typing import List
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
from rnn import *
from collections import deque

def get_corpus_raw(filename: str) -> List[str]:
    with open(filename) as f:
        return f.read().splitlines()


def load_corpus(corpus_text: List[str]) -> List[Doc]:
    # returns docs
    docs = []
    for text in corpus_text:
        docs.append(nlp(text))
    return docs

def embed_corpus(corpus: List[Doc]):
    # token embedding, pos tag, pos-1 tag
    label_encoder = create_label_encoder(corpus)
    pos_encoder = create_pos_label_encoder(corpus)

    return label_encoder, pos_encoder


def organize_prepped_data(prepped_data):
    x = np.vstack([row[0] for row in prepped_data])
    y = np.vstack([row[1] for row in prepped_data])
    return x, y


def prep_data_for_model(corpus: List[Doc], label_encoder: LabelEncoder, pos_encoder: DictVectorizer, min_dim):
    prepped_data = []
    # x = [ token | pos - 1]        y = [token + 1 label]
    doc_count = 0
    for doc in corpus:
        if doc_count % 10 == 0:
            print(".", end='')
        if doc_count % 100 == 0:
            print("")  # create new line
        prep_doc_for_model(doc, label_encoder, pos_encoder, prepped_data, min_dim)
        doc_count += 1

    return organize_prepped_data(prepped_data)

def make_timesteps(x, num_steps):
    nptimesteps = []
    for i in range(len(x) - num_steps):
        # next_step = np.roll(x, i, axis=0)
        nptimesteps.append(x[i:i + num_steps])
    steps = np.array(nptimesteps)
    return steps

def prep_doc_for_model(doc, label_encoder, pos_encoder, prepped_data, min_dim):
    # maybe add queue and pop left, add right
    # add tokens as <S>'s
    queue = fill_prev_pos_queue(min_dim)

    for tkn_idx in range(1, len(doc) - 1):
        # refactor to get observation vector as inputting tokens
        x = create_x_observation_vector(pos_encoder, doc[tkn_idx], doc[tkn_idx - 1], queue)
        t_plus_1_label_embedded = create_y_label(doc, label_encoder, tkn_idx + 1)
        prepped_data.append((x, t_plus_1_label_embedded))

        queue.popleft()
        queue.append(doc[tkn_idx])


def fill_prev_pos_queue(min_dim):
    queue = deque([])
    s_token = nlp("<S>")[0]
    for i in range(min_dim):
        queue.append(s_token)
    return queue


def create_x_observation_vector(pos_encoder, curr_token, prev_token, queue):
    token_embedded = curr_token.vector
    pos_m_1_embedded = pos_encoder.transform({prev_token.pos_: 1})[0]
    queue_pos = []
    for q in queue:
        queue_pos.append(pos_encoder.transform({q.pos_: 1})[0])
    queue_pos = np.array(queue_pos).reshape(len(queue_pos[0]) * len(queue_pos))
    x = np.concatenate([token_embedded, pos_m_1_embedded])
    x = np.concatenate([x, queue_pos])
    return x


def create_y_label(doc, label_encoder, tkn_idx):
    t_plus_1_label_embedded = label_encoder.transform([doc[tkn_idx].text])
    return t_plus_1_label_embedded


def create_label_encoder(corpus: List[Doc]) -> LabelEncoder:
    # for each document, get word and add to labels for output
    label_encoder = LabelEncoder()
    labels = {""}
    for doc in corpus:
        for token in doc:
            labels.add(token.text)
    return label_encoder.fit(list(labels))


def create_pos_label_encoder(corpus: List[Doc]) -> DictVectorizer:
    # dict vectorizer on transform({pos label: 1}) creates 1 x len(num parts of speech) row
    pos_encoder = DictVectorizer(sparse=False)
    pos_label_dicts = []
    poses = {"<S>"}
    for doc in corpus:
        for token in doc:
            poses.add(token.pos_)
    pos_label_dicts = [{label: 1} for label in poses]
    pos_encoder = pos_encoder.fit(pos_label_dicts)
    return pos_encoder



def is_input_lt_10(input_doc: Doc) -> bool:
    return len(input_doc) < 10


def fill_input_to_10(input_doc: Doc):
    num_tokens_deficient = 10 - len(input_doc)
    start_sequence = " ".join(["<S>" for i in range(num_tokens_deficient)])
    input_text = input_doc.text
    input_sequence = start_sequence + input_text
    return nlp(input_sequence)


nlp = spacy.load("en_core_web_sm")

corpus = get_corpus_raw("StardewContentsSmall.txt")
corpus_as_docs = load_corpus(corpus)

print("Embedding corpus", end='')
label_encoder, pos_encoder = embed_corpus(corpus_as_docs)
print("\n")

print("Preparing data for model", end='')
x, y = prep_data_for_model(corpus_as_docs, label_encoder, pos_encoder, 10)
x = make_timesteps(x, 10)
# get rid fo last 10 because I shifted time steps to fit the model
# might be able to change this later
y = y[-(len(y) -10):]
print("\n")

print(x.shape, y.shape)
n_dims = x.shape[2]

# x = x.reshape((x.shape[0], 5, x.shape[1]))
vocab_size = len(label_encoder.classes_)
model = create_model(n_dims, vocab_size)
model = train(model, x, y.squeeze())
model.save("model.keras")




def predict_tokens(input_text, starting_token, xdata, num_timesteps, queue):
    predictions = []
    output = []
    prev_token = starting_token
    for i in range(20):
        # create 20 new tokens,
        # 111 is num that spacy created, will change on trained text
        # xdata = xdata[-num_timesteps:].reshape(xdata.shape[0], num_timesteps, 1)
        # get best one
        # print("---")
        timestep_predictions = model.predict(xdata)
        y_pred = np.argmax(timestep_predictions[len(timestep_predictions) - 1:])  # get last timestep argmax prediction
        predictions.append(y_pred)
        # get ypred to word i.e. reverse engineer it
        word = label_encoder.inverse_transform([y_pred])[0]
        # print(f"<<:{word}:>>")
        # if str(word) == "":
        #     xdata[:, 9:, :] += np.random.normal(-0.001, 0.001, size=(1, 1, xdata.shape[2]))
        #     continue
        input_text += f" {str(word)}"
        token = nlp(input_text)[-1]
        # make word into vec
        x = create_x_observation_vector(pos_encoder, token, prev_token, queue)
        # token_embedded = token.vector
        # pos_m_1_embedded = pos_encoder.transform({prev_token.pos_: 1})[0]
        # x = np.concatenate([token_embedded, pos_m_1_embedded]).reshape(1, 1, n_dims)
        x = x.reshape(1, 1, n_dims)
        xdata = np.concatenate([xdata, x], axis=1)[:, -num_timesteps:, :]
        prev_token = token
        output.append(token.text)
        queue.popleft()
        queue.append(token)
        # get pos

        # append to xdata
    return predictions, output


def get_input_and_prep_for_model():
    input_text = "What is the maximum number of health points needed everyday"  # todo what is len < 10?
    docs = load_corpus([input_text])
    queue = fill_prev_pos_queue(10)

    xdata = []
    for tkn_idx in range(len(docs[0])):
        x = create_x_observation_vector(pos_encoder, docs[0][tkn_idx], docs[0][tkn_idx - 1], queue)
        xdata.append(x)
        queue.popleft()
        queue.append(docs[0][tkn_idx])

    xdata = np.vstack(xdata)
    print(xdata.shape)
    # number of rows
    num_timesteps = xdata.shape[0]
    last_token = docs[0][-1]
    num_timesteps = 10
    xdata = xdata[-num_timesteps:].reshape(1, num_timesteps, xdata.shape[1])
    return xdata, last_token, num_timesteps, input_text, queue


xdata, last_token, num_timesteps, input_text, queue = get_input_and_prep_for_model()
predictions, output = predict_tokens(input_text, last_token, xdata, num_timesteps, queue)

print(predictions)
print(" ".join(output))