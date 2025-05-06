import os
import sys

import gensim
import spacy

from parsewiki import load_file_contents, get_page_tokens, parse_wiki
from processing import embed_corpus, load_corpus, get_corpus_raw, prep_data_for_model, make_timesteps, \
    get_input_and_prep_for_model, predict_tokens
from rnn import create_model, train

def parse_args(args):
    key_args = {}
    for arg in args:
        pass

if __name__ == "__main__":

    # assuming you have already parsed and saved the wiki
    max_articles = 10
    filename = "StardewContentVerySmall.txt"

    # if filename not in os.listdir(os.getcwd()):
    #     parse_wiki(filename, max_articles)

    nlp = spacy.load("en_core_web_sm")

    corpus = get_corpus_raw(filename)
    corpus_as_docs = load_corpus(corpus, nlp)

    print("Embedding corpus", end='')
    label_encoder, pos_encoder = embed_corpus(corpus_as_docs)
    print("\n")

    print("Preparing data for model", end='')
    x, y = prep_data_for_model(corpus_as_docs, label_encoder, pos_encoder, 10, nlp)
    x = make_timesteps(x, 10)
    # get rid fo last 10 because I shifted time steps to fit the model
    # might be able to change this later
    y = y[-(len(y) - 10):]
    print("\n")

    print(x.shape, y.shape)
    n_dims = x.shape[2]

    # x = x.reshape((x.shape[0], 5, x.shape[1]))
    vocab_size = len(label_encoder.classes_)
    model = create_model(n_dims, vocab_size)
    model = train(model, x, y.squeeze())
    model.save("model.keras")

    # contents = load_file_contents(filename)
    # corpus = []
    # for wiki_page in contents:
    #     doc = get_page_tokens(wiki_page)
    #     corpus.append([token.text for token in doc])
    # model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
    # print(model.wv['Stardew'])
    xdata, last_token, num_timesteps, input_text, queue = get_input_and_prep_for_model(pos_encoder, nlp)
    predictions, output = predict_tokens(input_text, last_token, xdata, num_timesteps, queue, model,
                                         pos_encoder, label_encoder, n_dims, nlp)

    print(predictions)
    print(" ".join(output))

