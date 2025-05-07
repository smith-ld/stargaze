import os
import pickle
import sys

import keras
import spacy

from parsewiki import load_file_contents, get_page_tokens, parse_wiki
from processing import embed_corpus, load_corpus, get_corpus_raw, prep_data_for_model, make_timesteps, \
    get_input_and_prep_for_model, predict_tokens
from rnn import create_model, train

def parse_args(args):
    key_args = {}
    for arg in args:
        kargs = arg.split('=')
        key_args[kargs[0]] = kargs[1]
    return key_args


def model_exists(model_name):
    if os.path.realpath(os.path.dirname(__file__) + "/" + model_name):
        return True
    else:
        return False

if __name__ == "__main__":

    # assuming you have already parsed and saved the wiki
    max_articles = 10
    filename = "StardewContentsVerySmall.txt"
    args = parse_args(sys.argv[1:])

    nlp = spacy.load("en_core_web_md")

    if not args.get("use_existing_model", False):
        parse_wiki(filename, max_articles)
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

        vocab_size = len(label_encoder.classes_)
        print("Creating model", end='')
        model = create_model(n_dims, vocab_size)
        model = train(model, x, y.squeeze(), epochs=30)
        model.save("model.keras")
        print("Done")
        pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
        pickle.dump(pos_encoder, open("pos_encoder.pkl", "wb"))
        pickle.dump(n_dims, open("n_dims.pkl", "wb"))
    else:
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        pos_encoder = pickle.load(open("pos_encoder.pkl", "rb"))
        model = keras.models.load_model('model.keras')
        n_dims = pickle.load(open("n_dims.pkl", "rb"))


    input_text = input("Input text: ")
    while not input_text.lower().startswith("quit"):
        xdata, last_token, num_timesteps, input_text, queue = get_input_and_prep_for_model(pos_encoder, nlp, input_text)
        predictions, output = predict_tokens(input_text, last_token, xdata, num_timesteps, queue, model,
                                             pos_encoder, label_encoder, n_dims, nlp)

        # print(predictions)
        print(" ".join(output))
        input_text = input("Input text: ")

