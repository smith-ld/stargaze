from gensim.models import Word2Vec

def save(model, model_name):
    model.save(model_name)


def load_vectors(model_name):
    try:
        model = Word2Vec.load(model_name)
        return model.wv
    except:
        raise Exception("Model name not found")
