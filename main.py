import gensim

from parsewiki import load_file_contents, get_page_tokens

if __name__ == "__main__":
    filename = "StardewContentsSmall.txt"
    contents = load_file_contents(filename)
    corpus = []
    for wiki_page in contents:
        doc = get_page_tokens(wiki_page)
        corpus.append([token.text for token in doc])
    model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
    print(model.wv['Stardew'])


