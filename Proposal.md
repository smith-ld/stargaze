# Proposal - Modeling 

Summary: Web scrape and create a simple language model.

I would like to scrape text from the Stardew Valley wiki page, train a not so large language model for text prediction. After training, when you type a sentence you can ask a question related to Stardew Valley and it will try to finish the sentence for you. 

### Methods
- Web scrape the wiki pages
- Process data, create tokens
- Create word embeddings, was thinking word2vec
- Train a neural net (sklearn or pytorch), or try bag of words/similar.


Admittedly I do not expect this to be highly performant, am curious to see the progress that I can make.
