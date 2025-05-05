# stargaze


Steps
Originally I wanted to create a language model based in the proposal. 
After reviewing and testing, I adjusted my work to do what I was able to code. 
Admittedly, this is likely due to an inexperience with keras, which I have also discussed below, 
in learning improvements. 

Training

Training was completed using pythons `keras` module. I built a Recurrent Neural Network and trained
it on the wiki pages of Stardew. For this to work, I had to parse the wiki, save it, and then load 
it for the model to use. 

Tokenization was done by using the `spacy` module. I considered each document a wikipedia page. 
When storing the pages in a file, I kept one document per line, which made it easy
for parsing documents. 

Features and labels
- Each word is labeled via `labelencoder` and is a unique number
- The input vector (x) is a combination of:
  - The current token's spacy vector
  - The 10 previous parts of speech


Performance
Originally I wanted to utilize a f1 score, however encountered issues doing so with keras. 
Instead, I utilized accuracy. 

Testing on 46 documents, we find an accuracy of around 42%.

How to use it
- Currently I have not processed input as a feedback loop. That means that you would have to put\
your text sequence of what you want into code and run it
- I also have not made the ability to load a model. There are is a concern on the trainability side
of this. There are about 3,000 wiki pages; on training with around 250 of those pages, the 
training time on many epochs took a very long time. 


Issues along the way

`keras`

I found that keras has limited documentation for this specific use case. With limited 
knowledge about RNNs and text generation, it took some research to find a way to use
both RNNs and the `keras.Sequential` workflow to produce a RNN that actually worked. 

Some combinations that I tried that were unsuccessful
- Using the `model.add(layers.LSTM(3))` 
- Using more than 2 `SimpleRNN` layers. This provided me with issues on dimensionality, and I was unsure how to adjust the dimensionality to fit what the model wanted

Dimensionality
- I found this to be the biggest issue. Not the curse of dimensionality,
but rather providing the model with the correct dimensions that it needs.
I anticipated inputting a 2d vector into the sequential model. 
Each observation would represent a vectorized word, which would
then do into the RNN. However it became apparent that keras did not like that. 
Instead if required a 3 dimensional input, which from what I understand now
is a timestep. I adjusted the model to account for a 3d input.
  - Originally not knowing what a timestep was I used a timestep=1.
  I thought that looked a little weird and made no sense so I further researched why this might be used.
  - After some research, I found that it would be the previous tokens, so I adjusted
  the model to be an input of (1, # timesteps, # features). 


