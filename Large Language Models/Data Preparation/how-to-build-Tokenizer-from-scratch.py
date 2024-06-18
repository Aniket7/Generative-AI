import re

#----------------------- START - How to create Vocabulary (Dictionary) -------------------------------------
# Text corpus to prepare vocabulary
corpus = """Hey, hi how are you? what are you doing?
Let's go to the market to buy groceries. We will come back by 5 PM.
"""
#corpus = corpus.lower()

# Break down the corpus into small units know as word tokens
splt_data = re.split(r'([,.:;?@"()\']|--|\s)', corpus)
#print(len(splt_data), splt_data)

# remove whitespaces from each item and then filter out empty string
split_data = [word.strip() for word in splt_data if word.strip()]
#print(len(split_data), split_data)

# adding spacial tokens to handle out of vocab word.
split_data.extend(['<|EOS|>', '<|UNK|>'])
#print(len(split_data), split_data)

# sort and get the unique words
split_data = sorted(set(split_data))

# Create the vocabulary
vocab = {word:index for index,word in enumerate(split_data)}
#print(vocab)

#-------------------------------- End of Vocab creation------------------------------

# Define Tokenizer class
class Tokenizer:
    # initialize the constructor with vocab
    def __init__(self, vocab):
        self.text_to_numbers = vocab
        # it is used to decode the token id to word
        self.numbers_to_text = {index:word for word, index in vocab.items()}

    # it generate token id for the given input
    def encode(self, input):
        split_data = re.split(r'([,.:;?@"()\']|--|\s)', input)
        preprocessed = [word.strip() for word in split_data if word.strip()]
        tokens = [w if w in self.text_to_numbers else "<|UNK|>" for w in preprocessed]
        token_id = [self.text_to_numbers[w] for w in tokens]
        return token_id

    # it converts token id back to the original word
    def decode(self, tokens):
        text = " ".join([self.numbers_to_text[t] for t in tokens])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?@"()\'])', r'\1', text)
        return text

# sample test
t1 = Tokenizer(vocab)

input_1 = "hi, how are you?"
token_ids1 = t1.encode(input_1)
print("These are the token id's ", token_ids1)
print("These is the input ", t1.decode(token_ids1))
print()

input_2 = "hi, how are you? What r u playing"
token_ids2 = t1.encode(input_2)
print("These are the token id's ", token_ids2)
print("These is the input ", t1.decode(token_ids2))

 
'''
OUTPUT
These are the token id's  [19, 1, 20, 11, 27, 6]
These is the input  hi, how are you?

These are the token id's  [19, 1, 20, 11, 27, 6, 5, 5, 5, 5]
These is the input  hi, how are you? <|UNK|> <|UNK|> <|UNK|> <|UNK|>

'''


