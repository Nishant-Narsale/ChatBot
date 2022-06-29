import nltk
# if you're running tokenizing first time you might uncomment below line
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    bag = []
    for word in all_words:
        if word in tokenized_sentence:
            index = 1
        else:
            index = 0
        bag.append(index)
    return bag

# a = tokenize("Hello, thanks for visiting")
# print(a)
# a = [stem(w) for w in a]
# print(a)