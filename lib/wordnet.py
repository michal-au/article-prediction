from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()
tag_dict = {"N": wn.NOUN, "V": wn.VERB, "R": wn.ADV, "J": wn.ADJ}


def lemmatize(word, tag=""):
    # unifying all the numbers into one word
    try:
        float(word)
        return "<number>"
    except ValueError:
        pass


    if not tag:
        return word
    if tag[0] not in tag_dict:
        return word
    else:
        return lemmatizer.lemmatize(word, tag_dict[tag[0]])


def hypernyms(word):
    if len(wn.synsets(word, wn.NOUN)) > 0:
        return [s.name() for s in wn.synsets(word, wn.NOUN)[0].hypernyms()]
    else:
        return None
