import nltk

# This will print the path to the local punkt and stopwords if already downloaded
print("Punkt path:", nltk.data.find("tokenizers/punkt"))
print("Stopwords path:", nltk.data.find("corpora/stopwords"))
