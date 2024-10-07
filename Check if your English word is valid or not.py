import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load a list of words 
import nltk
nltk.download('words')
from nltk.corpus import words

word_list = words.words()

# Create a DataFrame with words and labels
df = pd.DataFrame({'word': word_list, 'label': 1})

# Create negative samples (non-words)
from random import choice, sample
import string

def generate_nonwords(word_list, num_samples=1000, length=5):
    nonwords = set()
    while len(nonwords) < num_samples:
        letters = ''.join(sample(string.ascii_lowercase, length))
        if letters not in word_list:
            nonwords.add(letters)
    return list(nonwords)

nonwords_list = generate_nonwords(word_list, 1000, 5)
df_nonwords = pd.DataFrame({'word': nonwords_list, 'label': 0})

# Combine and shuffle the dataset
df = pd.concat([df, df_nonwords]).sample(frac=1).reset_index(drop=True)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))  # Use character-level n-grams
X = vectorizer.fit_transform(df['word'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
def is_valid_word(word):
    word_vector = vectorizer.transform([word])
    return model.predict(word_vector)[0] == 1

!pip install python-Levenshtein  # Install the Levenshtein package
import nltk
nltk.download('words')
from nltk.corpus import words
import Levenshtein # Import after installation

english_words = set(words.words())

def is_valid_word(word):
    return word.lower() in english_words

def find_nearest_valid_word(word):
    min_distance = float('inf')
    nearest_word = None

    for valid_word in english_words:
        distance = Levenshtein.distance(word.lower(), valid_word)
        if distance < min_distance:
            min_distance = distance
            nearest_word = valid_word

    return nearest_word

while True:
    user_input = input("Enter a word (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    if is_valid_word(user_input):
        print("The word is valid")
    else:
        nearest_word = find_nearest_valid_word(user_input)
        print(f"The word is not valid. Did you mean '{nearest_word}'?")
