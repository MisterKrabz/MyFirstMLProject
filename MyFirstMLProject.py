from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class article: 
    def __init__(self, title, sent):
        self.title = title
        self.sent = sent

file_path = "./data/all-data.csv"

sentences = []
sentiments = []
data_lines = []

# stores article objects 
articles = []
print(len(articles))

try:
    with open(file_path, encoding='latin1') as f:
        data_lines = f.readlines()
        print(f"Successfully read {len(data_lines)} lines from {file_path}")

except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found. Please check the file path.")

for line in data_lines:
    parts = line.strip().split(',', 1)
    
    if len(parts) == 2:
        sentiment = parts[0]
        sentence = parts[1].strip(' "') 
        
        sentiments.append(sentiment)
        sentences.append(sentence)

        articles.append(article(sentence, sentiment))
        
file_path1 = "./data/sentences_AllAgree.txt"

data_lines1 = []

try:
    with open(file_path1, encoding='utf-8') as f:
        data_lines1 = f.readlines()
        print(f"Successfully read {len(data_lines1)} lines from {file_path1}")
except IOError as e:
    print(f"ERROR: An error occurred while writing to the file '{file_path1}': {e}")

for line in data_lines1: 
    parts = line.strip().split('.@', 1)

    if len(parts) == 2:
        sentiment = parts[1]
        sentence = parts[0]
        
        sentiments.append(sentiment)
        sentences.append(sentence)

        articles.append(article(sentence, sentiment))
        

# training the model
headline = [x.title for x in articles]
sentiment= [x.sent for x in articles]

vectorizer = CountVectorizer()
headline_vectors = vectorizer.fit_transform(headline)

clf_log = LogisticRegression(max_iter=1000)
clf_log.fit(headline_vectors, sentiment)

# prediction function 
def predictSentiment(headline):
    headline_vector = vectorizer.transform([headline])
    prediction = clf_log.predict(headline_vector)
    return prediction[0]