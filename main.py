import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
def preprocess(data):
    processed_features=[]
    for i in range(0,len(data)):
        processed_feature=re.sub(r'\W',' ',data[i])
        processed_feature=re.sub(r'^\s+',' ',processed_feature)
        processed_feature=re.sub(r'[0-9]',' ',processed_feature)
        processed_feature=re.sub(r'\s+[a-zA-Z]\s+',' ',processed_feature)
        processed_feature=re.sub(r'\s+',' ',processed_feature)
        processed_feature=processed_feature.lower()
        #processed_features.append(processed_feature)
        processed_feature=processed_feature.split()
        processed_feature= [WordNetLemmatizer().lemmatize(word) for word in processed_feature if not word in set(stopwords.words('english'))]
        processed_feature=" ".join(processed_feature)
        processed_features.append(processed_feature)
    
    #print(processed_features[0].shape)
    return processed_features

class Model:
    def __init__(self, datafile = "airline_sentiment_analysis.csv"):
        """
        Initialize the class
        """
        self.df = pd.read_csv(datafile)
        del self.df['Unnamed: 0']
        self.X=list(self.df['text'])
        self.y=self.df.iloc[:, 0].values
        self.modelsvr = svm.LinearSVC(class_weight='balanced',max_iter=2000)
        self.clf = GridSearchCV(self.modelsvr,{ 'C' : [1,5,10] },cv=5)
    
    def split(self, test_size):
        """
        Spiliting the dataset
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = 0)    
    
    def fit(self):
        """
        training the model
        """
        self.model = self.clf.fit(self.X_train, self.y_train)
    
    def predict(self, input_value=None):
        """
        prediction of y
        """
        if input_value == None:
            result = self.clf.predict(self.X_test)
        else: 
            result = self.clf.predict(np.array([input_values]))
        return result

    def save_model(self,path="finalized_model.hdf5"):
        """
        saving the model
        """
        with open(path, "wb") as open_file:
             pickle.dump(self.clf, open_file)    

if __name__ == '__main__':
    model = Model()
    processed_features=preprocess(model.X)
    vectorizer = TfidfVectorizer (max_features=2500, min_df=1, max_df=1.0)
    processed_features = vectorizer.fit_transform(processed_features).toarray()
    pickle.dump(vectorizer, open("tfidf.pickle", "wb"))
    model.X=processed_features
    model.split(0.2)
    model.fit()
    predictions=model.predict()
    #print(confusion_matrix(y_test, predictions))
    model.save_model()
    print(classification_report(model.y_test, predictions))    