import sklearn
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#Loading in dataset

moviedir = r'C:\Users\Ludvig Damberg\Desktop\Filer för plugg\TNM108\Labb 4\movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

print(len(movie.data))

print(movie.target_names)

#split data into training and test sets

docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.5, random_state = 12)


#Building a pipeline

text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf',SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])


#Train the SVM classifier model

text_clf.fit(docs_train, y_train)
# training SVM classifier
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == y_train))

print(metrics.classification_report(y_train, predicted,target_names=movie.target_names))

#Confusion matrix 
print(metrics.confusion_matrix(y_train, predicted))

#Parameter Tuning using grid search

parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(docs_train[:600], y_train[:600])


#Fake movie reviews

reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']


#Get best score and params from one review

print('\nBest score:',gs_clf.best_score_,'\n')

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    #Predict on estimator wiht the best found parameters

    pred = gs_clf.predict(reviews_new)

    # Print results of comments
print('\n')
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))