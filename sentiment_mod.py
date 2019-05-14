
#classifier predicts whether given review is positive or negative and then checks it against given one .. and we calculate accuracy

import nltk
import random
import pickle
#from nltk.corpus import movie_reviews
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk import word_tokenize

# #combining all algorithms 
class VoteClassifiers(ClassifierI):        #passing a list of classifiers 
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # def classify(self, features):
    #     votes =[]     #pos or neg list according to every classifier for a review
    #     for c in self._classifiers:
    #         v = c.classify(features)
    #         votes.append(v)
    #     return mode(votes)   #for a review...return pos or neg whichever is occuring more no. of times
 
    def classify(self, features):
        votes =[]     #pos or neg list according to every classifier for a review
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            try:
                ans = mode(votes)
            except Exception as e:
                ansl = ['pos','neg']
                ans = random.choice(ansl)
        return ans   #for a review...return pos or neg whichever is occuring more no. of times


    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(v) 
        conf = choice_votes/len(votes)
        return conf

# new words set from positive and negative tweets
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents = []
all_words = []

# for r in short_pos.split('\n'):
#     documents.append((r, "pos"))
# for r in short_neg.split('\n'):
#     documents.append((r, "neg"))

# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)

# for w in short_pos_words :
#     all_words.append(w.lower())
# for w in short_neg_words :
#     all_words.append(w.lower())

#new short method and allowing only adjectives here 
#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
# print(word_features)
#  print(all_words.most_common(25)) // will also print no. of occurences 

save_word_features = open("word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features                    #seperate review true false of all 3000 words 

featuresets =[(find_features(rev), category)for (rev, category) in documents]
random.shuffle(featuresets)

#print(featuresets)       #list of all reviews' ......words in word_features 3000 with true false ... dictionary and pos or neg

training_set = featuresets[:1900]
test_set = featuresets[1900:]

#train classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# #use saved classifier 
# classifier_f = open("naivebayes.pickle","rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

accuracy = nltk.classify.accuracy(classifier, test_set)
print(accuracy*100)
#classifier.show_most_informative_features(15)

#save classifier
# save_classifier = open("algos/naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

#open and use it  
classifier_f = open("algos/naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# all diff classifiers

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

#save classifier
# save_classifier = open("algos/MNB_classsifier.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("algos/MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()


# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)

#save classifier
# save_classifier = open("algos/Ber_classifier.pickle","wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("algos/Ber_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()


# LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='liblinear'))
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

#save classifier
# save_classifier = open("algos/log_classifier.pickle","wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("algos/log_classifer.pickle","rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()


# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

#save classifier
# save_classifier = open("algos/sgdc_classifier.pickle","wb")
# pickle.dump(SGDClassifier, save_classifier)
# save_classifier.close()

classifier_f = open("algos/sgdc_classifer.pickle","rb")
SGDC_classifier = pickle.load(classifier_f)
classifier_f.close()


# SVC_classifier = SklearnClassifier(SVC(gamma='scale'))
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

#save classifier
# save_classifier = open("algos/linear_svc.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()


# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)


# working new classifier
voted_classifier = VoteClassifiers(classifier,LinearSVC_classifier,SGDClassifier_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier)
# checking accuracy for entire test set
#print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)