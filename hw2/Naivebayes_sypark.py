import os
import math
import pickle
import fnmatch
import nltk
import sys
from os import listdir
from os.path import isfile, join
from nltk.tokenize import wordpunct_tokenize

classes = ['spam','ham']
path = os.getcwd()
training_ham_path = path + '/train/ham/'
training_spam_path = path + '/train/spam/'
test_ham_path = path + '/test/ham/'
test_spam_path = path + '/test/spam/'

spam_train_size = len(fnmatch.filter(os.listdir(training_spam_path), '*.txt'))
ham_train_size = len(fnmatch.filter(os.listdir(training_ham_path), '*.txt'))
spam_test_size = len(fnmatch.filter(os.listdir(test_spam_path), '*.txt'))
ham_test_size = len(fnmatch.filter(os.listdir(test_ham_path), '*.txt'))
total_size = spam_train_size + ham_train_size

def simple_probablity(Nc,N):
    return Nc/N

ham_prior = simple_probablity(ham_train_size,total_size)
spam_prior = simple_probablity(spam_train_size,total_size)

def get_words(message, flag):

    all_words = set(wordpunct_tokenize(message.replace('=\\n', '').lower()))
#    msg_words = [word for word in all_words if word not in stopwords.words() and len(word) > 2]
    if flag=='true':
        
        msg_words = [word for word in all_words if len(word) > 2 and word not in stop_words] 
    else:
        msg_words = [word for word in all_words if len(word) > 2]   
    return msg_words
    
def get_mail_from_file(file_name):

    message = ''
    
    with open(file_name, encoding="latin-1") as mail_file:
        
        for line in mail_file:
            for line in mail_file:
                message += line
                    
    return message
    
stop_words_message = get_mail_from_file(path+'/' +'stop_words_list.txt')
stop_words = get_words(stop_words_message, 'false')   
    
def make_training_set(path, flag):

    training_set = {}
    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))] 
    for mail_name in mails_in_dir:
        message = get_mail_from_file(path + mail_name)
        terms = get_words(message, flag)
        for term in terms:
            if term in training_set:
                training_set[term] = training_set[term] + 1
            else:
                training_set[term] = 1
    return training_set

def count_total_docs(training_set):
    total_count = 0
    for term in training_set.keys():
        total_count = total_count + training_set[term]
    return total_count
def calculate_conditional_probablities(training_set, total_count, smoothing):
    for term in training_set.keys():
        training_set[term] = float(training_set[term]+1) / (total_count + smoothing)
    return training_set       

def classify(path, flag):
    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))]
    classify = {}
    for mail_name in mails_in_dir:
        spam_probablity = math.log(spam_prior)
        ham_probablity = math.log(ham_prior)
        message = get_mail_from_file(path + mail_name)
        terms = get_words(message, flag)
        for term in terms:
            if term in spam_training_set:
                spam_probablity = spam_probablity + math.log(spam_training_set[term])
            else:
                spam_probablity = spam_probablity + math.log(1/(total_spam_count + smoothing_count))
            if term in ham_training_set:
                ham_probablity = ham_probablity + math.log(ham_training_set[term])
            else:
                ham_probablity = ham_probablity + math.log(1/(total_ham_count + smoothing_count))
        
        if spam_probablity < ham_probablity:
            classify[mail_name] = 'ham'
        else:
            classify[mail_name] = 'spam'
    return classify

       
exists = os.path.isfile(path + '/ham.file')
if exists:
    with open("spam.file", "rb") as f:
        spam_training_set = pickle.load(f)
    with open("ham.file", "rb") as f:
        ham_training_set = pickle.load(f)

else:
    spam_training_set = make_training_set(training_spam_path, 'false')
    
    ham_training_set = make_training_set(training_ham_path, 'false')
   
    spam_unique_word_count = len(spam_training_set.keys())
    ham_unique_word_count = len(ham_training_set.keys())
 
    spam_set = set(spam_training_set)
    ham_set = set(ham_training_set)
    common_count = 0
    for name in spam_set.intersection(ham_set):
        common_count = common_count + 1
    smoothing_count = spam_unique_word_count + ham_unique_word_count - common_count
    
    total_spam_count = 0
    total_ham_count = 0 
    total_spam_count = count_total_docs(spam_training_set)
    total_ham_count = count_total_docs(ham_training_set)
    spam_training_set = calculate_conditional_probablities(spam_training_set,total_spam_count, smoothing_count)
    ham_training_set = calculate_conditional_probablities(ham_training_set,total_ham_count, smoothing_count)
  
    with open("spam.file", "wb") as f:
        pickle.dump(spam_training_set, f, pickle.HIGHEST_PROTOCOL)
    with open("ham.file", "wb") as f:
        pickle.dump(ham_training_set, f, pickle.HIGHEST_PROTOCOL)


ham_exists = os.path.isfile(path + '/ham_classify.file')
if ham_exists:
    with open("spam_classify.file", "rb") as f:
        classification_spam = pickle.load(f)
    with open("ham_classify.file", "rb") as f:
        classification_ham = pickle.load(f)

else:
    classification_ham = classify(test_ham_path, 'false')
    classification_spam = classify(test_spam_path, 'false')
    with open("spam_classify.file", "wb") as f:
        pickle.dump(classification_spam, f, pickle.HIGHEST_PROTOCOL)
    with open("ham_classify.file", "wb") as f:
        pickle.dump(classification_ham, f, pickle.HIGHEST_PROTOCOL)
        
def accuracy(classifications, classes):
    result = 0
    for term in classifications.keys():
        if classifications[term] == classes:
            result = result + 1
    return result
        
ham_classified_as_ham = 0
ham_classified_as_spam = 0
spam_classified_as_spam = 0
spam_classified_as_ham = 0
ham_classified_as_ham = accuracy(classification_ham, 'ham')
ham_classified_as_spam = len(classification_ham) - ham_classified_as_ham
spam_classified_as_spam = accuracy(classification_spam, 'spam')
spam_classified_as_ham = len(classification_spam) - spam_classified_as_spam

print("Test Data Accuracy on Ham dataset: ", ham_classified_as_ham/( ham_classified_as_spam + ham_classified_as_ham)*100," %")
print("Test Data Accuracy on Spam dataset: ", spam_classified_as_spam/ (spam_classified_as_ham + spam_classified_as_spam)*100," %")
