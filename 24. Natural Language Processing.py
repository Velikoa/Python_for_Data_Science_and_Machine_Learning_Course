import nltk

#Need the below shell to choose which package you want to use from the nltk library.
#nltk.download_shell()

messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))
print(messages[50])

#A collection of text is called a 'corpus'.
#Print the first 10 messages and number them using enumrate.
for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)     #Printing the message number and the actual message itself separated by a line below.
    print('\n')

import pandas as pd

#Going to use pandas to read this as a csv file and split the data along each tab delimiter.
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.head())

print(messages.describe())

#Now want to see how many 'ham' versus 'spam' messages there are by groupby the 'label' column.
print(messages.groupby('label').describe())

#To identify whether a message is spam or ham it might be easier to find out the length of each message.
messages['length'] = messages['message'].apply(len)
print(messages.head())

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=150)
plt.show()

#From the graph can spot that the x-axis goes up to 1000. Below is the code to find this long message.
#Using thw describe below can see that the longest message is 910 characters.
print(messages['length'].describe())

print(messages[messages['length'] == 910]['message'].iloc[0])       #Use iloc to print out the entire string.


#Having a combined histogram/subplot with ham and spam together on same graph.
messages.hist(column='length', by='label', bins=60, figsize=(12, 4))
plt.show()

#Based on the above graphs- it appears that spam messages appear to be longer than normal legit messages.


#Part 2##############################################

#Need to convert the raw messages into vectors/sequence of numbers.

import string

mess = 'Sample message! Notice: it has punctuation.'
no_punc = [c for c in mess if c not in string.punctuation]
print(no_punc)          #Now all punctuation has been removed.

#Now going to remove stop words using the nltk library.
from nltk.corpus import stopwords


#Now the original message is back to how it started but without punctuation. Basically saying that after every item in the list you will have empty space.
no_punc = ''.join(no_punc)
print(no_punc)

#Now going to split up all the words within the list in order to be able to later identify them as stop words or not.
no_punc.split()
print(no_punc)

clean_mess = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
print(clean_mess)

#Creating a function of all the above instead of doing it one by one.
def text_process(mess):
    #1. remove punc
    #2. remove stop words
    #3. return list of clean text words

    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

#Now going to tokenise the list. i.e. taking a text string and converting it into tokens - this is a list of strings that are actually needed.
#Will use the .apply method in order to include the above function created within the overall 'messages' listing.
print(messages['message'].head(5).apply(text_process))
#What has happened here is that a list is created dof all the tokens needed - i.e. all punctuation is removed and all stopwords also removed.

#Going to use Vectorisation now.
from sklearn.feature_extraction.text import CountVectorizer

#basically going to create a very large matrix below where every single message in our original df is a column and every row is every single word in existence.
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

#Printing the total number of vocabulary words.
print(len(bow_transformer.vocabulary_))


mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])

#This will show that there are 7 unique words (after removing all the stop words) in message 4 and 2 of them appear twice.
print(bow4)
print(bow4.shape)

#How do you know which are the  unique words in message 4 which appear twice?
print(bow_transformer.get_feature_names()[4068])

####Part 3###########################

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)

#To find the amount of non-zero occurrences in a message use below code.
print(messages_bow.nnz)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

#If you want to find the tfidf for a specific word.
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

#Now going to convert the entire bag of words corpus into a tfidf corpus at once.
messages_tfidf = tfidf_transformer.transform(messages_bow)

#Use the Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

#If want to predict whether the tfidf4 message will be spam or ham.
print(spam_detect_model.predict(tfidf4)[0])

#If want to predict if all the messages are ham versus spam.
all_pred = spam_detect_model.predict(messages_tfidf)

#The correct way to actually do the above prediction however is to first split the sample into a test and train sample.
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

#Use the sklearn Pipeline feature to store the entire pipeline of data and work as opposed to having to repeat all the above steps all over again.
#When using real world data no need to do everything from the above first 3 videos - only need to build a pipeline like the one below and use that.
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(label_test, predictions))



