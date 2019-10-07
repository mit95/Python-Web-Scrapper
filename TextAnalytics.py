# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 23:44:50 2019

@author: mitk
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import nltk
import time
import os

#URl specification
url = "https://www.yelp.com/biz/marukame-udon-honolulu"

css_lst_pg_nmbr_lnk_slctr = ".review-pager .pagination-block .page-of-pages.arrange_unit.arrange_unit--fill"
css_review_slctr = ".review-wrapper .review-content > p"
css_rating_slctr = ".review-wrapper .review-content .biz-rating.biz-rating-large.clearfix > div > div"
css_nxt_pg_btn_slctr = ".review-pager .pagination-block .pagination-links.arrange_unit > div > div > a.next"

#for Opening browser
chromedriver = "/usr/local/bin/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get(url)

def get_pg_nmbrs():
    time.sleep(3)
    page = driver.page_source
    soup = BeautifulSoup(page)
    page_of_pages = soup.select(css_lst_pg_nmbr_lnk_slctr)[-1].get_text()
    last_page = page_of_pages.strip().split(' ')[-1]
    return last_page

#for Saving extracted info
total_pages = int(get_pg_nmbrs())
scraped_data = []
scraped_ratings = []

#it will get the reviews from the website
def extract_reviews():
    time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page)
    parent_div = soup.select(css_review_slctr)
    for child_element in parent_div:
        scraped_data.append(child_element.get_text())

#getting the ratings for the review
def extract_ratings():
    page = driver.page_source
    soup = BeautifulSoup(page)
    rating_div = soup.select(css_rating_slctr)
    for rating in rating_div:
        rating_in_title = rating["title"]
        rating_str = rating_in_title.strip().split(' ')[0]
        scraped_ratings.append(float(rating_str))
=
#this is for going to next page
def goToNxtPg():
    driver.find_element_by_css_selector(css_nxt_pg_btn_slctr).click()

def get_all_reviews():
    i = 0
    while i < total_pages:
        extract_reviews()
        extract_ratings()
        time.sleep(3)
        i = i + 1
        goToNxtPg()

get_all_reviews()


#all the reviews are being lemmetized here
from nltk.stem import WordNetLemmatizer

lemmatized_reviews_array = []
lemmatizer = WordNetLemmatizer()

def lemmatize_review(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        newwords = []
        for word in words:
            newwords.append(lemmatizer.lemmatize(word))
        sentences[i] = ' '.join(newwords)
    paragraph = '.'.join(sentences)
    lemmatized_reviews_array.append(paragraph)


def process_reviews():
    for data in scraped_data:
        lemmatize_review(data)

process_reviews()

#all the negative and positive words are being imported
def isNotNull(value):
    return value is not None and len(value)>0

dict_pstv = []
dict_ngtv = []
f = open('negative-words.txt','r')

for line in f:
    t= line.strip().lower();
    if (isNotNull(t)):
        dict_ngtv.append(t)
f.close()

f = open('positive-words.txt','r')
for line in f:
    t = line.strip().lower();
    if (isNotNull(t)):
        dict_pstv.append(t)
f.close()


#all the positive and negative review based on ratings are being calculated
pstv_content_rating_array = []
ngtv_content_rating_array = []

def calculate_review_rating(sentence, rating):
    if rating > 3.0 and rating <= 5.0:
        pstv_content_rating_array.append(sentence)
    elif rating < 3.0:
        ngtv_content_rating_array.append(sentence)

for i in range(len(scraped_ratings)):
    calculate_review_rating(scraped_data[i], scraped_ratings[i])


#all the positive and negative review based on sentiments are being calculated
pstv_content_sent_score_array = []
ngtv_content_sent_score_array = []
    
def calculate_sentiment(sentence):
    pos_cnt = 0
    neg_cnt = 0
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word in dict_pstv:
            pos_cnt = pos_cnt + 1
        if word in dict_ngtv:
            neg_cnt = neg_cnt + 1
    pstv_content_sent_score_array.append(pos_cnt)
	ngtv_content_sent_score_array.append(neg_cnt)
    
for data in scraped_data:
    calculate_sentiment(data)


#for Saving the separated reviews
import pandas as pd
import csv

#for creating a directory to store files
cwd = os.getcwd()
directory = cwd + '/dataset'

if not os.path.exists(directory):
    os.makedirs(directory)

write_or_create_positive_file = 'w'
if not os.path.exists(directory + '/positive_ratings.csv'):
    write_or_create_positive_file = 'x'

with open(directory + '/positive_ratings.csv', write_or_create_positive_file, newline='') as myfile:
    wr = csv.writer(myfile, dialect = 'excel', quoting=csv.QUOTE_ALL)
    for content in pstv_content_rating_array:
        wr.writerow([content])


write_or_create_negative_file = 'w'
if not os.path.exists(directory + '/negative_ratings.csv'):
    write_or_create_negative_file = 'x'

with open(directory + '/negative_ratings.csv', write_or_create_positive_file, newline='') as myfile:
    wr = csv.writer(myfile,  dialect = 'excel', quoting=csv.QUOTE_ALL)
    for neg_sent in ngtv_content_rating_array:
        wr.writerow([neg_sent])


#for creating dataset for algorithm
df = pd.DataFrame()
y_text = []
y = []
for data in scraped_data:
    if data in pstv_content_rating_array:
        y.append(1)
        y_text.append('positive')
    else:
        y.append(0)
        y_text.append('negative')
    
df['positive_score'] = pstv_content_sent_score_array
df['negative_score'] = ngtv_content_sent_score_array

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

model.fit(X_train, y_train)

predicted= model.predict(X_test)

print(confusion_matrix(y_test, predicted))

print(classification_report(y_test, predicted))

dataset = pd.DataFrame()
dataset['reviews'] = scraped_data
dataset['positive_score'] = pstv_content_sent_score_array
dataset['negative_score'] = ngtv_content_sent_score_array
dataset['result'] = y_text

dataset.to_csv('./dataset/text_analytics_dataset.csv')

df_pred = pd.DataFrame()
df_pred = X_test
df_pred['result'] = y_test
df_pred['predicted'] = predicted
df_pred.to_csv('./dataset/predicted_output_with_actual_result.csv')












