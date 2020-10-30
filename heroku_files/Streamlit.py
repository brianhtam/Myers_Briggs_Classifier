  
#import package
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

#preprocessing
import spacy
import re
import string

#import the data
image = Image.open("images/MBTI_people.png")

#intro
st.sidebar.write("This is an application for predicting your MBTI personality type with natural language processing!")

text = st.sidebar.text_area("Enter in someone's writting and I'll predict their Myer's Briggs")
st.sidebar.button("Predict")


Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
        .reportview-container .main .block-container{
            padding-top: 3em;
        }
        body {
            background-image:url('https://images2.alphacoders.com/692/692539.jpg');
            background-position-y: -200px;
        }
        @media (max-width: 1800px) {
            body {
                background-position-x: -500px;
            }
        }
        .Widget.stTextArea, .Widget.stTextArea textarea {
        height: 586px;
        width: 400px;
        }
        h1{
            color: brown
        }
        h2{
            color: cyan
        }
        .sidebar-content {
            width:25rem !important;
        }
        .Widget.stTextArea, .Widget.stTextArea textarea{

        }
        .sidebar.--collapsed .sidebar-content {
         margin-left: -25rem;
        }
        .streamlit-button.small-button {
        padding: .5rem 9.8rem;
        }
        .streamlit-button.primary-button {
        background-color: yellow;
        }
    </style> 
    
    <div>
        <h1>Welcome to the Myers Briggs Prediction App!</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

# Calculate our prediction
# import models
EI = pd.read_pickle('pickled_models/EI_Logistic Reg.pkl')
NS = pd.read_pickle('pickled_models/NS_Logistic Reg.pkl')
FT = pd.read_pickle('pickled_models/FT_Logistic Reg.pkl')
PJ = pd.read_pickle('pickled_models/PJ_Logistic Reg.pkl')

# import transformations
tfidf = pd.read_pickle('pickled_transformations/tfidf.pkl')
TopicModel = pd.read_pickle('pickled_transformations/NMF.pkl')
# parser = pd.read_pickle('pickled_transformations/parser.pkl')
stop_words = pd.read_pickle('pickled_transformations/stop_words.pkl')

# Load English tokenizer, tagger, parser, NER and word vectors
parser = spacy.load('en_core_web_sm')
# Create our list of punctuation marks
punctuations = string.punctuation
# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    # return preprocessed list of tokens
    return ' '.join(mytokens)
alphanumeric = lambda x: re.sub('\w*\d\w*', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())

new_text = pd.Series(text).apply(spacy_tokenizer).map(alphanumeric).map(punc_lower)
X_test_tfidf = tfidf.transform(pd.Series(new_text))

def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Takes in model and feature names and outputs 
    a list of string of the top words from each topic.
    """
    topics = []
    for ix, topic in enumerate(model.components_):
        topics.append(str(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])))
    return topics

topics = display_topics(TopicModel, tfidf.get_feature_names(), 3)
topic_word = pd.DataFrame(TopicModel.components_.round(3),
             index =  topics,
             columns = tfidf.get_feature_names())

X_test_topic_array = TopicModel.transform(pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names()))

X_test_topics = pd.DataFrame(X_test_topic_array.round(5),
             columns = topics)

pred_list = []
if EI.predict(X_test_topics) == 1:
    pred_list.append('E')
else:
    pred_list.append('I')
if NS.predict(X_test_topics) == 1:
    pred_list.append('N')
else:
    pred_list.append('S')
if FT.predict(X_test_topics) == 1:
    pred_list.append('F')
else:
    pred_list.append('T')
if PJ.predict(X_test_topics) == 1:
    pred_list.append('P')
else:
    pred_list.append('J')
prediction = ''.join(pred_list)

if text == '':
    st.image(image, use_column_width=True)
    st.markdown("<div class='title'><h1>Start by writing text on the left sidebar</h1></div>", unsafe_allow_html=True)
if text != '':
    st.header('We guess that you are:')
    predict_html = f"<div class='title'><h1>{prediction}</h1></div>"
    st.markdown(predict_html, unsafe_allow_html=True)
    st.image(f'images/{prediction}.png',width=340)
    st.header('Are we correct?')
    st.write('Find more information here:')
    st.write(f'https://www.16personalities.com/{prediction.lower()}-personality')

    try:
        # Generate WordCloud
        from wordcloud import WordCloud 

        # Generate a word cloud image
        wordcloud = WordCloud(width = 1000, height = 1000,
                        background_color ='white',
                        min_font_size = 20).generate(text)
        st.header('Word cloud of your text:',)
        # Display the generated image:
        # the matplotlib way:
        fig = plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)
    
        # fig2 = plt.figure(figsize=(10,10))
        st.header('Top 3 Topics in your writing')
        fig2 = plt.figure()
        plt.barh(X_test_topics.T.sort_values(0).tail(10).index, X_test_topics.T.sort_values(0).tail(10)[0])
        st.pyplot(fig2)

        # st.pyplot(fig2)
        st.image('Tableau_topics.gif', use_column_width=True)
        st.header('Here is your writing:')
        st.write(text)
    except ValueError:
        pass
    










