Check out the app for this project <a href="https://myers-briggs-nlp.herokuapp.com/" target="_blank">here</a> deployed on Heroku.


### Metis Project 3: Classification, NLP, and Web Apps

# Myer's Brigg Classification: by Brian Tam

## Context

Myers Briggs is a popular personality test that attempts to classify and describe people's unique ways of interacting with the world, making decisions, and process information. There are a total of 16 personalities based on 4 personality traits: 1st introversion or extroversion, 2nd intuitive or sensing, 3rd feeling or thinking, 4th judging or perceiving. Each of the 16 personalities have a unique sequence of 8 functions that describe their cognitive tendencies.

One's myers briggs is usually determined by taking a long online test. Such tests are often subject to confirmation bias as people select answers based on their existing perception of themselves. Additionally, many are unsure about their responses as they fill out these questionaires. Turns out, we do not know as much about ourselves as we think.

Thus, I attempt to use machine learning to learn and predict myers briggs based on something more reliable: people's writing.

## Goal

The goal of this project is to use machine learning classification methods to determine which of 16 personality types a person most closely falls into, based solely on their writing, to simplify the MBTI classification process. This is a multi-class problem; there are 16 different possibilities, but to simplify our model concept, we are going to split our problem into 4 seperate binary classifications, one for each letter.

## Methodologies
- This following dataset was used [MBTI kaggle dataset](https://www.kaggle.com/datasnaek/mbti-type)
- Imported and exported data as SQL tables using d6tstack
#### Data_Cleaning.ipynb
1. Preprocess so that each row is consists of all the posts from an individual person
2. Filter out links
3. Use spaCy to tolkenize, lemmetize, and remove stop words 
4. Perform EDA to understand class distribution and nature of texts
#### NLP.ipynb
5. Performed Train/Test/Val split to leave samples for testing my model
6. converted my text corpus into a Document-term matrix using TF-IDF vectorizer
7. Tested 6 different classification models.
8. Topic Modeled using Non-Negative Matrix Factorization
9. Labeling the Topic Matrix with the top 3 words in every topic for interpretation.
#### More_NLP.ipynb
10. Incoperated Vader sentiment Analysis into my model
11. Parts of speech counting into my model
#### Modeling
12. Address class imbalance with randomOverSampler
13. Created a pipline to run the following models:
    - Logistic Reg
    - Decision Tree
    - Random Forest
    - BernoulliNB
    - MultinomialNB
    - SVC
    - XGBoost
14. Created 4 seperate models to classify each letter
#### Streamlit & heroku 
15. Built an [app](https://myers-briggs-nlp.herokuapp.com/) that takes in a block of text and predicts the author's myer's Briggs
#### Tableau
16. Created interactive vizualization of the most frequent topics for each personality type. Published on [my tableau public](https://public.tableau.com/profile/bgood2me#!/vizhome/MyerBriggsTopics/Dashboard1?publish=yes)

## Findings and Conclusions

My four models had accuracy scores that ranged from 77% to 90%, improving the likelyhood of predicting someone's EXACT myers briggs from 6% to 50%, predicting the exact four letters of multiple participants who visted my [site](https://myers-briggs-nlp.herokuapp.com/).

## Deliverables

Sequentially:

- [Data Collection](https://github.com/anterra/yoga-class-ifying/tree/master/data_collection)
- [EDA and Feature Engineering](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/eda_feature_engineering.ipynb)
- [Pipeline Modules](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/pipeline_modules.py)
- [Classification Modeling](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/classification_modeling.ipynb)
- [Flask App](https://github.com/anterra/yoga-class-ifying/tree/master/flask_app)
- [Presentation Slides](https://github.com/anterra/yoga-class-ifying/blob/master/presentation/Yoga%20Classification.pdf)

## Project Team

- [Brian Tam](https://www.linkedin.com/in/brianhtam/)

## Technologies Used

- spaCy
- Tableau
- Vader
- gensim
- regex
- Jupyter Notebook
- Python
- Scikit-learn
- Streamlit
- Heroku
- HTML/CSS
- Pandas
- Matplotlib
- Seaborn


## Approaches and Skills

**NLP**
- TF-IDF
- Topic Modeling
- Dimensionality reduction
- sentiment analysis
- POS tagging

**Classification Algorithms**:
- Logistic Reg
- Decision Tree
- Random Forest
- BernoulliNB
- MultinomialNB
- SVC
- XGBoost


**Other**

- Classification Scoring Metrics
- Precision-Recall Trade-off
- EDA
- Tableau
