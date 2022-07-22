import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import time  
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_lemmatizer = WordNetLemmatizer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

def load_df():
    with open('saved_df.pkl','rb') as file:
        data=pickle.load(file)
    return data

def clean_text(text, ):
        if text is not None:
        #exclusions = ['RE:', 'Re:', 're:']
        #exclusions = '|'.join(exclusions)
                text = re.sub(r'[0-9]+','',text)
                text =  text.lower()
                text = re.sub('re:', '', text)
                text = re.sub('-', '', text)
                text = re.sub('_', '', text)
                text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE)
        # Remove text between square brackets
                text =re.sub('\[[^]]*\]', '', text)
        # removes punctuation
                text = re.sub(r'[^\w\s]','',text)
                text = re.sub(r'\n',' ',text)
                text = re.sub(r'[0-9]+','',text)
                #text = re.sub(r'[0-9]+','',text)
        # strip html 
                p = re.compile(r'<.*?>')
        # Replacing certain phrases or words    
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"can't", "cannot ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"I'm", "I am", text)
                text = re.sub(r" m ", " am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
        
                text = p.sub('', text)
# Function to convert text into tokens
        def tokenize_text(text,tokenizer=default_tokenizer):
            token = default_tokenizer.tokenize(text)
            return token
# Function to remove stopwords
        def remove_stopwords(text, stop_words=default_stopwords):
            tokens = [w for w in tokenize_text(text) if w not in stop_words]
            return ' '.join(tokens)
# Function to stem the text
        def stem_text(text, stemmer=default_stemmer):
            tokens = tokenize_text(text)
            return ' '.join([stemmer.stem(t) for t in tokens])
# Function to lemmatize the text
        def lem_text(text,lemmatizer = default_lemmatizer):
            tokens = tokenize_text(text)
            return ' '.join([lemmatizer.lemmatize(t) for t in tokens])


        text = stem_text(text) # stemming
        text=lem_text(text) #lemmatizing
        text = remove_stopwords(text) # remove stopwords
        #text.strip(' ') # strip whitespaces again?

        return text    

data=load_model()
data1=load_df()

classifier=data["model"]
sampdf=data1['dataframe']    

def show_detect_page():
    st.title("Detection of Fake Job Postings")
    company_description=st.text_area(label='Enter Company Description:')
    job_description=st.text_area(label='Enter Job Description:')
    job_requirements=st.text_area(label='Enter Job Requirements:')   
    col1, col2 = st.columns((2,1))
    with col1:
        st.write("")
        detect=st.button("Detect") 
    if detect:
        text=company_description+" "+job_description+" "+job_requirements
        df_dict={"text":[text]}
        df=pd.DataFrame.from_dict(df_dict)
        df['text'] = df['text'].apply(clean_text)
        df=df.append(sampdf,ignore_index=True)
        cv = TfidfVectorizer(max_features = 100)
        x = cv.fit_transform(df['text'])
        df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
        df1=df1.iloc[0]
        df1=df1.to_frame()
        df1=df1.T
        flag=0
        for i in df1.columns:
                if(df1[i][0]!=0):
                    flag=1
                    break        
        classifier_predict = classifier.predict(df1)
        output=""
        if(flag==0):
            output="Fake Job Posting"
        else:    
            if(classifier_predict[0]==1):
                output="Fake Job Posting"
            else:
                output="Genuine Job Posting"    
            st.write("")       
        c=st.empty()
        with c:
            my_bar=st.progress(0)
            for percent_complete in range(0,100,3):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()        

        with col2:
            st.write("")
            if(output=="Genuine Job Posting"):    
                st.success(output)
            else:
                st.error(output)    

        