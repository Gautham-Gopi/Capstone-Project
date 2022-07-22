import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


def show_explore_page():
    st.markdown('''
    <div style="text-align:center;width:100%;;font-family:sans-serif">
    <p style="align:center;font-size:70px;font-weight: bold;background: -webkit-linear-gradient(#F5E3E6, #C7E9FB);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent">CAPSTONE PROJECT</p><br>
  <p style="font-size:28px;align:center"> Fake Job Posting Detection Using Machine Learning And NLP</p>
    </div>
    <div style="width:100%;font-family:sans-serif">
    <h2>Introduction:</h2>
    <p style="align:left;font-size:20px">This project aims to create a classifier that will have the capability to identify fake and real jobs. Various Algorithms like Random Forest Classifier, XG Boost, K-Nearest Neighbors Classifier and Naive Bayes were tested. The algorithm which provided the best result was selected.</p>
    <br>
    <h2>About the data:</h2>
    <p style="align:left;font-size:20px">Number of columns: 18</p>
    <p style="align:left;font-size:28px">The columns are:</p>  
    <table style="width: 50%">
    <tr>
        <th>#</th>
        <th>Variable</th>
        <th>Datatype</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>1</td>
        <td>job_id</td>
        <td>int</td>
        <td>Identification number given to each job posting</td>
    </tr>
    <tr>
        <td>2</td>
        <td>title</td>
        <td>text</td>
        <td>A name that describes the position or job</td>
    </tr>
    <tr>
        <td>3</td>
        <td>location</td>
        <td>text</td>
        <td>Information about where the job is located</td>
    </tr>
    <tr>
        <td>4</td>
        <td>department</td>
        <td>text</td>
        <td>Information about the department this job is offered by</td>
    </tr>
    <tr>
        <td>5</td>
        <td>salary_range</td>
        <td>text</td>
        <td>Expected salary range</td>
    </tr>
    <tr>
        <td>6</td>
        <td>company_profile</td>
        <td>text</td>
        <td>Information about the company</td>
    </tr>
    <tr>
        <td>7</td>
        <td>description</td>
        <td>text</td>
        <td>A brief description about the position offered</td>
    </tr>
    <tr>
        <td>8</td>
        <td>requirements</td>
        <td>text</td>
        <td>Pre-requisites to qualify for the job</td>
    </tr>
    <tr>
        <td>9</td>
        <td>benefits</td>
        <td>text</td>
        <td>Benefits provided by the job</td>
    </tr>
    <tr>
        <td>10</td>
        <td>telecommuting</td>
        <td>boolean</td>
        <td>Is work from home or remote work allowed</td>
    </tr>
    <tr>
        <td>11</td>
        <td>has_company_logo</td>
        <td>boolean</td>
        <td>Does the job posting have a company logo</td>
    </tr>
    <tr>
        <td>12</td>
        <td>has_questions</td>
        <td>boolean</td>
        <td>Does the job posting have any questions</td>
    </tr>
    <tr>
        <td>13</td>
        <td>employment_type</td>
        <td>text</td>
        <td>5 categories – Full-time, part-time, contract, temporary and other</td>
    </tr>
    <tr>
        <td>14</td>
        <td>required_experience</td>
        <td>text</td>
        <td>Can be – Internship, Entry Level, Associate, Mid-senior level, Director, Executive or Not Applicable</td>
    </tr>
    <tr>
        <td>15</td>
        <td>required_education</td>
        <td>text</td>
        <td>Can be – Bachelor’s degree, high school degree, unspecified, associate degree, master’s degree, certification, some college coursework, professional, some high school coursework, vocational</td>
    </tr>
    <tr>
        <td>16</td>
        <td>Industry</td>
        <td>text</td>
        <td>The industry the job posting is relevant to</td>
    </tr>
    <tr>
        <td>17</td>
        <td>Function</td>
        <td>text</td>
        <td>The umbrella term to determining a job’s functionality</td>
    </tr>
    <tr>
        <td>18</td>
        <td>Fraudulent</td>
        <td>boolean</td>
        <td>The target variable  0: Real, 1: Fake</td>
    </tr>   
    </table> 
    <br>
    <p style="align:left;font-size:20px">Target Column: Fraudulent</p>
    <br>
    <h2>Visualization:</h2>
    </div>
    ''',unsafe_allow_html=True)
    st.markdown('''
    <br>
    ''',unsafe_allow_html=True)
    bar = Image.open('bar.jpg')
    st.image(bar)
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    pair = Image.open('pair.jpg')
    st.image(pair)
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    heatmap = Image.open('heatmap.jpg')
    st.image(heatmap,width=750)
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    ratio = Image.open('fake-to-real-ratio.jpg')
    st.image(ratio)
    st.markdown('''
    <br>
    <h2> Selection Of Algorithm:</h2>
    <br>
    ''',unsafe_allow_html=True)
    rfc = Image.open('rfc.jpg')
    st.image(rfc,caption="Random Forest Confusion Matrix")
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    xgb = Image.open('xgb.jpg')
    st.image(xgb,caption="XG Boost Confusion Matrix")
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    knn = Image.open('knn.jpg')
    st.image(knn,caption="KNN Confusion Matrix")
    st.markdown('''
    <br>
    <br>
    ''',unsafe_allow_html=True)
    nb = Image.open('nb.jpg')
    st.image(nb,caption="Naive Bayes Confusion Matrix")
    st.markdown('''
    <h3>Thus from observing the Confusion Matrices it is clear that Random Forest Classifier is the best algorithm for the data.</h3>
    ''',unsafe_allow_html=True)

