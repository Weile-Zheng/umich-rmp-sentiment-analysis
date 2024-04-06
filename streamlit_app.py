import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title='UMICH Rate My Professor Sentiment Data', page_icon='📈')
st.title('🧠 Interactive Data Explorer')
st.subheader('UMICH Rate My Professor Sentiment Analysis')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app provides visualization and interaction with our processed dataset webscraped from Ratemyprofessor.com')
  st.markdown('**How to use the app?**')
  st.warning("Select desired data information with the dropdown menu and checkboxes")
  
st.subheader('Course Review')

# Load data
course_review = pd.read_csv('./data/course_review_polarity_emotion_merged.csv')

# Input widgets
## Course selection
course_list = course_review["class"].unique()
course_selection = st.multiselect('Select course', course_list)
include_comment = st.checkbox("Include original comment")
include_emotion = st.checkbox("Include emotion labels")


# Display DataFrame
selected = course_review[course_review["class"].isin(course_selection)]


if not include_comment:
    selected = selected.drop(columns="comment")

if not include_emotion:
    selected = selected.drop(columns=["emotion1", "emotion2", "emotion3"])

st.dataframe(selected, height=212, use_container_width=True)

expander = st.expander("See explanation")

# Add sentiment column based on polarity_score
selected['sentiment'] = selected['polarity_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Display bar chart of sentiment counts
st.bar_chart(selected['sentiment'].value_counts())

expander.info("""Polarity Score: This is a numerical value that represents the 
                       sentiment of a text on a scale, typically from -1 to 1. A polarity 
                       score of -1 represents extremely negative sentiment, a score of 1
                        represents extremely positive sentiment, and a score of 0 represents
                        neutral sentiment. The score is calculated using [this] (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) 
               instance of DistilBERT Transformer model.""")

# Display chart
