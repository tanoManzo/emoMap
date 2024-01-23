import streamlit as st
from io import StringIO
from docx import Document
import nltk
import torch
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import sentencepiece

from transformers import pipeline
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

### Functions 
def getText(filename):
    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def collect_emotions(list_emotions):
    dict_emo = defaultdict(list)    
    for l_emo in list_emotions:
      for item in l_emo:
        if (item['score'] > 0.1) and (item['label']!='neutral'):
          dict_emo[item['label']].append(item['score'])
    return dict_emo



## General
st.title("EmoMap")
st.header("Retrieve Emotions from Transcripts")

## START PIPELINE Translation and Emotion

# Initialization
if 'key' not in st.session_state:
  st.session_state['key'] = 'value'
  
  # translation
  pipe_translation_es = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", max_length=512, truncation=True)
  pipe_translation_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", max_length=512, truncation=True)
  
  
  # emotion
  model_ckpt = "JuliusAlphonso/distilbert-plutchik"
  pipe_emotion = pipeline("text-classification",model=model_ckpt, top_k=None, max_length=512,truncation=True)

## File loader
uploaded_file = st.file_uploader("Upload a transcript")


option = st.selectbox(
   "Select Transcript Language?",
   ("French", "Spanish"),
   index=0,
   placeholder="Select language...",
)

if option=='French':
    pipe_translation = pipe_translation_fr
else:
    pipe_translation = pipe_translation_es
    
if uploaded_file and option!=0:
    st.write('File Uploaded')

    transcription = getText(uploaded_file).split('\n')

    ## Sentences from transcript
    list_parag_transcriptio = []

    for parag in transcription:
      sentences = sent_tokenize(parag)
      list_parag_transcriptio.extend(sentences)  

    ## get translations and emotions
    flag_process = True
    translations = []
    emotions = []
    with st.spinner('Please wait'):
        progress_text = "Emotion detection in progress .."
        my_bar = st.progress(0, text=progress_text)
        for idx, item in enumerate(list_parag_transcriptio):
            translated = pipe_translation(item)[0]['translation_text']
            translations.append(translated)
            emotions.append(pipe_emotion(translated)[0])
            my_bar.progress(int(100*(idx/len(list_parag_transcriptio))) + 1, text=progress_text)

        print(len(list_parag_transcriptio))
    
    ## Preocess results 
    dict_emo = collect_emotions(emotions)
    keys = dict_emo.keys()

    list_val = []
    list_val_max = []
    list_keys = []
    for k in keys:
      emo_dict = dict_emo
      if k in emo_dict:
        list_val.append(np.mean(dict_emo[k]))
        list_val_max.append(np.max(dict_emo[k]))
        list_keys.append(k)
    
    categories = list_keys

    fig = go.Figure()

    # change color based on the emotion
    green_list = ['joy','trust']
    red_list = ['fear', 'anger', 'sadness', 'disgust']
    
    if categories[np.argmax(list_val)] in green_list:
        color_fill = 'rgba(0,250,0,0.5)'
    elif categories[np.argmax(list_val)] in red_list:
        color_fill = 'rgba(250,0,0,0.5)'
    else:    
        color_fill = 'rgba(0,0,250,0.5)'
        

    
    fig.add_trace(go.Scatterpolar(
      r=list_val_max,
      theta=categories,
      fill='toself',
      line_color='rgba(0,0,200,0.7)',
      fillcolor='rgba(0,0,200,0.3)',
      name='Max Emotions'
    ))

    fig.add_trace(go.Scatterpolar(
        r=list_val,
        theta=categories,
        fill='toself',
        line_color='rgba(0,200,0,0.7)',
        fillcolor='rgba(0,200,0,0.3)',
        name='Mean Emotions'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
