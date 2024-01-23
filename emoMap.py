import streamlit as st


from io import StringIO
from docx import Document
import nltk
import torch


from nltk.tokenize import sent_tokenize
import sentencepiece

from transformers import pipeline
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0




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



## File loader
uploaded_file = st.file_uploader("Upload a transcript")

    
if uploaded_file:
    st.write('File Uploaded')  
    
    transcription = getText(uploaded_file).split('\n')

    ## Sentences from transcript
    list_parag_transcription = []

    for parag in transcription:
      sentences = sent_tokenize(parag)
      list_parag_transcription.extend(sentences)  
    
    ## Initialization

    
    flag_stop = 0
    language = ''
    if detect(list_parag_transcription[0])=='fr':
        language = 'FRENCH'
        if 'tran_fr' not in st.session_state:
          st.session_state['tran_fr'] = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", max_length=512, truncation=True)
        pipe_translation = st.session_state['tran_fr']
        st.write(f'Language detected: {language}')
    elif detect(list_parag_transcription[0])=='es':
        language = 'SPANISH'
        if 'tran_es' not in st.session_state:
          st.session_state['tran_es'] = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", max_length=512, truncation=True)
        pipe_translation = st.session_state['tran_es']
        st.write(f'Language detected: {language}')
    else:
        language = detect(list_parag_transcription[0])
        st.write(f'Language detected: {language} --> NOT SUPPORTED.')
        flag_stop = 1
    
    if 'pipe_emo' not in st.session_state:
      # emotion
      model_ckpt = "JuliusAlphonso/distilbert-plutchik"
      st.session_state['pipe_emo'] = pipeline("text-classification",model=model_ckpt, top_k=None, max_length=512,truncation=True)
      
    pipe_emotion = st.session_state['pipe_emo']
        
    ## get translations and emotions
    translations = []
    emotions = []
    with st.spinner('Please wait'):
        progress_text = "Emotion detection in progress .."
        my_bar = st.progress(0, text=progress_text)
        for idx, item in enumerate(list_parag_transcription):
            translated = pipe_translation(item)[0]['translation_text']
            translations.append(translated)
            emotions.append(pipe_emotion(translated)[0])
            my_bar.progress(int(100*(idx/len(list_parag_transcription))) + 1, text=progress_text)

    
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
        
#
    
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