from PyPDF2 import PdfReader
import os

## Importacion de librerias de langchain

from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

## Importacion de librerias STREAMLIT

import streamlit as st

#Configuracion de la biblioteca streamlit

st.set_page_config(page_title="Chatbot PDF", page_icon=":robot:", layout="wide")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)  # Tweak padding here'text-align: center;'>Chatbot PDF</style>", unsafe_allow_html=True)


## ingresa la apikey de openai
OPENAI_API_KEY = 'sk-sk-proj-kxJgH19FmINO7pQi8b1PT3BlbkFJ84yvC4nArG0wzwJ66tPx'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

## Funcion para crear la base de caracteristicas
def create_embedding(pdf):
   if pdf is not None:
      reader = PdfReader(pdf)
      text = ""
      for page in reader.pages:
         text += page.extract_text()
#Division del texto en fragmentos con espaciado 
      text_splitter = CharacterTextSplitter(
         separator="\n",
         chunk_size=1000,
         chunk_overlap=200,
         length_function=len
      )
      chunks = text_splitter.split_text(text)
#Uso de la biblioteca de OpenAI para crear los embeddings de los textos(es decir la
# conversión de texto a vectores)
      embeddings = OpenAIEmbeddings()
#Almacenamiento de los vectores de los textos en la libreria FAISS,para consultas
#Se pasa el texto divivido en chunks y los embeddings
      docsearch = FAISS.from_texts(chunks, embeddings)
      return docsearch
#Funcion de carga del documento pdf en el sidebar

st.sidebar.markdown("<H1 style='text-align: center;'>Chatbot PDF</H1>", unsafe_allow_html=True)
st.sidebar.write("Carga tu archivo PDF")

pdf = st.sidebar.file_uploader("", type="pdf")

st.sidebar.write("____________________")

#seccion del chat

st.markdown("<H2 style='text-align: center;'>En que te puedo ayudar?</H2>", unsafe_allow_html=True)
st.write("____________________")

#Container del historial de chats

textcontainer = st.container()

# Creacion de campo de entrada de texto

with textcontainer:
   user_input = st.text_input("Tu:", key="input")
   if user_input:
      docsearch = create_embedding(pdf)
      docs = docsearch.similarity_search(user_input)
      st.write(docs[0].page_content)

#Boton de enviar

submit = st.button("Enviar")

#spinner de carga

if submit:
   with st.spinner("Escribiendo..."):
      st.experimental_rerun()





