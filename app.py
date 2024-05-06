from PyPDF2 import PdfReader
import os


## Importacion de librerias de langchain

from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

## Importacion de librerias STREAMLIT

import streamlit as st

## Importacion de librerias OPENAI

import openai

## Importacion de librerias tiktoken
import tiktoken

#Configuracion de la biblioteca streamlit



st.set_page_config(page_title="Chatbot PDF", page_icon=":robot:", layout="wide")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)  # Tweak padding here'text-align: center;'>Chatbot PDF</style>", unsafe_allow_html=True)


## ingresa la apikey de openai(me piden la apikey porque no me dejo subir el repo con esa llave)


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

# Esta función busca los fragmentos iguales en los embeddings y los devuelve
def similarity_search(docsearch, query):
    if docsearch is not None and query is not None:
        return docsearch.similarity_search(query)
    else:
        return None

# Aquí definimos la función load_qa_chain
def load_qa_chain(llm, chain_type):
    if chain_type == "stuff":
        # Inicializa tu cadena de QA aquí
        qa_chain = "Tu cadena de QA inicializada"
        return qa_chain
    else:
        raise ValueError("Tipo de cadena no reconocido")

# Definicion  del contexto get_openai_callback
import contextlib

@contextlib.contextmanager
def get_openai_callback():
    try:
        # Preparación antes de la llamada a la API
        print("Preparando la llamada a la API de OpenAI")
        yield
    finally:
        # Limpieza después de la llamada a la API
        print("Llamada a la API de OpenAI completada")

# Definicion de la función chat_gpt

def chat_gpt(prompt):
    response = OPENAI_API_KEY.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
 
 #Se llega hasta este punto, ya que es necesario tener la app paga de openai




