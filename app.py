from PyPDF2 import PdfReader
import os

## langchain imports libraries

from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


## Set up your OpenAI API key here
OPENAI_API_KEY = 'sk-sk-proj-kxJgH19FmINO7pQi8b1PT3BlbkFJ84yvC4nArG0wzwJ66tPx'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

