# load pdf
# spilt into chunks
# create the embeddings
# store the chromadb

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

data = PyPDFLoader("/Users/henilbhavsar/Henil/RAG/document loders/deeplearning.pdf")
docs = data.load()


splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
)

chunks=splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model="mistral-embed")
vectorstore=Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory="chroma-db")
