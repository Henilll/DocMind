# Split Using TikToken Tokenization

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10

)
data=PyPDFLoader("/Users/henilbhavsar/Henil/RAG/document loders/deeplearning.pdf")
docs=data.load()
chunks=splitter.split_documents(docs)
print(len(chunks))

print(chunks[0].page_content)