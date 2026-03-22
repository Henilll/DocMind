# Split Using CharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

loader = TextLoader("/Users/henilbhavsar/Henil/RAG/document loders/notes.txt", encoding="utf-8")



docs = loader.load()
chunks=splitter.split_documents(docs)

print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()