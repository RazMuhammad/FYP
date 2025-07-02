import os
from langchain.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

file_path = "aup_website_data.txt"
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into smaller chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
print(f"Documents split into {len(split_docs)} chunks.")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "aup-website-data"
index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
uuids = [str(uuid4()) for _ in range(len(split_docs))]
vector_store.add_documents(documents=split_docs, ids=uuids)