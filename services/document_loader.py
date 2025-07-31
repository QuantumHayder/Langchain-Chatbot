
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(path="GermanCulture.pdf"):
    loader = PyPDFLoader("GermanCulture.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
    return text_splitter.split_documents(docs)


