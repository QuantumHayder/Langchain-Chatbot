
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import HierarchicalNodeParser

from pathlib import Path


def load_and_split_pdf(path="GermanCulture.pdf"):
    loader = PyPDFLoader("GermanCulture.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
    return text_splitter.split_documents(docs)


def llamaindex_file_loader(path):
    md_docs = PDFReader().load_data(Path(path))
    node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 256, 64]
    )
    md_nodes = node_parser.get_nodes_from_documents(md_docs)
    print(md_nodes[0])
    return md_nodes