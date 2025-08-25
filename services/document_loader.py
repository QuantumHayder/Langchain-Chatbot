
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import HierarchicalNodeParser

from pathlib import Path


class DocumentLoaderService:
    """ This is a wrapper class to encapsulate the document loading functionality. """
    
    def __init__(self,chunk_size: int = 100, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def load_and_split_pdf(self, path):
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
        return text_splitter.split_documents(docs)


    def llamaindex_file_loader(self, path):
        md_docs = PDFReader().load_data(Path(path))
        node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 256, 64]
        )
        md_nodes = node_parser.get_nodes_from_documents(md_docs)
        return md_nodes