from langchain_openai.embeddings import OpenAIEmbeddings
import os
import sys
from pinecone import Pinecone
from resources.playground_secret_key import PINECONE_KEY_2, SECRET_KEY
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline

os.environ['PINECONE_API_KEY'] = PINECONE_KEY_2
environment = os.environ.get('PINECONE_ENVIRONMENT')
os.environ['OPENAI_API_KEY'] = SECRET_KEY


class Index:

    __index = Pinecone().Index('rag')
    __embed_model = OpenAIEmbeddings(model = 'text-embedding-3-small',dimensions=512)

    @classmethod
    def get_embed_model(cls) -> OpenAIEmbeddings:
        return cls.__embed_model

    @classmethod
    def __generate_metadata(cls, docs, keyword: bool = True):

        extractor = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

        splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

        transformations = [splitter]

        if keyword:
            transformations.append(KeywordExtractor(llm=extractor))

        # obtain file text (probably in a different way)

        ing_pipeline = IngestionPipeline(transformations=transformations)
        nodes = ing_pipeline.run(documents=docs)

        metadata = [node.metadata for node in nodes]
        text = [node.text for node in nodes]

        ids = [node.id_ for node in nodes]
        return metadata, text, ids  # need to decide how/where to get output

    @classmethod
    def __get_size_of(cls,obj):
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum([cls.__get_size_of(v) for v in obj.values()])
            size += sum([cls.__get_size_of(k) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += cls.__get_size_of(obj.__dict__)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([cls.__get_size_of(i) for i in obj])
        return size

    @classmethod
    def populate(cls, filepaths: List[str] = None, directory : str = None):
        if directory is not None:
            docs = SimpleDirectoryReader(input_dir=directory).load_data()
        else:
            docs = SimpleDirectoryReader(input_files=filepaths).load_data()

        metadata, texts, ids = cls.__generate_metadata(docs)

        for j in range(len(metadata)):
            metadata[j]['text'] = texts[j]

        embeds = [cls.get_embed_model().embed_query(text) for text in texts]

        cur_batch = []
        cur_batch_size = 0
        batch_size_limit = 2 * 1024 * 1024

        for i in range(len(texts)):
            vector_size = cls.__get_size_of(embeds[i])

            if cur_batch_size+vector_size>batch_size_limit:
                cls.__index.upsert(vectors=cur_batch, namespace='ns1')
                cur_batch = []
                cur_batch_size = 0

            cur_batch.append({
                'id': ids[i],
                'values': embeds[i],
                'metadata': metadata[i]})
            cur_batch_size += vector_size

        if cur_batch:
            cls.__index.upsert(vectors=cur_batch, namespace='ns1')
        return

    @classmethod
    def add_file(cls):
        pass # TODO

    @classmethod
    def remove_file(cls):
        pass # TODO


if __name__ == '__main__':
    print(Index.populate(directory='../data/Annotated Handouts-20240310/full_chapters_annotated'))



