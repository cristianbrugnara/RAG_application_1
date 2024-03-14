from pinecone import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LcPinecone
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_openai import ChatOpenAI
import os
from pinecone import Pinecone
from resources.playground_secret_key import PINECONE_KEY_2, SECRET_KEY
from typing import List, Dict
from index import Index
from timeit import default_timer


os.environ['PINECONE_API_KEY'] = PINECONE_KEY_2
environment = os.environ.get('PINECONE_ENVIRONMENT')
os.environ['OPENAI_API_KEY'] = SECRET_KEY


class MainModel:
    """Class to encapsulate the Rag pipeline.

        Attributes:
            __index (Pinecone.Index) : contains the pinecone index in which the vectorized embeddings are stored
            __embed_model (OpenAIEmbeddings) : contains the model used to embed the text
            __vector_store (Pinecone) : contains the vectorstore used to do similarity search on the index
            __chat (ChatOpenAI)

        """
    __index = Index
    __embed_model = Index.get_embed_model()
    __vector_store = LcPinecone.from_existing_index('rag', __embed_model)
    __chat_model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo')

    @classmethod
    def __augment_prompt(cls, query: str, k: int):
        """Takes as input a user query and concatenate it with inherent documents taken from the vectorstore database, based on similarity

        :param query: query inserted from the user
        :param k: number of documents retrieved from the similarity search
        :return: the augmented prompt containing the user query + the inherent chunk of text retrieved from the vectorstore
        """
        source_knowledge = cls.__similarity_search(query, k)
        augmented_prompt = \
            f"""Using the context below, answer the query. 

        Contexts: 
        ###
        {source_knowledge} 
        ###
        
        Query: 
        {query}"""
        return augmented_prompt

    @classmethod
    def __similarity_search(cls, query: str, k: int, namespace='ns1'):
        """

        :param query: query inserted from the user
        :param k: number of documents retrieved from the similarity search
        :return: source knowledge from the vectorstore database that relates to the user query
        """
        start = default_timer()
        results = cls.__vector_store.similarity_search(query=query, namespace=namespace, k=k)
        end = default_timer()
        print(f"Similarity search: {end-start} s")
        source_knowledge = '\n'.join([f"Content:{x.page_content}\n Metadata: {x.metadata}" for x in results])
        return source_knowledge

    @classmethod
    def query(cls, user_query : str):
        """

        :param user_query: query inserted by the user
        :return: return the model's response informed by the knowledge base retrieved from the index + user_query
        """

        system_prompt = SystemMessage("""
        You are an helpful assistant that answers questions on machine learning and supervised learning.
        You only use the provided context, never use prior knowledge. If you don't know the answer, don't try to make it up.
        Whenever you answer a question, always provide a reference to the context, such as the file name, the page or any specific section.
        If you have to list something or define steps, use bullet points.
        Take some time to make the answer very clear and detailed.
        """)

        user_prompt = HumanMessage(
            content=cls.__augment_prompt(user_query, k=3))

        prompt = [system_prompt, user_prompt]
        print(prompt)
        start = default_timer()
        res = cls.__chat_model.invoke(prompt).content
        end = default_timer()
        print(f"Invoke: {end-start} s")
        print()
        return res

    @classmethod
    def populate_index(cls):
        pass

    @classmethod
    def add_file(cls):
        pass

    @classmethod
    def remove_file(cls):
        pass

    # metadata with llamaindex, maybe can be done with langchain


if __name__ == '__main__':
    MainModel.query('What are the metrics we can use to evaluate a model?')
    pass
