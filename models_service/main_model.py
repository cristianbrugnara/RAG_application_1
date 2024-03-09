from pinecone import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LcPinecone
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.chat_models import ChatOpenAI
import os
from pinecone import Pinecone
from resources.playground_secret_key import PINECONE_KEY,SECRET_KEY

os.environ['PINECONE_API_KEY'] = PINECONE_KEY
environment = os.environ.get('PINECONE_ENVIRONMENT')
os.environ['OPENAI_API_KEY'] = SECRET_KEY


class MainModel:
    __index = Pinecone().Index('rag')
    __embed_model = OpenAIEmbeddings(model='text-embedding-ada-002')
    __vector_store = LcPinecone.from_existing_index('rag', __embed_model)
    __chat_model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model='gpt-3.5-turbo')

    """Class to encapsulate the Rag pipeline. 
    
    Attributes:
        __index (Pinecone.Index) : contains the pinecone index in which the vectorized embeddings are stored 
        __embed_model (OpenAIEmbeddings) : contains the model used to embed the text
        __vector_store (Pinecone) : contains the vectorstore used to do similarity search on the index 
        __chat (ChatOpenAI)
    """
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
        {source_knowledge} 

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
        results = cls.__vector_store.similarity_search(query=query, namespace=namespace,k=k)
        source_knowledge = '\n'.join([x.page_content for x in results])
        return source_knowledge

    @classmethod
    def query(cls, user_query):
        """

        :param user_query: query inserted by the user
        :return: return the model's response informed by the knowledge base retrieved from the index + user_query
        """
        prompt = [HumanMessage(
            content=cls.__augment_prompt(user_query, 1)
        )]
        return cls.__chat_model.invoke(prompt).content

    @classmethod
    def populate_index(cls):
        pass

    @classmethod
    def add_file(cls):
        pass

    @classmethod
    def remove_file(cls):
        pass


print(MainModel.query('According to Michela Papandrea, what are the main steps of M.L. analysis'))