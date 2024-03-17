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

    __system_prompt = SystemMessage("""
        You are an helpful assistant that answers questions on machine learning and supervised learning.
        You only use the provided context, never use prior knowledge. If you don't know the answer, don't try to make it up.
        Whenever you answer a question, always provide a reference to the context, such as the file name, the page or any specific section in the format:
        <context_format> 
        {'file_name': 'context_document_file_name', 'file_path': 'context_document_file_path', 'page' : 'context_document_file_page' }
        </context_format> 
        and append this information to the end of your answer.
        If possible answer questions in a schematic way, making use of lists and bulletpoints.
        Take some time to make the answer very clear and detailed.
        <Important remarks>
        1) It is crucial that you answer in Italian at every question
        2) Be aware that if you add non relevant information the answer will not be accepted and you will not be paid
        3) It is crucial that your answers are based exclusively on the context given
        4) It is crucial that you do not answer a question unless you are 100% sure of the answer you are giving.  
        5) Do not add any formatting in the answer. Just your response to the query.     
        </Important remarks>

        """)

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

        <context>
        {source_knowledge} 
        </context>

        <query>
        {query}
        </query>
        
        """
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
        # print(f"Similarity search: {end-start} s")
        source_knowledge = '\n'.join([f"Content:{x.page_content}\n Metadata: {x.metadata}" for x in results])
        return source_knowledge

    @classmethod
    def query(cls, user_query : str, return_prompt : bool = False):
        """
        Asks the model to generate an answer for the user query.

        :param user_query: query inserted by the user
        :param return_prompt: helper parameter. Specifies if the function should return a tuple with the result of the
        query and the augmented prompt, instead of just the former.
        :return: return the model's response informed by the knowledge base retrieved from the index + user_query
        """

        # print('entered in query')
        # print()
        user_prompt = HumanMessage(
            content=cls.__augment_prompt(user_query, k=1))

        prompt = [cls.__system_prompt, user_prompt]

        start = default_timer()
        res = cls.__chat_model.invoke(prompt).content
        end = default_timer()

        # print(f"Invoke: {end-start} s")
        # print()

        if return_prompt:
            return res, user_prompt.content
        return res

    @classmethod
    def double_step_query(cls, user_query : str, return_prompt : bool = False):
        """
        Asks the model to generate an answer for the user query.
        Feeds the query back in and asks the model to refine it.

        :param user_query: query inserted by the user
        :param return_prompt: helper parameter. Specifies if the function should return a tuple with the result of the
        query and the augmented prompt, instead of just the former.
        :return: return the model's response informed by the knowledge base retrieved from the index, the user_query and a previews answer.
        """

        answer,prompt = cls.query(user_query, return_prompt=True)

        new_user_prompt = HumanMessage(f"""
        
        Given the following query:
        
        <query>
        {prompt}
        </query>
        
        We provided the following answer:
        <answer>
        {answer}
        </answer>
        
        
        You have the opportunity to improve the answer using the following context:
        <context>
        {cls.__similarity_search(user_query, k=5)}
        </context>
        
        
        If you don't believe the answer needs refinement return the original answer.
        Be aware that if you add non relevant information the answer will not be accepted and you will not be paid
        Do not mention that you refined or not the answer.
        
        """)

        prompt = [cls.__system_prompt, new_user_prompt]
        res = cls.__chat_model.invoke(prompt).content

        if return_prompt:
            return res, new_user_prompt.content

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
    print(MainModel.double_step_query('Quale organo ha espresso un preliminare parere di condivisione generale degli obiettivi e delle finalità del progetto "PNRR - M5C2 - INV. 2.2 Interventi di miglioramento della qualità ambientale del territorio – fascia Laguna Santa Gilla" come risposta alla nota del Comune di Elmas prot. n. 13250 del 20/09/2023?'))
    # print(MainModel.query('What is gradient descent?'))

# - Domanda: Quale organo ha espresso un preliminare parere di condivisione generale degli obiettivi e delle finalità del progetto "PNRR - M5C2 - INV. 2.2 Interventi di miglioramento della qualità ambientale del territorio – fascia Laguna Santa Gilla" come risposta alla nota del Comune di Elmas prot. n. 13250 del 20/09/2023?
# - Risposta: La Direzione Generale dell'Agenzia Regionale del Distretto Idrografico della Sardegna ha espresso un preliminare parere di condivisione generale degli obiettivi e delle finalità del progetto.