import streamlit as st
from langchain_community.retrievers import BM25Retriever
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import os


@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\JVC Store\\Downloads\\data\\data 6\\train.csv")
    df = df.drop(columns = ['Unnamed: 0','hs_code_2','hs_code_4'])
    documents = []

    for index, row in df.iterrows():
        text = row['full_description']
        hs_code = row['hs_code']
        documents.append(Document(page_content=text, metadata={'hs_code': hs_code}))

    splitter = CharacterTextSplitter(
        chunk_size=100,  
        chunk_overlap=0,
        separator = ' '
    )

    split_documents = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        #remove chunk split word
        word_chunks = []
        current_chunk = []
        
        for chunk in chunks:
            words = chunk.split()
            for word in words:
                if len(' '.join(current_chunk + [word])) <=100:
                    current_chunk.append(word)
                else:
                    word_chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
        if current_chunk:
            word_chunks.append(' '.join(current_chunk))
            
        split_documents.append(Document(page_content=word_chunks[0], metadata=doc.metadata))


    docs = []
    for doc in split_documents:
        metadata = doc.metadata
        metadata_str = str(metadata).strip('{}')
        page = doc.page_content
        docs.append([metadata_str + " " + page])


    cleaned_list = [item.replace('"','').replace("'",'') for items in docs for item in items]
    retriever = BM25Retriever.from_texts(cleaned_list)
    retriever.k = 5
    return retriever


def load_llm():

    api_key2 = "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"

    llm2 = ChatGroq(model = "llama-3.1-70b-versatile", temperature = 0,api_key = api_key2)
    return llm2


def predict(sentence,retriever,llm2):
    sentence = sentence.lower()
    context = retriever.get_relevant_documents(sentence)
    #print("context:",context)
    template2 = """
    You are an expert in HS Code classification. 
    Based on the provided product description, accurately determine and return only one 6-digit HS Code that best matches the description.
    Always return the HS Code as a 6-digit number only.
    example: 123456
    Context:\n {context} \n
    Description:\n {description} \n
    Answer:
    """
    prompt2 = PromptTemplate(template=template2, input_variables=['context','description'])
    chain = load_qa_chain(llm2, chain_type = 'stuff', prompt = prompt2)
    response = chain.invoke({'input_documents': context, 'description':sentence})
    answer = response.get("output_text")
    return answer


if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

if st.session_state.retriever is None:
    st.session_state.retriever = load_data()

if st.session_state.llm is None:
    st.session_state.llm = load_llm()

sentence = st.text_input("please enter description:")

if sentence !='':
    answer = predict(sentence,st.session_state.retriever,st.session_state.llm )
    st.write("answer:",answer)