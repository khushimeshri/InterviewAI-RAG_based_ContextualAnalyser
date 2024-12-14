
import assemblyai as aai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
#data ingestion or loading
from langchain_community.document_loaders import TextLoader

#the transormation of large data into chunks of data of contextual sizes
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np

#to store embeddings in vectorDB
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

#to set up an interface with hugging face and to use thw api key 
import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#this will help set up environment for generating contextual based summary from the text document
#setup environment for LLM(chatgroq)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain

import streamlit as st

# Replace with your API key
aai.settings.api_key = "ASSEMBYAI_API_KEY"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("videoplayback.mp4")

# Save the transcription to a .txt file
with open("transcript.txt", "w") as file:
    file.write(transcript.text)


#data ingestion or loading
loader=TextLoader("transcript.txt")
text_documents=loader.load()
##text_documents

#the transormation of large data into chunks of data of contextual sizes
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
f_documents=text_splitter.split_documents(text_documents)
##f_documents[2]


huggingface_embeddings=HuggingFaceBgeEmbeddings(
     model_name= "BAAI/bge-small-en-v1.5",
     model_kwargs ={'device':'cpu'},
     encode_kwargs ={'normalize_embeddings': True} # set True to compute cosine similarity
     )

#shows the embedded vectors
##print(np.array(huggingface_embeddings.embed_query(f_documents[2].page_content)))
#print(np.array(huggingface_embeddings.embed_query(f_documents[2].page_content)).shape)

#store all the embeddings or the vectors into vectorestore
vectorstore=FAISS.from_documents(f_documents[:100],huggingface_embeddings)

##give a query using similarity search in order to check wether it provides proper results or not 
query="What Kind of laurels he has earned for the country?"
relevant_docments=vectorstore.similarity_search(query)


retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})


#calling the api fromopensource model hugging face 
os.environ['HUGGINGFACEHUB_API_TOKEN']="HUGGINGFACE_API_KEY"

#using mistralai to answer the query based on the given prompt 
hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1,"max_length":500}

)
query="What is the final goal of Atmika?"
hf.invoke(query)


load_dotenv()
#I have used the ChatGroq LLM for advanced interview analysis by combining embeddings,context retrieval, and structured prompts to evaluate interview performance traits.
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.3,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    api_key="CHATGROQ_API_KEY"
)

prompt2= PromptTemplate.from_template("""
You are an expert in interview analysis. 
Please carefully evaluate the candidate's performance based on the traits provided in the input and accordingly generate the contextual summary 
and also provide an evaluation score (scaling from 1 to 10)


Here is the context from the interview:
{context}
""")

document_chain= LLMChain(llm=llm,prompt=prompt2)

retriever=vectorstore.as_retriever()
#it is an interface to vectordb

#use retriever and document chain to get the response by combining them making a retriever chain.
#this chain has both the things combined LLM and the prompt based on which it generates our output.
retrieval_chain=create_retrieval_chain(retriever,document_chain)

responses= retrieval_chain.invoke(
    {
        "input": "Please evaluate and provide a contextual summary focusing on the above traits. \
        \n Communication style : Clarity and effectiveness in expression.\
        \n Active Listening: The candidate’s attentiveness and responsiveness to questions.\
        \n Engagement with the Interviewer: The candidate’s level of interaction, rapport, and attentiveness.\
                  \n\nHere is the candidate's transcript: \
                  \n{transcript}"
    }
)


st.title("Interview Analysis")

# Step 1: Upload Video File
st.header("Step 1: Upload Interview Video")
video_file = st.file_uploader("Upload a video file (e.g., .mp4)", type=["mp4"])

if video_file:
    # Save the uploaded video locally
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Transcribe the video
    st.info("Transcribing video... Please wait.")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe("uploaded_video.mp4")

    if transcript.status == aai.TranscriptStatus.error:
        st.error("Transcription Error: {transcript.error}")
    else:
        st.success("Transcription completed!")

        # Generate embeddings for the transcript
        st.info("Processing transcript and generating embeddings...")

        # Generate contextual summary
        st.header("Contextual Analysis")
        if st.button("Generate Summary"):
            
            st.subheader("Contextual Summary")
            st.write(responses['answer']['text'])
