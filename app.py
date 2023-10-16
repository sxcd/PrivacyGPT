from dotenv import load_dotenv
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Privacy GPT Alpha")
    
    # Add a PNG image as the header
    st.image("logotrans.png", use_column_width=True)
    
 # Add a short description of the app with bullet points
    description = """
    Welcome to Privacy GPT Alpha! This app is trained on the most current version of Data Privacy laws focusing on the Middle East. Please keep in mind:
    - Currently only KSA is supported.
    - Do not send an influx of questions as it costs my OpenAI key.
    - Unlike ChatGPT, craft your questions purely towards the text of the law. Copy the output to ChatGPT for finetuning the text if needed.
    - Answers will dissapear on each query.
    """
    st.markdown(description)
    
    # create vector store
    vector_store = get_vector_store()
    
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask anything about Data Privacy laws in scope!")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")
    
        
if __name__ == '__main__':
    main()
