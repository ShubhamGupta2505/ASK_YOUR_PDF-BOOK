import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

def main():
    load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
    st.set_page_config(page_title="Ask Your Pdf")
    st.header("Ask Your Pdf ")
    pdf = st.file_uploader("Upload Your PDF ",type="pdf")
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
    #   st.write(text)

      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )

      chunk = text_splitter.split_text(text)

      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunk,embeddings)

      user_question = st.text_input("Ask a Question about your PDF")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.run(input_documents=docs, question=user_question)

        st.write(response)

    
    #   st.write(chunk)



if __name__ == '__main__':
    main()