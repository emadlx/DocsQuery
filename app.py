import os
import streamlit as st
from dotenv import load_dotenv


if not st.secrets:  
    load_dotenv()


OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

HUGGINGFACEHUB_API_TOKEN = st.secrets.get(
    "HUGGINGFACEHUB_API_TOKEN", 
    os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



from pdf2image import convert_from_bytes
import pytesseract
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS      
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub





def get_pdf_text(pdf_docs):
    text = ""
    for uploaded_file in pdf_docs:
        # Convert each page to a PIL image
        images = convert_from_bytes(uploaded_file.read())
        for img in images:
            # OCR the image to text
            page_text = pytesseract.image_to_string(img)
            if page_text:
                text += page_text + "\n"
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # 1) Inject your custom CSS
    st.markdown(css, unsafe_allow_html=True)

    # 2) Init session state if missing
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 3) Main interface
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    # 4) Only run the chain if we've processed documents already
    if user_question:
        if st.session_state.conversation is None:
            st.error(
                "⚠️ Please upload one or more PDF files and click **Process** before asking questions."
            )
        else:
            handle_userinput(user_question)

    # 5) Sidebar for uploads & processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click **Process**", 
            type=None,  # allow any type, we’ll validate ourselves
            accept_multiple_files=True
        )

        if st.button("Process"):
            # 5a) No files at all?
            if not pdf_docs:
                st.warning("⚠️ No files selected. Please upload at least one PDF file.")
                return

            # 5b) Reject any non-PDF by extension
            bad = [f.name for f in pdf_docs if not f.name.lower().endswith(".pdf")]
            if bad:
                st.error(
                    "❌ The following are not PDF files:\n\n" +
                    "\n".join(f"- {name}" for name in bad) +
                    "\n\nPlease remove them and upload only .pdf files."
                )
                return

            # 5c) Everything looks good → process
            with st.spinner("🔍 Processing PDF documents…"):
                raw_text    = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Documents processed! You can now ask questions above.")



if __name__ == '__main__':
    main()
