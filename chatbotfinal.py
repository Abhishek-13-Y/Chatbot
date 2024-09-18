import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from datetime import datetime
import tempfile
import os
import bcrypt
import re

# Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
OPENAI_API_BASE = "http://localhost:1234/v1"
OPENAI_API_KEY = "1234"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 50

USER_DATA = {
    "alice": {"password_hash": bcrypt.hashpw("password123".encode(), bcrypt.gensalt()), "department": "Finance", "sensitivity": "Confidential", "name": "Alice"},
    "bob": {"password_hash": bcrypt.hashpw("password123".encode(), bcrypt.gensalt()), "department": "HR", "sensitivity": "Internal", "name": "Bob"},
    "charlie": {"password_hash": bcrypt.hashpw("password123".encode(), bcrypt.gensalt()), "department": "Procurement", "sensitivity": "Public", "name": "Charlie"}
}

# Authentication
def authenticate(username, password):
    user = USER_DATA.get(username)
    if user and bcrypt.checkpw(password.encode(), user["password_hash"]):
        return user
    return None

# Streamlit Page Config
def configure_page():
    st.set_page_config(page_title="ChatBot", page_icon="🪖")
    st.title("ChatBot")

# Login
def handle_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.sidebar.header("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state.update(logged_in=True, **user)
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    else:
        st.sidebar.write(f"**Name:** {st.session_state.name}")
        st.sidebar.write(f"**Department:** {st.session_state.department}")
        st.sidebar.write(f"**Access Level:** {st.session_state.sensitivity}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

# Upload Files & Metadata
def upload_files():
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    metadata_list = []
    for file in uploaded_files:
        with st.sidebar.expander(f"Metadata for {file.name}"):
            sensitivity = st.selectbox("Select sensitivity", ["Public", "Internal", "Confidential", "Highly Confidential"], key=f"sensitivity_{file.name}")
            date = st.date_input("Select date", key=f"date_{file.name}").strftime("%Y-%m-%d")
            categories = st.multiselect("Select categories", ["Finance", "Procurement", "HR", "Operations"], key=f"categories_{file.name}")

            if not (sensitivity and date and categories):
                st.warning(f"All metadata fields are mandatory for {file.name}.")
                continue
            
            metadata_list.append({"file": file, "sensitivity": sensitivity, "date": date, "categories": categories})
    return metadata_list

# Process Documents
def process_documents(metadata_list):
    with tempfile.TemporaryDirectory() as temp_dir:
        docs = []
        for metadata in metadata_list:
            temp_filepath = os.path.join(temp_dir, metadata["file"].name)
            with open(temp_filepath, "wb") as f:
                f.write(metadata["file"].getvalue())
            loader = PyPDFLoader(temp_filepath)
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata.update({"source": metadata["file"].name, "sensitivity": metadata["sensitivity"], "date": metadata["date"], "categories": metadata["categories"]})
            docs.extend(file_docs)
        return docs

def filter_and_prioritize_docs(docs, department, sensitivity):
    sensitivity_levels = ["Public", "Internal", "Confidential", "Highly Confidential"]
    allowed_sensitivities = sensitivity_levels[:sensitivity_levels.index(sensitivity) + 1]
    filtered_docs = [doc for doc in docs if doc.metadata.get("sensitivity") in allowed_sensitivities and department in doc.metadata.get("categories", [])]
    return sorted(filtered_docs, key=lambda doc: datetime.strptime(doc.metadata.get("date", "1970-01-01"), "%Y-%m-%d"), reverse=True)

@st.cache_resource(ttl="1h")
def configure_retriever(metadata_list, department, sensitivity):
    docs = process_documents(metadata_list)
    filtered_docs = filter_and_prioritize_docs(docs, department, sensitivity)
    if not filtered_docs:
        st.warning("No documents available for your department and sensitivity level.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(filtered_docs)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectordb = FAISS.from_documents(splits, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})

# Callback Handlers
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token, **kwargs):
        if self.run_id_ignore_token == kwargs.get("run_id"):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized, query, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            self.status.write(f"**Document {idx + 1} from {doc.metadata.get('source', 'Unknown Source')} (Sensitivity: {doc.metadata.get('sensitivity', 'Unknown Sensitivity')}, Date: {doc.metadata.get('date', 'Unknown Date')}, Priority: {idx + 1})**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Prompt Sanitization
def sanitize_prompt(prompt):
    disallowed_patterns = [
        r"(?i)drop\s+table",   # SQL injection patterns
        r"(?i)union\s+select", # SQL injection patterns
        r"(?i)exec\s+sp_",    # Stored procedure injection patterns
        r"(?i)select\s+from",  # SQL injection patterns
        r"(?i)alert\(",        # XSS patterns
        r"(?i)javascript:",    # XSS patterns
        r"(?i)script\s*=",     # XSS patterns
    ]

    for pattern in disallowed_patterns:
        if re.search(pattern, prompt):
            st.warning("Your prompt contains disallowed content and cannot be processed.")
            return None
    return prompt

# Main App
def main():
    configure_page()
    handle_login()

    if not st.session_state.logged_in:
        return

    department = st.session_state.department
    sensitivity = st.session_state.sensitivity
    metadata_list = upload_files()
    if not metadata_list:
        st.info("Please upload PDF documents and fill in the metadata to continue.")
        return

    retriever = configure_retriever(metadata_list, department, sensitivity)
    if retriever is None:
        return

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OpenAI(openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY, temperature=0.3, streaming=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, chain_type='stuff', retriever=retriever, memory=memory)

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        sanitized_query = sanitize_prompt(user_query)
        if sanitized_query:
            st.chat_message("user").write(user_query)
            with st.chat_message("assistant"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(sanitized_query, callbacks=[retrieval_handler, stream_handler])

if __name__ == "__main__":
    main()
