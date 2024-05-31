import streamlit as st
import os
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import AstraDB
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Callback handler to stream responses
class ResponseStreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function to process and vectorize uploaded documents into Astra DB
def process_and_vectorize_document(uploaded_file, vector_db):
    if uploaded_file is not None:
        
        # Create a temporary file to store the uploaded document
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.getvalue())

        # Load the PDF file
        document_pages = []
        pdf_loader = PyPDFLoader(temp_file_path)
        document_pages.extend(pdf_loader.load())

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100
        )

        # Split the document and add it to the vector store
        split_pages = splitter.split_documents(document_pages)
        vector_db.add_documents(split_pages)
        st.info(f"{len(split_pages)} pages have been loaded into the database.")

# Cache prompt for reuse
@st.cache_data()
def get_chat_prompt():
    prompt_template = """You are a helpful AI assistant here to answer the user's questions.
You are friendly and respond extensively with multiple sentences. You prefer using bullet points to summarize.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", prompt_template)])
chat_prompt = get_chat_prompt()

# Cache OpenAI Chat Model for reuse
@st.cache_resource()
def get_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-3.5-turbo',
        streaming=True,
        verbose=True
    )
chat_model_instance = get_chat_model()

# Cache the Astra DB Vector Store for reuse
@st.cache_resource(show_spinner='Connecting to AstraDB')
def get_vector_store():
    vector_store_instance = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="datastax",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store_instance
vector_store_instance = get_vector_store()

# Cache the Retriever for reuse
@st.cache_resource(show_spinner='Initializing retriever')
def get_retriever():
    retriever_instance = vector_store_instance.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever_instance
retriever_instance = get_retriever()

# Initialize session state for messages
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display title and description
st.title("Study Buddy")
st.markdown("""
Welcome to **Study Buddy**, your ultimate exam preparation assistant!

Harnessing the power of Retrieval-Augmented Generation (RAG), Study Buddy combines the latest advancements in AI to enhance your study sessions and boost your productivity. With Study Buddy, you can efficiently tackle complex topics, get quick answers to your questions, and receive personalized guidance to help you excel in your exams.

Let's transform the way you study and achieve your academic goals together!
""")

# Sidebar for document upload
with st.sidebar:
    with st.form('upload_form'):
        uploaded_file = st.file_uploader('Upload a document for enhanced context', type=['pdf'])
        submit_button = st.form_submit_button('Save to AstraDB')
        if submit_button:
            process_and_vectorize_document(uploaded_file, vector_store_instance)

# Display chat messages
for msg in st.session_state.chat_history:
    st.chat_message(msg['role']).markdown(msg['content'])

# Input box for user questions
if user_question := st.chat_input("What's on your mind?"):
    
    # Store the user's question in the session state
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Display the user's question
    with st.chat_message('user'):
        st.markdown(user_question)

    # Placeholder for the assistant's response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate response using the OpenAI Chat Model
    input_data = RunnableMap({
        'context': lambda x: retriever_instance.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    })
    response_chain = input_data | chat_prompt | chat_model_instance
    generated_response = response_chain.invoke({'question': user_question}, config={'callbacks': [ResponseStreamHandler(response_placeholder)]})
    assistant_answer = generated_response.content

    # Store the assistant's answer in the session state
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_answer})

    # Display the final response
    response_placeholder.markdown(assistant_answer)
