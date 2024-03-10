import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader

# Initialize Streamlit
st.title("Question Answering System")

# Function to read PDF and extract text
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    raw_text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to initialize Cassio and LangChain components
def initialize_components():
    ASTRA_DB_APPLICATION_TOKEN = "AstraCS:RYSEsgeCsJnSdLwHLUehxGOB:6e06dfaea59380f6e0d422e74244110151e7ac0e8854d916ffa7eb0f4adc7f92"
    ASTRA_DB_ID = "d79f649d-bc75-4e8b-bc65-f34a431770fc"
    OPENAI_API_KEY = "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"
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return llm, embedding

# Initialize components
llm, embedding = initialize_components()

# Read PDF and extract text
pdf_file_path = 'COI...pdf'  # Provide the path to your PDF file
raw_text = read_pdf(pdf_file_path)

# Split text into chunks
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Initialize Cassandra vector store and add texts
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)
astra_vector_store.add_texts(texts[:50])

# Create vector index
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Main Streamlit app
first_question = True
while True:
    if first_question:
        query_text = st.text_input("Enter your question (or type 'quit' to exit):")
    else:
        query_text = st.text_input("What's your next question (or type 'quit' to exit):")

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    st.write(f"\nQUESTION: \"{query_text}\"")
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    st.write(f"ANSWER: \"{answer}\"")

    st.write("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        st.write(f"    [%0.4f] \"{doc.page_content[:300]} ...\"" % score)
