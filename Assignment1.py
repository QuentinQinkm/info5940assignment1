import streamlit as st
from openai import OpenAI
from os import environ
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import logging

st.set_page_config(page_title="📝 Document Q&A", layout="wide")
st.title("📝 File Q&A with OpenAI")

@st.cache_resource
def get_openai_client():
    try:
        return OpenAI(api_key=environ.get('OPENAI_API_KEY'))
    except:
        return None

client = get_openai_client()
embedding_model = OpenAIEmbeddings(
    api_key=environ.get('OPENAI_API_KEY'),
    model="openai.text-embedding-3-small"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "documents" not in st.session_state:
    st.session_state.documents = {}

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

#Extract
def extract_text(uploaded_file):
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            return extract_text_from_pdf(uploaded_file.getvalue())
        
        elif file_type == 'txt' or file_type == 'md':
            return uploaded_file.getvalue().decode("utf-8")
        
        else:
            st.error("Error")
            return None
        

    except Exception as e:
        st.error("Error")
        return None
#Split
def chunk_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        print(f"Split text into {len(chunks)} chunks")
        
        return chunks
    except Exception as e:
        st.error("Error")
        return None

#vector
def update_vector_store():
    try:
        all_chunks = []
        all_metadatas = []
        
        for doc_id, doc_info in st.session_state.documents.items():
            for i, chunk in enumerate(doc_info["chunks"]):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_info["file_name"],
                    "chunk_id": i
                })
        
        if not all_chunks:
            st.session_state.vector_store = None
            return
            
        #combine
        st.session_state.vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=embedding_model,
            metadatas=all_metadatas
        )
    except Exception as e:
        st.error(f"Error")
        st.session_state.vector_store = None


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

##Multi
uploaded_files = st.file_uploader("Upload an article", type=("txt", "md", "pdf"), accept_multiple_files=True)
##

if uploaded_files:
    new_files = False
    
    for uploaded_file in uploaded_files:

        file_id = f"{uploaded_file.name}_{id(uploaded_file)}"
        

        if file_id not in st.session_state.documents:
            file_content = extract_text(uploaded_file)
            
            if file_content:
                chunks = chunk_text(file_content)
                
                if chunks:
                    st.session_state.documents[file_id] = {
                        "file_name": uploaded_file.name,
                        "content": file_content,
                        "chunks": chunks
                    }
                    new_files = True
    
    if new_files:
        update_vector_store()


question = st.chat_input(
    "Ask something about the article",
    disabled=len(st.session_state.documents) == 0,
)



if question and st.session_state.vector_store:

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        try:
            search_results = st.session_state.vector_store.similarity_search(
                question,
                k=3 
            )
            
            context = "\n\n".join([doc.page_content for doc in search_results])
            sources = set([doc.metadata["source"] for doc in search_results])
            
            messages = [
                {"role": "system", "content": f"Content of the file:\n\n{context}"},
                {"role": "user", "content": question}
            ]
            
            stream = client.chat.completions.create(
                model="openai.gpt-4o",
                messages=messages,
                stream=True
            )
            
            response = st.write_stream(stream)
            
            if len(sources) > 0:
                st.write(f"Sources: {', '.join(sources)}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})