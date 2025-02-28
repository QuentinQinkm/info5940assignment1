# info5940assignment1

Features:

Supports multiple documents
Supports PDF, TXT, and MD files
Uses vector search to find information
Shows answers in real-time
Tells you which documents were used
---------

I used:

Streamlit WebUI
GPT-4o model
PyPDF2 for processing PDF files
LangChain for processing text
FAISS for searching through document content


---------
Steps:
1. Upload one or more documents using the file uploader
2. Wait for the processing to complete
3. Type your question in the chat box
4. View the answer and which documents it came from

---------
Need to install:
pip install langchain PyPDF2 faiss-cpu openai streamlit



I didn't make any changes to the Docker or devcontainer configurations from the README.md in lecture5's repo.
