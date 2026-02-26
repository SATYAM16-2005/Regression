import os
import certifi
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fix for SSL certificate issues in some Windows environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self):
        self._embeddings = None
        self._llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = None

    def _get_embeddings(self):
        if not self._embeddings:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API Key not found. Please provide it in the sidebar.")
            self._embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        return self._embeddings

    def _get_llm(self):
        if not self._llm:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API Key not found. Please provide it in the sidebar.")
            self._llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        return self._llm

    def process_pdf(self, pdf_path):
        """
        Loads a PDF, splits text, and creates a FAISS vector store.
        """
        print(f"DEBUG: Processing PDF at path: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, self._get_embeddings())
        return len(chunks)

    def query(self, question):
        """
        Queries the RAG chain with a user question using LCEL.
        """
        if not self.vector_store:
            return "Please upload a document first."

        retriever = self.vector_store.as_retriever()
        
        template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self._get_llm()
            | StrOutputParser()
        )

        try:
            response = rag_chain.invoke(question)
            return response
        except Exception as e:
            return f"Error during query: {str(e)}"
