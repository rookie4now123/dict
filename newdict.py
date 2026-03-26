import streamlit as st
import torch
from transformers import pipeline
from google.cloud import translate_v3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os

# --- 1. GOOGLE TRANSLATOR ---
class PolishTranslator:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = translate_v3.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/global"

    def translate(self, text: str, target_lang: str) -> dict:
        response = self.client.translate_text(
            contents=[text],
            parent=self.parent,
            mime_type="text/plain",
            target_language_code=target_lang,
        )
        return {"translated_text": response.translations[0].translated_text}

# --- 2. PYTORCH: Linguistic Analysis ---
@st.cache_resource
def load_pytorch_analyzer():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment" 
    analyzer = pipeline("text-classification", model=model_name, framework="pt")
    return analyzer

# --- 3. LANGCHAIN & RAG: Example Sentence Retrieval ---
@st.cache_resource
def setup_polish_rag(file_path="PolskiA2B1.pdf"):
    # 1. INITIALIZE DATA
    final_docs = []

    if not os.path.exists(file_path):
        st.warning(f"File {file_path} not found. Loading basic sentences.")
        # We manually create Document objects so the code below stays the same
        texts = [
            "Cześć, jak się masz? (Hi, how are you?)",
            "Bardzo dziękuję za pomoc. (Thank you very much for the help.)",
            "Gdzie jest najbliższa stacja kolejowa? (Where is the nearest train station?)"
        ]
        final_docs = [Document(page_content=t) for t in texts]
    else:
        # 2. LOAD FROM FILE
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        
        raw_documents = loader.load()

        # 3. CHUNK THE DOCUMENTS
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        final_docs = text_splitter.split_documents(raw_documents)

    # 4. CREATE VECTOR STORE
    # We use HuggingFaceEmbeddings (PyTorch based)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # CRITICAL FIX: Use 'from_documents' specifically
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    
    return vectorstore

# --- STREAMLIT UI ---
def main():
    st.set_page_config(page_title="Polish Learner AI", page_icon="🇵l")
    st.title("Polish Language Learning Assistant")
    st.markdown("**PyTorch + LangChain + RAG**")
    
    # Initialize
    project_id = "banded-pager-420903" # Your Google Project ID
    translator = PolishTranslator(project_id)
    pytorch_analyzer = load_pytorch_analyzer()
    rag_engine = setup_polish_rag()

    # Layout
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("Input")
        user_text = st.text_input("Enter a word or sentence:", placeholder="e.g., Dzień dobry")
        mode = st.radio("Target:", ["Polish to English", "English to Polish"])
        
        target_lang = "en" if mode == "Polish to English" else "pl"
        analyze_button = st.button("Analyze & Learn")

    if analyze_button and user_text:
        with col_output:
            st.subheader("Results")
            
            # 1. Google Translation
            translation = translator.translate(user_text, target_lang)
            st.success(f"**Translation:** {translation['translated_text']}")

            # 2. PyTorch Analysis (Linguistic Emotion/Tone)
            # This shows you can handle PyTorch model outputs
            analysis = pytorch_analyzer(user_text)[0]
            st.write(f"**AI Tone Detection (PyTorch):** {analysis['label']} ({round(analysis['score']*100)}%)")

            # 3. LangChain RAG (Context Retrieval)
            # This fulfills the "RAG techniques for grounding" JD requirement
            st.write("**Related Examples (RAG Context):**")
            docs = rag_engine.similarity_search(user_text, k=2)
            for doc in docs:
                st.info(doc.page_content)

    # Technical Details for the Recruiter
    with st.expander("See Technical Architecture"):
        st.write("""
        - **Frontend:** Streamlit
        - **Translation:** Google Cloud Translate V3 API
        - **ML Framework:** PyTorch (via Transformers Pipeline)
        - **RAG Orchestration:** LangChain
        - **Vector Database:** FAISS (Facebook AI Similarity Search)
        - **Embeddings:** HuggingFace (running on PyTorch)
        """)

if __name__ == "__main__":
    main()