import os
custom_cache_dir = r"D:\MyAI_Models\huggingface_cache"
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = custom_cache_dir
import streamlit as st
import torch
from transformers import pipeline
from google.cloud import translate_v3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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

@st.cache_resource
def load_pytorch_analyzer():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment" 
    analyzer = pipeline("text-classification", model=model_name, framework="pt")
    return analyzer

@st.cache_resource
def setup_polish_rag(file_path="PolskiA2B1.pdf"):
    final_docs = []

    if not os.path.exists(file_path):
        st.warning(f"File {file_path} not found. Loading basic sentences.")
        texts = [
            "Cześć, jak się masz? (Hi, how are you?)",
            "Bardzo dziękuję za pomoc. (Thank you very much for the help.)",
            "Gdzie jest najbliższa stacja kolejowa? (Where is the nearest train station?)"
        ]
        final_docs = [Document(page_content=t) for t in texts]
    else:
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        final_docs = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    
    return vectorstore

def main():
    st.set_page_config(page_title="Polish Learner AI", page_icon="🇵l")
    st.title("Polish Language Learning Assistant")
    st.markdown("**PyTorch + LangChain + RAG**")
    
    project_id = "banded-pager-420903" # Your Google Project ID
    translator = PolishTranslator(project_id)
    pytorch_analyzer = load_pytorch_analyzer()
    rag_engine = setup_polish_rag()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", project=project_id)

    with st.sidebar:
        st.header("🔑 Authentication")
        user_api_key = st.text_input("Google AI API Key:", type="password", placeholder="AIza...")
        try:
            if user_api_key:
                # OPTION A: User provided their own API Key (Google AI Studio)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    google_api_key=user_api_key
                )
                st.sidebar.success("Using your API Key")
            else:
                # OPTION B: Default to your Project ID (Vertex AI)
                llm = ChatVertexAI(
                    model_name="gemini-2.5-flash-lite", 
                    project=project_id
                )
                st.sidebar.warning("Using Developer's Project ID")
        except Exception as e:
            st.error("Could not initialize AI. Please check your API key or Project settings.")
    
        st.divider()
        st.header("🛠️ Dictionary & Tools")
        lookup_text = st.text_area("Quick Translate & Search:")
        if lookup_text:
            trans = translator.translate(lookup_text, "en")
            st.success(f"**EN:** {trans['translated_text']}")
            
            st.write("**From Textbook:**")
            docs = rag_engine.similarity_search(lookup_text, k=2)
            for doc in docs:
                st.info(doc.page_content)
    


    # ==========================================
    # MAIN SCREEN: Chat Interface
    # ==========================================
    st.title("🇵🇱 Chat with your AI Polish Tutor")
    st.markdown("Practice your Polish! The AI will correct your grammar and reply to keep the conversation going.")

    # 1. Initialize Chat History in Session State
    if "messages" not in st.session_state:
        # The System Message tells the AI how to behave
        system_instructions = """
        You are a friendly and encouraging Polish language tutor. 
        When the user says something:
        1. First, judge if their Polish is grammatically correct. 
        2. If they made a mistake, gently correct them and briefly explain why (e.g., wrong case, wrong verb tense).
        3. Finally, reply to their message naturally in Polish to keep the conversation going. Include an English translation of your reply in parentheses.
        If they speak in English, help them translate it to Polish.
        """
        st.session_state.messages = [SystemMessage(content=system_instructions)]

    # 2. Display previous chat messages
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue # Don't show the secret system instructions on screen
        
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # 3. Chat Input area for the user
    if prompt := st.chat_input("Napisz coś po polsku... (e.g., Mam na imię Jan)"):
        
        # Add user message to memory and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Nauczyciel pisze... (Teacher is typing...)"):
                # Pass the whole conversation history to the AI
                response = llm.invoke(st.session_state.messages)
                st.write(response.content)
                
                # Add AI response to memory
                st.session_state.messages.append(AIMessage(content=response.content))

if __name__ == "__main__":
    main()