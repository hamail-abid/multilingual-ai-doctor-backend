import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

CHROMA_PATH = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
def query_rag(query_text: str, language: str = "Urdu"):
    # 1. Embeddings Model load karna
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2. Existing Chroma DB load karna
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 3. Query ke mutabiq top 3 matching chunks search karna
    results = db.similarity_search(query_text, k=3)
    
    if not results:
        return "System mein is query se mutalliq koi context nahi mila."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # 4. System Prompt with Guardrails
    system_prompt = f"""
    You are a professional Medical AI Assistant specializing in Cardiology (Heart Health/Hypertension) and General Health.
    Answer the user's query strictly using the provided context below.

    Context:
    {context_text}

    Instructions:
    - Respond in {language} language.
    - Keep the explanation clear, empathetic, and factual based on the context.
    - MANDATORY DISCLAIMER: Always end your response with this disclaimer in {language}:
      "Note: This information is for general awareness only and does NOT replace a certified doctor or cardiologist. Please consult a registered physician for official diagnosis."
    """

    # 5. Groq LLM Call (Updated Active Model Name)
    llm = ChatGroq(
        temperature=0.2,
        model_name="openai/gpt-oss-20b",
        groq_api_key=GROQ_API_KEY
    )

    response = llm.invoke(system_prompt + f"\n\nUser Question: {query_text}")
    return response.content

# Interactive Chat Loop Function
def start_terminal_chat():
    print("\n==========================================")
    print("  AI Doctor Terminal Chat Started!")
    print("  Type 'exit' or 'quit' to end the chat.")
    print("==========================================\n")
    
    while True:
        user_input = input("\nYou (Patient): ")
        if user_input.lower().strip() in ["exit", "quit", "stop"]:
            print("Chat ended. Take care!")
            break
            
        if not user_input.strip():
            continue

        print("\nAI Doctor (Searching context & thinking...)...")
        try:
            answer = query_rag(user_input, language="Urdu")
            print(f"\nAI Doctor: {answer}")
        except Exception as e:
            print(f"\nError aaya hai: {e}")

if __name__ == "__main__":
    start_terminal_chat()