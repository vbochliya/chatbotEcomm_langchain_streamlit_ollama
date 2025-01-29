import streamlit as st

# LangChain imports
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain

# FAISS vector store and Ollama embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Local FAQ data
from faq_data import FAQS

# 1. Initialize the Ollama LLM
llm = OllamaLLM(model="llama3.2:latest")

# 2. Store embeddings in session_state to avoid re-initialization on rerun
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


def create_documents_from_faqs(faq_list):
    """
    Convert each FAQ into a Document with 'Q' and 'A' combined.
    Storing question + answer in one chunk is a simple approach.
    """
    docs = []
    for faq in faq_list:
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        doc = Document(page_content=content, metadata={"source": "faq"})
        docs.append(doc)
    return docs


def create_embeddings():
    """
    Creates and saves the FAISS vector store for our FAQ documents.
    This should only be run once (or whenever FAQ data changes).
    """
    faq_docs = create_documents_from_faqs(FAQS)
    vectorstore = FAISS.from_documents(faq_docs, st.session_state.embeddings)
    vectorstore.save_local("faiss_index")
    st.success("FAISS index created and saved to 'faiss_index' folder!")


# Uncomment the following line if you need to create the FAISS index for the first time:
create_embeddings()

st.title("Chatbot for E-commerce")

# 3. Load the existing vector store from disk (FAISS index)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=st.session_state.embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever()

# 4. Create a Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Use session state to store conversation history
if "history" not in st.session_state:
    st.session_state["history"] = []

# 6. Text input for user queries
user_query = st.text_input("Ask a question based on the FAQs:", key="user_query")

if st.button("Ask"):
    # Ensure the question is not empty
    if user_query.strip():
        chat_history = st.session_state["history"]
        response = qa_chain({"question": user_query, "chat_history": chat_history})

        answer = response["answer"]
        source_docs = response["source_documents"]

        # Display the answer
        st.markdown(f"**Answer:** {answer}")

        # Display the relevant source chunks
        with st.expander("Sources"):
            for doc in source_docs:
                st.write(doc.page_content)

        # Update the conversation history
        chat_history.append((user_query, answer))
    else:
        st.warning("Please enter a question.")

# 7. Optionally, display the entire conversation
if st.session_state["history"]:
    st.write("### Conversation History")
    for i, (q, a) in enumerate(st.session_state["history"], 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
