import streamlit as st

from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from faq_data import FAQS

llm = OllamaLLM(model="llama3.2:latest")
st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


# 1. Convert FAQs to LangChain Documents
def create_documents_from_faqs(faq_list):
    """
    Each FAQ is turned into a Document with content = question + answer.
    You can also store metadata if you'd like.
    """
    docs = []
    for faq in faq_list:
        # Combine question + answer in a single chunk (optional approach).
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        doc = Document(page_content=content, metadata={"source": "faq"})
        docs.append(doc)
    return docs


def create_embeddings():
    faq_docs = create_documents_from_faqs(FAQS)

    vectorstore = FAISS.from_documents(faq_docs, st.session_state.embeddings)
    vectorstore.save_local("faiss_index")


st.title("Chatbot for eCommerce")

#create embeddings for first time
# create_embeddings()

# get vectorstore from local storage
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=st.session_state.embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=True
)


# Streamlit session state for storing conversation history
if "history" not in st.session_state:
    st.session_state["history"] = []

# 6. Simple input box for user query
user_query = st.text_input("Ask a question based on the FAQs:", key="user_query")

# 7. Handle user query
if st.button("Ask"):
    if user_query.strip():
        # We pass the conversation history to maintain context
        chat_history = st.session_state["history"]
        response = qa_chain({"question": user_query, "chat_history": chat_history})
        answer = response["answer"]
        source_docs = response["source_documents"]

        # Display the answer
        st.write(f"**Answer**: {answer}")

        # Optionally display source documents
        with st.expander("Sources"):
            for doc in source_docs:
                st.write(doc.page_content)

        # Update chat history
        chat_history.append((user_query, answer))
    else:
        st.warning("Please enter a question.")

# 8. Optionally, display the entire conversation for clarity
if st.session_state["history"]:
    st.write("### Conversation History")
    for i, (q, a) in enumerate(st.session_state["history"], 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
