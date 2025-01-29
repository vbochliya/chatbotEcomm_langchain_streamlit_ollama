# Chatbot E-commerce Project

**Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Contributing](#contributing)  
7. [License](#license)

---

## Overview

This project provides an **e-commerce chatbot** designed to assist customers with common inquiries, such as **order tracking**, **return policies**, **payment methods**, and more. The chatbot uses:

- **Streamlit** for the user interface  
- **LangChain** for building a conversational retrieval pipeline  
- **Ollama** for local large language model (LLM) embeddings and inference  
- **FAISS** as a vector store for semantic search over FAQ documents

---

## Features

- **FAQ-based Conversational Chat**: The bot answers user queries using a pre-loaded FAQ knowledge base.  
- **Embeddings and Vector Search**: All FAQ documents are turned into embeddings for accurate semantic matching.  
- **Local LLM Support**: Utilizes [Ollama](https://github.com/jmorganca/ollama) for running an LLM such as `llama3.2:latest` on your machine.  
- **Easy to Extend**: Add more FAQ entries or integrate with other data sources as needed.

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/vbochliya/chatbotEcomm_langchain_streamlit_ollama.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd chatbotEcomm_langchain_streamlit_ollama
    ```
3. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

> **Note**: You must also have [Ollama](https://github.com/jmorganca/ollama) installed and running locally for this to work.

---

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
