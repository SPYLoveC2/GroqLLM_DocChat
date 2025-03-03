# GroqLLM_DocChat

## Overview

GroqLLM_DocChat is an AI-powered document chat system that extracts text from PDFs, generates vector embeddings, stores them in a vector database, and allows users to query information using natural language. It leverages LangChain, FAIS and google's embeddings to provide an efficient document retrieval and question-answering system.

## Features

- 📄 Extracts text from PDFs, PPTs, etc.
- 🔍 Stores embeddings in a vector database (FAISS)
- 🤖 Uses AI embeddings for semantic search
- 💬 Allows users to ask natural language questions about the document
- 🚀 Fast and scalable retrieval-augmented generation (RAG)
- ⚡ Uses Groq API for fast inferencing

## Tech Stack

- Python (Core language)
- LangChain (Document processing and retrieval)
- FAISS (Vector database)
- google/models/text-embedding-004 (Embeddings model)
- Docling (Document text extraction)
- Streamlit (API or UI for interaction)

## Installation

Clone the repository:

```sh
git clone https://github.com/SPYLoveC2/GroqLLM_DocChat.git 
cd GroqLLM_DocChat
```

Create a virtual environment and install dependencies:

```sh
conda env create -f environment.yml
```

## Usage

Start the chatbot interface:

```sh
streamlit run app.py  # If using Streamlit UI
```

Ask questions about the document in the UI or API.

## Configuration

Set your API keys (if using OpenAI embeddings) in an `.env` file:

```sh
GOOGLE_API_KEY=your_api_key
GROQ=your_api_key
```

## Roadmap

- ✅ Basic Doc processing & embedding storage
- ✅ Q&A system with retrieval
- 🔜 Multi-document support
- 🔜 Fine-tuned LLM integration

## Contributing

Feel free to submit issues or pull requests to improve the project.

