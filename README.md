# University AI Assistant

An intelligent assistant designed to help university students and staff with academic questions, university-specific information, and document analysis.

---

## Features

- **Smart Query Routing**: Automatically determines the best way to answer each question.
- **University Knowledge Base**: Access to university-specific information through vector search.
- **Web Search Integration**: Retrieves up-to-date information from the web when needed.
- **Academic Support**: Provides thorough explanations of academic concepts and theories.
- **Document Analysis**: Uploads and analyzes PDFs, Word documents, text files, and CSVs.
- **Fast Response Caching**: Improves performance with intelligent response caching.

---

## Visual Overview

### Data Scraping Process
![Data Scraping Process](images/Image%20that%20show%20the%20data%20scraping%20process.webp)

### Chunking and Embedding Steps
![Chunking and Embedding](images/Image%20that%20show%20chunking%20and%20embedding%20steps.webp)

### Embedding Visualization in 2D Space
![Embedding Visualization](images/Image%20the%20show%20the%20embedding%20visualization%20in%202D%20space.png)

### Final RAG Structure
![RAG Structure](images/Image%20that%20show%20our%20final%20RAG%20structure.png)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/university-ai-assistant.git
   cd university-ai-assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory and add your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key
     TAVILY_API_KEY=your_tavily_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

---

## Usage

1. **Start the application:**
   ```bash
   streamlit run main.py
   ```

2. **Access the app:**
   - Open your browser and go to the local URL provided by Streamlit (usually http://localhost:8501).

3. **How to use:**
   - Upload any documents (optional).
   - Type your question in the chat.
   - Get intelligent responses.

---

## Project Structure

```
.
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
├── app/
│   ├── agents/
│   │   ├── generalQA.py
│   │   ├── uni_agent.py
│   │   └── web_agent.py
│   ├── core/
│   │   └── decision_maker.py
│   └── utils/
│       └── embeddings.py
├── images/                # Project images and diagrams
└── ...
```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
