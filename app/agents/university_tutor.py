from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Union
import os
import tempfile


def university_tutor(query: str, mode: str = "general", file_paths: Optional[List[str]] = None) -> str:
    """
    Process academic queries using an AI tutor.

    Args:
        query: The user's academic question
        mode: The mode of operation - "general" for general questions, "file" for file-based summarization
        file_paths: List of file paths for summarization (required if mode is "file")

    Returns:
        A comprehensive academic response
    """
    if mode == "general":
        template = """<think>
        You are analyzing a student's academic question. Consider:
        1. What field of study does this question belong to?
        2. What level of detail would be appropriate?
        3. What key concepts should be explained?
        4. What examples would help illustrate these concepts?
        5. Are there any common misconceptions to address?
        </think>

        You are a highly knowledgeable university professor with expertise across multiple disciplines.

        INSTRUCTIONS:
        - Provide comprehensive, academically rigorous explanations
        - Break down complex topics into clear, understandable components
        - Use specific examples and analogies to illustrate concepts
        - Structure your response with appropriate headings and subheadings
        - Include relevant equations, theories, or models when applicable
        - Address common misconceptions related to the topic
        - When appropriate, suggest further reading or related concepts
        - Use academic language while remaining accessible

        Student question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b", 
            temperature=0.2
        )

        chain = (
            {"question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        def get_response(input_query):
            return chain.invoke(input_query)

        response = get_response(query)
        return response

    elif mode == "file":
        if not file_paths:
            raise ValueError("File paths must be provided in 'file' mode.")

        documents = []
        for file_path in file_paths:
            try:
                loader = get_file_loader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                return f"Error processing file: {os.path.basename(file_path)}. {str(e)}"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        consolidated_text = "\n\n".join([chunk.page_content for chunk in chunks])

        MAX_CONTEXT_LENGTH = 32000
        if len(consolidated_text) > MAX_CONTEXT_LENGTH:
            consolidated_text = consolidated_text[:MAX_CONTEXT_LENGTH] + "..."

        template = """<think>
        You are analyzing files uploaded by a student. Consider:
        1. What are the main topics covered in these documents?
        2. What specific information does the student want to know?
        3. How can you best organize the information to answer their query?
        4. What key insights from the documents are most relevant?
        </think>

        You are a highly skilled academic assistant helping a student understand the content of their documents.

        DOCUMENT CONTENT:
        {document_content}

        STUDENT QUESTION:
        {question}

        INSTRUCTIONS:
        - Answer the student's question using ONLY information from the provided documents if document information is not available use your own knowledge to answer user query
        - Provide a comprehensive, well-structured response
        - Use headings and sections to organize your response when appropriate
        - Include relevant details, quotes, data, or examples from the documents
        - If the student's question cannot be answered from the documents, state this clearly
        - When referencing specific content, indicate which document or section it came from
        - Maintain academic rigor while ensuring clarity and accessibility
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.1,
            max_tokens=2048
        )

        chain = (
            {"document_content": lambda x: consolidated_text, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        def get_response(input_query, content):
            return chain.invoke(input_query)

        response = get_response(query, consolidated_text[:100])

        file_names = [os.path.basename(path) for path in file_paths]
        response += f"\n\n_Analysis based on {len(file_names)} document(s): {', '.join(file_names)}_"

        return response

    else:
        raise ValueError("Invalid mode. Use 'general' or 'file'.")

def get_file_loader(file_path: str):
    """Get the appropriate document loader based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    elif file_extension == '.csv':
        return CSVLoader(file_path)
    elif file_extension in ['.docx', '.doc']:
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def summarize_file(query: str, file_paths: List[str]) -> str:
    """
    Summarize content from uploaded files and answer questions about them.
    
    Args:
        query: The user's question about the document
        file_paths: Paths to the uploaded files
        
    Returns:
        A response addressing the query in the context of the uploaded files
    """
    # Load and process documents
    documents = []
    for file_path in file_paths:
        try:
            loader = get_file_loader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            return f"Error processing file: {os.path.basename(file_path)}. {str(e)}"
    
    # Split text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Prepare consolidated text for the model
    consolidated_text = "\n\n".join([chunk.page_content for chunk in chunks])
    
    # If text is too long, trim it while keeping as much as possible
    MAX_CONTEXT_LENGTH = 32000
    if len(consolidated_text) > MAX_CONTEXT_LENGTH:
        consolidated_text = consolidated_text[:MAX_CONTEXT_LENGTH] + "..."
    
    # Generate summary or answer question about the document
    template = """<think>
    You are analyzing files uploaded by a student. Consider:
    1. What are the main topics covered in these documents?
    2. What specific information does the student want to know?
    3. How can you best organize the information to answer their query?
    4. What key insights from the documents are most relevant?
    </think>
    
    You are a highly skilled academic assistant helping a student understand the content of their documents.
    
    DOCUMENT CONTENT:
    {document_content}
    
    STUDENT QUESTION:
    {question}
    
    INSTRUCTIONS:
    - Answer the student's question using ONLY information from the provided documents if document information is not available use your own knowledge to answer user query
    - Provide a comprehensive, well-structured response
    - Use headings and sections to organize your response when appropriate
    - Include relevant details, quotes, data, or examples from the documents
    - If the student's question cannot be answered from the documents, state this clearly
    - When referencing specific content, indicate which document or section it came from
    - Maintain academic rigor while ensuring clarity and accessibility
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Use a more powerful model for document analysis
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.1,
        max_tokens=2048
    )
    
    chain = (
        {"document_content": lambda x: consolidated_text, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    def get_response(input_query, content):
        return chain.invoke(input_query)
    
    response = get_response(query, consolidated_text[:100])  # Use first 100 chars of content as cache key
    
    # Include info about processed files
    file_names = [os.path.basename(path) for path in file_paths]
    response += f"\n\n_Analysis based on {len(file_names)} document(s): {', '.join(file_names)}_"
    
    return response