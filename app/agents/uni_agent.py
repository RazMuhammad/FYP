import pinecone
import os
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

def uni_agent(query: str, embedding_model: Embeddings) -> str:
    """
    Handle university-specific inquiries using a retrieval-augmented generation approach.
    
    Args:
        query: The user's university-related question
        embedding_model: The embedding model for vector search
        
    Returns:
        A comprehensive response based on university knowledge base
    """
    # Get Pinecone API key from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        return "Error: Pinecone API key not found. Please check your environment variables."

    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    
    # Multi-query generation prompt
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""<think>
        I need to generate multiple versions of this question to improve vector search retrieval.
        I should consider:
        1. Different phrasings of the same core question
        2. More specific versions of the question
        3. More general versions of the question
        4. Questions that focus on different aspects of the topic
        5. Questions that use different terminology for the same concepts
        </think>
        
        Generate five different versions of the given question to improve retrieval of relevant documents.
        Create variations that might match different ways the information could be stored in a university knowledge base.
        
        Original question: {question}
        
        Provide exactly 5 alternative questions, each on a new line, without numbering or explanation:
        """,
    )

    # Response generation prompt
    template = """<think>
    I'm analyzing a university-related question and the retrieved context. I need to:
    1. Identify the specific information request
    2. Determine if the context contains the answer
    3. Plan how to structure my response to be most helpful
    4. Identify any key university policies, procedures, or details to include
    </think>
    
    You are a specialized university information assistant with access to the institution's knowledge base.
    
    CONTEXT INFORMATION:
    {context}
    
    STUDENT QUESTION:
    {question}
    
    INSTRUCTIONS:
    - Answer the question comprehensively using ONLY information from the provided context
    - Structure your response with clear headings and organized sections
    - Include specific details, dates, requirements, and procedures relevant to the question
    - When referencing forms or applications, include how to access them
    - Specify relevant departments, offices, or contact information when applicable
    - If the context doesn't fully answer the question, clearly state what information is available
      and what's missing, suggesting where the student might find complete information
    - Maintain a helpful, informative tone throughout your response
    """

    # Set up the language model
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b", 
        max_tokens=2048
    )

    # Set up vector store and retriever
    try:
        index_name = "aup-website-data"
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
        
        # Base retriever with improved parameters
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,  # Increase number of results
                "score_threshold": 0.4,  # Slightly lower threshold for better recall
            },
        )
        
        # Multi-query retriever for improved results
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=QUERY_PROMPT
        )
        
        # Define the retrieval function with error handling
        def get_relevant_documents(query_str: str) -> List[Document]:
            try:
                return retriever.get_relevant_documents(query_str)
            except Exception as e:
                # Return a document with the error information
                return [Document(
                    page_content="Error retrieving information from the university knowledge base. " +
                                "The system may be experiencing technical difficulties.",
                    metadata={"error": str(e)}
                )]
        
        # Set up the chain
        chain = (
            {"context": lambda x: get_relevant_documents(x), "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        
        def get_response(input_query):
            return chain.invoke(input_query)
        
        response = get_response(query)
        return response
        
    except Exception as e:
        return f"I encountered an issue with the university knowledge base: {str(e)}. Please try again later or contact technical support if the problem persists."