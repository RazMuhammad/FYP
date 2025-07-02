from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def web_agent(query: str) -> str:
    """
    Handle queries requiring current web information using search and summarization.
    
    Args:
        query: The user's question requiring web search
        
    Returns:
        A comprehensive response based on web search results
    """
    # Check if Tavily API key is available
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Web search capabilities are currently unavailable. Please ask a different type of question."
    
    template = """<think>
    I'm analyzing web search results to answer a user question. I need to:
    1. Identify the most relevant information from the search results
    2. Synthesize information from multiple sources
    3. Determine how to present the information clearly
    4. Note any inconsistencies or gaps in the search results
    5. Organize my response logically
    </think>
    
    You are a web research specialist providing accurate, up-to-date information.
    
    SEARCH RESULTS:
    {context}
    
    USER QUESTION:
    {question}
    
    INSTRUCTIONS:
    - Synthesize a comprehensive answer based ONLY on the provided search results
    - Structure your response with clear headings and logical organization
    - Include relevant facts, figures, and details from the search results
    - Cite sources for specific information using [Source: X] notation
    - When information from different sources conflicts, acknowledge this and present both perspectives
    - If the search results don't adequately answer the question:
      1. Clearly state what information is missing
      2. Provide the partial information available
    - Maintain a balanced, informative tone throughout
    - Format your response for maximum readability and comprehension
    """

    # Configure Tavily search with improved parameters
    tool = TavilySearch(
        max_results=1,  # Increased for better coverage
        include_domains=None,  # Allow all domains
        exclude_domains=None,
        include_raw_content=True,  # Get full text
        include_images=False,
        include_image_descriptions=False,
        search_depth="advanced",  # Use advanced search for better results
        time_range="year",  # Wider time range for more comprehensive results
    )
    
    # Define search function with error handling
    def perform_web_search(query_str: str):
        try:
            return tool.invoke(query_str)
        except Exception as e:
            return [{
                "content": f"Error performing web search: {str(e)}. The search service may be unavailable.",
                "url": "https://example.com/error"
            }]
    
    
    def get_search_results(input_query):
        return perform_web_search(input_query)
    
    # Get web search results
    web_result = get_search_results(query)

    # Set up the response generation
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b", 
            temperature=0.2
        )

    chain = (
        {"context": lambda x: web_result, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
 
    def get_response(input_query):
        return chain.invoke(input_query)
    
    # Use a hash of the first search result as part of the cache key
    response = get_response(query)
    
    return response