from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def take_decision(query: str) -> str:
    """
    Decide which agent should handle the user's query.
    Returns: "university", "web search", or "general"
    """
    template = """<think>
    Analyzing user query:
    1. Is this about university life, academics, campus resources, or specific institutional knowledge?
    2. Does this require current information, facts that change regularly, or recent events?
    3. Is this a general academic, theoretical, or conceptual question?
    
    Key indicators:
    - University: mentions specific university programs, campus services, enrollment, etc.
    - Web search: requires current information like weather, news, prices, etc.
    - General: academic concepts, theories, explanations, how-to questions
    </think>
    
    You are a specialized classifier determining which AI system should handle a user query.
    
    INSTRUCTIONS:
    Analyze the query and return EXACTLY ONE of these three classifications:
    - "university" - for questions about specific university programs, policies, campus resources, 
                    student services, administrative procedures, etc.
    - "web search" - for questions requiring current information, news, events, prices, 
                    weather, or other frequently changing data
    - "general" - for academic concept explanations, theoretical questions, study advice,
                career guidance, and other general knowledge questions
    
    IMPORTANT: Return ONLY the classification word without any additional text or explanation.
    
    User query: {question}
    Classification:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(
        model_name="gemma2-9b-it",
        max_tokens=10
    )
    
    chain = (
        {"question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    def get_response(input_query):
        return chain.invoke(input_query).strip().lower()
    
    result = get_response(query)
    
    # Validate response
    valid_responses = ["university", "web search", "general"]
    if result not in valid_responses:
        return "general"  # Default to general if invalid response
        
    return result