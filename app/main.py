# import os

# from app.core.decision_maker import take_decision
# from app.agents.uni_agent import uni_agent
# from app.agents.web_agent import web_agent
# from app.agents.university_tutor import university_tutor
# from app.utils.embeddings import set_embeddings

# # Get environment variables
# os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
# os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")

# if __name__ == "__main__":
#     query = "Who is the VC of Agriculture University Peshawar"
#     embeddings = set_embeddings()
#     llm_output = take_decision(query)

#     if llm_output == "university":
#         result = uni_agent(query, embeddings)
#         print(result)
#     elif llm_output == "web search":
#         result = web_agent(query)
#         print(result)
#     elif llm_output == "general":
#         result = university_tutor(query)
#         print(result)
#     else:
#         print("error")
