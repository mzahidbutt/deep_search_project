import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, ModelSettings, set_tracing_disabled, RunResult

from  dotenv import load_dotenv, find_dotenv
from tavily import AsyncTavilyClient
from dataclasses import dataclass

#set_tracing_disabled(disabled=True)

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

#set_tracing_disabled(True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


gemini_api_key = os.getenv("GEMINI_API_KEY")

tavil_api_key = os.getenv("TAVIL_API_KEY")

tavily_client = AsyncTavilyClient(api_key=tavil_api_key)

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


# 1. Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Model
llm_model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.5-flash"
)



# Define DataClasses
#user_profile = {"name":"Ali","city":"Lahore","topic":"AI"}

@dataclass
class UserProfile:
    username: str
    city: str
    topic: str | None = None
    input_chat: list[dict] = None

@function_tool
async def Search(query: str) -> str:
    print("[TOOLS....] searching for:", query)
  #  response = tavily_client.search(query)
    response = await tavily_client.search(query)
    print("[TOOLS....] search result:", response)
    return response


@function_tool
async def extract_content(urls: list) -> dict:
    print("/n/n")
    print("[TOOLS....] extracting content from:", urls)
    response = await tavily_client.extract_content(urls)
    print("[TOOLS....] extracted content:", response)
    print("/n/n")
    return response



def dynamic_instructions() -> str:
    prompt = """You are an expert in managing complex research tasks. 
                    Your role is to coordinate specialized agents to break down an open research question into focused subqueries, 
                    assign tasks to Researcher, Synthesizer, and Writer agents, and ensure the final report is comprehensive and well-cited."""
    return prompt


# Define agents
# Researcher Agent
researcher_agent = Agent(
    name="Researcher Agent",
    instructions="""You are an expert researcher. Use web_search and browse_page tools to gather comprehensive information 
                    from multiple sources on the given topic. Collect data with sources for citation. 
                    Once you have sufficient information (at least 3-5 sources), handoff to the Synthesizer Agent.""",
  ##  tools=[web_search, browse_page],  Temporary Commit
    model=llm_model
)

synthesizer_prompt = """
                    You are an expert Synthesis Agent, skilled at analyzing and synthesizing complex research data into organized, 
                    insightful conclusions. Your task is to process the provided research data on a given topic, 
                    organize findings into 2-3 key themes, resolve any conflicts between sources,
                      and provide deep insights with clear reasoning. You do not use other agents or external tools; 
                      rely solely on the input data and your analytical capabilities.
                    """	
  
#  instructions="""You are an expert at synthesizing information. Take the research data from the conversation history, 
#                     understand, reason, and synthesize key insights about the topic. Provide deep analysis. 
#                     Once synthesized, handoff to the Writer Agent."""

# Synthesizer Agent
synthesizer_agent = Agent(
    name="Synthesizer Agent",
    instructions=synthesizer_prompt,
    model=llm_model
)

writer_prompt = """
                You are the expert Writer Agent. Your task is to create a clear, concise, and well-structured response based 
                on the input from the Synthesizer Agent and the Research Agent. 
                The Synthesizer Agent provides a consolidated summary of the research findings, 
                while the Research Agent provides detailed results for each sub-task. Combine these inputs to produce 
                a coherent and polished output that directly addresses the user's original query. 
                Ensure the response is informative, engaging, and tailored to the user's context (e.g., market analysis focus).
                Format the output as text format.
                """
# """You are an expert writer. Use the synthesized insights and original research data from history to produce a detailed, 
#                     well-structured report on the topic. Ensure perfect citation accuracy using inline citations like [1], [2] 
#                     based on sources collected. Include a references section at the end listing sources with links."""

#  Writer Agent
writer_agent = Agent(
    name="Writer Agent",
    instructions=writer_prompt,
    model=llm_model
)

research_as_tools = researcher_agent.as_tool(
                            tool_name ="Researcher_Tool",
                            tool_description ="Tool to gather comprehensive information on a topic using web search and browsing."
)

synthesizer_as_tools = synthesizer_agent.as_tool(
                            tool_name="Synthesizer_Tool",
                            tool_description="Tool to synthesize research data into key insights."
                        )

writer_as_tools = writer_agent.as_tool(
                            tool_name="Writer_Tool",
                            tool_description="Tool to write a detailed report based on synthesized insights and research data."
                        )

#Orchestrator Agent
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions= dynamic_instructions(),
                    model=llm_model,
                    tools=[research_as_tools, synthesizer_as_tools, writer_as_tools,Search, extract_content                       
                    ]
)


planner_prompt = """
                You are the Planner Agent. Take the research requirement from requirements gathering Agent
                and break it down into specific, manageable sub-tasks for research. 
                Create a plan with 3-5 sub-tasks to cover different angles. 
                Output the plan as a JSON list of tasks. Then handoff to the Orchestrator Agent.
                """	

#"""                You are the Planner Agent. Take the research requirement and break it down into specific, 
 #               manageable sub-tasks for research. Create a plan with 3-5 sub-tasks to cover different angles. 
 #               Output the plan text format and than handoff to the Orchestrator Agent.""",

planner_agent = Agent(
    name="Planner Agent",
    instructions= planner_prompt,
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=provider),
     tools=[],
   # model=llm_model,
    handoffs=[orchestrator_agent]  # Handoff to Orchestrator Agent after planning
)


# Define main agents with handoffs

requirement_gatherer = Agent(
    name="Requirement Gatherer Agent",
    instructions="""
                You are the starting point for deep research. Gather and clarify the user's research requirement or topic. 
                If unclear, ask questions, but for now, assume the input is the topic. Then handoff to the Planner Agent.""",
    model=llm_model,
    tools=[],
    handoffs=[planner_agent]  # Handoff to Planner Agent after gathering requirement
)


# async def main():
   
#     user_profile = UserProfile(username="Ali", city="Lahore", topic="AI")

#     PROMPT_MSG = "what is trend electric vs hubrid vs gas cars."
#     #"Compare the environmental impact of electric vs hybrid vs gas cars."

#     result = await Runner.run(
#                             requirement_gatherer,
#                             PROMPT_MSG,
#                             context=user_profile,
#                             max_turns=20
#                             )

   
#     print("BOT Output:" , result.final_output)

user_chat: list[dict] = []

async def main():

    market_keywords = ["market", "analysis", "stocks", "economy", "finance", "investment", "trading", "business", "industry", "trend"]

    user_profile = UserProfile(username="Ali", city="Lahore", topic="AI")

    while True:
        user_input = input("Enter your input (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'view':
            print("\nCurrent Chat History:", user_chat)
    
            user_message = {"role": "user", "content": user_input}

        
            user_chat.append(user_message)
    

        is_market_related = any(keyword in user_input.lower() for keyword in market_keywords)
    
        if is_market_related:          
        #  print("\nInput is related to market analysis. Processing...",user_input)
            res: RunResult = await Runner.run(starting_agent=requirement_gatherer, input=user_input, context=user_profile, max_turns=20)
            user_chat = res.to_input_list()
            print("\nAGENT RESPONSE:", res.final_output)
            break
        else:
            print("\nInput is not related to market analysis. Please provide a market-related query.")


if __name__ == "__main__":
    asyncio.run(main())