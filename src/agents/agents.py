from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import scrape_url, web_search
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0)

def build_search_agent():
    agent = create_agent(
        model=llm,
        tools=[web_search]
    )

    return agent


def build_scrape_agent():
    agent = create_agent(
        model=llm,
        tools=[scrape_url]
    )

    return agent

#LCEL CHAINS FROM WRITER AND REVIEWER AGENTS

#WRITER AGENT

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write a comprehensive humanized research report based on the information provided. Ensure clarity, coherence, and proper structure."),
    ("human", """Writw a detailed research report based on the following information.

        Topic: {topic}

        Research Gathered:
        {research}

        Structure the report as:
        - Introduction
        - Key Findings (minimum 3 well-explained points)
        - Conclusion
        - Sources (list all URLs found in the research)

        Be detailed, factual, humanized and professional."""),
        ])

writer_chain=writer_prompt | llm | StrOutputParser()

#REVIEWER AGENT

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

        Report:
        {report}

        Respond in this exact format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...

        One line verdict:
        ..."""),
        ])

critic_chain=critic_prompt | llm | StrOutputParser()
