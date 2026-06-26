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
    ("system", "You are an expert research writer and editor. Your job is to turn raw research into a polished, insightful, and evidence-based report that directly addresses critique feedback. You must ground every claim in the provided research."),
    ("human", """Write or revise a high-quality research report based on the following information.

        Topic: {topic}

        Research Gathered (includes URLs, snippets, and scraped content):
        {research}

        Critic Feedback to Address:
        {critic_feedback}

        Previous Draft (if any):
        {existing_report}

        Requirements:
        1. Produce a report that is clear, structured, and professionally written.
        2. If critic feedback is provided, use it aggressively to improve depth, clarity, balance, and usefulness.
        3. Include a short executive summary at the top.
        4. Add at least 4 well-explained key findings, each directly supported by specific data, statistics, or quotes from the research provided.
        5. After every claim or statistic, cite the source inline like this: (Source: <URL>).
        6. Include context, nuance, and implications where the research supports them.
        7. Highlight limitations or uncertainty when the evidence is weak — do NOT invent facts.
        8. Add a Methodology section briefly describing what sources were used and how.
        9. Include a critical perspective — limitations of LLMs, ethical concerns, risks, or regulatory challenges if present in research.
        10. End with a concise conclusion.
        11. End with a Sources section — for each URL found in the research, list:
            - Title (if available)
            - Full URL
            - One line describing what it contributed to the report
        
        STRICT RULES:
        - Every key finding MUST reference at least one specific URL from the research.
        - If the research does not contain URLs, explicitly state: "No URLs were returned by the search/scrape tools" and note this as a pipeline limitation.
        - Do not use phrases like "based on search results" without citing what those results actually said.
        - Do not invent statistics, forecasts, or expert quotes.
        - Make the report strong enough to earn at least an 8/10 from a strict critic who penalizes unsupported claims.

        Structure the report as:
        - Executive Summary
        - Methodology
        - Introduction
        - Key Findings (each with inline citations)
        - Critical Perspectives & Limitations
        - Conclusion
        - Sources (Title | URL | Contribution)"""),
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
