from src.agents.agents import build_search_agent, build_scrape_agent,writer_chain,critic_chain

def run_research_pipeline(topic: str)->dict:

    state={}

    # Step 1: Use the search agent to gather information
    print("\n"+"Running search agent..."+"\n")

    search_agent = build_search_agent()
    search_results = search_agent.invoke({"messages": [("user", f"Find recent and reliable information on the topic: {topic}. Provide a list of relevant URLs.")]})

    state['search_results'] = search_results['messages'][-1].content
    print("\n"+"Search Results:"+"\n", state['search_results'])



    # Step 2: Use the scrape agent to extract content from the URLs found
    print("\n"+"Running scrape agent..."+"\n")

    scrape_agent = build_scrape_agent()

    scraped_contents = scrape_agent.invoke({
        "messages": [("user", 
                      f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_contents'] = scraped_contents['messages'][-1].content
    print("\n"+"Scraped Contents:"+"\n", state['scraped_contents'])
    


    # Step 3: Use the writer chain to generate a research report
    print("\n"+"Running writer chain..."+"\n")

    research_combined=(
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"SCRAPE CONTENTS:\n{state['scraped_contents']}"
    )
    initial_report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
        "critic_feedback": "",
        "existing_report": ""
    })

    state["report"] = initial_report

    print("\n"+"Generated Report:"+"\n", state["report"])


    # Step 4: Use the critic chain to review and improve the report
    print("\n"+"Running critic chain..."+"\n")

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n"+"Critic Feedback:"+"\n", state["feedback"])

    # Step 5: Rewrite the report using the critic feedback
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
        "critic_feedback": state["feedback"],
        "existing_report": initial_report
    })

    print("\n"+"Improved Report:"+"\n", state["report"])

    return state