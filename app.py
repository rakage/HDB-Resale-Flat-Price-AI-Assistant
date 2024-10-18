import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import os

# Load HDB resale summary data from chromadb
def query_chromadb(query):
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory="/chroma-data",  # Where to save data locally, remove if not necessary
    )

    results = vector_store.similarity_search_with_score(
        query=query,
        k=5,
    )

    return results

@st.cache_resource
def load_data():
    df = pd.read_csv("merged_hdb_resale_prices.csv")
    return df

df = load_data()

def query_llm(prompt):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    message = [
            (
                "system",
                "You are an expert assistant for HDB resale flat buyers in Singapore. Use the provided data context to inform your responses.",
            ),
            (
                "human",
                prompt,
            )
        ]
    

    response = llm.invoke(prompt)
   
    return response

def main():
    st.title("HDB Resale Flat Buyer's AI Assistant")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Home", "About Us", "Methodology", "Personalized Advice", "Market Insights"])
    
    if page == "Home":
        home_page()
    elif page == "About Us":
        about_us_page()
    elif page == "Methodology":
        methodology_page()
    elif page == "Personalized Advice":
        personalized_advice()
    elif page == "Market Insights":
        market_insights()

def personalized_advice():
    st.header("Personalized HDB Resale Advice")

    # Collect user information
    st.subheader("Tell us about your situation")
    budget = st.number_input("Your budget (SGD)", min_value=0, step=10000)
    preferred_towns = st.multiselect("Preferred towns", sorted(df['town'].unique()))
    flat_type = st.selectbox("Preferred flat type", sorted(df['flat_type'].unique()))
    
    if st.button("Get Personalized Advice"):
        query = f"Advice for buying a {flat_type} flat in {' or '.join(preferred_towns)} with a budget of ${budget:,.2f}"
        relevant_context = query_chromadb(query)
        
        prompt = f"""
        Given the following summarized market information:
        {relevant_context}
        
        Provide personalized advice for an HDB resale flat buyer looking for a {flat_type} flat in {' or '.join(preferred_towns)} with a budget of ${budget:,.2f}.
        Consider their budget, preferred locations, and flat type. 
        Suggest suitable options, potential challenges, and any relevant policies or grants they should be aware of.
        """
        
        response = query_llm(prompt)
        st.write(response.content)

def market_insights():
    st.header("HDB Resale Market Insights")

    st.subheader("Ask about market trends")
    user_question = st.text_input("What would you like to know about the HDB resale market?")
    
    if st.button("Get Insights"):
        relevant_context = query_chromadb(user_question)

        prompt = f"""
        Given the following summarized market information:
        {relevant_context}
        
        Please answer the following question about the HDB resale market in Singapore:
        {user_question}
        
        Provide insights based on the given context and your knowledge of Singapore's housing market.
        """
        
        response = query_llm(prompt)
        st.write(response.content)

def home_page():
    st.header("Welcome to HDB Resale Flat Buyer's Assistant")
    st.write("""
    This application is designed to help you navigate the HDB resale market in Singapore. 
    Use the sidebar to explore different features:
    
    - Learn about our project in the 'About Us' section
    - Understand our approach in the 'Methodology' section
    - Get personalized advice based on your preferences
    - Explore market insights and trends
    
    Start your journey to finding the perfect HDB resale flat today!
    """)

def about_us_page():
    st.header("About Us")
    st.markdown("""
    ## Project Scope
    Our project aims to provide a comprehensive and user-friendly LLM-powered assistant for potential HDB resale flat buyers in Singapore. By leveraging advanced AI technologies and up-to-date market data, we strive to offer personalized advice and valuable insights to help users make informed decisions in their property search.

    ## Objectives
    1. To simplify the HDB resale flat buying process for Singaporeans and PRs.
    2. To provide accurate, data-driven insights into the HDB resale market.
    3. To offer personalized recommendations based on individual preferences and budget constraints.
    4. To keep users informed about relevant policies, grants, and market trends.

    ## Data Sources
    Our system utilizes a variety of reliable data sources to ensure the most accurate and up-to-date information:

    - HDB Resale Price Index
    - Transaction data from Data.gov.sg
    - URA Master Plan information
    - Historical pricing data from various towns and estates

    ## Features
    1. **Personalized Advice**: Get tailored recommendations based on your budget, preferred locations, and flat type.
    2. **Market Insights**: Ask questions about current market trends, pricing, and policies.
    3. **Interactive Interface**: User-friendly Streamlit application for easy navigation and data visualization.
    4. **AI-Powered Analysis**: Utilizes advanced language models and embeddings for intelligent responses.
    5. **Real-time Data Processing**: Continuously updated information to reflect the latest market conditions.

    Our team is committed to providing a valuable tool for HDB resale flat buyers, combining cutting-edge technology with comprehensive market knowledge to simplify your property search journey.
    """)

def methodology_page():
    st.header("Methodology")
    st.write("""
    Our HDB Resale Flat Buyer's AI Assistant employs a sophisticated methodology to provide accurate and personalized information. Here's an overview of our approach:
    """)
    
    st.image("methodology.jpg", caption="HDB Resale Assistant Methodology Flowchart")
    
    st.markdown("""
    1. **Data Collection and Processing:** We extract data from multiple HDB Resale Data sources. Using Python, we merge the data and create summaries to ensure comprehensive and up-to-date information.
    
    2. **Data Embedding:**We apply a language model to create embeddings using OpenAI's technology. This process allows for better context understanding and more efficient querying.
    
    3. **Data Storage:** The embedded data is stored in ChromaDB, a database optimized for vector search. This enables efficient querying and retrieval of relevant information.
    
    4. **Similarity Search:** The system uses the embedded user query to search within ChromaDB. It extracts the top k most relevant results based on the embedded data.
    
    5. **Language Model Processing:** The retrieved results are passed to a Language Learning Model (LLM). The LLM constructs a prompt incorporating these findings to generate personalized insights or recommendations.
    
    6. **Response Generation:** The LLM generates a tailored response based on the user's query and the relevant data. This response is then presented to the user through the Streamlit interface. 

    This methodology allows us to provide highly relevant and personalized advice by combining large-scale data processing, advanced natural language understanding, and efficient information retrieval. By leveraging cutting-edge AI technologies, we offer a unique and valuable service to HDB resale flat buyers in Singapore, helping them make informed decisions based on the latest market data and trends.
    """)


if __name__ == "__main__":
    main()