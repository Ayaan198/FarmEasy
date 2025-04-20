from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Set up the Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the GroqLLM 
groq_llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt_template = """
Use paragraphs to divide each topic into sections. Do not use special characters.
Write a guide for Indian farmers on growing {crop_name}. Include:

- Ideal climate and soil
- Sowing and harvesting times
- Common pests and diseases
- Tips to maximize yield
- Market trends and selling advice

Highlight the mentioned topics in BOLD
"""


# Create the prompt template instance
template = PromptTemplate(input_variables=["crop_name"], template=prompt_template)

# Create an LLM chain to run the model with the prompt
llm_chain = LLMChain(llm=groq_llm, prompt=template)

def get_crop_info_from_groq(crop_name):
    try:
        # Fetch the response from Groq using LangChain
        crop_info = llm_chain.run(crop_name=crop_name)
        return crop_info
    except Exception as e:
        return f"Error fetching AI insights: {e}"
