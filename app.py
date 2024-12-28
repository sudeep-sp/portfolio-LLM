import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

model = ChatGroq(model_name="llama-3.1-8b-instant",
                 groq_api_key=os.environ["GROQ_API_KEY"],
                 temperature=0)

about_sudeep = {
    "name": "Sudeep S Patil",
    "title": "AI Enthusiast | Machine Learning Researcher | Developer",
    "about_me": "I am currently pursuing an MS in AI at BTU Cottbus, Germany. My expertise lies in core machine learning, neural networks, and generative AI. I have hands-on experience with frameworks like PyTorch and tools like langchain, crewai. I’m passionate about creating AI-driven applications and exploring the latest trends in technology.",
    "education": [
        {
            "degree": "MS in Artificial Intelligence",
            "institution": "BTU Cottbus, Germany",
            "year": "Present"
        },
        {
            "degree": "Bachelors in CSE",
            "institution": "VTU, India",
            "year": "2019 - 2023"
        }
    ],
    "internships": [
        {
            "role": "associate data analyst",
            "company": "Contriver",
            "duration": "sep 2022 - nov 2022",
        }
    ],
    "projects": [
        {
            "name": "AI Blogging Website",
            "description": "Developed a platform learnaiwithus.codes that generates AI-assisted blogs using four collaborative agents, adhering to ethical content guidelines.",
            "technologies": ["React", "Next.js", "FastAPI", "Streamlit"]
        },
        {
            "name": "Federated Learning Simulator",
            "description": "Built a Streamlit app to simulate and log results of federated learning experiments.",
            "technologies": ["Python", "Streamlit", "Flow"]
        }
    ],
    "work_experience": [
        None
    ],
    "contact_details": {
        "email": "sudeep.subhashchandra.patil@gmail.com",
        "linkedin": "https://www.linkedin.com/in/sudeepspatil/",
        "github": "https://github.com/sudeep-sp",
    }
}

template = """
You are an AI assistant helping users learn about Sudeep S Patil, an AI enthusiast and developer. Respond to the user's queries accurately and concisely  for the following categories.:

1. About Me
2. Education
3. Projects
4. Work Experience
5. internships
6. Contact Details

use {about_me} to get the above information.

User Query: "{query}"

Use the structured data provided about Sudeep S Patil to respond. If the question relates to and dont say query again :
- **About Me**: Summarize his background, interests, and expertise in AI and development.
- **Education**: List his degrees, institutions, and ongoing academic pursuits.
- **Projects**: Highlight specific projects he has worked on, including names, descriptions, and technologies used also give link of my github to see my more projects.
- **work experence**: say i am fresher so i have no work experience but i have done internships and projects and give details about them.
- **Contact Details**: Provide his email, LinkedIn, GitHub.

Give respone in structured manner so easy to read and understand.
Provide direct answers with no preambles or additional context.
If the query is unclear or outside scope, say "i only trained to answer about sudeep" and  ask for clarification politely.

"""
prompt_template = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = prompt_template | model | output_parser

# Streamlit UI
st.title("Sudeep S Patil Portfolio AI Assistant")
st.write("Ask any query about Sudeep's portfolio (e.g., 'Tell me about your projects').")

# Input query
query = st.text_input("Enter your query:")

# Process query and display result
if st.button("Submit"):
    if query:
        try:
            result = chain.invoke({"query": query, "about_me": about_sudeep})
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to proceed.")
