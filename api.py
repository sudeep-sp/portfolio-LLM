from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from your Next.js app's domain (e.g., localhost:3000)
origins = [
    "http://localhost:3000",  # Allow your Next.js app running on localhost
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows only the specified origin(s)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Model configuration
model = ChatGroq(model_name="llama-3.1-8b-instant",
                 groq_api_key=os.environ.get("GROQ_API_KEY"),
                 temperature=0)

# Portfolio data
about_sudeep = {
    "me": "Sudeep S Patil",
    "about me": "I am currently pursuing an MS in AI at BTU Cottbus, Germany. My expertise lies in core machine learning, neural networks, and generative AI. I have hands-on experience with frameworks like PyTorch and tools like langchain, crewai. I’m passionate about creating AI-driven applications and exploring the latest trends in technology.",
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
            "technologies": ["React", "Next.js", "FastAPI", "Streamlit"],
            "github": ["https://github.com/sudeep-sp/learnaiwithus_next-js", "https://github.com/sudeep-sp/learnaiwithus-Ask_AI_RAG", "https://github.com/sudeep-sp/learnaiwithus-blog_gen_agents"]
        },
        {
            "name": "Federated Learning Simulator",
            "description": "Built a Streamlit app to simulate and log results of federated learning experiments.",
            "technologies": ["Python", "Streamlit", "Flow"],
            "github": "https://github.com/sudeep-sp/FL_sim"
        }
    ],
    "work_experience": [
        "I have no work experience yet. but i done internships and projects."
    ],
    "contact_details": {
        "email": "sudeep.subhashchandra.patil@gmail.com",
        "linkedin": "https://www.linkedin.com/in/sudeepspatil/",
        "github": "https://github.com/sudeep-sp",
    },
    "skills": ["Python", "PyTorch", "TensorFlow", "Streamlit", "React", "Next.js", "FastAPI", "Flow", "langchain", "crewai"]
}

# Template setup
template = """
You are an AI assistant specializing in providing accurate and concise information about Sudeep S Patil.

Use {about_me} to extract the relevant information for answering the query in summarized manner.

User Query: "{query}"

Guidelines for Response:

- Structured Format:

    Provide answers in a professional, easy-to-read and should be in markdown format.
    Use headings or bullet points when appropriate.
    
- Direct and Focused:
    Answer directly with no preambles or additional context.
    For example, if asked "Who is Sudeep S Patil?", respond with a concise paragraph like:
    [Answer here] no preambles needed.
    
- Clickable Links:
    Ensure links are clickable.
    For example, if providing a GitHub link, format it as [GitHub](https://github.com/repo)

- Scope Limit:
    If the query is unclear or unrelated to Sudeep, respond with:
    "I am only trained to answer questions about Sudeep S Patil. Could you please clarify your query?"
"""

prompt_template = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = prompt_template | model | output_parser
# FastAPI Request/Response Models


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str

# Define API endpoints


@app.post("/ask", response_model=QueryResponse)
async def ask_portfolio(query_request: QueryRequest):
    query = query_request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Process query using the language model chain
        result = chain.invoke({"query": query, "about_me": about_sudeep})
        return QueryResponse(response=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
