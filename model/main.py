
import os
import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

os.environ["GROQ_API_KEY"] = "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"
os.environ["HF_TOKEN"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["WATSONX_APIKEY"] = "4iF_n7glQTZsEol5RIFbNFZAQOIqaKVr031080rz2paG"

## Load data using json

loader = JSONLoader(
    file_path='output.json',
    jq_schema='.',
    text_content=False)

docs_list = loader.load()

### Build Index

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="Simpli-Gov-V1.0",
    embedding=HuggingFaceEmbeddings(),
)
retriever = vectorstore.as_retriever()

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 1500,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

llm_granite = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="102027a1-943a-4272-a1c2-1d8be5853e8a",
    params=parameters,
)

### Router

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = ChatGroq(temperature=0, model="llama3-70b-8192")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to Indian government schemes available to its citizens, about the different Indian government forms like taxes and how to fill them.
Preassume that the user is an Indian citizen who needs help on doing their work.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

### Retrieval Grader

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
llm = ChatGroq(temperature=0, model="llama3-70b-8192")
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s).semantic meaning or enough content related to the user question so as to answer it, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant and contains enough context to answer the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Generate

prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only the context provided.
    If the answer is somehow related to the context but not given in the context, you can search the web for the answer.
    When the user asks about facts you need to provide a proper answer in less than 40 words
    But for everything else, especially instructions, think step by step and provide a detailed answer.
    Give well structured response which is comprehensible even for a 70 year old man.
    You are forbidden from refering to the fact that you have a context document.
    You need to remember what the user asked and the answers you gave them.
    <context>
    {context}
    </context>
    Question : {question}
    """
)

llm = llm_granite

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

### Hallucination Grader

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

llm = ChatGroq(temperature=0, model="llama3-70b-8192")
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

### Answer Grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

llm = ChatGroq(temperature=0, model="llama3-70b-8192")
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

### Search

web_search_tool = TavilySearchResults(k=5,search_depth="advanced", max_results=5)

### State

class GraphState(TypedDict):

    question : str
    generation : str
    web_search : str
    documents : List[str]

### Nodes

def retrieve(state):

    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):

    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):

    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.score
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):

    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

### Conditional edge

def route_question(state):

    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == 'web_search':
        return "websearch"
    elif source.datasource == 'vectorstore':
        return "vectorstore"

def decide_to_generate(state):

    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.score
    if grade == "yes":
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

app = workflow.compile()

output = {}
question = input("User: ")
inputs = {"question": question}
for output in app.stream(inputs):
  continue
result = output["generate"]
print(result["generation"])

