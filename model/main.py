import os
import json
from langchain import hub
from typing import Literal
from langchain_groq import ChatGroq
from langchain_ibm import WatsonxLLM
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from typing import List, Annotated, Sequence
from langchain_community.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

loader = JSONLoader(
    file_path='output.json',
    jq_schema='.',
    text_content=False)

docs_list = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

from langchain_ibm import WatsonxLLM

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 1500,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Router
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...,description="Given a user question choose to route it to web search or a vectorstore.",)

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
question = "PM Kisan"
docs = retriever.get_relevant_documents(question)

### Generate

prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only the context provided.
    If the answer is somehow related to the context but not given in the context, you can search the web for the answer.
    When the user asks about facts you need not answer step by step.
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

#llm = ChatGroq(temperature=0, model="llama3-70b-8192")
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="102027a1-943a-4272-a1c2-1d8be5853e8a",
    params=parameters,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"context": docs, "question": question})

### Hallucination Grader

class GradeHallucinations(BaseModel):
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
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader

class GradeAnswer(BaseModel):
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
answer_grader.invoke({"question": question,"generation": generation})

### Search
web_search_tool = TavilySearchResults(k=5,search_depth="advanced", max_results=5)

# Update GraphState to include messages
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The conversation history"]

# Helper function to update conversation history
def update_conversation(state: GraphState, role: str, content: str) -> GraphState:
    new_state = state.copy()
    if role == "human":
        new_state["messages"].append(HumanMessage(content=content))
    elif role == "ai":
        new_state["messages"].append(AIMessage(content=content))
    return new_state

# Modify existing functions to use and update conversation history

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    new_state = update_conversation(state, "human", question)
    new_state["documents"] = documents
    return new_state

def generate(state):
    question = state["question"]
    documents = state["documents"]

    # Update prompt to include conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided context to answer the question."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    # RAG generation with conversation history
    generation = rag_chain.invoke({
        "context": documents,
        "question": question,
        "messages": state["messages"]
    })

    new_state = update_conversation(state, "ai", generation)
    new_state["generation"] = generation
    return new_state

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

    new_state = state.copy()
    new_state["documents"] = filtered_docs
    new_state["web_search"] = web_search
    return new_state

def web_search(state):
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    new_state = state.copy()
    if documents is not None:
        new_state["documents"] = documents + [web_results]
    else:
        new_state["documents"] = [web_results]
    return new_state

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == 'web_search':
        return "websearch"
    elif source.datasource == 'vectorstore':
        return "vectorstore"

def decide_to_generate(state):
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"

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

# Create the graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

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

# Compile the graph
app = workflow.compile()

def run_conversation(question, previous_state=None):
    if previous_state is None:
        initial_state = {
            "question": question,
            "generation": "",
            "web_search": "",
            "documents": [],
            "messages": []
        }
    else:
        initial_state = previous_state.copy()
        initial_state["question"] = question

    result = app.invoke(initial_state)
    return result

result = run_conversation(input("user: "))
print("assistant:", result["generation"])