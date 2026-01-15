from typing import TypedDict,Optional,List,Annotated
import os
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from uuid import uuid4
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from pinecone import Pinecone,ServerlessSpec
import logging
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s"
)
logger=logging.getLogger("RAG-BACKEND")


llm=ChatOpenAI(model_name='gpt-4o-mini',temperature=0.2)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

checkpointer=InMemorySaver()
reranker=CohereRerank(
    model='rerank-english-v3.0',
    top_n=10,)

logger.info("Initializing Pinecone database")
pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name='rag-prod-index'

pc_indexes=[i['name'] for i in pc.list_indexes()]
logger.info("Using Pinecone index: %s",index_name)
if index_name not in pc_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index=pc.Index(index_name)


def get_vector_database(namespace:str):
    '''Return named pinecone vectorstore'''
    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key='text',
        namespace=namespace
    )

class ChatState(TypedDict):
    uploaded_document:Optional[str]
    has_source:Optional[bool]
    pasted_text:Optional[str]
    query:str
    retrieved_docs:Optional[List[dict]]
    messages:Annotated[List[BaseMessage],add_messages]
    final_docs:Optional[List[dict]]
    answer:Optional[dict]


def validate_documents(state: ChatState):
    logger.info("Validating Documents...")

    if state.get("has_source"):
        return {}

    if state.get("uploaded_document") or state.get("pasted_text"):
        return {}

    return {
        "messages": [
            AIMessage(content="No source provided. Please upload a document or paste text.")
        ],
        "answer": {
            "Answer": "No source provided. Please upload a document or paste text.",
            "Citations": [],
            "context": ""
        }
    }

    

def router(state:ChatState):
    return 'continue' if state.get("uploaded_document") or state.get("pasted_text") else 'end'

def load_and_index_documents(state: ChatState, config):
    logger.info("Index and Load document node")

    if state.get("has_source"):
        logger.info("Source already indexed for this thread")
        return {}

    namespace = config["configurable"]["thread_id"]
    vector_db = get_vector_database(namespace)

    documents = []

    if state.get("uploaded_document"):
        loader = PyPDFLoader(state["uploaded_document"])
        documents.extend(loader.load())

    if state.get("pasted_text"):
        documents.append(
            Document(
                page_content=state["pasted_text"],
                metadata={"source": "pasted_text"}
            )
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    logger.info("Indexing %d chunks", len(chunks))
    vector_db.add_documents(chunks)

    return {"has_source": True}




def retrieved_docs(state: ChatState, config):
    logger.info("Retrieving contextual documents")

    vector_db = get_vector_database(config["configurable"]["thread_id"])
    docs = vector_db.similarity_search(query=state["query"], k=8)

    logger.info("Retrieved %d documents", len(docs))

    return {
        "retrieved_docs": [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]
    }


def reranked_docs(state: ChatState):
    logger.info("Reranking the retrieved documents")

    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        logger.info("No documents retrieved")
        return {"final_docs": []}

    rerank_results = reranker.rerank(
        query=state["query"],
        documents=[doc["content"] for doc in retrieved]
    )

    reranked = [
        {
            "text": retrieved[r["index"]]["content"],
            "score": r["relevance_score"],
            "metadata": retrieved[r["index"]]["metadata"]
        }
        for r in rerank_results
    ]

    confident = [d for d in reranked if d["score"] >= 0.05]

    if confident:
        logger.info("Using reranked documents")
        return {"final_docs": confident[:5]}

    logger.warning("Low reranker confidence, falling back to retriever")
    return {
        "final_docs": [
            {
                "text": d["content"],
                "score": 1.0,
                "metadata": d["metadata"]
            }
            for d in retrieved[:5]
        ]
    }





def generate_response(state:ChatState):
    logger.info("Generating response...")
    final_docs = state.get("final_docs", [])

    if not final_docs:
        return {
            "messages": [
                AIMessage(
                    content="Sorry, I don't know the answer based on the provided documents."
                )
            ],
            "answer": {
                "Answer": "Sorry, I don't know the answer based on the provided documents.",
                "Citations": [],
                "context": ""
            }
        }
    top_docs=state['final_docs'][:5]
    context_docs="\n\n".join(f"[{i+1}] {doc['text']}" for i,doc in enumerate(top_docs))
    sources_docs='\n'.join(f"[{i+1}] Source:{doc['metadata'].get('source','unknown')}"
                           f"Section: {doc['metadata'].get('section','unknown')}" for i,doc in enumerate(top_docs))
    template=ChatPromptTemplate.from_messages(
        [('system','''You are a factual question-answering assistant.

    INSTRUCTIONS:
    You may rephrase or normalize bullet points and headings
    into complete sentences, as long as you use ONLY the provided context
    and do not add any new information.
    - Every factual claim MUST have an inline citation like [1], [2] .
    Headings like "Project:" indicate the name of the project.

    - If the answer cannot be derived from the context, respond exactly with:
    "Sorry,I don't know the answer based on the provided documents."
    
    If a general question is asked about a document like what is this document talking about try to summarise the content and answer.
          
    PROCESS:
    Step 1: Identify which chunks contain relevant evidence.
    Step 2: Extract factual statements from those chunks.
    Step 3: Write the answer using only those statements and cite each one.
    stick to the format

    Respond with ONLY the answer text.
    Use inline citations like [1], [2].
    Do NOT include headings like "Answer:" or "Sources: or Context:"
          '''),
    MessagesPlaceholder(variable_name='messages'),
    ('human','''
     Context:
    {context_docs}

    Current Question:
    {query}

    Sources Reference:
    {sources_docs} '''),
    ])

    prompt=template.invoke({
        'context_docs':context_docs,
        'query':state['query'],
        'sources_docs':sources_docs,
        'messages':state['messages']
    })
    logger.info("LLM CONTEXT:\n%s", context_docs)


    response=llm.invoke(prompt)
    full_answer=response.content
    citations=[
        {
            'id':i+1,
            'source':doc['metadata'].get('source','unknown'),
            'section':doc['metadata'].get('section','unknown')
        }
        for i,doc in enumerate(top_docs)
    ]
    
    return {
            'messages':[AIMessage(content=full_answer)],
            'answer':{
                'Answer':full_answer,
                'Citations':citations,
                "context": [
        {
            "id": i + 1,
            "content": doc["text"],
            "source": doc["metadata"].get("source", "unknown"),
            "section": doc["metadata"].get("section", "unknown")
        }
        for i, doc in enumerate(top_docs)
    ]
            }
        }


graph=StateGraph(ChatState)
graph.add_node('validate_documents',validate_documents)
graph.add_node('document_index',load_and_index_documents)
graph.add_node('document_retriever',retrieved_docs)
graph.add_node('document_ranker',reranked_docs)
graph.add_node('response_generator',generate_response)


graph.add_edge(START,'validate_documents')
graph.add_conditional_edges('validate_documents',router,{'end':END,'continue':'document_index'})
graph.add_edge('document_index','document_retriever')
graph.add_edge('document_retriever','document_ranker')
graph.add_edge('document_ranker','response_generator')
graph.add_edge('response_generator',END)
workflow=graph.compile(checkpointer=checkpointer)

