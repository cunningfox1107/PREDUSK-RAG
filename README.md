# PREDUSK-RAG

## Introduction
This Retrieval Augumented Generation chatbot is based on Langgraph orchestration workflow using Pinecone vector database,OpenAI API key and streamlit.
The user can upload a single document or a chunk of text in a single session and the workflow retrives the contextual data based on user's query and returns the response along with the citations.The user can also ask several follow-up questions for the uploaded document.<br>

#### Demo Video: 

## Architecture Overview
With the user uploading a document or a textual excerpt in the text box and asking his query, the RAG workflow initiates with the document or the text being passed for loading,chunking indexing and being stored as embeddings in the vector database(Pinecone).
#### Core Components:
**Frontend**: Streamlit <br>
**Orchestration**: Langgraph <br>
**LLM**: OpenAI (gpt-4o-mini) <br>
**Embeddings**: OpenAI embedding (text-embedding-3-large) <br>
**Vector DB**: Pinecone <br>
**Reranking**: Cohere Rerank <br>
**Persistence**: InMemorySaver()<br>

### LangGraph workflow
The workflow consists of multiple nodes for different tasks,including **validate_document**,**document_index**,**document_retriver**,**document_ranker** and **response_generator**.<br><br>
The **validate_document** node checks whether a document or a text has been provided to the state or not and passes to router function which decides whether to continue the workflow or not based upon whether documents have been uploaded or not.<br><br>
The **document_index** node is used to index and embed the uploaded content in pinecone vector database.<br><br>
The  **document_retriever** node is used to retrieve the desired contexts from the content related to the query.<br><br>
The **document_reranker** node ranks the retrieved documents based upon the similarity with the query.It also includes a fallback mechanism to document_retriver node if the retrieved docs have a very low index score.<br><br>
The **response_generator** node generates the final response based upon the prompt and returns the answer along with the citations.
<br><br>
A schematic of the entire workflow has been attached here:
<br>

<img width="250" height="678" alt="rag_workflow" src="https://github.com/user-attachments/assets/7416ebc5-ea35-417b-9e5c-0426d5d9eaba" />

### Retrieval and Reranking 
- Similarity Search with Pinecone(k value=8)
- Cohere rank
- Confidence threshold : If reranker confidence>=threshold use reranked docs else fall to retrieved docs.
- Final context limited to top 5 chunks
<br><br>
This ensures precision when reranker's confidence is high and recall when the confidence is low

### Frontend 
Based in streamlit, the UI offers several features like uplaoding a pdf or a text chunk.
The functionality is limited to one document per session. On clicking the **New Chat** button, a new thread id is generated in the chat history panel. **The chatbot also displays the complete execution time and the rough token usage estimate.** Also the answer is displayed in the form of a text along with its citations.
<br><br>
**The concept of persistence allows to save the state parameters in the RAM which allows a user to visit his older chats belonging to a different thread while the server is executing.**
<br><br>
For state handling,InMemorySaver helps. Also Pinecone namespace is defined per chat thread id.**Also the feature of memory in a single chat session is also implemented using MessagePlaceHolder function so that the user can ask follow-up questions one after the other related to the content and all the messages are stored as BaseMessage in the messages list.**
This helps the llm to remember all the previous conversations.

### Index Configuration
**Pinecone Index:**
- Dimensions:3072
- Metric: Cosine
- Namespace: thread_id generated per new chat
- spec:ServerlessSpec(cloud='aws',region='us-east-1')
<br><br>

**Cohere Reranker:**
- model: rerank-english-v3.0
- top_n: 10
<br><br>

**Chunking parameters**
- chunk_size: 1200
- chunk_overlap: 200

## Limitations and Tradeoff

### Current Limitations
- In-memory state: The chats vanish on restarting the server
- Streaming not available: The response is returned after full completion
- Single document per thread-id: Only a single doc or a text chunk can be uploaded per chat session(design restrictions).

### Future Improvements
- PostgreSQL checkpointer can be enabled to store the state parameters of the chat sessios in a Postgre database.
- Streaming implemntation would improve the UX and improves the latency of the bot.
- Uploading multiple documents per chat session would improve the efficiency and the utility of the chatbot.

## RAG Evaluation 
### To validate the correctness ,and robustness of the system, a golden dataset of questions and answers is used.

#### Knowledge excerpt uploaded on the chatbot

Project: Smart Resume Analyzer
<br><br>
The Smart Resume Analyzer is an AI-powered system designed to automate resume screening and candidate evaluation. The project was developed as part of a capstone initiative focused on applying Natural Language Processing (NLP) techniques to real-world recruitment workflows.
<br><br>
The system ingests resumes in PDF format and extracts structured information such as skills, education, work experience, and certifications. Text extraction is performed using PDF parsing utilities, after which the content is cleaned and normalized. Named Entity Recognition (NER) is applied to identify key entities including programming languages, frameworks, universities, companies, and job titles.
<br><br>
A retrieval-augmented generation (RAG) pipeline is implemented to enable semantic querying over resumes. The extracted text is chunked into overlapping segments and embedded using a transformer-based embedding model. These embeddings are stored in a vector database to support similarity-based retrieval.
<br><br>
During inference, user queries are embedded and matched against stored vectors to retrieve relevant resume sections. A reranking model is applied to improve relevance by reordering retrieved chunks based on semantic similarity to the query. Only high-confidence chunks are passed to the language model for answer generation.
<br><br>
The system supports explainability by returning citations that reference the exact resume sections used to generate each answer. This ensures transparency and helps recruiters validate the model’s decisions.
<br><br>
Technologies Used:
<br><br>
Python
<br><br>
spaCy
<br><br>
Pinecone
<br><br>
OpenAI embedding and language models
<br><br>
Streamlit
<br><br>
Key Outcomes:
<br><br>
Reduced resume screening time by over 60%
<br><br>
Improved consistency in candidate shortlisting
<br><br>
Enabled natural-language queries over unstructured resume data
<br><br>

### Questions and answers
- Q1 What is the name of the project?
  - Answer: The name of the project is the Smart Resume Analyzer [2]<br>
    Citations: [2] source:pasted text,section:unknown'
<br><br>
- Q2 what is the file format uploaded
   - Answer: The uploaded file format is PDF [2].<br>
     Citations:  [2] source:pasted text,section:unknown'
- Q3 what is the database used to store the vectors
    - Answer: The database used to store the vectors is Pinecone [1].<br>
      Citations: [1] source:pasted text,section:unknown'
- Q4 what is the package used for natural entity recognition of the documents
   - Answer: Sorry, I don't know the answer based on the provided documents.
- Q5 What is the main purpose of the Smart Resume Analyzer?
    - Answer: The main purpose of the Smart Resume Analyzer is to automate resume screening and   candidate evaluation using AI technology, specifically by applying Natural Language Processing (NLP) techniques to enhance recruitment workflows [1].<br>
    Citations: [1] source:pasted_text section:unknown

##### Performance 
As we can see from the golden set, the chatbot performs decently on the provided excerpt with being able to provide 4 out of 5 right answers.

## Conclusion
This project showcased the implementation of a production oriented RAG system with retrieval and reranking along with clean UI and UX constraints.








