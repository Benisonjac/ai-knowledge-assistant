# Artificial Intelligence – Key Concepts and Technologies (2024-2026)

## Large Language Models (LLMs)

Large Language Models are neural networks trained on massive text corpora to understand and generate human-like text. Modern LLMs like GPT-4o, Claude, and Gemini use the Transformer architecture with billions of parameters. They excel at tasks such as text generation, summarization, translation, code writing, and reasoning.

### Key Concepts:
- **Tokenization**: Breaking text into subword units (tokens) that the model processes.
- **Attention Mechanism**: The self-attention mechanism allows the model to weigh the importance of different parts of the input when generating each output token.
- **Context Window**: The maximum number of tokens the model can process at once. Modern models support 128K+ tokens.
- **Fine-tuning**: Adapting a pre-trained model to specific tasks using domain-specific data.
- **RLHF (Reinforcement Learning from Human Feedback)**: A training technique that aligns model outputs with human preferences.

## Retrieval Augmented Generation (RAG)

RAG is a technique that enhances LLM responses by first retrieving relevant documents from a knowledge base, then using those documents as context for the LLM to generate answers. This approach reduces hallucinations and grounds responses in factual, up-to-date information.

### RAG Pipeline:
1. **Document Ingestion**: Loading documents (PDF, DOCX, etc.) and extracting text.
2. **Chunking**: Splitting documents into smaller, overlapping text chunks for embedding.
3. **Embedding**: Converting text chunks into dense vector representations using models like sentence-transformers.
4. **Vector Storage**: Storing embeddings in a vector database (ChromaDB, Pinecone, Weaviate) for fast similarity search.
5. **Retrieval**: Finding the most relevant chunks for a user query using cosine similarity.
6. **Reranking**: Using cross-encoder models to re-score and reorder retrieved documents for better precision.
7. **Generation**: Feeding retrieved context to an LLM to generate a grounded answer.

### Advanced RAG Techniques:
- **Hybrid Search**: Combining dense (vector) and sparse (BM25) retrieval for better coverage.
- **Query Rewriting**: Using an LLM to rewrite ambiguous queries before retrieval.
- **Agentic RAG**: Using LLM agents (e.g., LangGraph) to dynamically decide retrieval strategies.
- **Self-Corrective RAG**: Checking generated answers for hallucinations and retrying if not grounded.

## Vector Databases

Vector databases are specialized databases optimized for storing and querying high-dimensional vectors. They enable fast approximate nearest neighbor (ANN) search, which is essential for RAG systems.

### Popular Vector Databases:
- **ChromaDB**: Open-source, easy to use, supports local persistence.
- **Pinecone**: Managed cloud vector database with high scalability.
- **Weaviate**: Open-source with built-in vectorization and GraphQL API.
- **Qdrant**: High-performance vector database with filtering capabilities.
- **FAISS**: Facebook's library for efficient similarity search (not a full database).

## Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning. Similar texts have vectors that are close together in vector space.

### Popular Embedding Models:
- **sentence-transformers/all-MiniLM-L6-v2**: Lightweight, fast, good for general use.
- **text-embedding-3-small (OpenAI)**: High quality, API-based.
- **BAAI/bge-large-en**: Top performing open-source embedding model.
- **Cohere embed-v3**: Multilingual, high quality embedding model.

## LangChain & LangGraph

**LangChain** is a framework for building applications powered by LLMs. It provides abstractions for chains, tools, memory, and agents.

**LangGraph** extends LangChain with a graph-based framework for building stateful, multi-step LLM applications. It models workflows as state machines where:
- **Nodes** are processing steps (retrieve, generate, validate).
- **Edges** define the flow between steps.
- **Conditional edges** allow the graph to branch based on runtime decisions.

## Observability & Evaluation

### LangSmith
LangSmith is a platform for debugging, testing, and monitoring LLM applications. It provides:
- Trace visualization for multi-step chains.
- Latency and cost tracking.
- Comparison of different prompt/model configurations.

### RAGAS (RAG Assessment)
RAGAS is a framework for evaluating RAG systems with metrics like:
- **Faithfulness**: Is the answer supported by the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved documents relevant?
- **Context Recall**: Was all necessary information retrieved?

## Prompt Engineering

Prompt engineering is the practice of designing effective prompts to guide LLM behavior. Key techniques include:
- **System prompts**: Setting the LLM's role and behavior rules.
- **Few-shot learning**: Providing examples in the prompt.
- **Chain-of-thought**: Asking the model to reason step by step.
- **Output formatting**: Specifying the desired output structure.

## AI Safety & Responsible AI

- **Hallucination mitigation**: Using RAG, grounding, and validation to prevent fabricated information.
- **Content filtering**: Implementing guardrails to prevent harmful outputs.
- **Bias detection**: Monitoring for and mitigating biases in model outputs.
- **Transparency**: Providing source citations and confidence indicators.
