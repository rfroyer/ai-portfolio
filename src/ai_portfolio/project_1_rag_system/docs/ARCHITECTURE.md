# AutoRAG: System Architecture Design

---

## Document Information

| Field | Value |
|-------|-------|
| **Project Name** | AutoRAG (Autonomous Retrieval-Augmented Generation) |
| **Document Type** | System Architecture Design |
| **Version** | 1.0 |
| **Last Updated** | January 25, 2026 |
| **Status** | Final |

---

## 1. Architecture Overview

The AutoRAG system is designed as a modular, end-to-end solution for building and evaluating a Retrieval-Augmented Generation pipeline. The architecture is composed of two primary workflows: the **RAG Workflow** for answering user queries and the **Evaluation Workflow** for autonomously assessing the system's performance.

The design prioritizes simplicity, modularity, and adherence to the defined MVP scope, while allowing for future expansion. It leverages a local-first approach for core components (FAISS, SQLite) to minimize costs and complexity, consistent with the project constraints.

---

## 2. System Architecture Diagram

Below is a diagram representing the high-level architecture of the AutoRAG system. It illustrates the key components and their interactions across the two main workflows.

```mermaid
graph TD
    subgraph "User Interaction"
        A[User via CLI/API] --> B{RAG Pipeline};
    end

    subgraph "RAG Workflow"
        B --> C["1. Retriever"];
        C --> D["Vector DB FAISS"];
        B --> E["2. Generator LLM"];
        C --> E;
        E --> F["Generated Response"];
        F --> A;
    end

    subgraph "Data Ingestion Offline"
        G["Knowledge Base - Text Files"] --> H["Document Loader"];
        H --> I["Text Splitter"];
        I --> J["Embedding Model"];
        J --> D;
    end

    subgraph "Evaluation Workflow Autonomous"
        K["Evaluation Agent Scheduled"] --> B;
        B --> L["Response for Eval"];
        L --> K;
        K --> M["Evaluation Logic"];
        M --> N["Evaluation DB SQLite"];
    end

    style A fill:#cde4ff
    style G fill:#d5e8d4
    style K fill:#fff2cc
```

**How to View the Diagram:**
*   If you have the **Markdown Preview Enhanced** extension in VS Code, this diagram will render automatically in the preview pane.
*   Alternatively, you can copy the `mermaid` code block and paste it into an online editor like the [**Mermaid Live Editor**](https://mermaid.live) to see the visual diagram.

---

## 3. Component Breakdown

This section provides a detailed description of each component outlined in the architecture diagram.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **1. Data Ingestion Pipeline** | Python, LangChain | An offline script responsible for loading documents from a local directory, splitting them into manageable chunks, generating embeddings using an OpenAI model, and storing them in the vector database. |
| **2. Vector Database** | FAISS | A local, file-based vector store that indexes the document embeddings for efficient similarity search. It is simple to set up and ideal for the MVP. |
| **3. RAG Pipeline** | LangChain | The core of the system. It orchestrates the retrieval of relevant document chunks from the Vector DB and the generation of a final answer by an LLM, using the retrieved context. |
| **4. User Interface** | Python (CLI/FastAPI) | A simple command-line interface (CLI) or a basic FastAPI endpoint to allow a user to submit questions to the RAG Pipeline and receive answers. |
| **5. Autonomous Evaluation Agent** | CrewAI / Python Scheduler | A scheduled process that autonomously runs a set of predefined questions against the RAG pipeline, captures the responses, and passes them to the evaluation logic. |
| **6. Evaluation Logic** | Python, LangChain | A module that assesses the quality of the RAG system's responses based on metrics like accuracy and relevance. It uses an LLM to perform the assessment. |
| **7. Evaluation Database** | SQLite | A local, file-based SQL database used to log the results of each evaluation run, including the question, response, score, and any feedback from the evaluation logic. |

---

## 4. Data Flow

This section describes the sequence of data movement within the system for the two primary workflows.

### 4.1. RAG Workflow (Querying)

1.  **Query Input:** A user submits a question through the CLI or API.
2.  **Retrieval:** The RAG Pipeline's Retriever takes the user's query, creates an embedding for it, and queries the FAISS Vector Database to find the most similar (i.e., relevant) document chunks.
3.  **Context Augmentation:** The retrieved document chunks are compiled into a context string.
4.  **Generation:** The context and the original query are passed to the Generator (LLM), which produces a final, human-readable answer.
5.  **Response Output:** The generated answer is returned to the user via the CLI or API.

### 4.2. Evaluation Workflow (Autonomous Assessment)

1.  **Trigger:** A scheduler (e.g., a cron job or a Python `schedule` library) activates the Evaluation Agent.
2.  **Question Execution:** The agent iterates through a predefined list of evaluation questions and sends each one to the RAG Pipeline.
3.  **Response Capture:** The agent receives the generated response from the pipeline for each question.
4.  **Quality Assessment:** The agent invokes the Evaluation Logic, providing the question, the ground truth (if available), and the generated response. The logic uses an LLM to score the response on accuracy and relevance.
5.  **Logging:** The agent logs the complete evaluation record (question, response, scores, timestamp) into the SQLite Evaluation Database for later analysis.

---

## 5. Technology Stack

This table summarizes the technology choices for the project, aligning with the MVP scope and constraints.

| Category | Technology | Justification |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | The standard for AI/ML development with extensive library support. |
| **AI Framework** | LangChain | Simplifies the creation of RAG pipelines and agentic workflows. |
| **Agent Framework** | CrewAI | Provides a straightforward structure for creating the autonomous evaluation agent. |
| **LLM Provider** | OpenAI | Offers high-quality models for embedding and generation, accessible via a simple API. |
| **Vector Database** | FAISS | A powerful, local-first vector search library that is free and easy to set up for the MVP. |
| **Evaluation Database** | SQLite | A serverless, file-based database perfect for simple, local logging without requiring a separate database server. |
| **User Interface** | Python `argparse` / FastAPI | Provides a simple and effective way to interact with the system for the MVP, fulfilling the CLI/API requirement. |
| **Code Editor** | Visual Studio Code | A versatile and powerful editor with excellent Python and Git integration. |
| **Version Control** | Git & GitHub | Industry standard for version control and collaborative development. |

---

## 6. Design Principles

*   **Modularity:** Each component (ingestion, RAG, evaluation) is designed to be independent, allowing for easier testing, maintenance, and future upgrades.
*   **Simplicity (MVP First):** The architecture intentionally uses simple, local-first technologies (FAISS, SQLite) to meet the project constraints and ensure rapid development.
*   **Extensibility:** The design allows for future enhancements, such as swapping the local vector database for a cloud-based one or adding a more sophisticated UI, as outlined in the Post-MVP scope.

---

## 7. Future Considerations (Post-MVP)

This architecture provides a solid foundation for the out-of-scope features defined in the `SCOPE.md` document. Future work could include:

*   **Replacing FAISS:** The `Vector Database` component can be replaced with a production-grade service like Pinecone or a cloud-native solution like Snowflake Vector Search with minimal changes to the RAG pipeline.
*   **Replacing SQLite:** The `Evaluation Database` can be migrated to a more robust cloud database like PostgreSQL on AWS RDS or Snowflake for better scalability and analytics.
*   **Adding a UI:** A web interface (e.g., using Streamlit or React) can be built on top of the FastAPI endpoint to provide a richer user experience.
