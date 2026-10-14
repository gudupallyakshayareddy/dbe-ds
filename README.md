# DSE & DBD Project  

# ResearchHub — Research Knowledge Repository
        
## Aim
To develop a research knowledge repository that stores research papers, extracts their text, generates embeddings, and enables semantic search to retrieve relevant research information.

## Objectives
- Upload and manage research papers in PDF format.
- Extract and preprocess text from research papers.
- Generate text embeddings using machine learning techniques.
- Store paper metadata, extracted text, and embeddings in PostgreSQL with pgvector.
- Perform semantic search and rank papers by relevance.
- Implement database operations and maintain activity records.
- Provide a user-friendly web interface for research paper management.

## Review 1 – Project Foundation
- Defined the problem statement and project scope.
- Identified the required technologies and architecture.
- Designed the initial database and web application structure.
- Started implementing PDF upload, text extraction, and database storage.

## Review 2 – Core Implementation
- Implemented PDF text extraction and embedding generation.
- Integrated PostgreSQL and pgvector for storing paper information and embeddings.
- Developed the research repository interface.
- Worked on semantic search and relevance-based retrieval.

## Review 3 – Advanced Implementation and Integration
- Improved research paper management and database operations.
- Worked on update and delete functionality and database activity tracking.
- Integrated machine learning components and analytics.
- Continued testing search, database operations, and application navigation.
- 

## Project Structure

```text
Research_Knowledge_Repository/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── search.html
│   ├── repository.html
│   ├── paper.html
│   ├── compare.html
│   ├── update.html
│   ├── placement.html
│   ├── ml_dashboard.html
│   ├── research_chat.html
│   ├── research_gaps.html
│   └── chatbot.html
│
├── static/
│   ├── style.css
│   ├── script.js
│   ├── chatbot.css
│   └── chatbot.js
│
├── services/
│   ├── embedding_service.py
│   └── document_service.py
│
├── ml/
│   ├── search_evaluation.py
│   ├── evaluate_real_search.py
│   ├── run_real_evaluation.py
│   └── artifacts/
│       └── model_metrics.json
│
└── database/
    └── schema.sql
```

> **Note:** This is a reference structure. Include only files and folders that actually exist in your repository.

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Database:** PostgreSQL
- **Semantic Search:** Sentence Transformers, pgvector
- **Machine Learning:** Python ML libraries
- **Document Processing:** PDF text extraction and embedding generation

## Main Features

- User registration and authentication
- Research paper upload and text extraction
- Embedding generation and database storage
- Semantic search and relevance ranking
- Research paper repository and comparison
- Research paper summarization and research gap analysis
- Placement prediction and ML evaluation
- Rule-based doubt-clearing chatbot

## Project Goals

ResearchHub aims to organize research papers, retrieve relevant documents using semantic similarity, and provide tools for exploring and evaluating research information.

## Running the Project

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Configure PostgreSQL and the required database settings.
4. Configure environment variables locally.
5. Start the Flask application using `python app.py`.
6. Open `http://127.0.0.1:5000` in your browser.

## Security

Do not commit API keys, passwords, `.env` files, virtual environments, or other sensitive information to GitHub.

## Conclusion
The Research Knowledge Repository combines database management, machine learning, and semantic search to organize research papers and retrieve relevant information efficiently. The project demonstrates how text preprocessing, embeddings, vector similarity, and database operations can work together in a practical research application.
