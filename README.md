### 📈 Industrial Data Analysis & RAG System

This repository contains a comprehensive data science project that demonstrates the use of machine learning and natural language processing to solve real-world industrial problems. The project is divided into two key tasks: a data analysis pipeline for predictive maintenance and an intelligent retrieval system for technical documentation.

***

### ⚙️ Task 1: Predictive Maintenance Analysis

This module processes time-series data from an industrial cyclone to generate actionable insights and forecast key operational metrics.

#### **Methodology**

1.  **Data Preprocessing**: The `main_analysis.py` script starts by cleaning the raw sensor data, handling **missing values** using a combination of forward-fill and interpolation. It then ensures a **consistent 5-minute frequency** and detects **outliers** with a robust Modified Z-score.
2.  **Operational Insights**:
    * **K-Means clustering** is used to identify distinct operational states (e.g., high-load, low-load).
    * **Isolation Forest** detects **contextual anomalies** that are unusual for a specific operational state, helping to reduce false alarms.
    * A rule-based method identifies **system shutdowns**, providing key metrics on downtime.
3.  **Forecasting**: A **Random Forest Regressor** model is trained on historical data to predict future temperature readings. The model's performance is compared against a simple **persistence baseline** to demonstrate its value.

#### **How to Run**

1.  Ensure you have `data.xlsx` in the `Task1_Sensor_Analysis` directory.
2.  Install the required Python libraries.
3.  Run the script: `python main_analysis.py`
4.  The output will be saved to the `outputs` and `plots` folders.

***

### 🧠 Task 2: Intelligent Knowledge Assistant

This module implements a Retrieval-Augmented Generation (RAG) system to allow users to query technical documents using natural language.

#### **Architecture**

1.  **Ingestion**: Documents from the `documents` folder are processed into chunks. These chunks are converted into numerical **embeddings** using a **Hugging Face `sentence-transformer` model**.
2.  **Indexing**: The embeddings are stored in a **FAISS vector store**, which is optimized for fast similarity searches.
3.  **Generation**: When a user asks a question, the system retrieves the most relevant document chunks. These chunks are then used as context for a **Flan-T5 LLM** to generate a factual, grounded answer.

#### **How to Run**

1.  Place your technical PDF files in the `documents` folder within the `Task2_RAG_Pipeline` directory.
2.  Install the required Python libraries.
3.  Run the script: `python rag_pipeline.py`
4.  The system will process the documents and then prompt you for a question.

***

### 🛡️ Guardrails & Key Features

* **Hallucination Prevention**: The RAG system is designed to provide answers strictly based on the provided documents, minimizing fabricated responses.
* **Scalable Architecture**: Both systems are modular, allowing for easy updates and scaling with more data or documents.
* **Open-Source**: The entire project is built using free, open-source models and tools, ensuring accessibility and transparency.
