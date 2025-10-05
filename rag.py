import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import PyPDF2
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter 

class DocumentProcessor:
    """Process and chunk PDF documents"""
    
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
                return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        # Clean text
        text = text.replace('\n\n', ' ').replace('\n', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within next 100 chars
                sentence_ends = ['.', '!', '?', '\n']
                best_break = end
                
                for i in range(end, min(end + 100, len(text))):
                    if text[i] in sentence_ends and i + 1 < len(text) and text[i + 1] == ' ':
                        best_break = i + 1
                        break
                
                end = best_break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': f"{doc_name}_chunk_{chunk_id}",
                    'text': chunk_text,
                    'doc_name': doc_name,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            start = end - self.overlap if end < len(text) else end
        
        return chunks
    
    def process_documents(self, doc_folder: str) -> List[Dict]:
        """Process all PDFs in folder"""
        print(f"Processing documents from: {doc_folder}")
        
        all_chunks = []
        doc_files = list(Path(doc_folder).glob("*.pdf"))
        
        print(f"Found {len(doc_files)} PDF files")
        
        for pdf_path in doc_files:
            doc_name = pdf_path.stem
            print(f"    Processing: {doc_name}")
            
            text = self.extract_text_from_pdf(str(pdf_path))
            
            if text:
                chunks = self.chunk_text(text, doc_name)
                all_chunks.extend(chunks)
                print(f"      Created {len(chunks)} chunks")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks

class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize with embedding model
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks = None
        
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from chunks"""
        print("\nBuilding vector index...")
        
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
        
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        
        return results
    
    def save(self, index_path: str, chunks_path: str):
        """Save index and chunks"""
        faiss.write_index(self.index, index_path)
        
        with open(chunks_path, 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        print(f"Saved index to {index_path}")
        print(f"Saved chunks to {chunks_path}")
    
    def load(self, index_path: str, chunks_path: str):
        """Load index and chunks"""
        self.index = faiss.read_index(index_path)
        
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunks")

class AnswerGenerator:
    """Generate answers using LLM with retrieved context"""
    
    def __init__(self, model_name='google/flan-t5-base'):
        """
        Initialize LLM
        """
        print(f"\nLoading LLM: {model_name}")
        
        device = 0 if torch.cuda.is_available() else -1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        
    def generate_answer(self, query: str, contexts: List[Dict], 
                        max_length: int = 256) -> Dict:
        """
        Generate answer from query and retrieved contexts
        
        Returns dict with answer, sources, and confidence
        """
        # Construct prompt with contexts
        context_text = "\n\n".join([
            f"[Source: {ctx['doc_name']}]\n{ctx['text']}" 
            for ctx in contexts
        ])
        
        prompt = f"""Answer the following question based on the provided context. 
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
Always cite the source document names in your answer.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Truncate if too long
        max_input_length = 512
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_input_length)
        prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                max_length=max_input_length).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract unique source documents
        sources = list(set([ctx['doc_name'] for ctx in contexts]))
        
        return {
            'answer': answer,
            'sources': sources,
            'num_contexts': len(contexts),
            'query': query
        }

class RAGGuardrails:
    """Guardrails for safe and faithful RAG responses"""
    
    def __init__(self):
        self.blocked_keywords = ['password', 'credential', 'secret', 'api_key']
        
    def check_query_safety(self, query: str) -> Tuple[bool, str]:
        """Check if query is safe to process"""
        query_lower = query.lower()
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword in query_lower:
                return False, f"Query contains blocked keyword: {keyword}"
        
        # Check query length
        if len(query) < 3:
            return False, "Query too short"
        
        if len(query) > 500:
            return False, "Query too long (max 500 characters)"
        
        return True, "OK"
    
    def validate_retrieval(self, results: List[Tuple[Dict, float]], 
                          threshold: float = 1.5) -> List[Dict]:
        """
        Filter retrieved results by relevance threshold
        """
        relevant = []
        
        for chunk, distance in results:
            if distance < threshold:
                relevant.append(chunk)
        
        return relevant
    
    def check_answer_quality(self, answer: str, contexts: List[Dict]) -> Dict:
        """Check answer quality and faithfulness"""
        # Check for hallucination indicators
        cannot_answer_phrases = [
            "cannot find",
            "not in the",
            "no information",
            "don't know",
            "unable to"
        ]
        
        has_uncertainty = any(phrase in answer.lower() for phrase in cannot_answer_phrases)
        
        # Check if answer cites sources
        source_names = [ctx['doc_name'] for ctx in contexts]
        has_citation = any(source in answer for source in source_names)
        
        return {
            'has_uncertainty': has_uncertainty,
            'has_citation': has_citation,
            'length': len(answer),
            'num_contexts_used': len(contexts)
        }

class RAGSystem:
    """Complete RAG system with all components"""
    
    def __init__(self, 
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='google/flan-t5-base'):
        """Initialize RAG system"""
        
        self.doc_processor = DocumentProcessor(chunk_size=500, overlap=100)
        self.vector_store = VectorStore(embedding_model)
        self.answer_generator = AnswerGenerator(llm_model)
        self.guardrails = RAGGuardrails()
        
        self.is_indexed = False
        
    def ingest_documents(self, doc_folder: str):
        """Ingest and index documents"""
        print("="*60)
        print("DOCUMENT INGESTION")
        print("="*60)
        
        # Process documents
        chunks = self.doc_processor.process_documents(doc_folder)
        
        # Build index
        self.vector_store.build_index(chunks)
        
        self.is_indexed = True
        print("\nDocument ingestion complete!")
        
    def query(self, query: str, top_k: int = 3, 
              relevance_threshold: float = 1.5) -> Dict:
        """
        Process a query through the RAG pipeline
        """
        if not self.is_indexed:
            return {
                'error': 'System not initialized. Please ingest documents first.',
                'answer': None
            }
        
        # 1. Query safety check
        is_safe, safety_msg = self.guardrails.check_query_safety(query)
        if not is_safe:
            return {
                'error': f'Query safety check failed: {safety_msg}',
                'answer': None
            }
        
        # 2. Retrieve relevant chunks
        results = self.vector_store.search(query, k=top_k)
        
        # 3. Filter by relevance
        relevant_contexts = self.guardrails.validate_retrieval(
            results, 
            threshold=relevance_threshold
        )
        
        # 4. Handle no relevant results
        if len(relevant_contexts) == 0:
            return {
                'answer': 'I could not find relevant information in the provided documents to answer this question.',
                'sources': [],
                'confidence': 'low',
                'num_contexts': 0,
                'query': query
            }
        
        # 5. Generate answer
        response = self.answer_generator.generate_answer(query, relevant_contexts)
        
        # 6. Validate answer quality
        quality_metrics = self.guardrails.check_answer_quality(
            response['answer'], 
            relevant_contexts
        )
        
        # Determine confidence
        if quality_metrics['has_citation'] and not quality_metrics['has_uncertainty']:
            confidence = 'high'
        elif quality_metrics['has_citation'] or len(relevant_contexts) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        response['confidence'] = confidence
        response['quality_metrics'] = quality_metrics
        response['retrieval_distances'] = [dist for _, dist in results[:len(relevant_contexts)]]
        
        return response
    
    def batch_query(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries"""
        results = []
        
        for query in queries:
            print(f"\nProcessing: {query}")
            result = self.query(query)
            results.append(result)
            
        return results
    
    def save_system(self, save_dir: str):
        """Save system state"""
        os.makedirs(save_dir, exist_ok=True)
        
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        chunks_path = os.path.join(save_dir, 'chunks.json')
        
        self.vector_store.save(index_path, chunks_path)
        
    def load_system(self, save_dir: str):
        """Load system state"""
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        chunks_path = os.path.join(save_dir, 'chunks.json')
        
        self.vector_store.load(index_path, chunks_path)
        self.is_indexed = True

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
    def evaluate_retrieval(self, test_queries: List[Dict]) -> pd.DataFrame:
        """
        Evaluate retrieval quality
        """
        results = []
        
        for item in test_queries:
            query = item['query']
            relevant_docs = set(item['relevant_docs'])
            
            # Retrieve
            retrieved = self.rag_system.vector_store.search(query, k=5)
            retrieved_docs = set([chunk['doc_name'] for chunk, _ in retrieved])
            
            # Calculate metrics
            true_positives = len(relevant_docs & retrieved_docs)
            false_positives = len(retrieved_docs - relevant_docs)
            false_negatives = len(relevant_docs - retrieved_docs)
            
            precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
            recall = true_positives / len(relevant_docs) if relevant_docs else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'query': query,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'retrieved_docs': len(retrieved_docs),
                'relevant_docs': len(relevant_docs),
                'true_positives': true_positives
            })
        
        df = pd.DataFrame(results)
        
        print("\n=== RETRIEVAL EVALUATION ===")
        print(f"Average Precision: {df['precision'].mean():.3f}")
        print(f"Average Recall: {df['recall'].mean():.3f}")
        print(f"Average F1: {df['f1'].mean():.3f}")
        
        return df
    
    def evaluate_end_to_end(self, test_qa_pairs: List[Dict]) -> pd.DataFrame:
        """
        Evaluate full RAG pipeline
        """
        results = []
        
        for item in test_qa_pairs:
            query = item['query']
            expected_keywords = item['expected_answer_keywords']
            
            # Get answer
            response = self.rag_system.query(query)
            answer = response.get('answer', '')
            
            # Check keyword presence
            keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
            keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
            
            results.append({
                'query': query,
                'answer_length': len(answer),
                'keyword_coverage': keyword_score,
                'num_sources': len(response.get('sources', [])),
                'confidence': response.get('confidence', 'unknown'),
                'has_answer': len(answer) > 20
            })
        
        df = pd.DataFrame(results)
        
        print("\n=== END-TO-END EVALUATION ===")
        print(f"Average Keyword Coverage: {df['keyword_coverage'].mean():.3f}")
        print(f"Queries with answers: {df['has_answer'].sum()}/{len(df)}")
        print(f"Average sources per query: {df['num_sources'].mean():.1f}")
        
        return df

def create_sample_documents(output_dir='sample_docs'):
    """Create sample technical documents for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample document 1: Cyclone Operation Manual
    doc1_text = """
    CYCLONE SEPARATOR OPERATION MANUAL
    
    1. INTRODUCTION
    The cyclone separator is a critical component in the material processing system. 
    It uses centrifugal force to separate particles from gas streams.
    
    2. NORMAL OPERATING CONDITIONS
    - Inlet Gas Temperature: 250-350°C
    - Outlet Gas Temperature: 200-300°C
    - Inlet Draft: -50 to -150 Pa
    - Material Temperature: 80-120°C
    
    3. SHUTDOWN PROCEDURES
    A shutdown should be initiated when:
    - Inlet temperature exceeds 400°C for more than 10 minutes
    - Draft pressure drops below -200 Pa
    - Material temperature exceeds 150°C
    
    Emergency shutdown: Close inlet valve, open bypass, reduce fan speed gradually.
    
    4. TROUBLESHOOTING
    
    Issue: High Inlet Temperature
    Cause: Excessive feed rate or inadequate cooling
    Solution: Reduce feed rate by 10-20%, check cooling system
    
    Issue: Low Draft Pressure
    Cause: Blockage in cone section or fan malfunction
    Solution: Inspect cone for buildup, verify fan operation
    
    Issue: Temperature Fluctuations
    Cause: Inconsistent feed or upstream process variation
    Solution: Stabilize feed rate, check upstream conditions
    """
    
    # Sample document 2: Maintenance Guide
    doc2_text = """
    CYCLONE MAINTENANCE GUIDE
    
    ROUTINE MAINTENANCE SCHEDULE
    
    Daily Checks:
    - Verify all temperature readings are within normal range
    - Check for unusual vibrations or noise
    - Inspect seals for leaks
    
    Weekly Maintenance:
    - Clean cone section if material buildup detected
    - Lubricate fan bearings
    - Check draft pressure sensors for calibration
    
    Monthly Maintenance:
    - Inspect inlet ducting for erosion
    - Replace filters if pressure drop exceeds 100 Pa
    - Test emergency shutdown systems
    
    COMMON FAILURE MODES
    
    Draft Surge:
    A sudden increase in negative pressure can indicate partial blockage clearing.
    This may cause temporary temperature spikes. Monitor for 30 minutes.
    
    Sensor Drift:
    Temperature sensors may drift over time. Recalibrate every 6 months.
    Typical drift: ±5°C per year.
    
    Fan Belt Wear:
    Worn fan belts cause reduced draft and inefficient separation.
    Replace belts showing cracks or 10% stretch.
    """
    
    # Sample document 3: Anomaly Response Guide
    doc3_text = """
    ANOMALY RESPONSE PROCEDURES
    
    TEMPERATURE ANOMALIES
    
    Sudden Inlet Temperature Drop:
    - Likely causes: Feed interruption, draft surge, cooling system malfunction
    - Immediate action: Check feed system, verify draft readings
    - If draft spike coincides: Probable blockage clearing event
    - Monitor for 15-30 minutes before intervention
    
    Gradual Temperature Increase:
    - Likely causes: Material accumulation, reduced airflow, process drift
    - Immediate action: Inspect cone section, check fan operation
    - Initiate controlled shutdown if temperature >380°C
    
    DRAFT ANOMALIES
    
    High Negative Pressure:
    - Indicates blockage or excessive fan speed
    - Reduce fan speed by 10%, inspect for blockages
    - Do not exceed -250 Pa for extended periods
    
    Low or Positive Pressure:
    - Indicates leak, fan failure, or bypass valve issue
    - Check all access points for leaks
    - Verify fan operation and bypass valve position
    
    COMBINED ANOMALIES
    
    Temperature drop + Draft spike:
    - Classic signature of blockage clearing
    - Usually self-correcting
    - Document event and inspect during next maintenance
    
    Temperature spike + Draft drop:
    - Serious condition indicating possible fan failure
    - Initiate emergency shutdown procedures
    - Do not restart until root cause identified
    """
    
    # Write PDFs (simplified - in real implementation use reportlab or similar)
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    docs = [
        ('cyclone_operation_manual.pdf', doc1_text),
        ('maintenance_guide.pdf', doc2_text),
        ('anomaly_response.pdf', doc3_text)
    ]
    
    for filename, text in docs:
        filepath = os.path.join(output_dir, filename)
        c = canvas.Canvas(filepath, pagesize=letter)
        
        # Simple text rendering
        y_position = 750
        for line in text.split('\n'):
            if y_position < 50:
                c.showPage()
                y_position = 750
            c.drawString(50, y_position, line[:80])  # Truncate long lines
            y_position -= 15
        
        c.save()
        print(f"Created: {filepath}")

def run_demo():
    """Run complete RAG system demo"""
    
    print("="*60)
    print("RAG SYSTEM DEMO")
    print("="*60)
    
    # 1. Create sample documents
    print("\n1. Creating sample documents...")
    create_sample_documents('sample_docs')
    
    # 2. Initialize RAG system
    print("\n2. Initializing RAG system...")
    rag = RAGSystem(
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        llm_model='google/flan-t5-base'  # Use 'flan-t5-small' for faster inference
    )
    
    # 3. Ingest documents
    print("\n3. Ingesting documents...")
    rag.ingest_documents('sample_docs')
    
    # 4. Test queries
    print("\n4. Testing queries...")
    print("="*60)
    
    test_queries = [
        "What does a sudden draft drop indicate?",
        "What are the normal operating temperature ranges?",
        "How should I respond to high inlet temperature?",
        "What maintenance should be done weekly?",
        "What causes temperature fluctuations?"
    ]
    
    results = []
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = rag.query(query, top_k=3)
        
        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources: {', '.join(response['sources'])}")
        print(f"Confidence: {response['confidence']}")
        print(f"Contexts used: {response['num_contexts']}")
        
        results.append(response)
    
    # 5. Save results
    print("\n5. Saving results...")
    results_df = pd.DataFrame([
        {
            'query': r['query'],
            'answer': r['answer'],
            'sources': ', '.join(r['sources']),
            'confidence': r['confidence'],
            'num_contexts': r['num_contexts']
        }
        for r in results
    ])
    
    results_df.to_csv('demo_results.csv', index=False)
    print("Results saved to demo_results.csv")
    
    # 6. Save system
    print("\n6. Saving RAG system...")
    rag.save_system('rag_index')
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)

def run_evaluation():
    """Run evaluation on RAG system"""
    
    # Initialize system
    rag = RAGSystem()
    rag.ingest_documents('sample_docs')
    
    # Evaluation test cases
    test_retrieval = [
        {
            'query': 'What are normal operating temperatures?',
            'relevant_docs': ['cyclone_operation_manual']
        },
        {
            'query': 'How to handle draft anomalies?',
            'relevant_docs': ['anomaly_response']
        },
        {
            'query': 'Weekly maintenance tasks?',
            'relevant_docs': ['maintenance_guide']
        }
    ]
    
    test_qa = [
        {
            'query': 'What does a sudden draft drop indicate?',
            'expected_answer_keywords': ['leak', 'fan', 'failure', 'bypass']
        },
        {
            'query': 'Normal inlet temperature range?',
            'expected_answer_keywords': ['250', '350', 'temperature']
        }
    ]
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    
    retrieval_results = evaluator.evaluate_retrieval(test_retrieval)
    qa_results = evaluator.evaluate_end_to_end(test_qa)
    
    # Save evaluation
    retrieval_results.to_csv('evaluation_retrieval.csv', index=False)
    qa_results.to_csv('evaluation_qa.csv', index=False)
    
    print("\nEvaluation complete! Results saved.")

if __name__ == "__main__":
    # Run demo
    run_demo()
    
    # Optionally run evaluation
    # run_evaluation()