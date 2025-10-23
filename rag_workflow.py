import streamlit as st
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph # Responsible for state management and workflow orchestration and Logging to langsmith

from state import GraphState
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance
from topic_router import TopicRouter 

class RAGWorkflow:
    """Manages the RAG workflow using LangGraph without online search"""
    
    def __init__(self):
        self.graph = None
        self.retriever = None
        self._current_session_retriever_key = None
        self.topic_router = TopicRouter()
    
    def get_graph(self):
        """Get or create the graph instance (cached for performance)"""
        if 'graph_instance' not in st.session_state or st.session_state.graph_instance is None:
            st.session_state.graph_instance = self._create_graph()
        return st.session_state.graph_instance
    
    def set_retriever(self, retriever):
        """Set the document retriever"""
        self.retriever = retriever
        if retriever is not None:
            current_file_key = st.session_state.get('processed_file')
            self._current_session_retriever_key = current_file_key
            print(f"Retriever set for file: {current_file_key}")
        else:
            self._current_session_retriever_key = None
            print("Retriever cleared")
    
    def get_current_retriever(self):
        """Get the current retriever, with fallback to session state"""
        if self.retriever is not None:
            return self.retriever
        session_retriever = st.session_state.get('retriever')
        if session_retriever is not None:
            print("Using retriever from session state")
            self.retriever = session_retriever
            return session_retriever
        return None
    
    def process_question(self, question):
        """Process a question through the RAG workflow"""
        print(f"STARTING RAG WORKFLOW for question: '{question}'")
        current_retriever = self.get_current_retriever()
        self.set_retriever(current_retriever)
        graph = self.get_graph()
        result = graph.invoke(input={"question": question})
        print(f"RAG WORKFLOW COMPLETED")
        return result
    
    def _create_graph(self):
        """Create and configure the state graph for handling queries"""
        workflow = StateGraph(GraphState)
        # Add nodes
        workflow.add_node("Detect Topic", self._detect_topic)
        workflow.add_node("Retrieve Documents", self._retrieve)
        #workflow.add_node("Grade Documents", self._evaluate)
        workflow.add_node("Generate Answer", self._generate_answer)
        
        # Set entry point
        workflow.set_entry_point("Detect Topic")
        
        # Define edges
        workflow.add_edge("Detect Topic", "Retrieve Documents")
        workflow.add_edge("Retrieve Documents", "Generate Answer")
        #workflow.add_edge("Retrieve Documents", "Grade Documents")
        #workflow.add_edge("Grade Documents", "")
        workflow.add_edge("Generate Answer", END)
        
        return workflow.compile()
    
    def _detect_topic(self, state: GraphState):
        """Detect the topic of the question and store it in state"""
        question = state["question"]
        topic = self.topic_router.detect_topic(question)
        state["topic"] = topic
        print(f"GRAPH STATE: Detected topic -> {topic}")
        return {"question": question, "topic": topic}

    
    def _retrieve(self, state: GraphState):
        """Retrieve documents relevant to the user's question"""
        print("GRAPH STATE: Retrieve Documents")
        question = state["question"]    
        current_retriever = self.get_current_retriever()
        if current_retriever is None:
            print("No retriever available, returning empty document list")
            return {"documents": [], "question": question}
        try:
            documents = current_retriever(question)
            print(f"Retrieved {len(documents)} documents")
            return {"documents": documents, "question": question}
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            self.retriever = None
            st.session_state.retriever = None
            return {"documents": [], "question": question}
    
    def _evaluate(self, state: GraphState):
        """Filter documents based on their relevance to the question"""
        print("GRAPH STATE: Grade Documents")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        document_evaluations = []
        for document in documents:
            response = evaluate_docs.invoke({"question": question, "document": document.page_content})
            document_evaluations.append(response)
            if response.score.lower() == "yes":
                filtered_docs.append(document)
        print(f"Filtered to {len(filtered_docs)} relevant documents")
        return {"documents": filtered_docs, "question": question, "document_evaluations": document_evaluations}
    
    def _generate_answer(self, state: GraphState):
        """Generate an answer based on the retrieved documents"""
        print("GRAPH STATE: Generate Answer")
        question = state["question"]
        documents = state["documents"]
        solution = generate_chain.invoke({"context": documents, "question": question})
        print(f"Answer generated: {len(solution)} characters")
        
        # Check relevance
        #doc_relevance_score = document_relevance.invoke({"documents": documents, "solution": solution})
        #question_relevance_score = question_relevance.invoke({"question": question, "solution": solution})
        #state["document_relevance_score"] = doc_relevance_score
        #state["question_relevance_score"] = question_relevance_score
        
        return {"documents": documents, "question": question, "solution": solution}
