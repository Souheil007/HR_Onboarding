"""
Document evaluation chain for LangGraph RAG workflows using Google Gemini

This module handles document relevance evaluation as part of the LangGraph RAG
pipeline. It determines whether retrieved documents contain enough relevant
information to answer a user's question effectively using Gemini LLM.

Used within the LangGraph workflow to make routing decisions about whether
to proceed with document-based answers or fall back to online search.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

class EvaluateDocs(BaseModel):
    """
    Document evaluation results for LangGraph RAG workflows

    Structures evaluation results when assessing whether retrieved documents
    are sufficient for answering a question.
    """
    score: str = Field(
        description="Whether documents are relevant to the question - 'yes' if sufficient, 'no' if insufficient"
    )
    relevance_score: float = Field(
        default=0.5,
        description="Relevance score between 0.0 and 1.0 indicating how well documents match the query",
        ge=0.0,
        le=1.0
    )
    coverage_assessment: str = Field(
        default="",
        description="Assessment of how well the documents cover the query requirements"
    )
    missing_information: str = Field(
        default="",
        description="Description of key information missing from documents (if any)"
    )

# Wrap LLM with structured output
structured_output = llm.with_structured_output(EvaluateDocs)

# System prompt describing evaluation instructions
system_prompt = """You are an expert document relevance evaluator for a RAG (Retrieval-Augmented Generation) system. 
Assess whether retrieved documents contain sufficient information to answer a user's query effectively.

EVALUATION FRAMEWORK:
1. TOPICAL RELEVANCE:
   - Do the documents directly address the main subject of the query?
   - Are the key concepts and themes aligned with what the user is asking?

2. INFORMATION SUFFICIENCY:
   - Is there enough detail to provide a comprehensive answer?
   - Are specific facts, data, or examples present when needed?
   - Can the query be answered without requiring external knowledge?

3. INFORMATION QUALITY:
   - Is the information accurate and credible?
   - Are there conflicting statements within the documents?
   - Is the information current and relevant to the query context?

4. COMPLETENESS ASSESSMENT:
   - Does the document set cover all aspects of the query?
   - Are there obvious gaps in information that would prevent a complete answer?

SCORING CRITERIA:
- Score 'yes' if documents provide sufficient, relevant information
- Score 'no' if documents lack key information or are off-topic
- Provide a relevance score (0.0-1.0)
- Assess coverage and identify missing critical information
"""

# Combine system + human prompts into a ChatPromptTemplate
evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", """Please evaluate whether the retrieved documents are sufficient to answer the user's query.

USER QUERY:
{question}

RETRIEVED DOCUMENTS:
{document}

EVALUATION REQUIRED:
1. Primary Score: 'yes' or 'no'
2. Relevance Score: 0.0-1.0
3. Coverage Assessment: How well do the documents address the query requirements?
4. Missing Information: Key missing info, if any

Provide your comprehensive evaluation based on the framework above."""),
    ]
)

# Final runnable chain
evaluate_docs: RunnableSequence = evaluate_prompt | structured_output
