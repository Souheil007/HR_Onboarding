from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# Initialize the Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

# Define structured output model
class DocumentRelevance(BaseModel):
    """Model for document relevance evaluation results"""
    
    binary_score: bool = Field(
        description="Whether the answer is grounded in the documents - true if supported, false if not supported"
    )
    
    confidence: float = Field(
        default=0.5,
        description="Confidence score between 0.0 and 1.0 indicating how certain the evaluation is",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the answer is or isn't grounded in the documents"
    )

# Attach structured output to the LLM
structured_output = llm.with_structured_output(DocumentRelevance)

# System prompt for relevance evaluation
system_prompt = """You are an expert document relevance evaluator. Your task is to determine whether an LLM-generated answer is properly grounded in the provided source documents.

EVALUATION CRITERIA:
- The answer must be directly supported by information found in the source documents
- Key facts, claims, and details should be traceable to the provided documents
- The answer should not contain information that contradicts the source documents
- Minor paraphrasing or reasonable inference from the documents is acceptable
- The answer should not include fabricated information or external knowledge not present in the documents

SCORING GUIDELINES:
- Score 'yes' (true) if the answer is well-supported by the documents
- Score 'no' (false) if the answer contains unsupported claims, contradictions, or fabricated information

Be strict in your evaluation to ensure answer quality and prevent hallucinations.
"""

# Human prompt with placeholders for documents and LLM solution
human_prompt = """Please evaluate whether the LLM generation is grounded in the provided documents.

SOURCE DOCUMENTS:
{documents}

LLM GENERATION TO EVALUATE:
{solution}

Provide:
1. A binary score (true/false) indicating if the answer is grounded in the documents
2. A confidence score (0.0-1.0) for your evaluation
3. A brief reasoning explaining your decision

Based on the evaluation criteria, is this answer properly grounded in the source documents?
"""

# Combine into a prompt template
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Combine prompt + structured output into a runnable sequence
document_relevance: RunnableSequence = relevance_prompt | structured_output
