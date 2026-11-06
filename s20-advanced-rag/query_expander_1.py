import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


## Part 1 -- Only focus on the QueryExpander class and getting the query expansion to work
class QueryExpander:
    """
    A class to expand a single query into multiple semantically similar variations
    to improve retrieval coverage.
    """

    def __init__(self, temperature: float = 0.3):
        """
        Initialize the QueryExpander.

        Args:
            temperature: Controls randomness in LLM response. Lower values make output more focused.
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-4o-mini")

        # Prompt template for query expansion
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question, generate 3 different versions of the question 
            that capture different aspects and perspectives of the original question. 
            Make the variations semantically diverse but relevant.
            
            Original Question: {question}
            
            Generate variations in the following format:
            1. [First variation]
            2. [Second variation]
            3. [Third variation]
            
            Only output the numbered variations, nothing else.""",
        )

    def expand_query(self, question: str) -> List[str]:
        """
        Expand a single query into multiple variations.

        Args:
            question: The original question to expand.

        Returns:
            List of query variations including the original question.

        Example:
            Original: "What are the effects of climate change?"
            Expanded: [
                "How does global warming impact our environment?",
                "What are the consequences of rising temperatures on Earth?",
                "What environmental changes are caused by greenhouse gases?",

            ]
        """
        try:
            # Get variations from LLM
            response = self.llm.invoke(
                self.query_expansion_prompt.format(question=question)
            )

            # Parse numbered list from response
            variations = [
                line.split(". ")[1] for line in response.content.strip().split("\n")
            ]

            # Add original question to variations
            variations.append(question)

            return variations

        except Exception as e:
            print(f"Error in query expansion: {e}")
            # If there's an error, return just the original question
            return [question]


def main():
    """
    Example usage of QueryExpander
    """
    # Initialize QueryExpander
    expander = QueryExpander()

    # Example questions
    questions = [
        "What are the main causes of global warming?",
        "How does exercise affect mental health?",
        "What are the benefits of renewable energy?",
    ]

    # Test query expansion
    for original_question in questions:
        print(f"\nOriginal Question: {original_question}")
        print("Expanded Queries:")

        expanded_queries = expander.expand_query(original_question)

        for i, query in enumerate(expanded_queries, 1):
            if query != original_question:
                print(f"{i}. {query}")

        print(f"Original: {original_question}")


if __name__ == "__main__":
    main()
