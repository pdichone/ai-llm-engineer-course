import os
from typing import Optional
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama  # Updated import for chat

from langchain.schema import Document
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


class NewsArticleSummarizer:
    def __init__(
        self,
        api_key: str = None,
        model_type: str = "openai",
        model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize the summarizer with choice of model
        Args:
            api_key: OpenAI API key (required for OpenAI models)
            model_type: 'openai' or 'ollama'
            model_name: specific model name
        """
        self.model_type = model_type
        self.model_name = model_name

        # Setup LLM based on model type
        if model_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_type == "ollama":
            # Using ChatOllama with proper configuration
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",  # Optional: for structured output
                timeout=120,  # Increased timeout for longer generations
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize text splitter for long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        """
        Fetch article content using newspaper3k
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """
        Create LangChain documents from text
        """
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        return docs

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        """
        Main summarization pipeline
        """
        # Fetch article
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        # Create documents
        docs = self.create_documents(article.text)

        # Define prompts based on summary type
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:  # concise summary
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        # Create prompts
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        # Create and run chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        # Generate summary
        summary = chain.invoke(docs)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main():
    # Example of using both models
    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-cambricons-first-profit-lands/"

    # Initialize both summarizers
    openai_summarizer = NewsArticleSummarizer(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_type="openai",
        model_name="gpt-4o-mini",
    )

    ollama_summarizer = NewsArticleSummarizer(
        model_type="ollama", model_name="llama3.2"  # or "llama3.2"
    )

    # Get summaries from both models
    print("\nGenerating OpenAI Summary...")
    openai_summary = openai_summarizer.summarize(url, summary_type="detailed")

    print("\nGenerating Llama Summary...")
    llama_summary = ollama_summarizer.summarize(url, summary_type="detailed")

    # Print results
    for summary, model in [(openai_summary, "OpenAI"), (llama_summary, "Llama")]:
        print(f"\n{model} Summary:")
        print("-" * 50)
        print(f"Title: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        # Print first document content
        print("\nFirst Document Content:")
        print(summary["summary"]["input_documents"][0].page_content)

        print("\Second Document Content:")
        print(summary["summary"]["input_documents"][1].page_content)


if __name__ == "__main__":
    main()
