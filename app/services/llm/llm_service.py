from typing import List, Dict
import google.generativeai as genai
from config import Config


class LLMService:
    def __init__(self):
        """
        Initialize the LLM service with the appropriate model
        """
        # Configure the API key
        genai.configure(api_key=Config.GEMINI_API_KEY)

        # Initialize the model
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def create_context_prompt(self, query: str, relevant_texts: List[Dict]) -> str:
        """
        Create a prompt that includes the query and relevant context
        """
        # Format the relevant texts
        context = "\n".join(
            [f"Context {i+1}: {text['text']}" for i, text in enumerate(relevant_texts)]
        )

        # Create the prompt template
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.
        If the context doesn't contain relevant information, say so and provide a general answer.

        Context:
        {context}

        User Question: {query}

        Please provide a detailed answer based on the context above.
        """

        return prompt

    def get_response(self, query: str, relevant_texts: List[Dict]) -> str:
        """
        Get a response from the LLM using the query and relevant context
        """
        # Create the prompt
        prompt = self.create_context_prompt(query, relevant_texts)

        # Get response from the model
        response = self.model.generate_content(prompt)

        return response.text
