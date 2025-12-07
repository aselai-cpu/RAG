"""
OpenAI Service - Infrastructure layer for LLM interactions.

This service encapsulates all interactions with OpenAI's API,
including chat completions and embeddings.
"""
from typing import List, Dict, Optional
from openai import OpenAI
import os
import time
from src.infrastructure.logging import RAGLogger

# Get logger for OpenAI API calls
logger = RAGLogger.get_logger('openai_api')


class OpenAIService:
    """
    Service for interacting with OpenAI's API.

    This class handles chat completions, streaming responses,
    and manages the OpenAI API client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
    ):
        """
        Initialize the OpenAI service.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: The model to use for completions
            temperature: Temperature for response generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response using OpenAI's chat completion API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context to inject into the system message
            model: Override the default model
            temperature: Override the default temperature

        Returns:
            The generated response text
        """
        start_time = time.time()
        used_model = model or self.model
        used_temp = temperature or self.temperature

        logger.info(f"Starting OpenAI API call: model={used_model}, temperature={used_temp}, num_messages={len(messages)}, has_context={context is not None}")

        try:
            # Build the messages list
            api_messages = []

            # Add system message with context if provided
            if context:
                context_length = len(context)
                logger.debug(f"Adding context to system message: context_length={context_length}")
                system_message = {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant that answers questions based on the provided context.
Use the following context to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context:
{context}
""",
                }
                api_messages.append(system_message)
            else:
                # Default system message
                api_messages.append(
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides accurate and informative responses.",
                    }
                )

            # Add conversation messages
            api_messages.extend(messages)

            # Log user query
            if messages:
                last_message = messages[-1]
                logger.info(f"User query: '{last_message.get('content', '')[:200]}...'")

            # Generate response
            logger.debug(f"Sending request to OpenAI API: total_messages={len(api_messages)}")
            response = self.client.chat.completions.create(
                model=used_model,
                messages=api_messages,
                temperature=used_temp,
            )

            # Extract response
            response_text = response.choices[0].message.content

            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(f"OpenAI API usage: prompt_tokens={response.usage.prompt_tokens}, "
                           f"completion_tokens={response.usage.completion_tokens}, "
                           f"total_tokens={response.usage.total_tokens}")

            elapsed = time.time() - start_time
            logger.info(f"OpenAI API call completed: response_length={len(response_text)}, time={elapsed:.2f}s")
            logger.debug(f"Response preview: '{response_text[:200]}...'")

            return response_text

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in OpenAI API call: error={str(e)}, time={elapsed:.2f}s", exc_info=True)
            raise

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Generate a streaming response using OpenAI's chat completion API.

        This allows for real-time display of the response as it's being generated.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context to inject into the system message
            model: Override the default model
            temperature: Override the default temperature

        Yields:
            Chunks of the response text as they are generated
        """
        # Build the messages list (same as generate_response)
        api_messages = []

        if context:
            system_message = {
                "role": "system",
                "content": f"""You are a helpful AI assistant that answers questions based on the provided context.
Use the following context to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context:
{context}
""",
            }
            api_messages.append(system_message)
        else:
            api_messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that provides accurate and informative responses.",
                }
            )

        api_messages.extend(messages)

        # Generate streaming response
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=api_messages,
            temperature=temperature or self.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        This is a rough estimate using the rule of thumb that
        1 token ≈ 4 characters in English.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4
