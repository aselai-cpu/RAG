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
        temperature: float = 0
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

        # Log request details
        logger.info("=" * 80)
        logger.info(f"OpenAI API Request - model={used_model}, temperature={used_temp}, num_messages={len(messages)}, has_context={context is not None}")
        logger.info("-" * 80)

        try:
            # Build the messages list
            api_messages = []

            # Add system message with context if provided
            if context:
                context_length = len(context)
                logger.debug(f"Adding context to system message: context_length={context_length} chars")
                system_message = {
                    "role": "system",
                    "content": f"""You are a RAG (Retrieval-Augmented Generation) assistant. Your role is to answer questions STRICTLY based on the provided context from the knowledge base.

CRITICAL RULES:
1. ONLY use information from the provided context below to answer questions
2. DO NOT use any external knowledge or general information
3. If the answer cannot be found in the context, explicitly state: "I cannot answer this question based on the provided documents. The information is not available in the knowledge base."
4. If the context is partially relevant, only answer the parts that are covered by the context
5. Cite which source/document the information comes from when possible

Context from Knowledge Base:
{context}

Remember: You are a RAG system. Your answers must be grounded in the provided context only.""",
                }
                api_messages.append(system_message)
                logger.debug("System message with context added:")
                logger.debug(f"Full Context ({context_length} chars):")
                logger.debug(context)
            else:
                # No context available - check if this is entity extraction
                is_entity_extraction = (
                    len(messages) > 0 and 
                    isinstance(messages[0], dict) and
                    ("Analyze the following text and extract" in messages[0].get("content", "") or
                     "Extract key entities" in messages[0].get("content", ""))
                )
                
                if is_entity_extraction:
                    # For entity extraction, don't add RAG restrictions
                    logger.debug("Entity extraction detected - skipping RAG restrictions")
                else:
                    # For regular RAG queries without context, enforce restrictions
                    api_messages.append(
                        {
                            "role": "system",
                            "content": """You are a RAG (Retrieval-Augmented Generation) assistant. Your role is to answer questions based on documents in the knowledge base.

IMPORTANT: No relevant documents were found in the knowledge base for this query.

You MUST respond with: "I cannot answer this question because no relevant information was found in the knowledge base. Please try rephrasing your question or ensure that relevant documents have been uploaded to the system."

DO NOT attempt to answer the question using general knowledge. You are a RAG system and can only answer based on the knowledge base.""",
                        }
                    )
                    logger.debug("System message added (no context - RAG restriction enforced)")

            # Add conversation messages
            api_messages.extend(messages)

            # Log full request structure
            logger.info("Request Messages:")
            for i, msg in enumerate(api_messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_length = len(content)
                logger.info(f"  [{i+1}] {role.upper()} ({content_length} chars):")
                logger.info(content)
                logger.info("")  # Empty line for readability
            
            # Log user query separately for easy reference
            if messages:
                last_message = messages[-1]
                user_query = last_message.get('content', '')
                query_length = len(user_query)
                logger.info(f"User Query ({query_length} chars):")
                logger.info(user_query)

            # Generate response
            logger.debug(f"Sending request to OpenAI API: total_messages={len(api_messages)}")
            request_timestamp = time.time()
            response = self.client.chat.completions.create(
                model=used_model,
                messages=api_messages,
                temperature=used_temp,
            )
            response_timestamp = time.time()
            api_latency = response_timestamp - request_timestamp

            # Extract response
            response_text = response.choices[0].message.content

            # Log full response
            logger.info("-" * 80)
            logger.info("OpenAI API Response:")
            logger.info(f"Full Response ({len(response_text)} chars):")
            logger.info(response_text)
            logger.info("-" * 80)

            # Log token usage
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Token Usage: prompt_tokens={response.usage.prompt_tokens}, "
                           f"completion_tokens={response.usage.completion_tokens}, "
                           f"total_tokens={response.usage.total_tokens}")
                
                # Calculate estimated cost (approximate, varies by model)
                # GPT-4 Turbo: ~$0.01/1K input tokens, ~$0.03/1K output tokens
                if "gpt-4" in used_model.lower():
                    estimated_cost = (response.usage.prompt_tokens / 1000 * 0.01) + (response.usage.completion_tokens / 1000 * 0.03)
                    logger.info(f"Estimated Cost: ${estimated_cost:.4f}")

            elapsed = time.time() - start_time
            logger.info(f"Request completed: response_length={len(response_text)}, api_latency={api_latency:.2f}s, total_time={elapsed:.2f}s")
            logger.info("=" * 80)

            return response_text

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("-" * 80)
            logger.error(f"OpenAI API Error: error={str(e)}, model={used_model}, time={elapsed:.2f}s")
            logger.error(f"Request details: num_messages={len(api_messages)}, has_context={context is not None}")
            if messages:
                last_message = messages[-1]
                user_query = last_message.get('content', '')
                query_length = len(user_query)
                logger.error(f"User query that failed ({query_length} chars):")
                logger.error(user_query)
            if context:
                logger.error(f"Context that was used ({len(context)} chars):")
                logger.error(context)
            logger.error("-" * 80)
            logger.error("Full error traceback:", exc_info=True)
            logger.error("=" * 80)
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
        start_time = time.time()
        used_model = model or self.model
        used_temp = temperature or self.temperature

        # Log request details
        logger.info("=" * 80)
        logger.info(f"OpenAI API Streaming Request - model={used_model}, temperature={used_temp}, num_messages={len(messages)}, has_context={context is not None}")
        logger.info("-" * 80)

        # Build the messages list (same as generate_response)
        api_messages = []

        if context:
            context_length = len(context)
            logger.debug(f"Adding context to system message: context_length={context_length} chars")
            system_message = {
                "role": "system",
                "content": f"""You are a RAG (Retrieval-Augmented Generation) assistant. Your role is to answer questions STRICTLY based on the provided context from the knowledge base.

CRITICAL RULES:
1. ONLY use information from the provided context below to answer questions
2. DO NOT use any external knowledge or general information
3. If the answer cannot be found in the context, explicitly state: "I cannot answer this question based on the provided documents. The information is not available in the knowledge base."
4. If the context is partially relevant, only answer the parts that are covered by the context
5. Cite which source/document the information comes from when possible

Context from Knowledge Base:
{context}

Remember: You are a RAG system. Your answers must be grounded in the provided context only.""",
            }
            api_messages.append(system_message)
            logger.debug("System message with context added:")
            logger.debug(f"Full Context ({context_length} chars):")
            logger.debug(context)
        else:
            # No context available - check if this is entity extraction
            is_entity_extraction = (
                len(messages) > 0 and 
                isinstance(messages[0], dict) and
                ("Analyze the following text and extract" in messages[0].get("content", "") or
                 "Extract key entities" in messages[0].get("content", ""))
            )
            
            if is_entity_extraction:
                # For entity extraction, don't add RAG restrictions
                logger.debug("Entity extraction detected in stream - skipping RAG restrictions")
            else:
                # For regular RAG queries without context, enforce restrictions
                api_messages.append(
                    {
                        "role": "system",
                        "content": """You are a RAG (Retrieval-Augmented Generation) assistant. Your role is to answer questions based on documents in the knowledge base. IMPORTANT: No relevant documents were found in the knowledge base for this query.

You MUST respond with: "I cannot answer this question because no relevant information was found in the knowledge base. Please try rephrasing your question or ensure that relevant documents have been uploaded to the system."

DO NOT attempt to answer the question using general knowledge. You are a RAG system and can only answer based on the knowledge base.""",
                    }
                )
                logger.debug("System message added (no context - RAG restriction enforced)")

        api_messages.extend(messages)

        # Log full request structure
        logger.info("Request Messages:")
        for i, msg in enumerate(api_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_length = len(content)
            logger.info(f"  [{i+1}] {role.upper()} ({content_length} chars):")
            logger.info(content)
            logger.info("")  # Empty line for readability
        
        # Log user query separately
        if messages:
            last_message = messages[-1]
            user_query = last_message.get('content', '')
            query_length = len(user_query)
            logger.info(f"User Query ({query_length} chars):")
            logger.info(user_query)

        # Track streaming response
        full_response = ""
        chunk_count = 0
        first_chunk_time = None

        try:
            # Generate streaming response
            logger.debug(f"Starting streaming request to OpenAI API: total_messages={len(api_messages)}")
            request_timestamp = time.time()
            stream = self.client.chat.completions.create(
                model=used_model,
                messages=api_messages,
                temperature=used_temp,
                stream=True,
            )

            logger.info("-" * 80)
            logger.info("OpenAI API Streaming Response:")
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    chunk_count += 1
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        time_to_first_token = first_chunk_time - request_timestamp
                        logger.info(f"First token received: time_to_first_token={time_to_first_token:.2f}s")
                    
                    # Log every 10th chunk or if it's a significant chunk
                    if chunk_count % 10 == 0 or len(chunk_content) > 100:
                        logger.debug(f"Chunk {chunk_count}: length={len(chunk_content)}, total_length={len(full_response)}")
                    
                    yield chunk_content

            # Log completion details
            elapsed = time.time() - start_time
            logger.info(f"Streaming completed: total_chunks={chunk_count}, response_length={len(full_response)}, total_time={elapsed:.2f}s")
            
            # Log full response
            if full_response:
                logger.info(f"Full Response ({len(full_response)} chars):")
                logger.info(full_response)
            logger.info("=" * 80)
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("-" * 80)
            logger.error(f"OpenAI API Streaming Error: error={str(e)}, model={used_model}, time={elapsed:.2f}s")
            logger.error(f"Request details: num_messages={len(api_messages)}, has_context={context is not None}")
            logger.error(f"Streaming progress: chunks_received={chunk_count}, partial_response_length={len(full_response)}")
            if messages:
                last_message = messages[-1]
                user_query = last_message.get('content', '')
                query_length = len(user_query)
                logger.error(f"User query that failed ({query_length} chars):")
                logger.error(user_query)
            if context:
                logger.error(f"Context that was used ({len(context)} chars):")
                logger.error(context)
            if full_response:
                logger.error(f"Partial response received ({len(full_response)} chars):")
                logger.error(full_response)
            logger.error("-" * 80)
            logger.error("Full error traceback:", exc_info=True)
            logger.error("=" * 80)
            raise

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
