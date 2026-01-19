"""
Entity Extraction Service - Extracts entities and relationships from text.

This service uses LLM to dynamically extract entities and their relationships
from document content, building an ontology on the fly.
"""
from typing import List, Dict, Optional
import json
import re
from src.infrastructure.llm.openai_service import OpenAIService
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('entity_extraction')


class EntityExtractionService:
    """
    Service for extracting entities and relationships from text using LLM.
    
    This service dynamically builds an ontology by extracting:
    - Entities (Person, Organization, Concept, Location, etc.)
    - Relationships between entities
    - Entity mentions in chunks
    """

    def __init__(self, llm_service: OpenAIService):
        """
        Initialize the entity extraction service.
        
        Args:
            llm_service: LLM service for entity extraction
        """
        self.llm_service = llm_service

    def extract_entities_and_relationships(
        self,
        text: str,
        chunk_id: Optional[str] = None,
    ) -> Dict:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text content to extract from
            chunk_id: Optional chunk ID for logging
            
        Returns:
            Dictionary with:
            - entities: List of entity dictionaries
            - relationships: List of relationship dictionaries
        """
        logger.info(f"Extracting entities from text: chunk_id={chunk_id}, text_length={len(text)}")
        
        prompt = f"""Analyze the following text and extract all entities and their relationships.

Text:
{text[:3000]}  # Limit to avoid token limits

Extract:
1. Entities: People, Organizations, Concepts, Locations, Events, Products, Technologies, etc.
2. Relationships: How entities relate to each other

Return a JSON object with this structure:
{{
  "entities": [
    {{
      "id": "unique_id",
      "name": "Entity Name",
      "type": "Person|Organization|Concept|Location|Event|Product|Technology|Other",
      "properties": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source_entity_name": "Entity Name 1",
      "target_entity_name": "Entity Name 2",
      "type": "WORKS_FOR|LOCATED_IN|RELATED_TO|PART_OF|CREATED_BY|etc",
      "description": "Brief description"
    }}
  ]
}}

Focus on:
- Named entities (people, organizations, places)
- Key concepts and topics
- Important relationships between entities
- Be specific with entity names (use full names when possible)

Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo-preview",
                temperature=0,
            )
            
            # Parse JSON response
            # Sometimes LLM adds markdown code blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            
            # Validate structure
            if "entities" not in result:
                result["entities"] = []
            if "relationships" not in result:
                result["relationships"] = []
            
            logger.info(
                f"Extracted {len(result['entities'])} entities and "
                f"{len(result['relationships'])} relationships: chunk_id={chunk_id}"
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}, response={response[:500]}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return {"entities": [], "relationships": []}

    def extract_entities_from_query(self, query: str) -> List[Dict]:
        """
        Extract entities from a user query to help with graph traversal.
        
        Args:
            query: User query text
            
        Returns:
            List of entity dictionaries
        """
        logger.debug(f"Extracting entities from query: query_length={len(query)}")
        
        prompt = f"""Extract key entities (people, organizations, concepts, locations, etc.) from this query:

Query: {query}

Return a JSON array of entities:
[
  {{
    "name": "Entity Name",
    "type": "Person|Organization|Concept|Location|Event|Product|Technology|Other"
  }}
]

Return ONLY valid JSON array, no additional text."""

        try:
            response = self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo-preview",
                temperature=0,
            )
            
            # Parse JSON response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            entities = json.loads(response)
            
            if not isinstance(entities, list):
                entities = []
            
            logger.debug(f"Extracted {len(entities)} entities from query")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities from query: {str(e)}")
            return []

    def generate_entity_id(self, name: str, entity_type: str) -> str:
        """
        Generate a unique entity ID from name and type.
        
        Args:
            name: Entity name
            entity_type: Entity type
            
        Returns:
            Unique entity ID
        """
        # Normalize name: lowercase, remove special chars, replace spaces with underscores
        normalized_name = re.sub(r'[^a-z0-9\s]', '', name.lower())
        normalized_name = re.sub(r'\s+', '_', normalized_name.strip())
        
        return f"{entity_type.lower()}_{normalized_name}"
