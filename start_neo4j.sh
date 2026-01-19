#!/bin/bash

# Script to start Neo4j for Graph RAG

echo "Starting Neo4j for Graph RAG..."
docker-compose up -d

echo ""
echo "Neo4j should now be running!"
echo ""
echo "Access Neo4j Browser at: http://localhost:7474"
echo "Default credentials: neo4j / password"
echo ""
echo "To check status: docker-compose ps"
echo "To view logs: docker-compose logs -f neo4j"
echo "To stop: docker-compose down"
