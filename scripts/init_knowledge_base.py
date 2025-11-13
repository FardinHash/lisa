import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from dotenv import load_dotenv

from app.services.rag import rag_service

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    logger.info("Starting knowledge base initialization...")

    try:
        num_chunks = rag_service.load_and_index_documents()
        logger.info(f"Successfully indexed {num_chunks} document chunks")
        logger.info("Knowledge base initialization complete!")

        logger.info("\nTesting retrieval...")
        test_query = "What are the types of life insurance?"
        results = rag_service.search(test_query, k=2)

        logger.info(f"\nTest query: {test_query}")
        logger.info(f"Found {len(results)} relevant results")

        for idx, result in enumerate(results, 1):
            logger.info(f"\nResult {idx}:")
            logger.info(f"Source: {result['source']}")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Content preview: {result['content'][:200]}...")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
