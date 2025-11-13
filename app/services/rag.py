import logging
import os
import warnings
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.llm import llm_service

warnings.filterwarnings("ignore", message=".*Relevance scores must be between.*")

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.embeddings = llm_service.get_embedding_model()
        self.vectorstore: Optional[Chroma] = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        try:
            if os.path.exists(settings.chroma_persist_dir):
                self.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_dir,
                    embedding_function=self.embeddings,
                )
                logger.info("Loaded existing vector store")
            else:
                logger.warning(
                    "Vector store not found. Run initialization script first."
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            self.vectorstore = None

    def load_and_index_documents(self) -> int:
        try:
            os.makedirs(settings.chroma_persist_dir, exist_ok=True)

            loader = DirectoryLoader(
                settings.knowledge_base_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document chunks")

            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )

            logger.info("Vector store created and persisted successfully")
            return len(splits)

        except Exception as e:
            logger.error(f"Error loading and indexing documents: {str(e)}")
            raise

    def search(
        self, query: str, k: int = 4, score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k
            )

            results = []
            for doc, score in docs_with_scores:
                if score >= score_threshold:
                    results.append(
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "score": score,
                        }
                    )

            logger.info(f"Found {len(results)} relevant documents for query")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def search_with_metadata_filter(
        self, query: str, metadata_filter: Dict[str, Any], k: int = 4
    ) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []

        try:
            docs = self.vectorstore.similarity_search(
                query, k=k, filter=metadata_filter
            )

            results = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]

            return results

        except Exception as e:
            logger.error(f"Error searching with metadata filter: {str(e)}")
            return []

    def get_relevant_context(self, query: str, k: int = 4) -> str:
        results = self.search(query, k=k)

        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for idx, result in enumerate(results, 1):
            source = os.path.basename(result["source"])
            context_parts.append(f"[Source {idx}: {source}]\n{result['content']}\n")

        return "\n".join(context_parts)


rag_service = RAGService()
