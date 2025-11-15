from unittest.mock import MagicMock, patch

import pytest

from app.services.rag import RAGService


class TestRAGService:
    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_initialize_vectorstore_existing(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        service = RAGService()

        assert service.vectorstore is not None
        mock_exists.assert_called()
        mock_chroma.assert_called_once()

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_initialize_vectorstore_missing(self, mock_exists, mock_chroma):
        mock_exists.return_value = False

        service = RAGService()

        mock_exists.assert_called()

    @patch("app.services.rag.Chroma.from_documents")
    @patch("app.services.rag.DirectoryLoader")
    @patch("app.services.rag.os.makedirs")
    def test_load_and_index_documents(
        self, mock_makedirs, mock_loader, mock_from_documents
    ):
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance

        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore

        service = RAGService()
        result = service.load_and_index_documents()

        assert isinstance(result, int)
        assert result > 0
        mock_makedirs.assert_called_once()
        mock_loader_instance.load.assert_called_once()

    def test_search_no_vectorstore(self):
        service = RAGService()
        service.vectorstore = None

        results = service.search("test query", k=3)

        assert results == []

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_search_with_results(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_doc = MagicMock()
        mock_doc.page_content = "Test insurance content"
        mock_doc.metadata = {"source": "test.txt"}

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.9)
        ]
        mock_chroma.return_value = mock_vectorstore

        service = RAGService()
        results = service.search("test query", k=3)

        assert len(results) == 1
        assert results[0]["content"] == "Test insurance content"
        assert results[0]["source"] == "test.txt"
        assert results[0]["score"] == 0.9

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_search_with_score_threshold(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "High relevance content"
        mock_doc1.metadata = {"source": "test1.txt"}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Low relevance content"
        mock_doc2.metadata = {"source": "test2.txt"}

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (mock_doc1, 0.9),
            (mock_doc2, 0.3),
        ]
        mock_chroma.return_value = mock_vectorstore

        service = RAGService()
        results = service.search("test query", k=3, score_threshold=0.5)

        assert len(results) == 1
        assert results[0]["score"] == 0.9

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_get_relevant_context(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_doc = MagicMock()
        mock_doc.page_content = "Test insurance content"
        mock_doc.metadata = {"source": "/path/to/test.txt"}

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.9)
        ]
        mock_chroma.return_value = mock_vectorstore

        service = RAGService()
        context = service.get_relevant_context("test query", k=2)

        assert "Test insurance content" in context
        assert "test.txt" in context
        assert "[Source 1:" in context

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_get_relevant_context_no_results(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = []
        mock_chroma.return_value = mock_vectorstore

        service = RAGService()
        context = service.get_relevant_context("test query", k=2)

        assert "No relevant information found" in context

    @patch("app.services.rag.Chroma")
    @patch("app.services.rag.os.path.exists")
    def test_search_with_metadata_filter(self, mock_exists, mock_chroma):
        mock_exists.return_value = True
        mock_doc = MagicMock()
        mock_doc.page_content = "Filtered content"
        mock_doc.metadata = {"source": "claims.txt", "type": "claims"}

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        mock_chroma.return_value = mock_vectorstore

        service = RAGService()
        results = service.search_with_metadata_filter(
            "test query", metadata_filter={"type": "claims"}, k=3
        )

        assert len(results) == 1
        assert results[0]["content"] == "Filtered content"
        assert results[0]["metadata"]["type"] == "claims"
