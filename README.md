# Life Insurance Support Assistant

AI-powered life insurance assistant using LangGraph, RAG, and OpenAI. Provides intelligent responses about policies, coverage, premiums, eligibility, and claims through CLI and REST API.

## Overview

This system uses a multi-stage LangGraph agent with Retrieval-Augmented Generation to answer life insurance queries with domain-specific knowledge. The agent classifies intent, retrieves relevant context from a vector store, executes specialized tools when needed, and generates accurate responses maintaining conversational context.

## Architecture

**Agent Pipeline** (LangGraph StateGraph):
1. Intent Classification - Categorize query type (policy, eligibility, premiums, claims, coverage, general)
2. Knowledge Retrieval - RAG search via ChromaDB vector store
3. Tool Execution - Conditional routing to specialized tools
4. Response Generation - Context-aware LLM generation with conversation history

**Specialized Tools**:
- Premium Calculator - AI-powered estimation using rating criteria from knowledge base
- Eligibility Checker - Underwriting analysis with health/age/occupation factors
- Policy Comparator - Multi-policy comparison from vector search

**Tech Stack**:
- LangGraph + LangChain for agent orchestration
- OpenAI GPT-4o-mini for reasoning
- ChromaDB with text-embedding-3-small for vector search
- FastAPI for REST API
- SQLAlchemy with SQLite/PostgreSQL for session persistence
- Rich for CLI interface

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key

### Local Setup

```bash
# Clone and setup
git clone git@github.com:FardinHash/lisa.git
cd lisa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Initialize knowledge base
python scripts/init_knowledge_base.py

# Verify installation
pytest
```

### Docker Setup

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

docker-compose up -d
```

API available at: `http://localhost:8000`
Documentation: `http://localhost:8000/docs`

## Usage

### CLI Interface

Start interactive chat:
```bash
python cli/chat.py
```

Available commands:
- `help` - Display command reference
- `clear` - Reset conversation history
- `history` - Show full session transcript
- `new` - Start new session
- `quit` / `exit` - Exit application

Example queries:
```
What types of life insurance are available?
Calculate premium for 35 year old, $500k coverage, 20 year term
Can I qualify for insurance if I have diabetes?
Compare term life and whole life insurance
What documents do I need to file a claim?
```

### REST API

Start API server:
```bash
uvicorn app.main:app --reload
```

**Create Session**
```bash
curl -X POST http://localhost:8000/api/v1/chat/session \
  -H "Content-Type: application/json" \
  -d '{"user_id": "optional_user_id"}'
```

Response:
```json
{
  "session_id": "uuid-string",
  "created_at": "2025-01-01T00:00:00",
  "message_count": 0
}
```

**Send Message**
```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid-from-above",
    "message": "What is term life insurance?"
  }'
```

Response:
```json
{
  "session_id": "uuid-string",
  "message": "Term life insurance is...",
  "sources": ["policy_types.txt"],
  "agent_reasoning": "Intent: POLICY_TYPES | Tools Used: None",
  "timestamp": "2025-01-01T00:00:00"
}
```

**Get Session History**
```bash
curl http://localhost:8000/api/v1/chat/session/{session_id}
```

**Delete Session**
```bash
curl -X DELETE http://localhost:8000/api/v1/chat/session/{session_id}
```

**Health Check**
```bash
curl http://localhost:8000/health
```

## Configuration

All settings via environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `ENVIRONMENT` | Deployment environment | `local` |
| `LLM_MODEL` | OpenAI model | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Response creativity | `0.7` |
| `LLM_MAX_TOKENS` | Max response length | `800` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_URL` | Session database | `sqlite:///./data/conversations.db` |
| `CHROMA_PERSIST_DIR` | Vector store path | `./data/chroma_db` |
| `KNOWLEDGE_BASE_DIR` | Source documents | `./knowledge_base` |
| `RAG_CHUNK_SIZE` | Document chunk size | `1000` |
| `RAG_CHUNK_OVERLAP` | Chunk overlap | `200` |
| `RAG_SEARCH_K` | Retrieved documents | `3` |
| `MEMORY_MAX_HISTORY` | Max messages stored | `10` |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

For production:
- Set `ENVIRONMENT=production`
- Use PostgreSQL: `DATABASE_URL=postgresql://user:pass@host:5432/db`
- Reduce logging: `LOG_LEVEL=WARNING`

## Project Structure

```
lisa/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Pydantic settings
│   ├── models.py                # Request/response models
│   ├── database.py              # SQLAlchemy ORM
│   ├── api/
│   │   └── chat.py              # REST endpoints
│   ├── agents/
│   │   ├── graph.py             # LangGraph agent workflow
│   │   ├── tools.py             # Specialized agent tools
│   │   └── prompts.py           # Prompt templates
│   └── services/
│       ├── llm.py               # OpenAI client wrapper
│       ├── rag.py               # ChromaDB vector search
│       └── memory.py            # Session management
├── cli/
│   └── chat.py                  # CLI interface
├── knowledge_base/              # Life insurance documents
│   ├── policy_types.txt
│   ├── eligibility_underwriting.txt
│   ├── claims_process.txt
│   ├── risk_assessment_criteria.txt
│   └── faq.txt
├── scripts/
│   └── init_knowledge_base.py   # Vector store initialization
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Testing

Run test suite:
```bash
# All tests
pytest

# With coverage report
pytest --cov=app --cov-report=html

# Specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only

# Single test file
pytest tests/unit/test_rag_service.py

# View coverage
open htmlcov/index.html
```

Code formatting:
```bash
black app/ tests/ cli/
isort app/ tests/ cli/
```

Current coverage: 85%+

## Agent Workflow Details

**State Machine Flow**:
```
User Query
    ↓
[Analyze Intent]
    ↓
[Retrieve Information] → RAG Search (ChromaDB)
    ↓
Should Use Tools? ← Intent + Keywords
    ├─ Yes → [Use Tools] → Premium Calc / Eligibility Check / Policy Compare
    └─ No  →
    ↓
[Generate Answer] ← Context + Tool Results + History
    ↓
Response + Sources + Reasoning
```

**Intent Categories**:
- `POLICY_TYPES` - Policy types, features, comparisons
- `ELIGIBILITY` - Qualification criteria, underwriting
- `PREMIUMS` - Cost calculations, pricing factors
- `CLAIMS` - Filing process, beneficiaries, documentation
- `COVERAGE` - Coverage amounts, limitations, riders
- `GENERAL` - General inquiries, greetings

**Tool Activation Logic**:
- Premium calculator: Keywords like "calculate", "estimate", "cost", "how much" OR intent `PREMIUMS`
- Eligibility checker: Keywords like "eligible", "qualify", "approved" OR intent `ELIGIBILITY`
- Policy comparator: Keywords like "compare", "versus", "difference"

## Knowledge Base

The system includes 5 curated documents covering:
- Life insurance policy types (term, whole, universal, variable)
- Eligibility and underwriting criteria
- Claims filing process and documentation
- Risk assessment and rating factors
- Frequently asked questions

To update knowledge base:
1. Add/modify `.txt` files in `knowledge_base/`
2. Run: `python scripts/init_knowledge_base.py`
3. Restart application

## Deployment

### Production Checklist
- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure PostgreSQL database URL
- [ ] Set secure `DATABASE_URL` with strong credentials
- [ ] Use production OpenAI API key with rate limits
- [ ] Configure logging level (`LOG_LEVEL=WARNING`)
- [ ] Enable HTTPS/TLS for API
- [ ] Set up monitoring and health checks
- [ ] Configure backup for database and vector store
- [ ] Implement rate limiting on API endpoints
- [ ] Review and adjust resource limits in `docker-compose.yml`

### Docker Production Deploy

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Health check
curl http://localhost:8000/health

# Stop services
docker-compose down
```

### Database Migration

SQLite (local) to PostgreSQL (production):
```bash
# Export sessions from SQLite
sqlite3 data/conversations.db .dump > backup.sql

# Update .env with PostgreSQL URL
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Application auto-creates tables on startup
```

## Troubleshooting

**Vector store not initialized**
```bash
python scripts/init_knowledge_base.py
```

**OpenAI API errors**
- Verify `OPENAI_API_KEY` in `.env`
- Check API key permissions and quota
- Confirm network connectivity

**Import/dependency errors**
```bash
pip install --upgrade -r requirements.txt
```

**Database locked (SQLite)**
- Ensure only one process accessing database
- Consider switching to PostgreSQL for production

**Poor response quality**
- Adjust `LLM_TEMPERATURE` (0.3-0.9)
- Increase `RAG_SEARCH_K` for more context
- Review and improve knowledge base documents

**Memory leaks in long sessions**
- Sessions auto-limit to `MEMORY_MAX_HISTORY` messages
- Use `/api/v1/chat/session/{id}` DELETE endpoint to clear old sessions

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs with `LOG_LEVEL=DEBUG`
3. Create GitHub issue with reproduction steps
