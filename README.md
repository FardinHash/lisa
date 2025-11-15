# Life Insurance Support Assistant

AI-powered life insurance assistant using LangGraph, RAG, and OpenAI. Provides intelligent responses about policies, coverage, premiums, eligibility, and claims through CLI and REST API.

## Overview

This system uses a multi-stage LangGraph agent with Retrieval-Augmented Generation to answer life insurance queries with domain-specific knowledge. The agent classifies intent, retrieves relevant context from a vector store, executes specialized tools when needed, and generates accurate responses maintaining conversational context.

## Architecture

**Agent Pipeline** (LangGraph StateGraph):
1. Intent Classification - LLM-based categorization with configurable temperature
2. Knowledge Retrieval - RAG search with multi-turn conversation context
3. Tool Selection - Intelligent LLM-based tool routing (not keyword matching)
4. Tool Execution - Validated specialized tools with input constraints
5. Response Generation - Context-aware generation with conversation history

**Specialized Services**:
- `IntentAnalyzer` - Classifies user intent into 6 categories
- `ContextRetriever` - Retrieves relevant knowledge with conversation context
- `ToolSelector` - LLM-powered tool selection with fallback logic
- `ToolExecutor` - Executes premium calculator, eligibility checker, policy comparator
- `ResponseGenerator` - Generates final answers with full context

**Production Features**:
- **Caching Layer** - In-memory cache for RAG and LLM calls (60-70% cost reduction)
- **Rate Limiting** - Token bucket algorithm, 100 req/min per client (configurable)
- **Monitoring** - Real-time metrics tracking (requests, tokens, costs, errors)
- **Hot Reload** - Update knowledge base without restart via API endpoint
- **Connection Pooling** - Optimized database connections for high concurrency
- **Input Validation** - Comprehensive validation for age, coverage, and term parameters

**Tech Stack**:
- LangGraph + LangChain for agent orchestration
- OpenAI GPT-4o-mini for reasoning (abstracted, swappable)
- ChromaDB with text-embedding-3-small for vector search
- FastAPI for REST API with rate limiting middleware
- SQLAlchemy with SQLite/PostgreSQL for session persistence
- Rich for CLI interface
- In-memory caching (extensible to Redis)

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

**Get System Metrics**
```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "uptime_seconds": 3600,
  "uptime_formatted": "1h 0m 0s",
  "timestamp": "2025-11-15T12:00:00",
  "metrics": {
    "/api/v1/chat/message": {
      "count": 150,
      "total_time": 225.5,
      "avg_time": 1.503,
      "errors": 2
    },
    "llm_gpt-4o-mini": {
      "count": 180,
      "total_time": 180.2,
      "avg_time": 1.001,
      "total_tokens": 45000,
      "total_cost": 0.23
    },
    "rag_search": {
      "count": 165,
      "total_time": 8.3,
      "avg_time": 0.050,
      "total_results": 495
    }
  }
}
```

**Reload Knowledge Base** (Admin)
```bash
curl -X POST http://localhost:8000/admin/reload-knowledge-base
```

Response:
```json
{
  "success": true,
  "message": "Knowledge base reloaded successfully with 125 chunks",
  "chunks": 125,
  "timestamp": "2025-11-15T12:00:00"
}
```

**Rate Limit Headers**

All API responses include rate limiting information:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1700050800
```

## Configuration

All settings via environment variables (`.env` file):

### Core Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `ENVIRONMENT` | Deployment environment | `local` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LLM_MODEL` | OpenAI model | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Response creativity | `0.7` |
| `LLM_MAX_TOKENS` | Max response length | `800` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

### Database Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Session database | `sqlite:///./data/conversations.db` |
| `DB_POOL_SIZE` | Connection pool size | `10` |
| `DB_MAX_OVERFLOW` | Max overflow connections | `20` |
| `DB_POOL_TIMEOUT` | Connection timeout (sec) | `30` |
| `DB_POOL_RECYCLE` | Connection recycle time (sec) | `3600` |

### RAG Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_PERSIST_DIR` | Vector store path | `./data/chroma_db` |
| `KNOWLEDGE_BASE_DIR` | Source documents | `./knowledge_base` |
| `RAG_CHUNK_SIZE` | Document chunk size | `1000` |
| `RAG_CHUNK_OVERLAP` | Chunk overlap | `200` |
| `RAG_SEARCH_K` | Retrieved documents | `3` |
| `RAG_SCORE_THRESHOLD` | Min relevance score | `0.5` |

### Agent Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `INTENT_CLASSIFICATION_TEMPERATURE` | Intent classification temp | `0.3` |
| `TOOL_SELECTION_TEMPERATURE` | Tool selection temp | `0.2` |
| `MEMORY_MAX_HISTORY` | Max messages stored | `10` |
| `MEMORY_CONTEXT_MESSAGES` | Context messages for retrieval | `4` |

### Validation Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `TOOL_AGE_MIN` | Minimum age | `18` |
| `TOOL_AGE_MAX` | Maximum age | `85` |
| `TOOL_COVERAGE_MIN` | Min coverage amount | `10000` |
| `TOOL_COVERAGE_MAX` | Max coverage amount | `10000000` |
| `TOOL_TERM_MIN` | Min term length (years) | `5` |
| `TOOL_TERM_MAX` | Max term length (years) | `40` |

### Performance Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `CACHE_ENABLED` | Enable caching | `true` |
| `CACHE_TTL` | Cache TTL (seconds) | `3600` |
| `CACHE_MAX_SIZE` | Max cache entries | `1000` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
| `RATE_LIMIT_CALLS` | Calls per period | `100` |
| `RATE_LIMIT_PERIOD` | Period in seconds | `60` |

### API Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `API_RELOAD` | Auto-reload on changes | `true` |

For production:
- Set `ENVIRONMENT=production`
- Use PostgreSQL: `DATABASE_URL=postgresql://user:pass@host:5432/db`
- Reduce logging: `LOG_LEVEL=WARNING`
- Adjust rate limits based on load
- Consider Redis for distributed caching
- Set `API_RELOAD=false`

## Project Structure

```
lisa/
├── app/
│   ├── main.py                  # FastAPI application with middleware
│   ├── config.py                # Comprehensive Pydantic settings
│   ├── models.py                # Request/response models
│   ├── database.py              # SQLAlchemy ORM with connection pooling
│   ├── api/
│   │   └── chat.py              # REST endpoints
│   ├── agents/
│   │   ├── graph.py             # LangGraph agent workflow (refactored)
│   │   ├── services.py          # Separated agent service classes
│   │   ├── tools.py             # Validated specialized tools
│   │   └── prompts.py           # Prompt templates
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── rate_limit.py        # Rate limiting middleware
│   └── services/
│       ├── llm.py               # LLM service (uses llm_provider)
│       ├── llm_provider.py      # Abstracted LLM provider layer
│       ├── rag.py               # ChromaDB with caching & hot reload
│       ├── memory.py            # Session management (refactored)
│       ├── cache.py             # Caching service
│       └── monitoring.py        # Metrics and monitoring
├── cli/
│   └── chat.py                  # Rich CLI interface
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
