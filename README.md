# BizAgents – Multi-Agent Business Automation

> **Current status: RESEARCH AGENT PHASE.** The text-based Intake Agent,
> persistent ChromaDB business memory, and source-aware Research Agent are
> implemented. Other agents and product functionality have not been
> implemented.

## What BizAgents is

BizAgents is a planned multi-agent business automation system. A coordinated
team of AI agents is intended to help users move from a business request to
research, analysis, outreach, reporting, and follow-up.

## Problem it aims to solve

Business workflows often require people to repeat and coordinate research,
lead analysis, outreach preparation, and follow-up work across separate tools.
BizAgents is intended to organize those steps into a traceable workflow with
specialized AI agents.

## Target users

The planned audience is businesses and business teams that need support with
research and outreach workflows, including sales, growth, and operations
teams.

## Planned AI agent team

The plan names the following agents and their intended responsibilities:

- **Intake Agent** — understand and structure a user's business request
  (text-only implementation completed in Phase 2).
- **Manager Agent** — coordinate the workflow and delegate tasks.
- **Research Agent** — gather relevant company and market information.
- **Analysis Agent** — synthesize research into useful findings.
- **Outreach Agent** — prepare business outreach.
- **Report Agent** — assemble results into a clear report.
- **Follow-up Agent** — manage planned follow-up steps.

## High-level planned workflow

1. Receive and structure a business request.
2. Coordinate research and analysis across the agent team.
3. Prepare relevant outreach and follow-up material.
4. Present the results in a report.

This is a high-level description only. The text-based Intake Agent is
implemented; the other planned agent workflows are not yet implemented.

## Planned technology stack

- **LLM:** Claude API
- **Agent framework:** LangGraph or CrewAI
- **Backend:** Python and FastAPI
- **Frontend:** React or Next.js
- **Database:** PostgreSQL
- **Retrieval-augmented generation:** ChromaDB initially
- **Speech-to-text:** Whisper or Deepgram
- **Text-to-speech:** ElevenLabs or Google TTS
- **Search:** Tavily or Serper
- **Company data:** Apollo, Crunchbase, or a similar provider
- **Email:** Gmail API or SendGrid
- **Monitoring and evaluation:** Arize Phoenix / Arize AX
- **Development tools:** VS Code Agent, Git, and GitHub

The scaffold intentionally does not install every planned integration. Those
dependencies should be selected and introduced when their implementation
phase begins.

## Project structure

```text
bizagents/
├── agents/          # Text-based Intake and provider-driven Research Agents
├── rag/             # Persistent ChromaDB business profile memory
├── voice/           # Planned speech components; currently empty
├── api/             # Planned backend API; currently empty
├── frontend/        # Planned frontend; currently empty
├── evals/           # Planned evaluation and monitoring; currently empty
├── tests/           # Intake Agent, RAG memory, and Research Agent tests
├── data/chroma/     # Local ChromaDB storage (ignored by Git)
├── .env.example     # Placeholder names for future service configuration
├── requirements.txt # Python foundation and ChromaDB dependency
└── README.md
```

## Development phases

**Phase 1 — Project Foundation:** establish the repository structure, minimal
Python dependencies, environment-variable template, and project documentation.

**Phase 2 — Intake Agent:** collect and confirm a business owner's offering,
target customers, location, budget, and goals through a text conversation.

**Phase 3 — RAG Memory:** persist confirmed business profiles in
ChromaDB and retrieve relevant profile fields semantically. Local persistent
storage is used at `data/chroma/`; generated vector database files are ignored
by Git. Future agents will consume this reusable memory interface. The
Analysis, Report, and Manager agents are not implemented. Queries without a
sufficiently relevant semantic match return no results.

```python
from rag import BusinessMemory, BusinessProfile

# Pass the completed, owner-confirmed IntakeSession.
profile = BusinessProfile.from_intake_session(
    completed_session,
    business_profile_id="development-business",
)
memory = BusinessMemory()  # Persists locally in data/chroma/
try:
    memory.save_business_profile(profile)
    matches = memory.retrieve(
        "Who are the target customers?",
        business_profile_id=profile.business_profile_id,
    )
finally:
    memory.close()
```

**Phase 4 — Research Agent (current):** accept a validated `BusinessProfile`,
derive deterministic business, market, competitor, and lead research queries,
and execute them through an injected `ResearchProvider` protocol. Results are
structured and source-aware: findings reference normalized source records,
and missing or unusable source URLs are reported rather than fabricated.
Provider failures and malformed results are represented in the result status
and issues. Unit tests use a deterministic fake provider and make no network
calls. No vendor search or company-data integrations are implemented.

The uploaded **BizAgents Complete Project Plan** is the authoritative Master
Plan. Development must follow its defined architecture and sequence:

1. Project Foundation
2. Intake Agent
3. RAG Memory
4. Research Agent (implemented)
5. Analysis Agent
6. Report Agent
7. Manager Agent
8. Arize Tracing
9. Arize Evaluations
10. Frontend
11. Voice and multilingual support
12. Outreach Agent with human approval
13. Follow-up Agent
14. Login, database, and per-user business memory
15. Testing, fixing, and deployment

Future implementation must not silently change this architecture or sequence.

## Security principles

- Keep API keys, OAuth credentials, and database passwords out of source
  control.
- Use `.env.example` only to document variable names; it contains placeholders,
  not usable credentials.
- Store real credentials only in an appropriately protected local environment
  or secret manager.
- Apply least-privilege access to service credentials and avoid logging
  secrets.
- Review external-data handling, consent, and access controls as integrations
  are implemented.

## Foundation dependencies

`requirements.txt` includes FastAPI, its ASGI server, settings/environment
support, and ChromaDB for the implemented local RAG memory layer. Agent
frameworks, LLM clients, database drivers, voice, search, company-data, email,
and observability packages are deferred until their planned implementation
phases. ChromaDB's built-in ONNX-backed `all-MiniLM-L6-v2` embedding function
runs locally after its model files are downloaded on first use; it does not
require a paid embedding API or API key.
