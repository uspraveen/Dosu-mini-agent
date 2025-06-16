## 🏗️  Mini-Dosu Architecture (v0.1)

This project glues together four key loops:

1. **Identity & Duplicate Detection** – unify every human + de-dupe repeat tickets.  
2. **Event Ingestion** – GitHub → webhook → SQS → worker pool (bursty but never drops).  
3. **Fact & Graph RAG** – persist verified triples in Aura-Neo4j; answer from graph first.  
4. **Agentic Web Search** – when graph misses, spawn a targeted web/Stack Overflow crawl.

### ✨  Big picture (Mermaid)

![image](https://github.com/user-attachments/assets/4fd93164-012f-4dbc-bd9d-528b38ee44d2)

2. Component Glossary
Layer	Role	Speed Tricks
GitHub App (dosubot)	Receives Issue / PR events.	JWT → install token cached 55 min.
API Gateway + Lambda	Verifies HMAC, drops JSON into SQS, responds <50 ms.	No DB I/O on this path.
SQS	Buffers bursts; guarantees at-least-once.	FIFO queue optional for strict order.
EC2 Worker Pool	Resolves user, de-dupes, labels, writes facts.	AMI with weights; autoscale on queue depth.
Supabase Postgres	Tables: users, aliases, tickets.	psycopg2 connection pool.
Aura Neo4j	Stores (Subject)-[:REL]->(Object) + source_url.	FULLTEXT index on subject, object.
Long-CTX LLM	Jamba 1.5 Large, 256 K tokens; invoked only on gap path.	vLLM + Flash-Attn2: ~7 ms/token.
Agentic Web Search	DuckDuckGo/StackOverflow scrapers; extracts new facts.	Detached asyncio.Task, 3 s budget.

3. Data Flow in Six Steps

i. Webhook → API GW → Lambda (<50 ms).

ii. Lambda pushes payload to SQS.

iii. EC2 worker pops message, /resolve → unified user_id.

Creates or flags duplicate ticket; writes facts to Neo4j.

If fact graph misses info ► calls LLM; LLM may trigger Web Search.

Worker posts comment / label back to GitHub.
