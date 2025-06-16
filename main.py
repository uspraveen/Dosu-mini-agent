"""
Simple Dosu Identity Service - Pure psycopg2 (NO SQLAlchemy!)
Using YOUR working database connection approach
"""
import os, uuid, re, datetime as dt
from itertools import islice
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from datasketch import MinHash
from dotenv import load_dotenv
import numpy as np

# Optional imports (graceful degradation)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence_transformers not installed - semantic similarity disabled")
    print("   Install with: pip install sentence-transformers")

# Load environment variables (EXACTLY like your working code)
load_dotenv()

# Use YOUR working environment variables
USER = os.environ.get("DB_USER")
PASSWORD = os.getenv("DB_PASS")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")

# Database connection manager
@contextmanager
def get_db_connection():
    """Get database connection using YOUR working approach"""
    connection = None
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME,
            cursor_factory=RealDictCursor  # Returns dict-like rows
        )
        yield connection
    except Exception as e:
        if connection:
            connection.rollback()
        raise e
    finally:
        if connection:
            connection.close()

# Enhanced MinHash + Semantic Similarity
class EnhancedMatcher:
    """
    Advanced matching with multiple signals:
    1. MinHash for text similarity
    2. Optional semantic embeddings for meaning similarity
    """

    def __init__(self):
        # Load sentence transformer only if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic similarity enabled")
            except Exception as e:
                print(f"⚠️  Could not load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            print("📝 Running with MinHash similarity only")

    def ngrams(self, words: List[str], n=3):
        iters = [islice(words, i, None) for i in range(n)]
        return zip(*iters)

    def minhash(self, text: str, num_perm: int = 64) -> bytes:
        tokens = re.split(r"\W+", text.lower())
        m = MinHash(num_perm=num_perm)
        for sh in self.ngrams(tokens, 3):
            m.update(" ".join(sh).encode())
        return m.hashvalues.tobytes()

    def semantic_embedding(self, text: str) -> Optional[bytes]:
        """Generate semantic embedding (optional)"""
        if not self.embedding_model:
            return None

        try:
            embedding = self.embedding_model.encode([text])[0]
            return embedding.astype(np.float32).tobytes()
        except Exception as e:
            print(f"⚠️  Failed to generate embedding: {e}")
            return None

    def jaccard_similarity(self, a: bytes, b: bytes) -> float:
        from array import array
        arr_a = array("L")
        arr_b = array("L")
        arr_a.frombytes(a)
        arr_b.frombytes(b)
        same = sum(x == y for x, y in zip(arr_a, arr_b))
        return same / len(arr_a)

    def cosine_similarity(self, a: bytes, b: bytes) -> float:
        """Cosine similarity for semantic embeddings"""
        if not a or not b:
            return 0.0

        try:
            vec_a = np.frombuffer(a, dtype=np.float32)
            vec_b = np.frombuffer(b, dtype=np.float32)

            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception:
            return 0.0

    def combined_similarity(self, fingerprint_a: bytes, embedding_a: Optional[bytes],
                           fingerprint_b: bytes, embedding_b: Optional[bytes]) -> float:
        """Combine MinHash + semantic similarity"""

        text_sim = self.jaccard_similarity(fingerprint_a, fingerprint_b)

        semantic_sim = 0.0
        if embedding_a and embedding_b and self.embedding_model:
            semantic_sim = self.cosine_similarity(embedding_a, embedding_b)

        # Weighted combination
        if semantic_sim > 0 and text_sim > 0:
            return 0.6 * semantic_sim + 0.4 * text_sim  # Favor semantic
        elif semantic_sim > 0:
            return semantic_sim
        else:
            return text_sim

# Identity Service with raw SQL + Enhanced Matching
class IdentityService:
    def __init__(self):
        self.matcher = EnhancedMatcher()

    def resolve_identity(
        self,
        provider: str,
        external_id: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """Core identity resolution with raw SQL"""

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 1. Check if alias already exists
            cursor.execute("""
                SELECT user_id FROM aliases 
                WHERE provider = %s AND external_id = %s
            """, (provider, external_id))

            alias = cursor.fetchone()
            if alias:
                return {
                    "user_id": alias['user_id'],
                    "match_type": "exact_alias",
                    "confidence": "high"
                }

            # 2. Try to match by email
            if email:
                cursor.execute("""
                    SELECT user_id FROM users WHERE primary_email = %s
                """, (email,))

                user = cursor.fetchone()
                if user:
                    # Create new alias for existing user
                    cursor.execute("""
                        INSERT INTO aliases (user_id, provider, external_id, email, display_name, username, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (user['user_id'], provider, external_id, email, display_name, username, dt.datetime.utcnow()))

                    conn.commit()

                    return {
                        "user_id": user['user_id'],
                        "match_type": "email_match",
                        "confidence": "high"
                    }

            # 3. Create new user
            user_id = str(uuid.uuid4())

            # Create user
            cursor.execute("""
                INSERT INTO users (user_id, primary_name, primary_email, created_at)
                VALUES (%s, %s, %s, %s)
            """, (user_id, display_name or username or email, email, dt.datetime.utcnow()))

            # Create alias
            cursor.execute("""
                INSERT INTO aliases (user_id, provider, external_id, email, display_name, username, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, provider, external_id, email, display_name, username, dt.datetime.utcnow()))

            conn.commit()

            return {
                "user_id": user_id,
                "match_type": "new_user",
                "confidence": "high"
            }

    def create_ticket(
        self,
        user_id: str,
        source: str,
        external_id: str,
        title: str,
        body: str
    ) -> Dict[str, Any]:
        """Create ticket with enhanced duplicate detection (MinHash + Semantic)"""

        # Generate fingerprints
        text_content = (title or "") + " " + body
        fingerprint = self.matcher.minhash(text_content)
        semantic_embedding = self.matcher.semantic_embedding(text_content)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check for duplicates from same user
            cursor.execute("""
                SELECT ticket_id, fingerprint, semantic_embedding FROM tickets 
                WHERE user_id = %s AND status != 'duplicate'
            """, (user_id,))

            existing_tickets = cursor.fetchall()

            best_match = None
            best_score = 0.0

            for existing in existing_tickets:
                if existing['fingerprint']:
                    # Use enhanced similarity (MinHash + Semantic)
                    score = self.matcher.combined_similarity(
                        bytes(existing['fingerprint']),
                        bytes(existing['semantic_embedding']) if existing['semantic_embedding'] else None,
                        fingerprint,
                        semantic_embedding
                    )

                    if score > best_score:
                        best_score = score
                        best_match = existing

            # Create ticket
            ticket_id = str(uuid.uuid4())

            if best_match and best_score > 0.85:
                # Mark as duplicate
                cursor.execute("""
                    INSERT INTO tickets (ticket_id, user_id, source, external_id, title, body, 
                                       fingerprint, semantic_embedding, status, duplicate_of, similarity_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (ticket_id, user_id, source, external_id, title, body, fingerprint,
                      semantic_embedding, 'duplicate', best_match['ticket_id'], best_score, dt.datetime.utcnow()))

                conn.commit()

                return {
                    "ticket_id": ticket_id,
                    "duplicate_of": best_match['ticket_id'],
                    "similarity_score": best_score,
                    "status": "duplicate",
                    "detection_method": "enhanced_similarity"
                }
            else:
                # Create normal ticket
                cursor.execute("""
                    INSERT INTO tickets (ticket_id, user_id, source, external_id, title, body, 
                                       fingerprint, semantic_embedding, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (ticket_id, user_id, source, external_id, title, body, fingerprint,
                      semantic_embedding, 'open', dt.datetime.utcnow()))

                conn.commit()

                return {
                    "ticket_id": ticket_id,
                    "status": "created"
                }

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user info"""

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get user
            cursor.execute("""
                SELECT * FROM users WHERE user_id = %s
            """, (user_id,))

            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # Get aliases
            cursor.execute("""
                SELECT provider, external_id, email, username FROM aliases 
                WHERE user_id = %s
            """, (user_id,))
            aliases = cursor.fetchall()

            # Get ticket count
            cursor.execute("""
                SELECT COUNT(*) as count FROM tickets WHERE user_id = %s
            """, (user_id,))
            ticket_count = cursor.fetchone()['count']

            return {
                "user_id": user['user_id'],
                "primary_name": user['primary_name'],
                "primary_email": user['primary_email'],
                "created_at": user['created_at'].isoformat(),
                "total_aliases": len(aliases),
                "total_tickets": ticket_count,
                "aliases": [dict(alias) for alias in aliases]
            }

# Test database connection and create tables
def setup_database():
    try:
        # Test connection (YOUR working approach)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")  # Simple test query
            result = cursor.fetchone()
            print("✅ Database connection successful")

            # Check what tables already exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            existing_tables = cursor.fetchall()

            if existing_tables:
                print(f"📋 Found {len(existing_tables)} existing tables:")
                for table in existing_tables:
                    table_name = table['table_name']
                    try:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                        row_count = cursor.fetchone()['count']
                        print(f"   - {table_name}: {row_count} rows")
                    except Exception:
                        print(f"   - {table_name}: (unable to count)")
            else:
                print("📋 No existing tables found - will create new ones")

            # Create tables if they don't exist (with better error handling)
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR PRIMARY KEY,
                        primary_name TEXT,
                        primary_email TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print("✅ Users table created/verified")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS aliases (
                        alias_id BIGSERIAL PRIMARY KEY,
                        user_id VARCHAR REFERENCES users(user_id) ON DELETE CASCADE,
                        provider TEXT NOT NULL,
                        external_id TEXT NOT NULL,
                        email TEXT,
                        display_name TEXT,
                        username TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(provider, external_id)
                    )
                """)
                print("✅ Aliases table created/verified")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tickets (
                        ticket_id VARCHAR PRIMARY KEY,
                        user_id VARCHAR REFERENCES users(user_id) ON DELETE SET NULL,
                        source TEXT NOT NULL,
                        external_id TEXT NOT NULL,
                        title TEXT,
                        body TEXT,
                        fingerprint BYTEA,
                        semantic_embedding BYTEA,
                        status TEXT DEFAULT 'open',
                        duplicate_of VARCHAR REFERENCES tickets(ticket_id),
                        similarity_score FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source, external_id)
                    )
                """)
                print("✅ Tickets table created/verified")

                conn.commit()

                # Check final state
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
                final_tables = cursor.fetchall()

                print(f"🎯 Final database state - {len(final_tables)} tables:")
                for table in final_tables:
                    table_name = table['table_name']
                    try:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                        row_count = cursor.fetchone()['count']
                        print(f"   - {table_name}: {row_count} rows")
                    except Exception:
                        print(f"   - {table_name}: (unable to count)")

                print("✅ All database tables ready!")

            except Exception as table_error:
                print(f"❌ Table creation failed: {table_error}")
                conn.rollback()

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()

# Run database setup immediately
setup_database()

# FastAPI App
app = FastAPI(title="Simple Dosu Identity Service", version="1.0.0")

# Service instance
identity_service = IdentityService()

# Pydantic Models
class ResolveRequest(BaseModel):
    provider: str
    external_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    username: Optional[str] = None

class TicketRequest(BaseModel):
    user_id: str
    source: str
    external_id: str
    title: str
    body: str

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": dt.datetime.utcnow().isoformat(),
        "semantic_similarity_enabled": HAS_SENTENCE_TRANSFORMERS
    }

@app.get("/api/test")
def simple_test():
    return {"message": "hello world"}

@app.get("/api/database/tables")
def list_database_tables():
    """Check what tables actually exist in the database"""

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get all tables in the public schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)

        tables = cursor.fetchall()

        # Count rows in each table
        table_info = []
        for table in tables:
            table_name = table['table_name']
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = cursor.fetchone()['count']
                table_info.append({
                    "table_name": table_name,
                    "row_count": row_count
                })
            except Exception as e:
                table_info.append({
                    "table_name": table_name,
                    "row_count": f"Error: {e}"
                })

        return {
            "total_tables": len(table_info),
            "tables": table_info
        }

@app.post("/resolve")
def resolve_identity(body: ResolveRequest):
    """Identity resolution endpoint"""
    result = identity_service.resolve_identity(
        provider=body.provider,
        external_id=body.external_id,
        email=body.email,
        display_name=body.display_name,
        username=body.username
    )
    return result

@app.post("/ticket")
def create_ticket(body: TicketRequest):
    """Create ticket with duplicate detection"""
    result = identity_service.create_ticket(
        user_id=body.user_id,
        source=body.source,
        external_id=body.external_id,
        title=body.title,
        body=body.body
    )
    return result

@app.get("/api/users/{user_id}")
def get_user(user_id: str):
    """Get user info"""
    return identity_service.get_user(user_id)

@app.get("/api/users/{user_id}/duplicates")
def find_user_duplicates(user_id: str, threshold: float = 0.75):
    """Find potential duplicate tickets using enhanced similarity"""

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get all non-duplicate tickets for user
        cursor.execute("""
            SELECT ticket_id, title, fingerprint, semantic_embedding 
            FROM tickets 
            WHERE user_id = %s AND status != 'duplicate'
        """, (user_id,))

        tickets = cursor.fetchall()

        duplicates = []

        for i, ticket_a in enumerate(tickets):
            for ticket_b in tickets[i+1:]:
                if ticket_a['fingerprint'] and ticket_b['fingerprint']:
                    similarity = identity_service.matcher.combined_similarity(
                        bytes(ticket_a['fingerprint']),
                        bytes(ticket_a['semantic_embedding']) if ticket_a['semantic_embedding'] else None,
                        bytes(ticket_b['fingerprint']),
                        bytes(ticket_b['semantic_embedding']) if ticket_b['semantic_embedding'] else None
                    )

                    if similarity >= threshold:
                        duplicates.append({
                            "ticket_a": ticket_a['ticket_id'],
                            "ticket_b": ticket_b['ticket_id'],
                            "similarity": similarity,
                            "title_a": ticket_a['title'],
                            "title_b": ticket_b['title']
                        })

        return {"duplicates": duplicates, "threshold": threshold}

@app.get("/api/database/tables")
def list_database_tables():
    """Check what tables actually exist in the database"""

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get all tables in the public schema
        cursor.execute("""
            SELECT table_name, 
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)

        tables = cursor.fetchall()

        # Count rows in each table
        table_info = []
        for table in tables:
            table_name = table['table_name']
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = cursor.fetchone()['count']
                table_info.append({
                    "table_name": table_name,
                    "row_count": row_count,
                    "size": table['size']
                })
            except Exception as e:
                table_info.append({
                    "table_name": table_name,
                    "row_count": f"Error: {e}",
                    "size": table['size']
                })

        return {
            "total_tables": len(table_info),
            "tables": table_info
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)