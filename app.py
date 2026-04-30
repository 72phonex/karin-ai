"""
Karin AI — Flask Server (HuggingFace Spaces edition)
=====================================================
Key difference from Render: HF Spaces disk is ephemeral.
We use HF Datasets Hub as a persistent store:
  - On startup: pull latest DB files from HF dataset → /app/karin_data/
  - Every N messages: push DB files back to HF dataset
  - On shutdown signal: final push

Set these secrets in your HF Space Settings:
  KARIN_OWNER_KEY   — your owner dashboard password
  HF_TOKEN          — your HF write token (Settings → Access Tokens)
  HF_DATASET_REPO   — e.g. "72phonex/karin-data" (create a private dataset repo)
"""

import os
import json
import hashlib
import sqlite3
import secrets
import threading
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, g
from karin_core import (
    EpisodicMemoryBank, EmotionalState, AutonomousReasoningLoop,
    TrinityContextBuilder, KarinResponseEngine
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

# ── Config ─────────────────────────────────────────────────────────────────────
OWNER_KEY      = os.environ.get("KARIN_OWNER_KEY", "phonex_is_the_creator_2026")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_DATASET     = os.environ.get("HF_DATASET_REPO", "")   # e.g. "72phonex/karin-data"
DATA_DIR       = Path("/app/karin_data")
DB_PATH        = str(DATA_DIR / "karin.db")
MEMORY_DB      = str(DATA_DIR / "memory.db")
BACKUP_EVERY   = 10   # push to HF every N messages

DATA_DIR.mkdir(parents=True, exist_ok=True)
_message_counter = 0
_backup_lock = threading.Lock()


# ── HF Persistence ─────────────────────────────────────────────────────────────

def pull_from_hf():
    """Download DB files from HF Dataset on startup."""
    if not HF_TOKEN or not HF_DATASET:
        print("[Karin] No HF persistence configured — running ephemeral.")
        return
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["karin.db", "memory.db"]:
            try:
                path = hf_hub_download(
                    repo_id=HF_DATASET,
                    filename=fname,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    local_dir=str(DATA_DIR)
                )
                print(f"[Karin] Restored {fname} from HF dataset.")
            except Exception as e:
                print(f"[Karin] {fname} not found on HF (first run?): {e}")
    except Exception as e:
        print(f"[Karin] HF pull failed: {e}")

def push_to_hf():
    """Upload DB files to HF Dataset for persistence."""
    if not HF_TOKEN or not HF_DATASET:
        return
    with _backup_lock:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            for fname in ["karin.db", "memory.db"]:
                fpath = DATA_DIR / fname
                if fpath.exists():
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=fname,
                        repo_id=HF_DATASET,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
            print(f"[Karin] DB backed up to HF at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[Karin] HF push failed: {e}")

def maybe_backup():
    global _message_counter
    _message_counter += 1
    if _message_counter % BACKUP_EVERY == 0:
        t = threading.Thread(target=push_to_hf, daemon=True)
        t.start()


# ── Startup: restore data ──────────────────────────────────────────────────────
pull_from_hf()


# ── Singletons ─────────────────────────────────────────────────────────────────
memory_bank = EpisodicMemoryBank(MEMORY_DB)
reasoning   = AutonomousReasoningLoop()
ctx_builder = TrinityContextBuilder()
engine      = KarinResponseEngine()


# ── User DB ────────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen TEXT,
                total_messages INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT
            );
            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                username TEXT,
                ip_address TEXT,
                user_agent TEXT,
                action TEXT,
                timestamp TEXT,
                success INTEGER
            );
            CREATE TABLE IF NOT EXISTS emotion_states (
                user_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

init_db()


# ── Helpers ────────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256((pw + "karin_salt_2026").encode()).hexdigest()

def get_ip() -> str:
    return request.headers.get("X-Forwarded-For",
           request.remote_addr or "unknown").split(",")[0].strip()

def log_action(user_id: str, username: str, action: str, success: bool):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO login_logs
            (user_id,username,ip_address,user_agent,action,timestamp,success)
            VALUES (?,?,?,?,?,?,?)
        """, (user_id, username, get_ip(),
              request.headers.get("User-Agent","")[:200],
              action, datetime.now().isoformat(), int(success)))

def load_emotion(user_id: str) -> EmotionalState:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT state_json FROM emotion_states WHERE user_id=?", (user_id,)
        ).fetchone()
    if row:
        try:
            data = json.loads(row[0])
            e = EmotionalState()
            for k, v in data.items():
                if hasattr(e, k):
                    setattr(e, k, v)
            return e
        except Exception:
            pass
    return EmotionalState()

def save_emotion(user_id: str, emotion: EmotionalState):
    state = {
        "valence": emotion.valence, "arousal": emotion.arousal,
        "dominance": emotion.dominance, "curiosity": emotion.curiosity,
        "warmth": emotion.warmth, "confidence": emotion.confidence,
        "creativity": emotion.creativity, "resilience": emotion.resilience,
        "loyalty": emotion.loyalty, "mood": emotion.mood,
        "mood_history": emotion.mood_history[-50:],
        "total_updates": emotion.total_updates
    }
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO emotion_states (user_id, state_json, updated_at)
            VALUES (?, ?, ?)
        """, (user_id, json.dumps(state), datetime.now().isoformat()))

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization","").replace("Bearer ","").strip()
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT s.user_id, u.username FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token=? AND s.expires_at > ?
            """, (token, datetime.now().isoformat())).fetchone()
        if not row:
            return jsonify({"error": "Invalid or expired session"}), 401
        request.user_id  = row[0]
        request.username = row[1]
        return f(*args, **kwargs)
    return decorated

def require_owner(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-Owner-Key","")
        if key != OWNER_KEY:
            return jsonify({"error": "Owner access denied"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Auth Routes ────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({"error": "Username must be 3–20 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if not username.replace("_","").isalnum():
        return jsonify({"error": "Letters, numbers, underscores only"}), 400
    user_id = hashlib.sha256(username.encode()).hexdigest()[:16]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (id,username,password_hash,created_at)
                VALUES (?,?,?,?)
            """, (user_id, username, hash_password(password),
                  datetime.now().isoformat()))
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already taken"}), 409
    log_action(user_id, username, "register", True)
    memory_bank.apply_decay(user_id)
    maybe_backup()
    return jsonify({"message": f"Welcome, {username}!", "user_id": user_id}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id,username FROM users WHERE username=? AND password_hash=?",
            (username, hash_password(password))
        ).fetchone()
    if not row:
        log_action("unknown", username, "login_fail", False)
        return jsonify({"error": "Invalid credentials"}), 401
    user_id  = row[0]
    username = row[1]
    token    = secrets.token_urlsafe(32)
    expires  = (datetime.now() + timedelta(days=30)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions (token,user_id,created_at,expires_at,ip_address)
            VALUES (?,?,?,?,?)
        """, (token, user_id, datetime.now().isoformat(), expires, get_ip()))
        conn.execute("UPDATE users SET last_seen=? WHERE id=?",
                     (datetime.now().isoformat(), user_id))
    log_action(user_id, username, "login", True)
    memory_bank.apply_decay(user_id)
    return jsonify({"token": token, "username": username,
                    "user_id": user_id, "expires_at": expires})


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    data    = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    user_id  = request.user_id
    username = request.username
    emotion  = load_emotion(user_id)
    memories = memory_bank.retrieve(user_id, message, top_k=5)
    goal     = reasoning.maybe_spawn_goal(message, emotion)
    context  = ctx_builder.build(message, memories, emotion, reasoning.active_goals)
    response = engine.generate(message, context, emotion, memories, username)
    emotion.update(0.1 if len(message) > 20 else 0.05,
                   0.05 if "?" in message else -0.02)
    save_emotion(user_id, emotion)
    memory_bank.store(
        user_id=user_id,
        content=f"User: {message}\nKarin: {response}",
        episode_type="conversation",
        emotional_valence=emotion.valence,
        importance=0.5 + 0.1*len(message)/100,
        creator_context=f"IP:{get_ip()}"
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET total_messages=total_messages+1 WHERE id=?",
                     (user_id,))
    reflection = reasoning.reflect(message, response)
    maybe_backup()
    return jsonify({
        "response": response,
        "mood": emotion.mood,
        "confidence": round(emotion.confidence, 2),
        "memories_used": len(memories),
        "goal_spawned": goal,
        "reflection_quality": reflection["quality"]
    })


# ── Status / Memory / Goals ────────────────────────────────────────────────────

@app.route("/api/status")
@require_auth
def status():
    emotion   = load_emotion(request.user_id)
    mem_stats = memory_bank.stats(request.user_id)
    return jsonify({
        "mood": emotion.mood,
        "valence": round(emotion.valence, 3),
        "arousal": round(emotion.arousal, 3),
        "confidence": round(emotion.confidence, 3),
        "total_mood_updates": emotion.total_updates,
        "memory": mem_stats,
        "active_goals": len([g for g in reasoning.active_goals
                              if g.get("status") == "active"])
    })

@app.route("/api/memory")
@require_auth
def user_memory():
    return jsonify({
        "stats":   memory_bank.stats(request.user_id),
        "recent_episodes": memory_bank.get_user_episodes(request.user_id, limit=20)
    })

@app.route("/api/goals")
@require_auth
def goals():
    active = [g for g in reasoning.active_goals if g.get("status") == "active"]
    return jsonify({"active_goals": active, "total": len(active)})


# ── Owner Routes ───────────────────────────────────────────────────────────────

@app.route("/api/owner/users")
@require_owner
def owner_users():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT id,username,created_at,last_seen,total_messages
            FROM users ORDER BY last_seen DESC
        """).fetchall()
    users = [{"id":r[0],"username":r[1],"created_at":r[2],
              "last_seen":r[3],"total_messages":r[4]} for r in rows]
    return jsonify({"users": users,
                    "memory_summaries": memory_bank.get_all_users_summary()})

@app.route("/api/owner/user/<user_id>")
@require_owner
def owner_user_detail(user_id):
    episodes = memory_bank.get_user_episodes(user_id, limit=100)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username,created_at,last_seen,total_messages FROM users WHERE id=?",
            (user_id,)
        ).fetchone()
    if not row:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "username": row[0], "created_at": row[1],
        "last_seen": row[2], "total_messages": row[3],
        "episodes": episodes,
        "memory_stats": memory_bank.stats(user_id)
    })

@app.route("/api/owner/logs")
@require_owner
def owner_logs():
    limit = int(request.args.get("limit", 100))
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT username,ip_address,user_agent,action,timestamp,success
            FROM login_logs ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
    logs = [{"username":r[0],"ip":r[1],"agent":r[2][:60],
             "action":r[3],"timestamp":r[4],"success":bool(r[5])} for r in rows]
    return jsonify({"logs": logs})

@app.route("/api/owner/backup", methods=["POST"])
@require_owner
def owner_backup():
    """Manually trigger a backup to HF."""
    t = threading.Thread(target=push_to_hf, daemon=True)
    t.start()
    return jsonify({"message": "Backup triggered."})


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "karin": "online",
        "phase": 0,
        "persistence": "hf_dataset" if HF_TOKEN else "ephemeral"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
