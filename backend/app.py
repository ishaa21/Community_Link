"""
CommunityLink — Flask Backend
All API routes, Firestore integration, Gemini AI matching
"""

import os
import json
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import firestore
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ── FIRESTORE INIT ────────────────────────────────────────────
db = None

def get_db():
    global db
    if db is None:
        try:
            service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if service_account_json:
                creds_dict = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                db = firestore.Client(
                    credentials=credentials,
                    project=os.getenv("GOOGLE_CLOUD_PROJECT", "communitylink-493702"),
                    database="communitylink-db"
                )
            else:
                db = firestore.Client(
                    project=os.getenv("GOOGLE_CLOUD_PROJECT", "communitylink-493702"),
                    database="communitylink-db"
                )
        except Exception as e:
            print(f"[Firestore] Could not connect: {e}")
            db = None
    return db


# ── GEMINI INIT ───────────────────────────────────────────────
gemini_model = None

def get_gemini():
    global gemini_model
    if gemini_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            print("[Gemini] GEMINI_API_KEY not set.")
    return gemini_model


# ── HELPERS ───────────────────────────────────────────────────
def ts_now():
    return datetime.now(timezone.utc).isoformat()


def firestore_to_dict(doc):
    d = doc.to_dict()
    d["id"] = doc.id
    # Convert Firestore Timestamps to ISO strings
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


# ── SERVE FRONTEND ────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("../frontend", path)


# ══════════════════════════════════════════════════════════════
# NEEDS API
# ══════════════════════════════════════════════════════════════

@app.route("/api/needs", methods=["GET"])
def get_needs():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        docs = db_client.collection("needs").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(100).stream()
        needs = [firestore_to_dict(d) for d in docs]
        return jsonify(needs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/needs", methods=["POST"])
def create_need():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        data = request.get_json()
        need_id = "N" + uuid.uuid4().hex[:8].upper()
        need = {
            **data,
            "id": need_id,
            "timestamp": ts_now(),
            "status": "Open",
        }
        db_client.collection("needs").document(need_id).set(need)
        # Log activity
        _add_activity(db_client, "need",
                       f"New need reported: \"{need['title']}\" in {need.get('area', '')}",
                       need.get("urgency", "Medium"))
        return jsonify(need), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/needs/<need_id>", methods=["PATCH"])
def update_need(need_id):
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        data = request.get_json()
        db_client.collection("needs").document(need_id).update(data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# VOLUNTEERS API
# ══════════════════════════════════════════════════════════════

@app.route("/api/volunteers", methods=["GET"])
def get_volunteers():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        docs = db_client.collection("volunteers").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(100).stream()
        vols = [firestore_to_dict(d) for d in docs]
        return jsonify(vols)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/volunteers", methods=["POST"])
def create_volunteer():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        data = request.get_json()
        vol_id = "V" + uuid.uuid4().hex[:8].upper()
        vol = {
            **data,
            "id": vol_id,
            "timestamp": ts_now(),
            "status": "Available",
            "matchCount": 0,
        }
        db_client.collection("volunteers").document(vol_id).set(vol)
        _add_activity(db_client, "volunteer",
                       f"{vol['name']} joined as volunteer in {vol.get('area', '')}",
                       "Low")
        return jsonify(vol), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# MATCHES API
# ══════════════════════════════════════════════════════════════

@app.route("/api/matches", methods=["GET"])
def get_matches():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        docs = db_client.collection("matches").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(50).stream()
        matches = [firestore_to_dict(d) for d in docs]
        return jsonify(matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/matches", methods=["POST"])
def create_match():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        data = request.get_json()
        match_id = "M" + uuid.uuid4().hex[:8].upper()
        match = {**data, "id": match_id, "timestamp": ts_now()}
        db_client.collection("matches").document(match_id).set(match)

        # Update need status to Assigned
        need_id = data.get("needId")
        vol_id  = data.get("volId")
        if need_id:
            db_client.collection("needs").document(need_id).update(
                {"status": "Assigned", "assignedTo": vol_id}
            )
        if vol_id:
            vol_ref = db_client.collection("volunteers").document(vol_id)
            vol_doc = vol_ref.get()
            if vol_doc.exists:
                cur = vol_doc.to_dict().get("matchCount", 0)
                vol_ref.update({"matchCount": cur + 1})

        _add_activity(db_client, "match",
                       f"Volunteer assigned to need (Score: {data.get('score', '?')}%)",
                       "High")
        return jsonify(match), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ACTIVITY API
# ══════════════════════════════════════════════════════════════

@app.route("/api/activity", methods=["GET"])
def get_activity():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        docs = db_client.collection("activity").order_by(
            "time", direction=firestore.Query.DESCENDING
        ).limit(30).stream()
        activity = [firestore_to_dict(d) for d in docs]
        return jsonify(activity)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _add_activity(db_client, type_, text, urgency):
    act_id = "A" + uuid.uuid4().hex[:8].upper()
    db_client.collection("activity").document(act_id).set({
        "id": act_id,
        "type": type_,
        "text": text,
        "urgency": urgency,
        "time": ts_now(),
    })


# ══════════════════════════════════════════════════════════════
# STATS API
# ══════════════════════════════════════════════════════════════

@app.route("/api/stats", methods=["GET"])
def get_stats():
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        needs_docs  = list(db_client.collection("needs").stream())
        vols_docs   = list(db_client.collection("volunteers").stream())
        match_docs  = list(db_client.collection("matches").stream())

        needs = [d.to_dict() for d in needs_docs]
        vols  = [d.to_dict() for d in vols_docs]

        people_helped = sum(
            int(n.get("peopleAffected", 0))
            for n in needs if n.get("status") == "Assigned"
        )

        return jsonify({
            "totalNeeds":      len(needs),
            "totalVolunteers": len(vols),
            "totalMatches":    len(list(match_docs)),
            "urgentNeeds":     sum(1 for n in needs if n.get("urgency") == "High" and n.get("status") == "Open"),
            "openNeeds":       sum(1 for n in needs if n.get("status") == "Open"),
            "availableVols":   sum(1 for v in vols if v.get("status") == "Available"),
            "peopleHelped":    people_helped,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# GEMINI AI MATCHING + INSIGHT ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/match", methods=["POST"])
def ai_match():
    """
    Real Gemini AI matching:
    POST { needs: [...], volunteers: [...] }
    Returns Gemini's ranked matches with explanations.
    """
    model = get_gemini()
    if not model:
        return jsonify({"error": "Gemini API not configured. Set GEMINI_API_KEY."}), 503

    body = request.get_json()
    needs = body.get("needs", [])
    vols  = body.get("volunteers", [])

    if not needs or not vols:
        return jsonify({"error": "Provide needs and volunteers"}), 400

    prompt = f"""
You are a community welfare coordination AI. Given the following community needs and available volunteers, produce the BEST matches.

COMMUNITY NEEDS (JSON):
{json.dumps(needs, indent=2)}

AVAILABLE VOLUNTEERS (JSON):
{json.dumps(vols, indent=2)}

For each need, find the top 3 volunteer matches. Score from 0-100 based on:
- Location proximity (35 pts): exact area > same district > nearby
- Skill match (40 pts): exact skill > partial > none
- Urgency (15 pts): High=15, Medium=8, Low=3
- Availability (10 pts): more hours = more points

Return ONLY valid JSON in this exact structure (no markdown, no explanation outside JSON):
{{
  "matches": [
    {{
      "needId": "...",
      "needTitle": "...",
      "topScore": 85,
      "volunteers": [
        {{
          "volId": "...",
          "volName": "...",
          "score": 85,
          "reasons": [
            {{"label": "Exact location match", "points": 35, "pct": 100, "type": "location"}},
            {{"label": "Exact skill match", "points": 40, "pct": 100, "type": "skill"}},
            {{"label": "Urgency: High", "points": 15, "pct": 100, "type": "urgency"}},
            {{"label": "Availability: 10-20 hours", "points": 8, "pct": 80, "type": "availability"}}
          ],
          "aiExplanation": "Short 1-sentence explanation of why this volunteer is ideal"
        }}
      ]
    }}
  ],
  "insight": "2-3 sentence summary of the overall matching situation and key recommendations",
  "urgentAlert": "Most critical action needed right now (1 sentence)"
}}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Gemini returned invalid JSON: {e}", "raw": text[:500]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/insight", methods=["POST"])
def ai_insight():
    """
    Real Gemini AI dashboard insight generation.
    POST { needs: [...], volunteers: [...], stats: {...} }
    """
    model = get_gemini()
    if not model:
        return jsonify({"insight": _fallback_insight(request.get_json())}), 200

    body   = request.get_json()
    needs  = body.get("needs", [])
    vols   = body.get("volunteers", [])
    stats  = body.get("stats", {})

    prompt = f"""
You are analyzing community welfare data for CommunityLink, a volunteer coordination platform.

STATS: {json.dumps(stats)}
NEEDS (last 10): {json.dumps(needs[:10], indent=2)}
VOLUNTEERS (last 10): {json.dumps(vols[:10], indent=2)}

Provide a concise, actionable dashboard insight in 2-3 sentences. Focus on:
1. Most critical situation right now
2. A specific recommendation (e.g., "Deploy Dr. X to area Y")
3. Overall system health

Be direct, factual, and use specific names/numbers from the data. No generic statements.
Return ONLY the insight text, no JSON, no preamble.
"""

    try:
        response = model.generate_content(prompt)
        return jsonify({"insight": response.text.strip()})
    except Exception as e:
        return jsonify({"insight": _fallback_insight(body)}), 200


def _fallback_insight(body):
    """Rule-based fallback when Gemini is unavailable."""
    needs = body.get("needs", [])
    vols  = body.get("volunteers", [])
    urgent = [n for n in needs if n.get("urgency") == "High" and n.get("status") == "Open"]
    avail  = [v for v in vols if v.get("status") == "Available"]
    parts  = []
    if urgent:
        parts.append(f"⚠️ {len(urgent)} critical need(s) require immediate deployment.")
    if avail:
        parts.append(f"✅ {len(avail)} volunteers available. Run Smart Match to deploy them.")
    if not parts:
        parts.append("Add needs and volunteers to see AI-powered recommendations.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════
# SEED DATA
# ══════════════════════════════════════════════════════════════

@app.route("/api/seed", methods=["POST"])
def seed_data():
    """Seeds Firestore with sample data (dev/demo only)."""
    db_client = get_db()
    if not db_client:
        return jsonify({"error": "Database unavailable"}), 503

    from seed_data import SAMPLE_NEEDS, SAMPLE_VOLUNTEERS, SAMPLE_ACTIVITY

    # Clear existing
    for col in ["needs", "volunteers", "matches", "activity"]:
        docs = db_client.collection(col).stream()
        for d in docs:
            d.reference.delete()

    for n in SAMPLE_NEEDS:
        db_client.collection("needs").document(n["id"]).set(n)
    for v in SAMPLE_VOLUNTEERS:
        db_client.collection("volunteers").document(v["id"]).set(v)
    for a in SAMPLE_ACTIVITY:
        act_id = "A" + uuid.uuid4().hex[:8].upper()
        db_client.collection("activity").document(act_id).set(a)

    return jsonify({"seeded": True, "needs": len(SAMPLE_NEEDS), "volunteers": len(SAMPLE_VOLUNTEERS)})


# ── HEALTH ────────────────────────────────────────────────────
@app.route("/health")
def health():
    db_ok     = get_db() is not None
    gemini_ok = os.getenv("GEMINI_API_KEY") is not None
    return jsonify({
        "status":  "ok",
        "db":      "connected" if db_ok else "not connected",
        "gemini":  "configured" if gemini_ok else "not configured",
        "version": "2.0.0",
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
