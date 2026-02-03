import os
import re
import sqlite3
import pandas as pd
import streamlit as st
from openai import OpenAI

ORANGE = "#f97316"  # RingCentral-ish orange
SOFT_ORANGE = "rgba(249,115,22,0.12)"
DOT_ORANGE = "rgba(249,115,22,0.18)"

st.set_page_config(page_title="Analytics Copilot", layout="wide")

st.markdown(
    f"""
    <style>
    /* ---------- Page background: tiny orange dots ---------- */
    .stApp {{
      background-color: #ffffff;
      background-image:
        radial-gradient({DOT_ORANGE} 1.6px, transparent 1.6px);
      background-size: 18px 18px;
      opacity: 0.9;
      background-position: 0 0;
    }}

    /* ---------- Widen main container (reduce side blank space) ---------- */
    .block-container {{
      max-width: 1400px;
      padding-top: 1.2rem;
      padding-bottom: 2rem;
    }}

    /* ---------- Title "Analytics Copilot" in orange ---------- */
    h1, h2, h3 {{
      color: {ORANGE} !important;
    }}

    /* ---------- Make chat input (textbox) orange accented ---------- */
    /* chat input container */
    div[data-testid="stChatInput"] {{
      border: none !important;
      padding: 0 !important;
      background: none !important;
      
    }}

    /* actual input */
    div[data-testid="stChatInput"] textarea {{
    
        height: 56px !important;
        min-height: 56px !important;
        max-height: 56px !important;
        background: #f97316 !important;     /* solid orange */
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        outline: none !important;
        padding: 14px 16px !important;
        resize: none !important;
        overflow: hidden !important;
        box-shadow: none !important;
        line-height: 1.2 !important;
    }}

    /* placeholder text color */
    div[data-testid="stChatInput"] textarea::placeholder{{
    color: rgba(255,255,255,0.85) !important;
}}

    div[data-testid="stChatInput"] textarea:focus {{
      border: none !important;
      box-shadow: none !important;
    }}

    /* send button (paper plane) */
    div[data-testid="stChatInput"] button {{
      border-radius: 12px !important;
      border: 1px solid {SOFT_ORANGE} !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
      border: 1px solid {ORANGE} !important;
    }}

    /* ---------- Chat bubbles styling & left/right alignment ---------- */
    /* Streamlit renders each chat message in a container; we style user vs assistant separately */




/* BOT → LEFT */
div[data-testid="stChatMessage"][data-testid*="assistant"] {{
    justify-content: flex-start;
}}

/* USER → RIGHT */
div[data-testid="stChatMessage"][data-testid*="user"] {{
    justify-content: flex-end;
}}



/* Bot bubble */
div[data-testid="stChatMessage"][data-testid*="assistant"] > div {{
    background: #f5f5f5;
}}

/* User bubble */
div[data-testid="stChatMessage"][data-testid*="user"] > div {{
    background: rgba(249,115,22,0.15);
}}

.chat-row{{width:100%;display:flex;margin:10px 0;}}
.chat-row.user{{justify-content:flex-end;}}
.chat-row.bot{{justify-content:flex-start;}}

.chat-bubble{{
  max-width:72%;
  padding:12px 14px;
  border-radius:16px;
  box-shadow:0 1px 2px rgba(0,0,0,0.06);
  white-space:pre-wrap;
  word-wrap:break-word;
  font-size:15px;
  line-height:1.4;
}}

.chat-row.user .chat-bubble{{
  background: rgba(249,115,22,0.18);
  border: 1px solid rgba(249,115,22,0.25);
  border-top-right-radius:6px;
}}

.chat-row.bot .chat-bubble{{
  background:#ffffff;
  border:1px solid rgba(0,0,0,0.08);
  border-top-left-radius:6px;
}}

""", unsafe_allow_html=True

)

def bubble(role: str, text: str):
    cls = "user" if role == "user" else "bot"
    txt = "" if text is None else str(text)
    st.markdown(f"""
    <div class="chat-row {cls}">
      <div class="chat-bubble">{txt}</div>
    </div>
    """, unsafe_allow_html=True)



def smart_chart(df):
    if df is None or df.empty:
        return

    df = df.copy()

    # detect date-like column
    date_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "month", "time"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="raise")
                date_col = c
                break
            except:
                pass

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # 1️⃣ Time series → line chart
    if date_col and numeric_cols:

        # ✅ if long format (date + category + value) → pivot to wide for multi-line
        if len(numeric_cols) == 1 and len(non_numeric_cols) >= 2:
            val = numeric_cols[0]

            # pick category column (not the date column)
            cat_candidates = [c for c in non_numeric_cols if c != date_col]
            cat = cat_candidates[0] if cat_candidates else non_numeric_cols[0]

            try:
                pivot_df = df.pivot(index=date_col, columns=cat, values=val)
                pivot_df = pivot_df.sort_index()
                st.line_chart(pivot_df)
                return
            except:
                pass

        # ✅ already wide format
        try:
            wide_df = df.set_index(date_col)[numeric_cols].sort_index()
            st.line_chart(wide_df)
            return
        except:
            pass

    # 2️⃣ Category vs numeric → bar chart
    if len(numeric_cols) == 1 and len(non_numeric_cols) >= 1:
        cat = non_numeric_cols[0]
        val = numeric_cols[0]
        try:
            temp = df[[cat, val]].copy()
            temp[val] = pd.to_numeric(temp[val], errors="coerce")
            temp = temp.dropna()
            if not temp.empty:
                st.bar_chart(temp.set_index(cat)[val])
            return
        except:
            pass

    # 3️⃣ Two numeric → scatter chart
    if len(numeric_cols) >= 2:
        try:
            st.scatter_chart(df[numeric_cols[:2]])
            return
        except:
            pass


        st.line_chart(df.set_index(date_col)[numeric_cols])

    # 2️⃣ Category vs numeric → bar chart
    elif len(numeric_cols) == 1 and len(non_numeric_cols) >= 1:
        cat = non_numeric_cols[0]
        val = numeric_cols[0]
        st.bar_chart(df.set_index(cat)[val])

    # 3️⃣ Two numeric → scatter chart
    elif len(numeric_cols) >= 2:
        st.scatter_chart(df[numeric_cols[:2]])



DB_PATH = "analytics.db"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SCHEMA = """
Tables:
users(user_id, signup_date, city)
orders(order_id, user_id, order_date, amount)
payments(payment_id, order_id, status, paid_amount)
"""

def clean_sql(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "").strip()

    if ";" in s:
        s = s.split(";")[0] + ";"
    return s.strip()


def fix_half_cte(sql: str) -> str:
    s = sql.strip()

    # If already starts with WITH, nothing to do
    if s.lower().startswith("with"):
        return s

    # Detect: starts with SELECT, later has "), <cte_name> AS ("
    m = re.search(r"\)\s*,\s*([A-Za-z_]\w*)\s+AS\s*\(", s, flags=re.IGNORECASE)
    if not m:
        return s

    if not s.startswith("select"):
        if not s.startswith("with"):
            return False
        return s

    second_cte_name = m.group(1)

    # Try to guess the missing first CTE name from references like "FROM monthly_revenue"
    ref = re.search(r"\bfrom\s+([A-Za-z_]\w*)\b", s[m.start():], flags=re.IGNORECASE)
    first_cte_name = ref.group(1) if ref else "cte0"

    # Split into: first_select_part + rest (starting after the comma)
    split_pos = m.start()  # points at ") , <cte> AS ("
    first_part = s[:split_pos].rstrip()
    rest = s[split_pos+1:].lstrip()  # skip the first ')', keep ", <cte> AS ( ..."

    # Wrap the first SELECT as the first CTE body
    fixed = f"WITH {first_cte_name} AS (\n{first_part}\n)\n{rest}"
    return fixed.strip()

def is_safe_select(sql: str) -> bool:
    print("DEBUG is_safe_select got:", type(sql), repr(str(sql)[:50]))

    if not isinstance(sql, str):
        return False

    s = sql.strip().lower()

    # ✅ allow select or with ignoring whitespace/comments
    if not re.match(r"^\s*(select|with)\b", s):
        return False

    blocked = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]

    return not any(b in s for b in blocked)

def run_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

def generate_sql(question: str) -> str:
    prompt = f"""
You are a senior analytics engineer.

Rules:
1) Use ONLY the given schema (tables/columns).
You are a senior analytics engineer.

SQL Generation Rules (STRICT):

1) Always generate query in CTE format. 
2) status = 'success' or 'failed' or 'pending' or 'refunded'

Schema:
{SCHEMA}

User question:
{question}

Return ONLY SQL.
"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return resp.output_text.strip()

def fix_sql(question: str, bad_sql: str, error_msg: str) -> str:
    prompt = f"""
You are a senior SQL expert.

The SQL query failed in SQLite.
Return a corrected SQLite SELECT query ONLY (no markdown, no explanation).

Schema:
{SCHEMA}

User question:
{question}

Failed SQL:
{bad_sql}

SQLite error:
{error_msg}

Return ONLY corrected SQL.
"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return resp.output_text.strip()

def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    preview = df.head(12).to_string(index=False)
    prompt = f"""
You are a professional business analyst.

User question:
{question}

SQL used:
{sql}

Result preview:
{preview if len(df) else "(empty)"}

Write a short professional answer in clear English:
- First line: direct answer
- Then 1-2 bullet insights
- If empty, explain likely reasons and suggest a better next question.
"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return resp.output_text.strip()

st.set_page_config(page_title="Analytics Copilot", layout="centered")
st.title("📊 Analytics Copilot")




st.caption("Chat-style analytics assistant • Ask in English • Get table + chart + insight")
if "messages" not in st.session_state:
    st.session_state.messages = []   # each item: {"role": "user"/"assistant", "content": "...", "sql": "...", "df": df, "err": "..."}


def render_result(df: pd.DataFrame):
    if df is None:
        st.error("No result returned.")
        return

    if df.empty:
        st.warning("No rows returned.")
        return

    st.dataframe(df, use_container_width=True)
    smart_chart(df)
 



if "history" not in st.session_state:
    st.session_state.history = []



for msg in st.session_state.messages:
    bubble(msg.get("role","assistant"), msg.get("content",""))

    # assistant ke niche SQL/df
    if msg.get("role") != "user":
        if msg.get("sql"):
            with st.expander("🔍 View Generated SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("err"):
            st.error(msg["err"])
        elif msg.get("df") is not None:
            render_result(msg["df"])



q = st.chat_input("Ask anything about your data… (type 'exit' to stop)")
clear_btn = st.button("🧹 Clear chat")

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if q and q.strip():
    question = q.strip()

    # save user msg
    st.session_state.messages.append({"role": "user", "content": question})

    raw_sql = generate_sql(question)
    sql = clean_sql(raw_sql)
    sql = fix_half_cte(sql)

    

# ✅ AUTO CTE FIX (important)
   # if " AS (" in sql.upper() and not sql.lower().startswith(("with", "select")):
    # sql = "WITH " + sql


    #if not is_safe_select(sql):
     #   err = "Blocked: Only safe SELECT queries are allowed."
      #  st.session_state.messages.append({
       #     "role": "assistant",
        #    "content": "",
         #   "sql": sql,
          #  "df": None,
           # "err": err
        #})
        #st.rerun()

    df = None
    err = None
    insight = ""

    # Try run + 2 auto-fixes
    for attempt in range(3):
        try:
            df = run_sql(sql)
            err = None
            break
        except Exception as e:
            err = str(e)
            if attempt < 2:
                fixed = clean_sql(fix_sql(question, sql, err))
                if not is_safe_select(fixed):
                    break
                sql = fixed

    # Generate insight only when no SQL error
    if not err:
        try:
            insight = explain_result(question, sql, df if df is not None else pd.DataFrame())
        except Exception as ex:
            insight = f"(Insight generation failed: {ex})"

    # Save assistant msg ALWAYS
    st.session_state.messages.append({
        "role": "assistant",
        "content": insight,
        "sql": sql,
        "df": df,
        "err": err
    })

    st.rerun()


