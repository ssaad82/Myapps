# ==========================================
# Streamlit Multilingual Sentiment Dashboard
# Balance of Opinion (BoO)
# Quarterly Report • Sector-weighted
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

st.set_page_config(page_title="Balance of Opinion Index", layout="wide")
st.title("📊  Balance of Opinion Index")
st.caption("Quarterly Report • Sector-weighted • Explainable")

# =========================
# Sidebar — Upload + controls
# =========================
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Required columns: text, source, date, sector"
)

thr = st.sidebar.slider(
    "Neutral threshold",
    min_value=0.05,
    max_value=0.4,
    value=0.2,
    step=0.05
)

# =========================
# Load data
# =========================
if uploaded:
    df = pd.read_csv(uploaded)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    st.info("Using demo data — upload a CSV to replace it.")
    df = pd.DataFrame({
        "text": [
            # Q1 2025
            "Demand is improving and orders are stronger",
            "Les coûts augmentent et les marges sont sous pression",
            "الطلب تحسن والاعمال افضل من قبل",

            # Q2 2025
            "Sales are stable, no major change",
            "New orders fell and liquidity is tight",
            "الطلب مستقر لكن السيولة ضيقة",

            # Q3 2025
            "Strong growth and recovery in activity",
            "Les ventes en baisse et pression sur les prix",
            "انتعاش كبير وارتفاع في الطلب",

            # Q4 2025
            "Very strong increase in revenues 🎉",
            "Crise et inflation, situation difficile",
            "الوضع سيء جدا والناس متشائمة 😢"
        ],
        "source": [
            "news","news","business",
            "business","news","business",
            "news","news","business",
            "business","news","social"
        ],
        "sector": [
            "industry","industry","finance",
            "industry","finance","finance",
            "services","industry","industry",
            "finance","services","households"
        ],
        "date": pd.to_datetime([
            "2025-01-15","2025-02-10","2025-03-05",
            "2025-04-20","2025-05-18","2025-06-10",
            "2025-07-12","2025-08-25","2025-09-03",
            "2025-10-14","2025-11-22","2025-12-05"
        ])
    })

required_cols = {"text","source","date","sector"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# =========================
# Sector filter
# =========================
sectors = sorted(df["sector"].dropna().unique())
selected_sectors = st.sidebar.multiselect(
    "Filter by sector",
    options=sectors,
    default=sectors
)
df = df[df["sector"].isin(selected_sectors)]

# =========================
# Arabic normalization + cleaning
# =========================
AR_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_arabic(text):
    text = AR_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ـ+", "", text)
    return text

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = normalize_arabic(s)
    s = re.sub(
        r"[^\p{Latin}\p{Arabic}\p{Number}\s!?.,\u2600-\u27BF\U0001F300-\U0001FAFF]",
        " ",
        s
    )
    return re.sub(r"\s+", " ", s).strip()

df["clean"] = df["text"].apply(clean_text)

# =========================
# Language detection
# =========================
def detect_lang_safe(text):
    try:
        return detect(text)
    except:
        return "unknown"

df["lang"] = df["clean"].apply(detect_lang_safe)

# =========================
# Lexicons
# =========================
POS_EN = {"improving","stronger","growth","increase","recovery","stable"}
NEG_EN = {"fell","decline","pressure","tight","liquidity","inflation","crisis"}

POS_FR = {"hausse","croissance","positif","reprise","stable"}
NEG_FR = {"baisse","pression","crise","inflation","couts"}

POS_AR = {"تحسن","ارتفاع","ايجابي","انتعاش","استقرار","افضل"}
NEG_AR = {"تراجع","ضغط","ازمه","تضخم","سيوله","مرتفعه","سيء","ضيقة"}

EMO_POS = {"😀","😊","👍","🎉","💪","✅"}
EMO_NEG = {"😡","😢","😭","👎","💔","❌","😂"}

NEGATORS_EN = {"not","no","never"}
NEGATORS_FR = {"ne","pas","jamais"}
NEGATORS_AR = {"لا","ليس","لم","لن"}

INTENSIFIERS_EN = {"very","too"}
INTENSIFIERS_FR = {"tres","très"}
INTENSIFIERS_AR = {"جدا","جداً","كثير"}

def lexicons_for_lang(lang):
    if lang == "fr":
        return POS_FR, NEG_FR, NEGATORS_FR, INTENSIFIERS_FR
    if lang == "ar":
        return POS_AR, NEG_AR, NEGATORS_AR, INTENSIFIERS_AR
    return POS_EN, NEG_EN, NEGATORS_EN, INTENSIFIERS_EN

# =========================
# Sentiment scoring
# =========================
def sentiment_score(text, lang):
    pos_lex, neg_lex, negators, intensifiers = lexicons_for_lang(lang)
    tokens = text.split()

    pos = 0.0
    neg = 0.0
    boost = 1.0
    negate = False

    for ch in text:
        if ch in EMO_POS:
            pos += 1
        elif ch in EMO_NEG:
            neg += 1

    for t in tokens:
        if t in intensifiers:
            boost = 1.5
            continue
        if t in negators:
            negate = True
            continue

        if t in pos_lex:
            if negate:
                neg += boost
            else:
                pos += boost
        elif t in neg_lex:
            if negate:
                pos += boost
            else:
                neg += boost

        boost = 1.0
        negate = False

    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total

df["score"] = df.apply(lambda r: sentiment_score(r["clean"], r["lang"]), axis=1)

# =========================
# Labels
# =========================
def label(score):
    if score >= thr: return "positive"
    if score <= -thr: return "negative"
    return "neutral"

df["label"] = df["score"].apply(label)

# =========================
# Weights (source × sector)
# =========================
source_w = {"business": 1.3, "news": 1.0, "social": 0.7}
sector_w = {"industry": 1.2, "finance": 1.3, "households": 0.8, "services": 1.0}

df["weight"] = (
    df["source"].map(source_w).fillna(1.0) *
    df["sector"].map(sector_w).fillna(1.0)
)

# =========================
# Quarterly Report – calendar quarters
# =========================
df["quarter"] = df["date"].dt.to_period("Q")

ts = (
    df.groupby("quarter")
      .apply(lambda x: (
          x.loc[x["label"] == "positive", "weight"].sum()
        - x.loc[x["label"] == "negative", "weight"].sum()
      ) / max(x["weight"].sum(), 1e-9))
      .reset_index(name="balance_of_opinion")
      .sort_values("quarter")
)

ts["index"] = 100 + 100 * ts["balance_of_opinion"]

# =========================
# KPIs
# =========================
st.subheader("📌 Key indicators (Quarterly Report)")
c1, c2, c3 = st.columns(3)
c1.metric("Latest Balance of Opinion", round(ts["balance_of_opinion"].iloc[-1], 3))
c2.metric("Latest Sentiment Index", round(ts["index"].iloc[-1], 1))
c3.metric("Observations", len(df))

# =========================
# Plot
# =========================
st.subheader("📈 Quarterly Report – Balance of Opinion Index")
fig, ax = plt.subplots()
ax.plot(ts["quarter"].astype(str), ts["index"], marker="o")
ax.set_ylabel("Index (0–200)")
ax.set_xlabel("Quarter")
ax.set_title("Sector-weighted Balance of Opinion Index")
plt.xticks(rotation=45)
st.pyplot(fig)

# =========================
# Table
# =========================
st.subheader("🧾 Detailed data (Quarterly Report)")
st.dataframe(
    df[["date","quarter","sector","text","lang","score","label","weight"]],
    use_container_width=True
)
