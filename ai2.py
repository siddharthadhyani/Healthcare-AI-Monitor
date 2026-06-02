# ============================================================
# 🩺 AI Smart Healthcare Monitoring System
# Powered by Groq AI (Free & Fast)
# ============================================================
# INSTALL:
#   pip install streamlit groq plotly pandas python-dotenv
#   pip install streamlit-mic-recorder SpeechRecognition pydub
#
# RUN:
#   streamlit run ai.py
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
import io

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Healthcare Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0a0f1e; color: #e2e8f0; }
    .main .block-container { padding-top: 1.5rem; }

    /* Welcome screen */
    .welcome-card {
        background: linear-gradient(135deg, #0f172a, #0c1a2e);
        border: 1px solid #1e293b;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        max-width: 500px;
        margin: 60px auto;
    }
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2dd4bf, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .welcome-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; font-weight: 700; font-family: monospace; }
    .metric-trend { font-size: 13px; margin-top: 4px; }

    .chat-user {
        background: linear-gradient(135deg, #0f766e, #0369a1);
        border-radius: 16px 16px 4px 16px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
        color: white;
    }
    .chat-bot {
        background: #1e293b;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 80%;
        color: #e2e8f0;
        line-height: 1.6;
    }
    .chat-time { font-size: 11px; color: #64748b; margin-top: 4px; }
    .blocked-msg {
        background: #450a0a;
        border: 1px solid #991b1b;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 80%;
        color: #fca5a5;
    }
    section[data-testid="stSidebar"] { background-color: #0f172a; }
    .stButton > button {
        background: linear-gradient(135deg, #0f766e, #0369a1);
        color: white; border: none; border-radius: 24px;
        padding: 8px 20px; font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.85; color: white; }
    .stTextArea textarea, .stTextInput input {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #2dd4bf !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    .stNumberInput input {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #2dd4bf !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #2dd4bf !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    /* Label text */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #2dd4bf !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GROQ API KEY
# ============================================================
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are a strict AI Healthcare Assistant. You ONLY answer questions related to:
- Human health, body, and anatomy
- Medical symptoms and diseases
- Medicines, drugs, and treatments
- Hospitals, clinics, and medical services
- Nutrition, diet, and healthy lifestyle
- Mental health and wellness
- Medical tests and reports
- First aid and emergencies
- Heart rate, blood pressure, sugar levels and other vitals

If the user asks about ANYTHING else (sports, movies, coding, politics, weather, jokes, general knowledge etc.),
you MUST respond with exactly:
"⚕️ I'm a Healthcare Assistant. I can only answer health, medicine, body, and hospital related questions. Please ask me something related to your health."

Always recommend consulting a real doctor for serious medical concerns.
Keep responses clear, empathetic, and under 150 words unless detailed explanation is needed.
Never diagnose definitively — always suggest professional medical consultation."""

# ============================================================
# HEALTH KEYWORDS
# ============================================================
HEALTH_KEYWORDS = [
    "health", "body", "pain", "doctor", "hospital", "medicine", "drug",
    "symptom", "disease", "fever", "headache", "heart", "blood", "pressure",
    "sugar", "diabetes", "cancer", "infection", "virus", "bacteria", "cold",
    "flu", "cough", "breathing", "lung", "kidney", "liver", "bone", "muscle",
    "skin", "eye", "ear", "throat", "stomach", "diet", "nutrition", "vitamin",
    "exercise", "weight", "sleep", "stress", "anxiety", "depression", "mental",
    "therapy", "vaccine", "allergy", "pregnancy", "period", "bp", "pulse",
    "oxygen", "calories", "protein", "fat", "carb", "clinic", "nurse", "surgery",
    "tablet", "capsule", "injection", "dose", "side effect", "treatment", "cure",
    "first aid", "emergency", "ambulance", "ache", "swelling", "rash", "wound",
    "fracture", "sprain", "bmi", "cholesterol", "thyroid", "hormone", "organ",
    "brain", "spine", "joint", "arthritis", "asthma", "migraine", "stroke",
    "normal", "rate", "level", "report", "test", "scan", "xray", "mri",
    "my heart", "my sugar", "my bp", "my health", "is it normal", "what is",
    "how to", "food", "eat", "drink", "water", "supplement"
]

def is_health_related(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in HEALTH_KEYWORDS)

# ============================================================
# LOAD GROQ CLIENT
# ============================================================
@st.cache_resource
def load_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)

# ============================================================
# GET AI RESPONSE
# ============================================================
def get_ai_response(user_message: str) -> tuple:
    if not is_health_related(user_message):
        return (
            "⚕️ I'm a Healthcare Assistant. I can only answer health, medicine, body, and hospital related questions. Please ask me something related to your health.",
            True
        )
    try:
        client = load_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in st.session_state.groq_history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        reply = response.choices[0].message.content

        st.session_state.groq_history.append({"role": "user", "content": user_message})
        st.session_state.groq_history.append({"role": "assistant", "content": reply})

        if len(st.session_state.groq_history) > 20:
            st.session_state.groq_history = st.session_state.groq_history[-20:]

        return (reply, False)

    except Exception as e:
        error = str(e)
        if "api_key" in error.lower() or "auth" in error.lower():
            return ("❌ Invalid Groq API Key. Update it in secrets.", False)
        elif "rate" in error.lower():
            return ("⚠️ Too many requests. Please wait a moment.", False)
        else:
            return (f"⚠️ Error: {error}", False)

# ============================================================
# HELPERS
# ============================================================
def add_message(role: str, content: str, blocked: bool = False):
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%I:%M %p"),
        "blocked": blocked
    })

def metric_card(label, value, unit, delta, color):
    trend_color = "#f87171" if delta > 0 else "#34d399" if delta < 0 else "#94a3b8"
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">
            {value}<span style="font-size:14px;color:#64748b"> {unit}</span>
        </div>
        <div class="metric-trend" style="color:{trend_color}">
            {arrow} {abs(delta)} from yesterday
        </div>
    </div>
    """, unsafe_allow_html=True)

def transcribe_audio(audio_bytes):
    import speech_recognition as sr
    from pydub import AudioSegment
    webm_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    wav_buffer = io.BytesIO()
    webm_audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# ============================================================
# SESSION STATE
# ============================================================
if "user_registered" not in st.session_state:
    st.session_state.user_registered = False
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 0
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "groq_history" not in st.session_state:
    st.session_state.groq_history = []
if "health_log" not in st.session_state:
    st.session_state.health_log = [
        {"Date": "Mon", "Heart Rate": 72, "Blood Pressure": 120, "Sugar Level": 90},
        {"Date": "Tue", "Heart Rate": 75, "Blood Pressure": 122, "Sugar Level": 95},
        {"Date": "Wed", "Heart Rate": 80, "Blood Pressure": 119, "Sugar Level": 100},
        {"Date": "Thu", "Heart Rate": 78, "Blood Pressure": 121, "Sugar Level": 92},
        {"Date": "Fri", "Heart Rate": 76, "Blood Pressure": 118, "Sugar Level": 97},
        {"Date": "Sat", "Heart Rate": 74, "Blood Pressure": 117, "Sugar Level": 88},
        {"Date": "Sun", "Heart Rate": 73, "Blood Pressure": 116, "Sugar Level": 91},
    ]

# ============================================================
# WELCOME / REGISTRATION SCREEN
# ============================================================
if not st.session_state.user_registered:

    st.markdown("""
    <div style='text-align:center; margin-top:40px; margin-bottom:10px'>
        <div style='font-size:60px'>🩺</div>
        <h1 style='background:linear-gradient(90deg,#2dd4bf,#38bdf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   font-size:2.2rem; margin-bottom:4px'>
            AI Healthcare Monitor
        </h1>
        <p style='color:#64748b; font-size:15px'>
            Your personal AI health assistant — powered by Groq
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Registration form centered
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div style='background:#0f172a; border:1px solid #1e293b;
                    border-radius:20px; padding:32px; margin-top:10px'>
            <h3 style='color:#2dd4bf; text-align:center; margin-bottom:4px'>
                👋 Welcome! Let's get started
            </h3>
            <p style='color:#64748b; text-align:center; font-size:13px; margin-bottom:20px'>
                Please enter your details to personalize your health experience
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("registration_form"):
            name = st.text_input(
                "👤 Your Full Name",
                placeholder="e.g. Sahil Kumar",
                max_chars=50
            )
            age = st.number_input(
                "🎂 Your Age",
                min_value=1,
                max_value=120,
                value=None,
                placeholder="Enter your age"
            )
            gender = st.selectbox(
                "⚧ Gender",
                ["Select", "Male", "Female", "Other"]
            )
            blood_group = st.selectbox(
                "🩸 Blood Group",
                ["Don't Know", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
            )

            submit = st.form_submit_button(
                "🚀 Start Health Monitoring",
                use_container_width=True
            )

            if submit:
                if not name.strip():
                    st.error("❌ Please enter your name.")
                elif age is None or age < 1:
                    st.error("❌ Please enter a valid age.")
                elif gender == "Select":
                    st.error("❌ Please select your gender.")
                else:
                    st.session_state.patient_name   = name.strip()
                    st.session_state.patient_age    = int(age)
                    st.session_state.patient_gender = gender
                    st.session_state.blood_group    = blood_group
                    st.session_state.user_registered = True
                    st.rerun()

    st.markdown("""
    <div style='text-align:center; margin-top:20px'>
        <small style='color:#334155'>
            🔒 Your data stays private • ⚕️ Only health questions answered •
            ⚠️ Not a substitute for professional medical advice
        </small>
    </div>
    """, unsafe_allow_html=True)

    st.stop()  # Stop here until registered

# ============================================================
# MAIN APP (shown after registration)
# ============================================================

patient_name   = st.session_state.patient_name
patient_age    = st.session_state.patient_age
patient_gender = st.session_state.patient_gender

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style='background:#0f172a; border:1px solid #1e293b;
                border-radius:12px; padding:16px; margin-bottom:16px'>
        <div style='font-size:24px; text-align:center'>👤</div>
        <div style='text-align:center; font-weight:700; color:#2dd4bf;
                    font-size:16px'>{patient_name}</div>
        <div style='text-align:center; color:#64748b; font-size:13px'>
            Age: {patient_age} • {patient_gender}
        </div>
        <div style='text-align:center; color:#64748b; font-size:12px'>
            Blood Group: {st.session_state.get('blood_group', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ➕ Log Today's Vitals")
    with st.form("vitals_form"):
        new_hr = st.number_input("Heart Rate (bpm)",     min_value=40,  max_value=200, value=75)
        new_bp = st.number_input("Blood Pressure (mmHg)",min_value=80,  max_value=200, value=120)
        new_sg = st.number_input("Sugar Level (mg/dL)",  min_value=60,  max_value=400, value=95)
        if st.form_submit_button("📥 Save Vitals"):
            st.session_state.health_log.append({
                "Date": datetime.now().strftime("%a %H:%M"),
                "Heart Rate": new_hr,
                "Blood Pressure": new_bp,
                "Sugar Level": new_sg
            })
            st.success("✅ Vitals saved!")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.groq_history = []
            st.rerun()
    with col_b:
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                padding:10px;font-size:12px;color:#64748b'>
    ⚕️ <b style='color:#94a3b8'>This bot only answers:</b><br>
    • Health & body questions<br>
    • Medicine & treatments<br>
    • Hospital & medical services<br>
    • Vitals & health reports<br><br>
    ⚠️ Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style='text-align:center; margin-bottom:20px'>
    <h1 style='background:linear-gradient(90deg,#2dd4bf,#38bdf8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:2rem; margin-bottom:4px'>
        🩺 Smart Healthcare Monitor
    </h1>
    <p style='color:#64748b'>
        Welcome back, <b style='color:#2dd4bf'>{patient_name}</b>!
        Age: {patient_age} • {patient_gender}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# METRIC CARDS
# ============================================================
log    = st.session_state.health_log
latest = log[-1]
prev   = log[-2] if len(log) >= 2 else latest

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("❤️ Heart Rate",     latest["Heart Rate"],     "bpm",
                latest["Heart Rate"]     - prev["Heart Rate"],     "#f472b6")
with col2:
    metric_card("💉 Blood Pressure", latest["Blood Pressure"], "mmHg",
                latest["Blood Pressure"] - prev["Blood Pressure"], "#38bdf8")
with col3:
    metric_card("🍬 Sugar Level",    latest["Sugar Level"],    "mg/dL",
                latest["Sugar Level"]    - prev["Sugar Level"],    "#a78bfa")

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["💬 AI Chat Assistant", "📊 Health Charts", "📋 Data Log"])

# ----------------------------------------------------------
# TAB 1: CHATBOT
# ----------------------------------------------------------
with tab1:

    if not st.session_state.chat_history:
        add_message(
            "assistant",
            f"Hello {patient_name}! 👋 I'm your AI Health Assistant. "
            f"I see you're {patient_age} years old. "
            "I can ONLY answer questions about health, medicine, body, hospitals, and medical topics. "
            "How can I help you today?"
        )

    chat_box = st.container(height=380)
    with chat_box:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 {msg["content"]}'
                    f'<div class="chat-time">{msg["time"]}</div></div>',
                    unsafe_allow_html=True
                )
            elif msg.get("blocked"):
                st.markdown(
                    f'<div class="blocked-msg">🚫 {msg["content"]}'
                    f'<div class="chat-time">{msg["time"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["content"]}'
                    f'<div class="chat-time">{msg["time"]}</div></div>',
                    unsafe_allow_html=True
                )

    st.markdown("#### ✍️ Type Your Health Question")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="e.g. Is my heart rate of 80 bpm normal? What causes high BP?",
            height=90,
            label_visibility="collapsed"
        )
        send = st.form_submit_button("🚀 Send")

    if send and user_input.strip():
        add_message("user", user_input.strip())
        with st.spinner("🤖 Thinking..."):
            reply, blocked = get_ai_response(user_input.strip())
        add_message("assistant", reply, blocked)
        st.rerun()

    st.markdown("#### 🎤 Voice Assistant")
    try:
        from streamlit_mic_recorder import mic_recorder
        import speech_recognition as sr
        from pydub import AudioSegment

        audio = mic_recorder(
            start_prompt="🎤 Click to Speak",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            key="voice_input"
        )

        if audio and audio.get("bytes"):
            try:
                webm = AudioSegment.from_file(
                    io.BytesIO(audio["bytes"]), format="webm"
                )
                wav = io.BytesIO()
                webm.export(wav, format="wav")
                wav.seek(0)
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav) as source:
                    audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)

                # Typing effect
                st.markdown("🗣️ **You said:**")
                words = transcript.split()
                display = st.empty()
                typed = ""
                import time
                for word in words:
                    typed += word + " "
                    display.markdown(
                        f"<div style='background:#1e293b; padding:10px 14px;"
                        f"border-radius:8px; color:#e2e8f0; font-size:15px'>"
                        f"{typed}</div>",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.12)

                add_message("user", transcript)
                with st.spinner("🤖 Thinking..."):
                    reply, blocked = get_ai_response(transcript)
                add_message("assistant", reply, blocked)
                st.rerun()

            except sr.UnknownValueError:
                st.warning("⚠️ Could not understand. Speak clearly.")
            except Exception as e:
                st.error(f"⚠️ Voice error: {e}")

    except ImportError:
        st.warning("Run: `pip install streamlit-mic-recorder SpeechRecognition pydub`")

    st.markdown("#### ⚡ Quick Health Questions")
    quick_questions = [
        "Is 80 bpm heart rate normal?",
        "What causes high blood pressure?",
        "How to lower sugar level?",
        "Signs of diabetes?",
        "Foods good for heart?",
    ]
    cols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q, key=f"quick_{i}", width='stretch'):
                add_message("user", q)
                with st.spinner("🤖 Thinking..."):
                    reply, blocked = get_ai_response(q)
                add_message("assistant", reply, blocked)
                st.rerun()

# ----------------------------------------------------------
# TAB 2: CHARTS
# ----------------------------------------------------------
with tab2:
    df = pd.DataFrame(st.session_state.health_log)
    st.markdown("### 📈 Weekly Health Trends")

    fig = go.Figure()
    for metric, color in [
        ("Heart Rate","#f472b6"),
        ("Blood Pressure","#38bdf8"),
        ("Sugar Level","#a78bfa")
    ]:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[metric],
            mode="lines+markers", name=metric,
            line=dict(color=color, width=2), marker=dict(size=7)
        ))
    fig.update_layout(
        plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
        hovermode="x unified", height=320,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, metric, color in zip(
        [c1, c2, c3],
        ["Heart Rate", "Blood Pressure", "Sugar Level"],
        ["#f472b6", "#38bdf8", "#a78bfa"]
    ):
        with col:
            bar = px.bar(df, x="Date", y=metric, title=metric,
                         color_discrete_sequence=[color])
            bar.update_layout(
                plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
                font=dict(color="#e2e8f0", size=11),
                height=200, margin=dict(l=5,r=5,t=30,b=5), showlegend=False
            )
            st.plotly_chart(bar, use_container_width=True)

    st.markdown("### 📌 Normal Ranges")
    ref = {
        "Metric":       ["Heart Rate", "Blood Pressure", "Sugar Level"],
        "Normal Range": ["60–100 bpm", "90–120 mmHg",   "70–100 mg/dL"],
        "Your Latest":  [
            f"{latest['Heart Rate']} bpm",
            f"{latest['Blood Pressure']} mmHg",
            f"{latest['Sugar Level']} mg/dL"
        ],
        "Status": [
            "✅ Normal" if 60 <= latest["Heart Rate"]     <= 100 else "⚠️ Consult doctor",
            "✅ Normal" if 90 <= latest["Blood Pressure"] <= 120 else "⚠️ Consult doctor",
            "✅ Normal" if 70 <= latest["Sugar Level"]    <= 100 else "⚠️ Consult doctor",
        ]
    }
    st.dataframe(pd.DataFrame(ref), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TAB 3: DATA LOG
# ----------------------------------------------------------
with tab3:
    st.markdown("### 📋 Complete Health Log")
    df_log = pd.DataFrame(st.session_state.health_log)
    st.dataframe(df_log, use_container_width=True, hide_index=True)

    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name=f"health_log_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    st.markdown("### 📊 Summary Statistics")
    st.dataframe(df_log.describe().round(2), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<center><small style='color:#475569'>"
    "🩺 AI Healthcare Monitor • Powered by Groq AI • "
    "⚕️ Only answers health & medical questions • "
    "⚠️ Not a substitute for professional medical advice"
    "</small></center>",
    unsafe_allow_html=True
)