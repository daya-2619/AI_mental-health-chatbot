import streamlit as st
import openai
from transformers import pipeline

openai.api_key = "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"  # Add your key here

MODEL_NAME = "gpt-4o-mini"  # you can use gpt-4o, gpt-4o-mini, or gpt-3.5-turbo


@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

emotion_model = load_emotion_model()
sentiment_model = load_sentiment_model()


def crisis_detector(text):
    danger = [
        "suicide", "kill myself", "end my life",
        "self harm", "hurt myself", "hopeless",
        "want to die", "can't go on"
    ]
    return any(w in text.lower() for w in danger)


if "memory" not in st.session_state:
    st.session_state.memory = []

def add_memory(user, bot):
    if len(st.session_state.memory) >= 6:
        st.session_state.memory.pop(0)
    st.session_state.memory.append(f"User: {user}\nBot: {bot}")


def generate_llm_response(prompt):
    completion = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content



def generate_response(user_input):

    # Crisis Detection
    if crisis_detector(user_input):
        return (
            "I'm really sorry you're feeling this way 💛\n\n"
            "Please reach out immediately to someone you trust, "
            "a counselor or your local helpline.\n\n"
            "You deserve care and support, and you are not alone."
        )

    # Emotion and Sentiment
    emotion = emotion_model(user_input)[0][0]["label"]
    sentiment = sentiment_model(user_input)[0]["label"]

    # Memory
    history = "\n".join(st.session_state.memory)

    prompt = f"""
You are a compassionate and emotionally intelligent mental health assistant.
Analyze and respond with empathy.

User Emotion: {emotion}
User Sentiment: {sentiment}

Conversation History:
{history}

User says: "{user_input}"

Please reply with:
- empathy
- emotional validation
- actionable advice
- gentle tone
- no medical diagnosis
"""

    bot_response = generate_llm_response(prompt)
    add_memory(user_input, bot_response)

    return bot_response


st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="💬")
st.title("💬 AI Mental Health Companion")
st.write("A supportive, emotion-aware mental wellness chatbot .")

user_input = st.text_input("How are you feeling today?", placeholder="You can share anything...")

if st.button("Send") and user_input.strip() != "":
    response = generate_response(user_input)
    st.write("### 🤖 Response:")
    st.write(response)

st.write("---")
st.write("### 🧠 Recent Memory")
for m in st.session_state.memory:
    st.write(m)
