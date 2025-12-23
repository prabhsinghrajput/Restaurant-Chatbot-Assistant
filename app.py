import json
import random
import re
from flask import Flask, request, jsonify, render_template_string, session
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import LancasterStemmer
import nltk
import os

nltk.download('punkt')

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key-change-this")

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

patterns, labels = [], []
tag_to_responses, tag_context_set, tag_context_filter = {}, {}, {}

for item in intents:
    tag = item["tag"]
    tag_to_responses[tag] = item.get("responses", [])
    if item.get("context_set"): tag_context_set[tag] = item["context_set"]
    if item.get("context_filter"): tag_context_filter[tag] = item["context_filter"]
    for p in item.get("patterns", []):
        patterns.append(p)
        labels.append(tag)

stemmer = LancasterStemmer()
def tokenize_and_stem(s):
    tokens = nltk.word_tokenize(s)
    return " ".join(stemmer.stem(t.lower()) for t in tokens if re.match(r'\w', t))

X = [tokenize_and_stem(p) for p in patterns]
le = LabelEncoder()
y = le.fit_transform(labels)

clf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("nb", MultinomialNB())
])
clf_pipeline.fit(X, y)

MIN_PROB = 0.05

def classify_and_respond(message, session_id):
    user_context = session.get(session_id, None)
    processed = tokenize_and_stem(message)
    probs = clf_pipeline.predict_proba([processed])[0]
    best_idx = probs.argmax()
    best_tag = le.inverse_transform([best_idx])[0]
    best_prob = probs[best_idx]

    if best_prob < MIN_PROB:
        return {"response": "😅 Sorry, I didn’t quite catch that. Could you rephrase?"}

    required_context = tag_context_filter.get(best_tag)
    if required_context and user_context != required_context:
        return {"response": "🤔 Hmm, I might need a bit more context. Can you elaborate?"}

    if best_tag in tag_context_set:
        session[session_id] = tag_context_set[best_tag]

    responses = tag_to_responses.get(best_tag, [])
    return {"response": random.choice(responses) if responses else "Okay."}


# --- ✨ Interactive Chat UI (HTML/CSS/JS) ---
CHAT_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Restaurant Chatbot Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body {
    margin: 0;
    font-family: "Inter", system-ui, sans-serif;
    background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    overflow-y: auto;
    position: relative;
}

/* Floating emoji background */
.food-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
}
.food-emoji {
    position: absolute;
    font-size: 4rem;
    opacity: 0.12;
    animation: floatEmoji 8s ease-in-out infinite;
}
@keyframes floatEmoji {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-25px) rotate(10deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

/* Chat box */
#chat {
    width: 420px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.25);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: fadeIn 0.8s ease-in;
}
header {
    background: #0b74ff;
    color: white;
    text-align: center;
    padding: 16px;
    font-size: 18px;
    font-weight: 600;
}

/* Messages */
#messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.msg {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    animation: fadeInUp 0.35s ease;
}
.user {
    align-self: flex-end;
    background: #007bff;
    color: white;
    border-bottom-right-radius: 4px;
}
.bot {
    align-self: flex-start;
    background: rgba(255,255,255,0.8);
    color: #222;
    border-bottom-left-radius: 4px;
}
.avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
}
.bot-avatar { background: #0b74ff; }
.user-avatar { background: #ffb400; }

/* Buttons */
#button-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    padding: 10px;
    background: rgba(255,255,255,0.9);
    border-top: 1px solid #eee;
}
#button-container button {
    background: #0b74ff;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 8px 14px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.3s;
}
#button-container button:hover {
    background: #0959c7;
    transform: scale(1.05);
}

/* Typing */
.typing {
    font-size: 12px;
    color: #777;
    font-style: italic;
    text-align: left;
    padding-left: 34px;
}

/* Animations */
@keyframes fadeInUp {
    from { transform: translateY(10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.98); }
    to { opacity: 1; transform: scale(1); }
}
</style>
</head>
<body>
<div class="food-bg">
    <div class="food-emoji" style="top:10%; left:15%;">🍔</div>
    <div class="food-emoji" style="top:20%; left:70%;">🍕</div>
    <div class="food-emoji" style="top:65%; left:25%;">🍣</div>
    <div class="food-emoji" style="top:75%; left:80%;">☕</div>
    <div class="food-emoji" style="top:40%; left:5%;">🍟</div>
    <div class="food-emoji" style="top:50%; left:60%;">🍜</div>
    <div class="food-emoji" style="top:85%; left:35%;">🍰</div>
    <div class="food-emoji" style="top:30%; left:85%;">🥗</div>
    <div class="food-emoji" style="top:15%; left:45%;">🥐</div>
</div>

<div id="chat">
  <header>💬 Restaurant Chatbot Assistant</header>
  <div id="messages">
    <div class="msg bot"><span class="avatar bot-avatar"></span>Hello! 👋 Welcome to our restaurant. What would you like to know?</div>
  </div>
  <form id="chat-form">
    <div id="button-container"></div>
  </form>
</div>

<script>
const mainMenu = [
  { text: "🍽️ Menu", query: "menu" },
  { text: "🚗 Delivery", query: "delivery" },
  { text: "🎁 Offers", query: "offers" },
  { text: "🕘 Hours", query: "hours" },
  { text: "📍 Location", query: "location" },
  { text: "📞 Reservation", query: "reservation" },
  { text: "💳 Payment", query: "payments" },
  { text: "💬 Feedback", query: "feedback" }
];

const menuOptions = [
  { text: "🍛 Today’s Special", query: "What's today's menu?" },
  { text: "🥗 Veg Options", query: "Do you have vegetarian options?" },
  { text: "🍗 Non-Veg Dishes", query: "What kind of food do you serve?" },
  { text: "⬅️ Back", query: "back" }
];

const deliveryOptions = [
  { text: "🚗 Home Delivery", query: "Do you offer home delivery?" },
  { text: "📦 Swiggy/Zomato", query: "Do you deliver through Zomato or Swiggy?" },
  { text: "🚶 Takeaway", query: "Do you provide takeaway or delivery?" },
  { text: "⬅️ Back", query: "back" }
];

const offerOptions = [
  { text: "💰 Discounts", query: "Any discounts today?" },
  { text: "🍰 Combos", query: "Do you have combo deals?" },
  { text: "⬅️ Back", query: "back" }
];

function loadButtons(set) {
  const container = document.getElementById('button-container');
  container.innerHTML = '';
  set.forEach(opt => {
    const btn = document.createElement('button');
    btn.type = "button";
    btn.textContent = opt.text;
    btn.onclick = () => handleOptionClick(opt.query);
    container.appendChild(btn);
  });
}

async function handleOptionClick(query) {
  if (query === "back") {
    loadButtons(mainMenu);
    return;
  }

  // submenu routing
  if (query === "menu") return loadButtons(menuOptions);
  if (query === "delivery") return loadButtons(deliveryOptions);
  if (query === "offers") return loadButtons(offerOptions);

  appendMessage(query, 'user');
  scrollToBottom();

  const typingEl = showTyping();

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: query })
    });
    const data = await resp.json();
    typingEl.remove();
    appendMessage(data.response, 'bot');
  } catch (err) {
    typingEl.remove();
    appendMessage("⚠️ Network error. Please try again.", 'bot');
  }

  scrollToBottom();
}

function appendMessage(text, who) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
  const avatar = document.createElement('span');
  avatar.className = 'avatar ' + (who === 'user' ? 'user-avatar' : 'bot-avatar');
  div.prepend(avatar);
  div.append(text);
  container.appendChild(div);
}

function showTyping() {
  const container = document.getElementById('messages');
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Bot is typing...';
  container.appendChild(typing);
  scrollToBottom();
  return typing;
}

function scrollToBottom() {
  const container = document.getElementById('messages');
  container.scrollTop = container.scrollHeight;
}

window.onload = () => loadButtons(mainMenu);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(CHAT_HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"response": "Please send a message."})
    session_id = "user_context"
    return jsonify(classify_and_respond(msg, session_id))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
