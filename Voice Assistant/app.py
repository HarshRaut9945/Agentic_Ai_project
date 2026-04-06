# Aiva (AI + Voice Assistant)
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

# ✅ Correct LLM init
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

conversation_history = []

def voice_assistance(user_input):
    global conversation_history

    try:
        prompt = f"""
        You are a smart AI voice assistant.
        Answer clearly and concisely.

        User: {user_input}
        """

        # ✅ Correct method
        response = llm.invoke(prompt)

        ai_response = response.content

        conversation_history.append({
            "user": user_input,
            "ai": ai_response
        })

        return ai_response

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_voice", methods=["POST"])
def process_voice():
    data = request.get_json()

    user_input = data.get("user_input")

    if not user_input:
        return jsonify({"error": "No input received"}), 400

    response = voice_assistance(user_input)

    return jsonify({
        "response": response,
        "conversation_history": conversation_history
    })


if __name__ == "__main__":
    app.run(debug=True)