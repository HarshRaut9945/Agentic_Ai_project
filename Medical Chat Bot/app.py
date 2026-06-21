from flask import Flask, render_template, request
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)

# ==========================
# Load Embeddings
# ==========================
print("Loading embeddings...")

embeddings = download_hugging_face_embeddings()

# ==========================
# Load FAISS
# ==========================
print("Loading FAISS index...")

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ==========================
# Gemini LLM
# ==========================
print("Loading Gemini model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

print("Medical Chatbot Ready!")

# ==========================
# Routes
# ==========================
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    user_message = request.form["msg"]

    # Retrieve relevant docs
    docs = retriever.invoke(user_message)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = f"""
{system_prompt}

Context:
{context}

Question:
{user_message}

Answer:
"""

    response = llm.invoke(final_prompt)

    return response.content


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )