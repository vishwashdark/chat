import json
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import boto3

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

# Handle user messages and invoke Claude
def bot_response(user_input):
    # Get or initialize chat history
    history = session.get("chat_history", [])
    
    # Properly format user message
    history.append({
        "role": "user", 
        "content": [
            {"type": "text", "text": user_input}
        ]
    })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "top_k": 250,
        "temperature": 1,
        "top_p": 0.999,
        "stop_sequences": [],
        "messages": history
    }

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())
        
        # Extract the assistant's message - this should be an array of content items
        assistant_content = result.get("content", [])
        
        # Add assistant response to history with proper structure
        history.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        # Store updated history
        session["chat_history"] = history
        
        # Extract text for display
        reply_text = ""
        for item in assistant_content:
            if item.get("type") == "text":
                reply_text += item.get("text", "")
        
        return reply_text

    except Exception as e:
        print(f"Claude invocation error: {e}")
        return f"Sorry, something went wrong while talking to Claude: {str(e)}"

# Routes
@app.route("/")
def home():
    # Initialize empty chat history when starting a new session
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    reply = bot_response(user_input)
    return jsonify({"response": reply})

@app.route("/reset", methods=["POST"])
def reset():
    # Clear chat history
    session["chat_history"] = []
    return jsonify({"status": "cleared"})


@app.route("/home")
def home_page():
    return render_template("home.html")

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True,port=5050)