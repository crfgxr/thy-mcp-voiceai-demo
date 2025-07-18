import os
import asyncio
import threading
import json
import base64
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from flask import Flask
from flask_socketio import SocketIO, emit
import requests
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from dotenv import load_dotenv
from openai import OpenAI

# MCP imports
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

import mcp_use

# mcp_use.set_debug(1)


# Import prompts configuration
from prompts import SYSTEM_PROMPT, ADDITIONAL_INSTRUCTIONS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder="public", static_url_path="")
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio_app = SocketIO(
    app, cors_allowed_origins="*", transports=["websocket"], async_mode="threading"
)

# Initialize clients
deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# MCP client configuration
mcp_config = {
    "mcpServers": {
        "turkish-airlines": {
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.turkishtechlab.com/mcp"],
        }
    }
}

# Global variables
socket_to_client = None
socket_to_deepgram = None
PORT = int(os.getenv("PORT", 3000))


# States
class STATES:
    AwaitingUtterance = "AwaitingUtterance"
    AwaitingBotReply = "AwaitingBotReply"


voicebot_state = STATES.AwaitingUtterance

# Transcript management
finalized_transcript = ""
unfinalized_transcript = ""
latest_finalized_word_end = float("inf")
latest_time_seen = 0.0
model_final_output_text = ""

# Memory file path
MEMORY_FILE = "conversation_memory.json"


def load_conversation_memory():
    """Load conversation history from file"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_conversation_memory(memory):
    """Save conversation history to file"""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Error saving memory: {e}")


def clean_old_memories(memory, days_to_keep=7):
    """Remove old conversation entries"""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    return [
        entry
        for entry in memory
        if datetime.fromisoformat(entry.get("timestamp", "1970-01-01")) > cutoff_date
    ]


# Initialize memory at startup
conversation_history = load_conversation_memory()


def reset_to_initial_state():
    """Reset all global variables to initial state"""
    global socket_to_client, socket_to_deepgram, voicebot_state
    global finalized_transcript, unfinalized_transcript
    global latest_finalized_word_end, latest_time_seen, model_final_output_text
    global conversation_history  # Add this line

    if socket_to_deepgram:
        try:
            socket_to_deepgram.finish()
        except:
            pass

    socket_to_client = None
    socket_to_deepgram = None
    voicebot_state = STATES.AwaitingUtterance
    finalized_transcript = ""
    unfinalized_transcript = ""
    latest_finalized_word_end = float("inf")
    latest_time_seen = 0.0
    model_final_output_text = ""
    conversation_history = []  # Reset conversation history


def change_voicebot_state(new_state):
    """Change voicebot state with logging"""
    global voicebot_state

    if new_state not in [STATES.AwaitingUtterance, STATES.AwaitingBotReply]:
        raise ValueError(f"Tried to change to invalid state: '{new_state}'")

    print(f"State change: {voicebot_state} -> {new_state}")
    voicebot_state = new_state


def init_dg_connection():
    """Initialize Deepgram connection using SDK 4.0+ API"""
    global socket_to_deepgram

    try:
        # Configure live transcription options
        options = LiveOptions(
            model="nova-2",  # nova-2
            version="latest",
            language="tr-TR",  # buraya multi de gelmis fakat sadece en ve es icin takip et
            smart_format=False,  # false yaptim
            punctuate=False,
            interim_results=True,  # bunu asla false yapma
            endpointing=False,  # 2000 di false yaptim
            # utterance_end_ms=1500,  # burayi duzenle tekrar
            vad_events=True,
            # sample_rate=48000,
            # encoding="linear16",
            # channels=2,
        )

        # Create live transcription connection - correct API for SDK 4.0+
        socket_to_deepgram = deepgram_client.listen.websocket.v("1")

        # Set up event handlers
        socket_to_deepgram.on(LiveTranscriptionEvents.Open, on_deepgram_open)
        socket_to_deepgram.on(LiveTranscriptionEvents.Close, on_deepgram_close)
        socket_to_deepgram.on(LiveTranscriptionEvents.Error, on_deepgram_error)
        socket_to_deepgram.on(
            LiveTranscriptionEvents.Transcript, on_deepgram_transcript
        )

        # Start the connection with options
        if socket_to_deepgram.start(options) == False:
            print("Failed to connect to Deepgram")
            return

        print("Deepgram connection initialized successfully")

    except Exception as e:
        print(f"Error initializing Deepgram connection: {e}")


def on_deepgram_open(self, *args, **kwargs):
    """Handle Deepgram connection open"""
    print("Opened websocket connection to Deepgram")


def on_deepgram_close(self, *args, **kwargs):
    """Handle Deepgram connection close"""
    print("Websocket to Deepgram closed")


def on_deepgram_error(self, *args, **kwargs):
    """Handle Deepgram errors"""
    error = args[0] if args else kwargs.get("error", "Unknown error")
    print(f"Error from Deepgram: {error}")


def on_deepgram_transcript(self, *args, **kwargs):
    """Handle Deepgram transcript results"""
    try:
        # Extract result from args or kwargs
        result = args[0] if args else kwargs.get("result", None)
        if result is None:
            print("No result found in transcript event")
            return

        # Handle both object and dictionary formats for SDK compatibility
        if hasattr(result, "channel"):
            # Object format (SDK 4.0+)
            channel = result.channel
            alternatives = getattr(channel, "alternatives", [])

            if alternatives:
                alternative = alternatives[0]
                transcript = getattr(alternative, "transcript", "")
                words = getattr(alternative, "words", [])

                start = getattr(result, "start", 0)
                duration = getattr(result, "duration", 0)
                is_final = getattr(result, "is_final", False)
                speech_final = getattr(result, "speech_final", False)
            else:
                return

        elif isinstance(result, dict) and result.get("type") == "Results":
            # Dictionary format (fallback)
            channel = result.get("channel", {})
            alternatives = channel.get("alternatives", [])

            if alternatives:
                alternative = alternatives[0]
                transcript = alternative.get("transcript", "")
                words = alternative.get("words", [])

                start = result.get("start", 0)
                duration = result.get("duration", 0)
                is_final = result.get("is_final", False)
                speech_final = result.get("speech_final", False)
            else:
                return
        else:
            print(f"Unexpected result format: {type(result)}")
            return

        """
        print("Deepgram result:")
        print(f"  is_final:     {is_final}")
        print(f"  speech_final: {speech_final}")
        print(f"  transcript:   {transcript}\n")
        """

        handle_dg_results(start, duration, is_final, speech_final, transcript, words)
    except Exception as e:
        print(f"Error processing transcript: {e}")


def handle_dg_results(start, duration, is_final, speech_final, transcript, words):
    """Handle Deepgram results based on current state"""
    global voicebot_state, finalized_transcript, unfinalized_transcript
    global latest_finalized_word_end, latest_time_seen

    if voicebot_state == STATES.AwaitingUtterance:
        # Emit transcript to client
        socketio_app.emit(
            "user-utterance-part", {"transcript": transcript, "isFinal": is_final}
        )

        update_transcript_state(transcript, is_final)
        update_silence_detection_state(start, duration, words, is_final)

        if finalized_transcript == "":
            return

        silence_detected = (
            unfinalized_transcript == ""
            and latest_time_seen - latest_finalized_word_end > 1.5
        )

        if silence_detected or speech_final:
            if speech_final:
                print("End of utterance reached due to endpoint")
            else:
                print("End of utterance reached due to silence detection")

            change_voicebot_state(STATES.AwaitingBotReply)
            socketio_app.emit("user-utterance-complete")

            # Process utterance in background thread
            threading.Thread(
                target=send_utterance_downstream, args=(finalized_transcript,)
            ).start()

    elif voicebot_state == STATES.AwaitingBotReply:
        # Discard user speech while bot is processing
        pass
    else:
        raise ValueError(f"Unexpected state: {voicebot_state}")


def update_transcript_state(transcript, is_final):
    """Update transcript state variables"""
    global finalized_transcript, unfinalized_transcript

    if is_final:
        unfinalized_transcript = ""
        if transcript != "":
            finalized_transcript = (finalized_transcript + " " + transcript).strip()
    else:
        unfinalized_transcript = transcript


def update_silence_detection_state(start, duration, words, is_final):
    """Update silence detection state variables"""
    global latest_finalized_word_end, latest_time_seen

    if is_final and words:
        last_word = words[-1]

        # Handle both object and dictionary formats for word data
        if hasattr(last_word, "word"):
            # Object format (SDK 4.0+)
            word_text = getattr(last_word, "word", "")
            word_end = getattr(last_word, "end", start + duration)
        else:
            # Dictionary format (fallback)
            word_text = last_word.get("word", "")
            word_end = last_word.get("end", start + duration)

        if len(word_text) > 1 and any(c.isdigit() for c in word_text):
            # Handle long number sequences
            latest_finalized_word_end = start + duration
        else:
            latest_finalized_word_end = word_end

    latest_time_seen = start + duration


async def mcp_generate_response(transcript):
    """Generate response using MCP server with manual conversation memory"""
    global model_final_output_text, conversation_history
    model_final_output_text = ""

    try:
        # Create MCPClient from configuration dictionary
        client = MCPClient.from_dict(mcp_config)

        # Create LLM
        llm = ChatOpenAI(
            model="o4-mini-2025-04-16",
            # model="gpt-4o-mini",
            # temperature=0.6,
        )

        # Create agent with memory disabled
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=5,
            memory_enabled=False,  # Disable buggy memory
            system_prompt=SYSTEM_PROMPT,
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            # verbose=True,
        )

        # Add conversation history to the transcript
        context_with_history = build_context_with_history(transcript)

        # Run the query with context
        result = await agent.run(context_with_history)

        model_final_output_text = str(result)

        # Update conversation history
        conversation_history.append(
            {
                "user": transcript,
                "assistant": model_final_output_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 5 exchanges to prevent context from getting too long
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]

        print(f"\nMCP assistant-> {model_final_output_text}")

    except Exception as e:
        print(f"Error in MCP generation: {e}")
        model_final_output_text = (
            "I'm sorry, I couldn't process that request through the MCP server."
        )


def build_context_with_history(current_transcript):
    """Build context with conversation history"""
    if not conversation_history:
        return current_transcript

    context = "Previous conversation:\n"
    for exchange in conversation_history[-3:]:  # Only use last 3 exchanges
        context += f"User: {exchange['user']}\n"
        context += f"Assistant: {exchange['assistant']}\n\n"

    context += f"Current user message: {current_transcript}"
    return context


async def openai_generated_audio(text):
    """Generate audio using OpenAI TTS"""
    try:
        start_time = time.time()

        response = openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            instructions="Speak and sound like a secretary.",
            input=text,
        )
        """
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",  # echo, fable, onyx, nova, shimmer
            speed=1.2,
            input=text,
        )
        """

        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"OpenAI's Voice latency: {latency:.0f}ms")

        # Return the audio content
        return response.content

    except Exception as e:
        print(f"Error in OpenAI TTS: {e}")
        return None


def strip_links_for_audio(text):
    """Remove or replace links in text for TTS while keeping readable content"""
    # Remove markdown links [text](url) and keep only the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove standalone URLs (http/https)
    text = re.sub(r"https?://[^\s]+", "", text)

    # Remove email addresses if needed
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def send_utterance_downstream(transcript):
    """Process utterance through the pipeline"""
    global model_final_output_text

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(mcp_generate_response(transcript))
        bot_generated_text = model_final_output_text

        # Strip links from text before sending to TTS
        audio_text = strip_links_for_audio(model_final_output_text)

        # Generate audio using the stripped text
        bot_generated_audio = loop.run_until_complete(
            openai_generated_audio(audio_text)
        )

        handle_bot_reply(bot_generated_text, bot_generated_audio)

    except Exception as e:
        print(f"Error in utterance processing: {e}")
    finally:
        loop.close()


def handle_bot_reply(text, audio):
    """Handle bot reply and reset state"""
    global voicebot_state, finalized_transcript, unfinalized_transcript
    global latest_finalized_word_end, latest_time_seen, model_final_output_text

    if voicebot_state != STATES.AwaitingBotReply:
        raise ValueError("Got bot reply in unexpected state")

    # Convert audio to base64 for transmission
    audio_base64 = base64.b64encode(audio).decode("utf-8") if audio else None

    socketio_app.emit("bot-reply", {"text": text, "audio": audio_base64})

    # Reset state
    finalized_transcript = ""
    unfinalized_transcript = ""
    latest_finalized_word_end = float("inf")
    latest_time_seen = 0.0
    model_final_output_text = ""
    change_voicebot_state(STATES.AwaitingUtterance)


def send_keep_alive_to_deepgram():
    """Send keep alive to Deepgram"""
    # Keep-alive might not be needed in SDK 4.0+
    # Disabled temporarily to avoid coroutine errors
    pass


# Start keep alive timer
def start_keep_alive():
    # Disabled keep-alive for SDK 4.0+ compatibility
    pass


# Socket.IO event handlers
@socketio_app.on("connect")
def handle_connect():
    """Handle client connection"""
    global socket_to_client
    print("Received websocket connection from client")

    socket_to_client = True
    reset_to_initial_state()
    init_dg_connection()


@socketio_app.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    print("User disconnected.")
    reset_to_initial_state()


@socketio_app.on("audio-from-user")
def handle_audio_from_user(data):
    """Handle audio data from user"""
    if socket_to_deepgram:
        try:
            # Skip if data length is 126 (as in original code)
            if len(data) != 126:
                socket_to_deepgram.send(data)
        except Exception as e:
            print(f"Error sending audio to Deepgram: {e}")


@app.route("/")
def index():
    """Serve the main HTML page"""
    return app.send_static_file("index.html")


if __name__ == "__main__":
    print(f"Server starting on port: {PORT}")
    start_keep_alive()
    socketio_app.run(app, host="0.0.0.0", port=PORT, debug=False)
