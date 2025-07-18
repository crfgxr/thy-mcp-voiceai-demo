import os
import asyncio
import threading
import json
import base64
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from flask import Flask
from flask_socketio import SocketIO, emit
import requests
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from dotenv import load_dotenv

# MCP imports
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

import mcp_use

mcp_use.set_debug(1)


# Import prompts configuration
from prompts import (
    SYSTEM_PROMPT,
    ADDITIONAL_INSTRUCTIONS,
    MCP_SYSTEM_PROMPT,
    MCP_ADDITIONAL_INSTRUCTIONS,
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder="public", static_url_path="")
socketio_app = SocketIO(
    app, cors_allowed_origins="*", transports=["websocket"], async_mode="threading"
)

# Initialize clients
deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))


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
available_tools = []  # Store available MCP tools
mcp_client = None  # Persistent MCP client
mcp_agent = None  # Persistent MCP agent


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

# Session-only conversation history (no persistent memory)
conversation_history = []


async def initialize_mcp_client():
    """Initialize persistent MCP client and agent"""
    global mcp_client, mcp_agent
    try:
        print("Initializing MCP client...")
        mcp_client = MCPClient.from_dict(mcp_config)
        print("MCP client initialized successfully")

        # Create persistent agent
        print("Creating persistent MCP agent...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

        mcp_agent = MCPAgent(
            llm=llm,
            client=mcp_client,
            memory_enabled=False,
            system_prompt=SYSTEM_PROMPT,
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
        )
        print("Persistent MCP agent created successfully")

    except Exception as e:
        print(f"Error initializing MCP client: {e}")
        import traceback

        traceback.print_exc()
        mcp_client = None
        mcp_agent = None


async def discover_available_tools():
    """Discover available MCP tools on startup"""
    global available_tools
    try:
        if not mcp_agent:
            print("No MCP agent available - tools discovery skipped")
            available_tools = []
            return

        print("Testing MCP connection and extracting tools...")

        # Test if MCP is working using the persistent agent
        try:
            result = await mcp_agent.run("What tools do you have available?")
            print(f"MCP Agent test result: {result}")

            # Now extract the actual tools from the initialized agent
            available_tools = []

            # Method 1: Try to get tools from agent's adapter (most likely location)
            if hasattr(mcp_agent, "adapter") and hasattr(mcp_agent.adapter, "tools"):
                tools_list = mcp_agent.adapter.tools
                print(f"Found {len(tools_list)} tools in agent.adapter.tools")

                for tool in tools_list:
                    if hasattr(tool, "name") and hasattr(tool, "description"):
                        available_tools.append(
                            {"name": tool.name, "description": tool.description}
                        )
                    elif hasattr(tool, "name") and hasattr(tool, "func"):
                        # Some tools might have func instead of description
                        desc = getattr(tool, "description", f"Tool: {tool.name}")
                        available_tools.append({"name": tool.name, "description": desc})

            # Method 2: Try to get tools from agent's tools attribute
            elif hasattr(mcp_agent, "tools") and mcp_agent.tools:
                tools_list = mcp_agent.tools
                print(f"Found {len(tools_list)} tools in agent.tools")

                for tool in tools_list:
                    if hasattr(tool, "name") and hasattr(tool, "description"):
                        available_tools.append(
                            {"name": tool.name, "description": tool.description}
                        )

            # Method 3: Try to get tools from agent's internal structure
            elif hasattr(mcp_agent, "_tools") and mcp_agent._tools:
                tools_list = mcp_agent._tools
                print(f"Found {len(tools_list)} tools in agent._tools")

                for tool in tools_list:
                    if hasattr(tool, "name") and hasattr(tool, "description"):
                        available_tools.append(
                            {"name": tool.name, "description": tool.description}
                        )

            # If we still don't have tools, try to parse from the agent response
            if not available_tools:
                print(
                    "No tools found in agent attributes, trying to parse from response..."
                )
                # This is a fallback - if tools aren't accessible, we'll have to work without them
                # but at least we know MCP is working
                print(
                    "MCP connection verified but couldn't extract tools - will work in conversation mode"
                )

            print(f"MCP connection verified - extracted {len(available_tools)} tools")

        except Exception as e:
            print(f"MCP test failed: {e}")
            available_tools = []

        print(f"Discovered {len(available_tools)} available tools:")
        for tool in available_tools:
            print(f"  - {tool['name']}: {tool['description']}")

    except Exception as e:
        print(f"Error discovering tools: {e}")
        import traceback

        traceback.print_exc()
        available_tools = []
        print(f"No tools available - system will work in conversation-only mode")


def get_tools_context():
    """Get context about available tools for the LLM"""
    if not available_tools:
        return ""

    tools_context = "\n\nAvailable Turkish Airlines tools:\n"
    for tool in available_tools:
        name = (
            tool.get("name", "Unknown")
            if isinstance(tool, dict)
            else getattr(tool, "name", "Unknown")
        )
        description = (
            tool.get("description", "No description")
            if isinstance(tool, dict)
            else getattr(tool, "description", "No description")
        )
        tools_context += f"- {name}: {description}\n"

    tools_context += (
        "\nUSE TOOLS ONLY when user specifically requests Turkish Airlines services.\n"
    )
    return tools_context


def build_context_with_history_and_tools(current_transcript):
    """Build context with conversation history and available tools"""
    context = ""

    # Add current date context so LLM can calculate dates properly
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_date_formatted = now.strftime("%B %d, %Y")
    tomorrow_date = (now + timedelta(days=1)).strftime("%d-%m-%Y")

    context += f"CURRENT DATE CONTEXT:\n"
    context += f"Today is {current_date_formatted} ({current_date})\n"
    context += f"Tomorrow is {tomorrow_date}\n\n"

    # Add conversation history
    if conversation_history:
        context += "Previous conversation:\n"
        for exchange in conversation_history[-10:]:  # Only use last 10 exchanges
            context += f"User: {exchange['user']}\n"
            context += f"Assistant: {exchange['assistant']}\n\n"

    # Always add tools context - let the LLM decide when to use them
    tools_context = get_tools_context()
    if tools_context:
        context += tools_context + "\n"

    # Add current user message
    context += f"Current user message: {current_transcript}"

    return context


def build_context_with_history(current_transcript):
    """Build context with conversation history - kept for backward compatibility"""
    if not conversation_history:
        return current_transcript

    context = "Previous conversation:\n"
    for exchange in conversation_history[-10:]:  # Only use last 10 exchanges
        context += f"User: {exchange['user']}\n"
        context += f"Assistant: {exchange['assistant']}\n\n"

    context += f"Current user message: {current_transcript}"
    return context


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
    """Initialize Deepgram connection"""
    global socket_to_deepgram

    try:
        # Configure live transcription options
        options = LiveOptions(
            model="nova-2",  # nova-2
            version="latest",
            language="en-US",  # buraya multi de gelmis fakat sadece en ve es icin takip et
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

        # Create live transcription connection
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
    """Generate response using regular ChatOpenAI with intelligent tool calling"""
    global model_final_output_text, conversation_history
    model_final_output_text = ""

    try:
        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
        )

        # Build context with conversation history and tools
        context_with_history = build_context_with_history_and_tools(transcript)

        # Create messages for ChatOpenAI
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\n" + ADDITIONAL_INSTRUCTIONS,
            },
            {"role": "user", "content": context_with_history},
        ]

        # Get response from LLM
        response = await llm.ainvoke(messages)
        llm_response = response.content.strip()

        # Parse the response for TOOL_NEEDED flag
        tool_needed, actual_response = parse_llm_response(llm_response)

        if tool_needed:
            # If LLM indicates it needs to call a tool, first respond with "checking" message
            print("LLM requested tool calling, accessing Turkish Airlines systems...")

            # Send immediate "checking" response with audio
            checking_message = "I will be checking with the system, please wait."
            checking_audio = await deepgram_generated_audio(checking_message)

            # Send checking message to user immediately
            audio_base64 = (
                base64.b64encode(checking_audio).decode("utf-8")
                if checking_audio
                else None
            )
            socketio_app.emit(
                "bot-reply", {"text": checking_message, "audio": audio_base64}
            )

            # Now call MCP tools
            model_final_output_text = await call_mcp_tool(context_with_history)
        else:
            # Use the LLM's direct response
            model_final_output_text = actual_response

        # Update conversation history
        conversation_history.append(
            {
                "user": transcript,
                "assistant": model_final_output_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 10 exchanges to prevent context from getting too long
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        print(f"\nAssistant-> {model_final_output_text}")

    except Exception as e:
        print(f"Error in response generation: {e}")
        model_final_output_text = (
            "I'm sorry, I couldn't process that request at the moment."
        )


def parse_llm_response(llm_response):
    """Parse the LLM response to extract TOOL_NEEDED flag and actual response"""
    try:
        lines = llm_response.split("\n")
        tool_needed = False
        actual_response = llm_response  # Default to full response if parsing fails

        for line in lines:
            if line.strip().startswith("TOOL_NEEDED:"):
                tool_value = line.split(":", 1)[1].strip().lower()
                tool_needed = tool_value == "true"
            elif line.strip().startswith("RESPONSE:"):
                actual_response = line.split(":", 1)[1].strip()
                break

        return tool_needed, actual_response

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        # If parsing fails, assume no tool needed and use full response
        return False, llm_response


async def call_mcp_tool(context):
    """Call MCP tool using a completely fresh MCPClient to avoid session conflicts"""
    try:
        # Create a completely fresh client for this tool call
        print("Creating fresh MCP client and agent for tool calling...")

        fresh_client = MCPClient.from_dict(mcp_config)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
        )

        # Create a new agent with fresh client using MCP-specific prompts
        tool_agent = MCPAgent(
            llm=llm,
            client=fresh_client,
            max_steps=5,  # Increased steps to allow for retry attempts
            memory_enabled=False,
            system_prompt=MCP_SYSTEM_PROMPT,
            additional_instructions=MCP_ADDITIONAL_INSTRUCTIONS,
        )

        # Run the query with context
        print("Calling MCP tools with fresh session...")
        print(f"DEBUG: Context being sent to MCP agent: {context[:200]}...")
        result = await tool_agent.run(context)
        print(f"DEBUG: FULL MCP TOOL RESULT: {result}")
        return str(result)

    except Exception as e:
        print(f"Error calling MCP tool: {e}")
        import traceback

        traceback.print_exc()
        return "I'm sorry, I couldn't access the Turkish Airlines booking system right now. Please try again in a moment."


async def deepgram_generated_audio(text):
    """Generate audio using Deepgram TTS"""
    try:
        endpoint = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json={"text": text})

        if response.status_code == 200:
            return response.content
        else:
            print(f"Deepgram TTS error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error in Deepgram TTS: {e}")
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


def clean_booking_links(text):
    """Clean booking links by removing trailing punctuation"""
    # Remove punctuation immediately after URLs
    text = re.sub(r"(https?://[^\s]+)[.,!?;]+", r"\1", text)
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

        # Clean booking links by removing trailing punctuation
        bot_generated_text = clean_booking_links(bot_generated_text)
        model_final_output_text = bot_generated_text  # Update the global variable

        # Strip links from text before sending to TTS
        audio_text = strip_links_for_audio(bot_generated_text)

        # Generate audio using the stripped text
        bot_generated_audio = loop.run_until_complete(
            deepgram_generated_audio(audio_text)
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

    # Discover tools if not already done
    if not available_tools:
        asyncio.run(discover_available_tools())


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


def test_tool_decision_logic():
    """Test the tool decision logic with sample inputs"""
    print("\nTool decision logic removed - LLM now decides dynamically")
    return True


async def initialize_server():
    """Initialize server with tool discovery"""
    print("Initializing server...")

    # Initialize MCP client first
    await initialize_mcp_client()

    # Then discover available tools
    await discover_available_tools()

    # Test tool decision logic
    test_tool_decision_logic()

    print("Server initialization complete")


if __name__ == "__main__":
    print(f"Server starting on port: {PORT}")

    # Initialize tools on startup
    asyncio.run(initialize_server())

    start_keep_alive()
    socketio_app.run(app, host="0.0.0.0", port=PORT, debug=False)
    # v4
