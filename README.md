# Turkish Airlines Voice AI Assistant Demo

A real-time voice AI assistant that connects to Turkish Airlines services through the Model Context Protocol (MCP). This demo showcases how to build a conversational AI agent that can help users with flight information, booking management, travel planning, and member services using voice interaction.

## 🎯 What This Solves

This project demonstrates how to create a voice-enabled AI assistant that can:

- **Process real-time voice input** using speech-to-text
- **Connect to airline services** through MCP (Model Context Protocol)
- **Provide intelligent responses** about flights, bookings, and travel information
- **Respond with natural speech** using text-to-speech
- **Handle complex queries** like flight status, booking details, and member services

## 🚀 Features

- **🎤 Voice Recognition**: Real-time speech-to-text using Deepgram
- **🔊 Voice Synthesis**: Natural text-to-speech responses
- **✈️ Flight Services**: Check flight status, search flights, create booking links
- **📋 Booking Management**: Retrieve booking details, check-in information
- **🌍 Travel Planning**: Get city guides, baggage information, promotions
- **👤 Member Services**: Access Miles&Smiles information, upcoming flights, expiring miles
- **🔄 Real-time Communication**: WebSocket-based live interaction

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for MCP client)
- **Deepgram API Key** (for speech services)
- **OpenRouter API Key** (for AI language model)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd thy-mcp-voiceai-test
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Node.js Dependencies

```bash
npm install
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
OPENAI_API_KEY=your_openrouter_api_key_here
PORT=3000
```

#### Getting API Keys:

- **Deepgram**: Sign up at [deepgram.com](https://deepgram.com) for speech services
- **OpenRouter**: Get your key at [openrouter.ai](https://openrouter.ai) for AI language model access

## 🚀 Usage

### Start the Server

```bash
python server.py
```

### Access the Application

Open your browser and navigate to:

```
http://localhost:3000
```

### Using the Voice Assistant

1. **Click the microphone button** to start voice recording
2. **Speak your question** (examples provided in the UI)
3. **Wait for the AI response** (both text and audio)
4. **Click the mic again** to ask follow-up questions

## 💬 Example Queries

### Flight Information

- "What's the status of flight TK1 on 2024-12-25?"
- "Show me flights from Istanbul to London tomorrow"
- "Create a booking link for flights to New York"

### Travel Planning

- "Get me a city guide for Paris"
- "What are the baggage allowance for my booking?"
- "What are the current promotions for international flights?"

### Booking Management

- "Get my booking details for PNR ABC123"
- "Help me find booking information for check-in"

### Member Services

- "Show me my upcoming flights"
- "What's my Miles&Smiles member information?"
- "Show me my expiring miles"

## 🏗️ Architecture

The system consists of:

- **Frontend**: HTML/CSS/JavaScript with WebSocket communication
- **Backend**: Flask server with SocketIO for real-time communication
- **Speech Processing**: Deepgram for STT/TTS
- **AI Processing**: DeepSeek model via OpenRouter
- **Service Integration**: MCP client connecting to Turkish Airlines services

## 🔧 Configuration

### MCP Server Configuration

The application connects to Turkish Airlines MCP server at:

```
https://mcp.turkishtechlab.com/mcp
```

### Customization

- Modify `mcp_config` in `server.py` to connect to different MCP servers
- Update the language model in the OpenAI configuration
- Adjust Deepgram settings for different speech models

## 🐛 Troubleshooting

### Common Issues:

1. **Microphone not working**: Check browser permissions for microphone access
2. **No audio response**: Verify Deepgram API key is valid
3. **AI not responding**: Check OpenRouter API key and model availability
4. **Connection errors**: Ensure all dependencies are installed and ports are available

## 📄 License

This project is for demonstration purposes. Check individual service terms for API usage.

## 🤝 Contributing

This is a demo project. For production use, consider:

- Adding authentication and authorization
- Implementing rate limiting
- Adding comprehensive error handling
- Setting up proper logging and monitoring
