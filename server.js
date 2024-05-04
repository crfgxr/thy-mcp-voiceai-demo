import "log-timestamp";
import "dotenv/config";
import path from "path";
import express from "express";
import { Deepgram, createClient, LiveTranscriptionEvents } from "@deepgram/sdk";
import { fileURLToPath } from "url";
import { createServer } from "http";
import { Server } from "socket.io";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import OpenAI from "openai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const openai = new OpenAI(process.env.OPENAI_API_KEY);
const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

let socketToClient = null;
let socketToDeepgram = null;

const PORT = process.env.PORT || 3000;

const STATES = {
  AwaitingUtterance: "AwaitingUtterance",
  AwaitingBotReply: "AwaitingBotReply",
};
let voicebotState = STATES.AwaitingUtterance;

/** The concatenated `is_final=true` results that comprise the current utterance. */
let finalizedTranscript = "";

/** The most recent `is_final=false` result for which we have not yet seen an `is_final=true` */
let unfinalizedTranscript = "";

/**
 * The timestamp in seconds that the last finalized word ended (or `Infinity` if there have been no
 * finalized words in the current utterance)
 */
let latestFinalizedWordEnd = Infinity;

/** The latest timestamp that we've seen included in a result (also known as the transcript cursor) */
let latestTimeSeen = 0.0;

let modelFinalOutputText = "";

function resetToInitialState() {
  if (socketToClient) {
    socketToClient.removeAllListeners();
  }
  if (socketToDeepgram) {
    socketToDeepgram.removeAllListeners();
  }
  socketToClient = null;
  socketToDeepgram = null;
  voicebotState = STATES.AwaitingUtterance;
  finalizedTranscript = "";
  unfinalizedTranscript = "";
  latestFinalizedWordEnd = Infinity;
  latestTimeSeen = 0.0;
  modelFinalOutputText = "";
}

function changeVoicebotState(newState) {
  if (!Object.values(STATES).includes(newState)) {
    throw new Error(`Tried to change to invalid state: '${newState}'`);
  }

  console.log(`State change: ${voicebotState} -> ${newState}`);

  voicebotState = newState;
}

function handleClientConnection(conn) {
  console.log(`Received websocket connection from client`);

  resetToInitialState();
  initDgConnection();
  socketToClient = conn;
  socketToClient.on("audio-from-user", handleAudioFromUser);
}

//changed to v3
function initDgConnection() {
  socketToDeepgram = openSocketToDeepgram();

  socketToDeepgram.on(LiveTranscriptionEvents.Open, () => {
    console.log("Opened websocket connection to Deepgram");
  });

  socketToDeepgram.on(LiveTranscriptionEvents.Close, (msg) => {
    console.log(
      `Websocket to Deepgram closed. Code: ${msg.code}, Reason: '${msg.reason}'`
    );
  });

  socketToDeepgram.on(LiveTranscriptionEvents.Error, (msg) => {
    console.log(`Error from Deepgram: ${JSON.stringify(msg)}`);
  });

  socketToDeepgram.on(LiveTranscriptionEvents.Transcript, (message) => {
    if (message.type === "Results") {
      let start = message.start;
      let duration = message.duration;
      let isFinal = message.is_final;
      let speechFinal = message.speech_final;
      let transcript = message.channel.alternatives[0].transcript;
      let words = message.channel.alternatives[0].words;

      console.log("Deepgram result:");
      console.log("  is_final:    ", isFinal);
      console.log("  speech_final:", speechFinal);
      console.log("  transcript:  ", transcript, "\n");

      handleDgResults(start, duration, isFinal, speechFinal, transcript, words);
    }
  });
}

//DgResults are the results from User
function handleDgResults(
  start,
  duration,
  isFinal,
  speechFinal,
  transcript,
  words
) {
  switch (voicebotState) {
    case STATES.AwaitingUtterance:
      // Give the transcript to the client for (optional) display
      socketToClient.emit("user-utterance-part", { transcript, isFinal });

      updateTranscriptState(transcript, isFinal);
      updateSilenceDetectionState(start, duration, words, isFinal);

      if (finalizedTranscript === "") {
        return;
      }

      let silenceDetected =
        unfinalizedTranscript === "" &&
        latestTimeSeen - latestFinalizedWordEnd > 1.25;

      if (silenceDetected || speechFinal) {
        if (speechFinal) {
          console.log("End of utterance reached due to endpoint");
        } else {
          console.log("End of utterance reached due to silence detection");
        }

        changeVoicebotState(STATES.AwaitingBotReply);
        socketToClient.emit("user-utterance-complete");
        sendUtteranceDownstream(finalizedTranscript);
      }

      break;
    case STATES.AwaitingBotReply:
      // Discard user speech since the bot is already processing a complete user utterance. Note
      // that more sophisticated approaches are possible. For example, we could analyze the
      // transcript, and if we conclude that the user is continuing their utterance, we could then
      // cancel Dialogflow processing and wait for a new complete utterance.
      break;
    default:
      throw new Error("Unexpected state: " + voicebotState);
  }
}

/** Updates `finalizedTranscript` and `unfinalizedTranscript` in light of a new result */
function updateTranscriptState(transcript, isFinal) {
  if (isFinal) {
    unfinalizedTranscript = "";
    if (transcript !== "") {
      finalizedTranscript = (finalizedTranscript + " " + transcript).trim();
    }
  } else {
    unfinalizedTranscript = transcript;
  }
}

/** Updates `latestFinalizedWordEnd` and `latestTimeSeen` in light of a new result */
function updateSilenceDetectionState(start, duration, words, isFinal) {
  if (isFinal && words.length > 0) {
    let lastWord = words.at(-1);

    if (lastWord.word.length > 1 && /\d/.test(lastWord.word)) {
      // Here we address a subtlety of the nova general model. The model assumes words cannot be
      // longer than 0.5 seconds. Essentially:
      //
      // `word_end(n) = min(word_start(n+1), word_start(n) + 0.5 sec)`
      //
      // This assumption is usually fine, but it breaks down when the user speaks a long string of
      // numbers and letters, as these are often grouped into a single word which takes far longer
      // than 0.5 seconds to pronounce. Therefore, if the last word is a string involving number(s),
      // we play it safe and consider the word to have ended all the way at the end of the result.

      latestFinalizedWordEnd = start + duration;
    } else {
      latestFinalizedWordEnd = lastWord.end;
    }
  }
  latestTimeSeen = start + duration;
}

//speech to text transcriber model configuration
function openSocketToDeepgram() {
  const dgConnection = deepgram.listen.live({
    model: "nova-2",
    language: "tr", //"en-US" "tr"
    smart_format: true,
    interim_results: true,
    endpointing: 500, //adjustable
    no_delay: true,
    utterance_end_ms: 1000, //adjustable
  });
  return dgConnection;
}

//getting finalizedTranscript from Deepgram
async function openAIGenerateResponse(finalizedTranscript) {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-3.5-turbo-0125",
    temperature: 0.2, //adjustable
    maxTokens: 100, //adjustable
    cache: true,
  });

  const messages = [
    new SystemMessage(
      "You are a helpful assistant, your name is Ayla. Be conversational, give short answers, open conversations."
    ),
    new HumanMessage(finalizedTranscript),
  ];

  /* burasi calisiyor
  const controller = new AbortController();
  if (finalizedTranscript.includes("Dur.")) {
    console.log("--------Aborted-------");
    return controller.abort();
  }
  */
  const stream = await model.stream(
    messages
    // { signal: controller.signal }
  );
  for await (const chunk of stream) {
    modelFinalOutputText += chunk.content;
    console.log("\nassistant->", modelFinalOutputText);
  }
}

async function openAIGenereateAudio(modelFinalOutputText) {
  try {
    let latency = 0;
    let startTime = Date.now(); // Store the current time
    const response = await openai.audio.speech.create({
      model: "tts-1",
      voice: "alloy", //echo, fable, onyx, nova, shimmer
      input: modelFinalOutputText,
    });

    /* for app
    const buffer = Buffer.from(await response.arrayBuffer());

    // Convert the Buffer to a base64 string
    const base64 = buffer.toString("base64");
    socketToClient.emit("audioBuffer", { buffer: base64, latency: latency });
    */

    latency = Date.now() - startTime; // Calculate the latency
    console.log("OpenAI's Voice latency:", latency);
    return await response.arrayBuffer(); //for web
  } catch (error) {
    console.log("Error in handleStreamVoice:", error);
  }
}

async function deepgramGenereateAudiov1(modelFinalOutputText) {
  const endpoint = "https://api.deepgram.com/v1/speak?model=aura-asteria-en";
  const apiKey = process.env.DEEPGRAM_API_KEY;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Token ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ text: modelFinalOutputText }),
  });
  return await response.arrayBuffer();
}

//pipeline
async function sendUtteranceDownstream(finalizedTranscript) {
  await openAIGenerateResponse(finalizedTranscript);
  let botGenereatedText = modelFinalOutputText;
  let botGenereatedAudio = await openAIGenereateAudio(modelFinalOutputText);
  //let botGenereatedAudio = await deepgramGenereateAudiov1(modelFinalOutputText);
  await handleBotReply(botGenereatedText, botGenereatedAudio);
}

function handleBotReply(text, audio) {
  if (voicebotState !== STATES.AwaitingBotReply) {
    throw new Error("Got bot reply in unexpected state");
  }
  socketToClient.emit("bot-reply", { text, audio });

  finalizedTranscript = "";
  unfinalizedTranscript = "";
  latestFinalizedWordEnd = Infinity;
  latestTimeSeen = 0;
  modelFinalOutputText = "";
  changeVoicebotState(STATES.AwaitingUtterance);
}

function handleAudioFromUser(event) {
  if (socketToDeepgram && socketToDeepgram.getReadyState() === 1) {
    if (event.length !== 126) {
      socketToDeepgram.send(event);
    }
  }
}

function sendKeepAliveToDeepgram() {
  if (socketToDeepgram && socketToDeepgram.getReadyState() === 1) {
    socketToDeepgram.send(
      JSON.stringify({
        type: "KeepAlive",
      })
    );
    console.log("Sent keep alive to Deepgram");
  }
}

setInterval(sendKeepAliveToDeepgram, 8000);

const app = express();
app.use(express.static("public"));
app.get("/", function (_req, res) {
  res.sendFile(__dirname + "/index.html");
});

const httpServer = createServer(app);

new Server(httpServer, {
  transports: "websocket",
  cors: {},
})
  .on("connection", handleClientConnection)
  .on("disconnect", () => console.log("User disconnected."));

httpServer.listen(PORT);

console.log("Server listening on port:", PORT);
