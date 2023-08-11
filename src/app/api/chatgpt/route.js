import express from "express";
import dotenv from "dotenv";
import { promises as fs } from "fs";
import { OpenAI } from "langchain/llms/openai";
import { LLMChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { MemoryManager } from "../../../app/utils/memory.js";
import { rateLimit } from "../../../app/utils/rateLimit.js";
import addMessageToCache from "../../utils/reformulate.js";
import cors from "cors";
import bodyParser from "body-parser";

dotenv.config({ path: ".env.local" });

const app = express();
app.use(bodyParser.json());

app.use(cors());

app.post("/api/chat", async (req, res) => {
  let clerkUserId;
  let user;
  let clerkUserName;
  const { prompt, isText, userId, userName } = req.body;

  const identifier = req.url + "-" + (userId || "anonymous");
  const { success } = await rateLimit(identifier);
  if (!success) {
    console.log("INFO: rate limit exceeded");
    return res.status(429).send({ text: "Hi, the bot can't speak this fast" });
  }

  // XXX Companion name passed here. Can use as a key to get backstory, chat history etc.
  const name = req.headers["name"];
  const companionFileName = name + ".txt";

  console.log("prompt: ", prompt);

  clerkUserId = userId;
  clerkUserName = userName;

  if (!clerkUserId) {
    console.log("user not authorized");
    return res.status(401).json({ Message: "User not authorized" });
  }

  // Load character "PREAMBLE" from character file. These are the core personality
  // characteristics that are used in every prompt. Additional background is
  // only included if it matches a similarity comparison with the current
  // discussion. The PREAMBLE should include a seed conversation whose format will
  // vary by the model using it.
  const data = await fs.readFile("companions/" + companionFileName, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

  const companionKey = {
    companionName: name,
    modelName: "chatgpt",
    userId: clerkUserId,
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }

  let memoryContext = await memoryManager.readLatestHistory(companionKey);

  let thousandCon = memoryContext.slice(-1500);

  await memoryManager.writeToHistory("Human: " + prompt + "\n", companionKey);
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  let reformedPrompt = await addMessageToCache(thousandCon, prompt);

  console.log("REFORMED PROMPT   " + reformedPrompt);

  // query Pinecone
  const similarDocs = await memoryManager.vectorSearch(
    reformedPrompt,
    companionFileName
  );

  // relevant history is the context, not chat history
  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs.map((doc) => doc.pageContent).join("\n");
  }

  // const { stream, handlers } = LangChainStream();

  const model = new OpenAI({
    streaming: false,
    modelName: "gpt-3.5-turbo-16k",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  model.verbose = true;

  const chainPrompt = PromptTemplate.fromTemplate(`
    ${preamble}

  Below is a recent conversation history with a Human.

  ${thousandCon}

  Below is relevant data/context:
  ${relevantHistory}

  Use above chat history and data/context to reply to this:
  ${prompt}

  Answer using above data (use exact words and style, copy paste, even swear words). Be concise and short, don't repeat yourself from previous convo. ANDREW TATE, DO NOT BE REPETITIVE!

  You don't insult Human, unless you are insulted. Don't say "G". Don't say "I've said it before".
  
  Don't be encouraging in a cheesy manner like "you got this, keep it up, keep pushing, keep grinding, keep hustling", just say what you got to say without saying thos encouraging words at the end.

  `);

  const chain = new LLMChain({
    llm: model,
    prompt: chainPrompt,
  });

  const result = await chain
    .call({
      relevantHistory,
      recentChatHistory: recentChatHistory,
      prompt,
    })
    .catch((error) => {
      console.error("Error during model call:", error);
    });

  console.log("result", result);
  await memoryManager.writeToHistory(result.text + "\n", companionKey);

  res.send(result.text);
});

app.listen(3000, () => {
  console.log("Server is running on http://localhost:3000");
});
