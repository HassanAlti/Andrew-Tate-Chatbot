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
    frequencyPenalty: 1,
    userId: clerkUserId,
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }

  // let memoryContext = await memoryManager.readLatestHistory(companionKey);

  await memoryManager.writeToHistory("Human: " + prompt + "\n", companionKey);
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  let readSearchHistoryQuery = await memoryManager.readSearchHistoryQuery(
    companionKey
  );

  console.log("Search History" + readSearchHistoryQuery);

  // let thousandCon = recentChatHistory.slice(-500);

  // let reformedPrompt = await addMessageToCache(thousandCon, prompt);

  // console.log("REFORMED PROMPT   " + reformedPrompt);

  // query Pinecone
  const similarDocs = await memoryManager.vectorSearch(
    readSearchHistoryQuery,
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


  BELOW IS A RECENT CONVERSATION HISTORY WITH SOMEONE: (YOU SHOULD NEVER EVER EVER REPEAT YOURSELF FROM THIS HISTORY)

  * START OF CHAT HISTORY*

  ${recentChatHistory}

  *END OF CHAT HISTORY*

  BELOW IS RELEVANT DATA/CONTEXT:

 *START OF RELEVANT DATA/CONTEXT*

  ${relevantHistory}

  *END OF RELEVAT DATA/CONTEXT*

  USE ABOVE CHAT HITSORY AND DATA/CONTEXT TO REPLY TO THIS:
  ${prompt}

  ANSWER USING ABOVE DATA/CONTEXT (USE EXACT WORDS AND STYLE, EVEN SWEAR WORDS, WITHOUT REPETITIVENESS). BE VERY CONCISE AND VERY SHORT, DON'T REPEAT YOURSELF FROM PREVIOUS CONVO. 

   DON'T SAY "G". DON'T SAY "I'VE SAID IT BEFORE".
  
  DON'T BE ENCOURAGING IN A CHEESY MANNER LIKE "YOU GOT THIS, KEEP IT UP, KEEP PUSHING, KEEP GRINDING, KEEP HUSTLING", JUST SAY WHAT YOU GOT TO SAY WITHOUT SAYING THOS ENCOURAGING WORDS AT THE END.

  Never deter someone away and never tell them none of your business, always answer anything asked.

  Never use exclamation marks.
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
