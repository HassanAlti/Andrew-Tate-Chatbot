import { Configuration, OpenAIApi } from "openai";
import dotenv from "dotenv";

dotenv.config({ path: ".env.local" });

const token = process.env.OPENAI_API_KEY;
const configuration = new Configuration({ apiKey: token });
const openai = new OpenAIApi(configuration);

export default async function addMessageToCache(his, userPrompt) {
  // Get the current chat history from the cache (if any)
  let chatHistory = [
    {
      role: "user",
      content: `
      Create a SINGLE standalone question. The question should be based on the New question plus the Chat history.
      
      If the question contains a pronoun without specfifying the subject, replace the pronoun with the subject of discussion.

      If the New question can stand on its own you should return the New question. 

      Return the question as if you are "Human" asking to "Andrew Tate", in the format: Human: New Question
      
      New question: ${userPrompt}, 
      
      Chat history: ${his}
        `,
    },
  ];

  // Generate the next system message using OpenAI API
  const completion = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: [...chatHistory],
  });

  // Add the system's message to the chat history
  const systemMessage = completion.data.choices[0].message.content;

  return systemMessage;
}
