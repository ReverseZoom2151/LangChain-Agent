import dotenv from "dotenv";
dotenv.config();

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import readline from 'readline';

async function setupAgent() {

    const searchTool = new TavilySearchResults();
    const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
    const rawDocs = await loader.load();
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const docs = await splitter.splitDocuments(rawDocs);
    
    const vectorstore = await MemoryVectorStore.fromDocuments(
        docs, 
        new OpenAIEmbeddings()
    );
    
    const retriever = vectorstore.asRetriever();
    
    const retrieverTool = createRetrieverTool(retriever, {
        name: "langsmith_search",
        description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    });
    
    const tools = [searchTool, retrieverTool];
    
    const llm = new ChatOpenAI({
        model: "gpt-4o",
        temperature: 0,
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
    
    const prompt = await pull("hwchase17/openai-functions-agent");
    
    const agent = await createOpenAIFunctionsAgent({
        llm,
        tools,
        prompt,
    });
    
    const agentExecutor = new AgentExecutor({
        agent,
        tools,
    });

    const messageHistory = new ChatMessageHistory();
    
    const agentWithChatHistory = new RunnableWithMessageHistory({
        runnable: agentExecutor,
        getMessageHistory: (_sessionId) => messageHistory,
        inputMessagesKey: "input",
        historyMessagesKey: "chat_history",
    });

    return agentWithChatHistory;

}

function getUserInput(prompt) {

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    return new Promise((resolve) => {
        rl.question(prompt, (answer) => {
            rl.close();
            resolve(answer);
        });
    });

}

async function main() {

    console.log("Setting up the agent...");
    const agent = await setupAgent();
    console.log("Agent is ready! You can start chatting. Type 'exit' to end the conversation.");
    const sessionId = "OperativeT"; 

    while (true) {

        const userInput = await getUserInput("You: ");
        
        if (userInput.toLowerCase() === 'exit') {
            console.log("Goodbye!");
            break;
        }

        const result = await agent.invoke(
            { input: userInput },
            { configurable: { sessionId } }
        );

        console.log("Agent:", result.output);

    }
}

main().catch(console.error);