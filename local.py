import os
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Load environment variables from .env file
load_dotenv()

modelPath = os.getenv('modelPath')

print(modelPath)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=modelPath,
    n_ctx=5000,
    n_gpu_layers=1,
    n_batch=512,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)


# Use 'invoke' instead of '__call__'
# llm(
#     "Question: What is the Presidents 42 Name? Answer:"
# )