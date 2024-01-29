import os
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain


# Load environment variables from .env file
load_dotenv()

modelPath = os.getenv('modelPath')

# print(modelPath)

repo_path = "storage/"

# Load
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()

# Chech how many docs in the file
# print(len(documents))


python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)


# Initialize callback manager and LlamaCpp language model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=modelPath,
    n_ctx=5000,
    n_gpu_layers=1,
    n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)




# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

question = "what does the print command output ?"
docs = texts

chain = load_qa_chain(llm,chain_type="stuff",prompt=QA_CHAIN_PROMPT)

result = chain.invoke({"input_documents":docs, "question":question},return_only_outputs=True)
print(result)

# # Use 'invoke' instead of '__call__'
# result = llm.invoke(
#     "Question: What is the Presidents 42 Name? Answer:"
# )




