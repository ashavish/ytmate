"""
installed : langchain, openai, tiktoken, faiss, fastapi,
uvicorn[standard]
"""
import openai
from fastapi import status
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import LLMChain, OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain import PromptTemplate
import tiktoken
import math
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document as LangchainDocument
from src.prompts import (
    transcript_summary_template,
    comment_summary_template,
    transcript_query_format,
)
from src.prompts import (
    intent_prompt_template,
    comment_summary_qa_template,
    transcript_summary_qa_template,
)
from src.settings import (
    MODEL_NAME_TRANSCRIPT,
    MAX_TOKENS_TRANSCRIPT,
    MAX_TOKENS_ALLOWED,
    MODEL_NAME_QA,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_index.node_parser import LangchainNodeParser
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index import Document
from llama_index.node_parser import SentenceSplitter
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
import urllib.error
from src import LOGGER
from src.utils import download_file, upload_file, upload_folder
from src.settings import BUCKET_NAME, LOCAL_DOWNLOAD_FOLDER
import json
from llama_index import Prompt
from src.utils import download_source_file_from_s3

# # dimensions of text-ada-embedding-002
d = 1536
# faiss_index = faiss.IndexFlatL2(d)


def calculate_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_documents(input_sentence, single_prompt, chunk_size):
    if single_prompt == 1:
        docs = [LangchainDocument(page_content=input_sentence)]
    else:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size)
        texts = text_splitter.split_text(input_sentence)
        docs = [LangchainDocument(page_content=t) for t in texts]
    return docs


def check_single_prompt(input_sentence, task, prompt_template):
    model_name = MODEL_NAME_TRANSCRIPT
    max_completion_tokens = MAX_TOKENS_TRANSCRIPT
    max_tokens_allowed = MAX_TOKENS_ALLOWED
    prompt_token_length = calculate_tokens(prompt_template, encoding_name="gpt2")
    input_text_prompt_token_length = calculate_tokens(
        input_sentence, encoding_name="gpt2"
    )
    permissible_chunk_size = math.floor(
        max_tokens_allowed - prompt_token_length - max_completion_tokens
    )
    if (
        prompt_token_length + input_text_prompt_token_length + max_completion_tokens
        < max_tokens_allowed
    ):
        return 1, permissible_chunk_size
    return 0, permissible_chunk_size


def get_summarize_chain(task, llm_model, input_sentence, prompt_template, prompt_text):
    flag_single_prompt, permissible_chunk_size = check_single_prompt(
        input_sentence, task, prompt_text
    )
    if flag_single_prompt == 1:
        llm_chain = load_summarize_chain(
            llm_model, chain_type="stuff", prompt=prompt_template
        )
    else:
        llm_chain = load_summarize_chain(
            llm_model,
            chain_type="map_reduce",
            map_prompt=prompt_template,
            combine_prompt=prompt_template,
        )
    return llm_chain, flag_single_prompt, permissible_chunk_size


def get_number_of_nodes(index):
    x = index.storage_context.docstore.to_dict()
    n = len(x["docstore/data"].keys())
    return n


def get_intent(question):
    llm_model = ChatOpenAI(temperature=0, model_name=MODEL_NAME_QA)
    prompt_template = PromptTemplate(
        input_variables=["query"], template=intent_prompt_template
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_model, verbose=False)
    llm_result = llm_chain.predict(query=question)
    return llm_result


def get_summary(text_json, source, video_id, force_train=False):
    try:
        # Return summary if already present
        if os.path.exists(f"./storage/{video_id}/{source}_summary.txt") and (
            force_train == False
        ):
            f = open(f"./storage/{video_id}/{source}_summary.txt", "r")
            summary = f.read()
            LOGGER.info("Getting summary from cache")
            return summary
        # check if json file needs to be downloaded
        if len(text_json) == 0:
            fpath = download_source_file_from_s3(video_id, source)
            text_json = json.load(open(fpath, "r"))
        if source == "video":
            prompt_text = transcript_summary_template
            text = " ".join(list(text_json["transcriptData"].values()))
        elif source == "comments":
            prompt_text = comment_summary_template
            text = "\n".join(
                [
                    f"{each['userName']}-{each['commentTime']}-{each['commentText']}"
                    for each in text_json
                ]
            )
        else:
            LOGGER.error(f"Source {source} not supported")
            raise status.HTTP_400_BAD_REQUEST

        llm_model = ChatOpenAI(temperature=0, model_name=MODEL_NAME_TRANSCRIPT)
        prompt_template = PromptTemplate(input_variables=["text"], template=prompt_text)

        llm_chain, flag_single_prompt, permissible_chunk_size = get_summarize_chain(
            "Summarize", llm_model, text, prompt_template, prompt_text
        )
        docs = get_documents(
            input_sentence=text,
            single_prompt=flag_single_prompt,
            chunk_size=10000,
        )
        llm_result = llm_chain.run(docs)
        # output = chatgpt_chain.predict(human_input=text)
        isExist = os.path.exists(f"./storage/{video_id}")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(f"./storage/{video_id}")
            LOGGER.info("The new directory is created!")
        f = open(f"./storage/{video_id}/{source}_summary.txt", "w")
        f.write(llm_result)
        f.close()
        # Upload summary to AWS s3
        upload_file(
            BUCKET_NAME,
            f"storage/{video_id}/{source}_summary.txt",
            f"storage/{video_id}/{source}_summary.txt",
        )
    except Exception as e:
        LOGGER.error(f"Error encountered : {e}")
        raise
    return llm_result


def train(text_json, source, video_id, force_train=False):
    try:
        # check if storage already exists
        if (not os.path.exists(f"./storage/{video_id}/{source}")) or (
            force_train == True
        ):
            # check if json file needs to be downloaded
            if len(text_json) == 0:
                fpath = download_source_file_from_s3(video_id, source)
                text_json = json.load(open(fpath, "r"))
            if source == "video":
                # text = " ".join(list(text_json['transcriptData'].values()))
                # Get text along with timestamp
                times = text_json["transcriptData"].keys()
                utterances = text_json["transcriptData"].values()
                text = ""
                for i, j in zip(times, utterances):
                    text = text + " (Time:" + i + ") " + j
            elif source == "comments":
                text = "\n".join(
                    [
                        f"{each['userName']}-{each['commentTime']}-{each['commentText']}"
                        for each in text_json
                    ]
                )
            elif source == "images":
                times = text_json.keys()
                captions = text_json.values()
                text = ""
                for i, j in zip(times, captions):
                    text = text + " (Time:" + i + ") " + j
            else:
                LOGGER.error(f"Source {source} not supported")
                raise status.HTTP_400_BAD_REQUEST
            # load the documents and create the index
            # documents = SimpleDirectoryReader("data").load_data()
            documents = [Document(text=text)]
            parser = LangchainNodeParser(
                RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    is_separator_regex=False,
                )
            )

            # parser = SentenceSplitter()
            nodes = parser.get_nodes_from_documents(documents)

            # Important to declare and use the faiss_index inside, so that we dont get irrevant nodes in query which then not found in storage context.
            # Refer error - https://github.com/run-llama/llama_index/issues/7684
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)

            # index = VectorStoreIndex.from_documents(
            #     documents, storage_context=storage_context
            # )

            # store it for later
            index.storage_context.persist(persist_dir=f"./storage/{video_id}/{source}")

            # Bug fix for Llama screwing up.
            os.rename(
                f"./storage/{video_id}/{source}/default__vector_store.json",
                f"./storage/{video_id}/{source}/vector_store.json",
            )

            # Upload index to s3
            upload_folder(
                BUCKET_NAME,
                f"storage/{video_id}/{source}",
                f"storage/{video_id}/{source}",
            )
        else:
            # load the existing index
            # vector_store = FaissVectorStore.from_persist_dir(f"./storage/{video_id}/{source}")
            # storage_context = StorageContext.from_defaults(
            #     vector_store=vector_store, persist_dir=f"./storage/{video_id}/{source}"
            # )
            # index = load_index_from_storage(storage_context)
            LOGGER.info("Index already exists")
    except Exception as e:
        LOGGER.error(f"Error encountered : {e}")
        raise
    return status.HTTP_200_OK


def get_answer_from_summary(question, source, video_id):
    summary = get_summary(
        text_json={}, source=source, video_id=video_id, force_train=False
    )
    llm_model = ChatOpenAI(temperature=0, model_name=MODEL_NAME_QA)
    if source == "video":
        template = transcript_summary_qa_template
    else:
        template = comment_summary_qa_template

    prompt_template = PromptTemplate(
        input_variables=["query", "summary"], template=template
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_model, verbose=False)
    llm_result = llm_chain.predict(query=question, summary=summary)
    return llm_result


def get_answer(question, source, video_id, k=0, similarity_cutoff=0.1):
    LOGGER.info(
        f"Request received with question:{question}, source:{source}, video: {video_id}"
    )
    if not os.path.exists(f"./storage/{video_id}/{source}"):
        LOGGER.error("Index not found.Please train first")
        raise urllib.error.HTTPError(
            url="", code=404, msg="Index Not Found", hdrs={}, fp=None
        )
    try:
        if source is None or source == "":
            tool = get_intent(question)
            print(f"Tool found {tool}")
            if "detailed" in tool.lower() and "video" in tool.lower():
                source = "video"
            elif "detailed" in tool.lower() and "comments" in tool.lower():
                source = "comments"
            elif "summary" in tool.lower() and "video" in tool.lower():
                source = "video"
                response = get_answer_from_summary(question, source, video_id)
                return response
            elif "summary" in tool.lower() and "comments" in tool.lower():
                source = "comments"
                response = get_answer_from_summary(question, source, video_id)
                return response
            else:
                source = "video"
            LOGGER.info(f"Using source as {source}")
        vector_store = FaissVectorStore.from_persist_dir(
            f"./storage/{video_id}/{source}"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=f"./storage/{video_id}/{source}"
        )
        index = load_index_from_storage(storage_context)
        # configure retriever
        if k == 0:
            # k is dynamic
            n = get_number_of_nodes(index)
            LOGGER.info(f"Number of nodes in Index {n}")
            if n <= 5:
                k = 1
            elif n <= 10:
                k = 2
            elif n > 10:
                k = 5
            else:
                k = n
        LOGGER.info(f"k set to {k}")
        # retriever = VectorIndexRetriever(
        #     index=index,
        #     similarity_top_k=k,
        #     node_ids=list(index.index_struct.nodes_dict.keys()),
        #     doc_ids=list(index.index_struct.nodes_dict.values())
        # )
        retriever = index.as_retriever(similarity_top_k=k)

        # configure response synthesizer
        if source in ["video", "images"]:
            CHANGED_QA_PROMPT = Prompt(transcript_query_format)
            response_synthesizer = get_response_synthesizer(
                text_qa_template=CHANGED_QA_PROMPT
            )
        else:
            response_synthesizer = get_response_synthesizer()

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            ],
        )
        # Fire query
        response = query_engine.query(question)
        final_response = response.response
        LOGGER.info(
            f"Question: {question}, video: {video_id}, source: {source}, Response: {final_response}"
        )
        # final_response = "DUMMY"
    except Exception as e:
        LOGGER.error(f"Error encountered : {e}")
        raise
    return final_response
