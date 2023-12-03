from langchain.chat_models import ChatOpenAI

from src.scripts.settings import (
    MODEL_NAME_TRANSCRIPT,
    MAX_TOKENS_TRANSCRIPT,
    MAX_TOKENS_ALLOWED,
    MODEL_NAME_QA,
)
from langchain import PromptTemplate
from src.scripts.prompts import (
    transcript_summary_template,
    comment_summary_template,
    transcript_query_format,
)
from src.scripts.prompts import (
    intent_prompt_template,
    comment_summary_qa_template,
    transcript_summary_qa_template,
)
from dataclasses import dataclass
from langchain.chains.summarize import load_summarize_chain
from langchain import LLMChain

# llm models
llm_model_longtext = ChatOpenAI(temperature=0, model_name=MODEL_NAME_TRANSCRIPT)
llm_model_qa = ChatOpenAI(temperature=0, model_name=MODEL_NAME_QA)

# prompt templates
prompt_template_comments_summary_qa = PromptTemplate(
    input_variables=["query", "summary"], template=comment_summary_qa_template
)
prompt_template_transcript_summary_qa = PromptTemplate(
    input_variables=["query", "summary"], template=transcript_summary_qa_template
)

prompt_template_intent = PromptTemplate(
    input_variables=["query"], template=intent_prompt_template
)

prompt_template_transcript_summary = PromptTemplate(
    input_variables=["text"], template=transcript_summary_template
)
prompt_template_comments_summary = PromptTemplate(
    input_variables=["text"], template=comment_summary_template
)


@dataclass
class llmChains:
    # Chains for summary
    llm_chain_stuff_transcript_summary = load_summarize_chain(
        llm_model_longtext,
        chain_type="stuff",
        prompt=prompt_template_transcript_summary,
    )
    llm_chain_stuff_comments_summary = load_summarize_chain(
        llm_model_longtext, chain_type="stuff", prompt=prompt_template_comments_summary
    )
    llm_chain_map_reduce_transcript_summary = load_summarize_chain(
        llm_model_longtext,
        chain_type="map_reduce",
        map_prompt=prompt_template_transcript_summary,
        combine_prompt=prompt_template_transcript_summary,
    )
    llm_chain_map_reduce_comments_summary = load_summarize_chain(
        llm_model_longtext,
        chain_type="map_reduce",
        map_prompt=prompt_template_comments_summary,
        combine_prompt=prompt_template_comments_summary,
    )
    # Chains for Summary QA
    llm_chain_transcript_summary_qa = LLMChain(
        prompt=prompt_template_transcript_summary_qa, llm=llm_model_qa, verbose=False
    )
    llm_chain_comments_summary_qa = LLMChain(
        prompt=prompt_template_comments_summary_qa, llm=llm_model_qa, verbose=False
    )
    # Chain for intent
    llm_chain_intent = LLMChain(
        prompt=prompt_template_intent, llm=llm_model_qa, verbose=False
    )


chains = llmChains()
