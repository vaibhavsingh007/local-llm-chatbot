from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",  # Need to specify orca is compatible with llama2
    max_new_tokens=20  # Max tokens (words) model should return
)

prompt_template = """### System:\nYou are an AI assistant that gives short and concise answers.
            \n\n### User:\n{instruction}\n\n### Response:\n"""
prompt = PromptTemplate(template=prompt_template, input_variables=["instruction"])
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke({"instruction": "What is the capital of India?"}))
