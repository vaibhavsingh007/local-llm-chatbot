from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
                                           model_file="llama-2-7b-chat.Q4_K_M.gguf")


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives short and concise answers."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


question = "What is the capital of India?"
# "Question: What is the capital of India called? Answer: The capital of India is"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()
