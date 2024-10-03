import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM

# Model config
model_in_use = "orca"  # Default model
llm_config = {
    "orca": {
        "path": "zoltanctoth/orca_mini_3B-GGUF",
        "file": "orca-mini-3b.q4_0.gguf"
    },
    "llama2": {
        "path": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q4_K_M.gguf"
    }
}


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives short and concise answers."
    prompt = ""

    if model_in_use == "orca":
        prompt = f"### System:\n{system}\n\n### User:\n"

        if len(history) > 0:
            prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "

        prompt += f"{instruction}\n\n### Response:\n"
    elif model_in_use == "llama2":
        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>"

        if len(history) > 0:
            prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "

        prompt += f"{instruction}\n\n{instruction} [/INST]"
    return prompt


def handle_special_message(msg: str) -> None:
    global model_in_use

    if msg == "use llama2":
        model_in_use = 'llama2'
    elif msg == "use orca":
        model_in_use = "orca"
    else:
        pass

    set_llm()


# Global model setter
def set_llm() -> None:
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=llm_config[model_in_use].path,
        model_file=llm_config[model_in_use].file)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    set_llm('orca')


@ cl.on_message
async def on_message(message: cl.Message):
    handle_special_message(message.content)
    history = cl.user_session.get("message_history")
    prompt = get_prompt(message.content, history)
    msg = cl.Message(content="")
    await msg.send()

    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    history.append(response)
