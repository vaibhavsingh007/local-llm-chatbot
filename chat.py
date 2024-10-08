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
        prompt = f"### System:\n{system} "

        if len(history) > 0:
            prompt += f"Take this context into account when answering the question: {''.join(history)}"

        prompt += f"\n\n### User:\n{instruction}\n\n### Response:\n"
    elif model_in_use == "llama2":
        prompt = f"<s>[INST] <<SYS>>\n{system} "

        if len(history) > 0:
            prompt += f"Take this context into account when answering the question: {''.join(history)}"

        prompt += f"\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


def handle_special_message(msg: str) -> str | None:
    global model_in_use

    if msg == "use llama2":
        model_in_use = 'llama2'
        set_llm()
        return "Model changed to Llama"
    elif msg == "use orca":
        model_in_use = "orca"
        set_llm()
        return "Model changed to Orca"
    elif msg == "which model":
        return model_in_use
    elif msg == "history":
        return cl.user_session.get("message_history")
    elif msg == "forget everything":
        cl.user_session.set("message_history", [])
        return "Uh oh, I've just forgotten our conversation history"
    else:
        return None


# Global model setter
def set_llm() -> None:
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=llm_config[model_in_use]['path'],
        model_file=llm_config[model_in_use]['file'])


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    set_llm()


@ cl.on_message
async def on_message(message: cl.Message):
    handler_response = handle_special_message(message.content)

    if handler_response is not None:
        msg = cl.Message(content=handler_response)
        await msg.send()
    else:
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
