import os
from dataclasses import asdict
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from interface import GenerationConfig, generate_interactive
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

logger = logging.get_logger(__name__)
print('pip install modelscope websockets')
os.system(f'pip install modelscope websockets==11.0.3')

base_path = './model/Ancient_Books'
os.system(f'git clone hhttps://openxlab.org.cn/models/detail/element/Ancient_Books.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

gradient_text_html = """
<style>
.container {
    position: relative;
    /* 可能需要调整的高度，以避免内容重叠 */
    padding-top: 50px; 
}

.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
    /* 使用相对定位并上移 */
    position: relative;
    top: -115px;
}
</style>
<div class="container">
    <div class="gradient-text">甘肃政法大学古籍解读</div>
</div>
"""
st.markdown(gradient_text_html, unsafe_allow_html=True)

class InternLM_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model = self.model.eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        system_prompt =  """
            你是古籍解读助手，精通古代文献和典籍，可以提供关于古籍解读、古文翻译和历史文化背景的专业建议和信息。无论是想了解古籍中的智慧，还是寻找特定古文的翻译和注释，都能提供帮助。
            """
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"

def load_chain(model, tokenizer):
    embeddings = HuggingFaceEmbeddings(model_name="/group_share/Ancient_Books/model/sentence-transformer")
    persist_directory = '/group_share/Ancient_Books/dataset/vector_db/chroma'
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever_chroma = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = InternLM_LLM(model, tokenizer)
    template = """你【甘肃政法大学古籍解读助手】可以参考以下上下文进行思考，并回答最后的问题。不要表明思考过程，直接返回答案。如果你不知道答案，就说你不知道，不要试图编造答
    案。请提供详细并且结构清晰的回答，并尽量避免简单带过问题。
    {context}
    问题: {question}
    有用的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_chroma, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

def on_btn_click():
    if "messages" in st.session_state:
        del st.session_state.messages

@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True).cuda())
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        st.title("甘肃政法大学古籍解读助手")
        st.subheader("目前支持功能")
        st.markdown("- 💖 古诗赏析")
        st.markdown("- 💬 文言文")
        st.markdown("- 📊 成语")
        st.markdown("- 📊 论语")
        st.markdown("- 📊 百家姓")
        with st.container(height=200, border=True):
            st.subheader("模型配置")
            max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
            top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)
    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature, repetition_penalty=1.002)
    return generation_config

user_prompt = 'user\n{user}\n'
robot_prompt = 'assistant\n{robot}\n'
cur_query_prompt = 'user\n{user}\nassistant\n'

def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('你是【甘肃政法大学古籍解读助手】。你会包括但不限于唐诗、宋词、论语等古籍，你还可以让我文言文翻译等。'
                        '【甘肃政法大学古籍解读助手】 can understand and communicate fluently in the language chosen by the user such as English and 中文.')
    total_prompt = f"<s>system\n{meta_instruction}\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def main():
    print('加载模型中...')
    model, tokenizer = load_model()
    qa_chain = load_chain(model, tokenizer)
    print('模型加载完毕.')
    user_avatar = 'assets/user.png'
    robot_avator = 'assets/logo.png'
    # st.title("甘肃政法大学古籍解读助手")

    generation_config = prepare_generation_config()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("assistant")):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user", avatar='user',):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": 'user'})

        with st.chat_message('robot', avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,
            'avatar': robot_avator,
        })
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
