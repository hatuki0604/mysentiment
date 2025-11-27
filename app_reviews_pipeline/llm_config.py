"""
LLM Configuration & Factory
Organized with section comments for clarity and maintainability.
"""

# ===========================
# Imports & Environment
# ===========================
from __future__ import annotations
import os
import requests
from typing import Callable, List, Dict
from dotenv import load_dotenv

load_dotenv()

# ===========================
# Type Definitions
# ===========================
ChatMessage = Dict[str, str]
ChatFn = Callable[[List[ChatMessage], float, int], str]

# ===========================
# OpenAI LLM
# ===========================
def openai_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for OpenAI Chat Completions.
    Env: OPENAI_API_KEY
    """
    from openai import OpenAI  # lazy import
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    def chat(
            messages: List[ChatMessage],
            temperature: float = 0.0,
            max_tokens: int = 512,
            with_logprobs: bool = False,
            top_logprobs: int = 5,
    ) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=with_logprobs,   #bật/tắt logprobs
            top_logprobs=top_logprobs if with_logprobs else None,
        )
        # Nếu không cần logprobs → giữ API y như cũ: trả string
        if not with_logprobs:
            return (resp.choices[0].message.content or "").strip()
        
        # Nếu cần logprobs → trả cả resp để xử lý bên ngoài
        return resp
    return chat

# ===========================
# Mistral LLM
# ===========================
def mistral_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for Mistral.
    Env: MISTRAL_API_KEY
    """
    from mistralai import Mistral  # lazy import
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in environment")

    client = Mistral(api_key=api_key)
    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    def chat(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # mistralai returns .choices[0].message.content
        return (resp.choices[0].message.content or "").strip()

    return chat

# ===========================
# Llama 2 via HF Inference API
# ===========================
def _messages_to_llama2_prompt(messages: List[ChatMessage]) -> str:
    """Minimal Llama-2 chat template: single-turn with optional system block."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user").strip()
    sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
    return f"<s>[INST] {sys_block}{user} [/INST]"

def llama2_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for Meta Llama-2 chat models
    using HF Inference API (hosted by Hugging Face).
    Env: HF_TOKEN (or HUGGINGFACE_TOKEN)
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGINGFACE_TOKEN) not found in environment")

    model = model or os.getenv("LLAMA2_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    def chat(messages: List[ChatMessage], temperature: float = 0.2, max_tokens: int = 512) -> str:
        prompt = _messages_to_llama2_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        r = requests.post(api_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Inference API commonly returns [{"generated_text": "..."}]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        # TGI-compatible responses may vary:
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)
    return chat

# ===========================
# LLAMA 3 via Ollama (local)
# ===========================
def llama3_ollama_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function that calls a local Ollama server
    running LLaMA 3 or any Ollama model.

    No API key is required.
    Default URL: http://localhost:11434/api/chat
    """
    import json

    model = model or os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")

    def chat(messages: List[ChatMessage], temperature: float = 0.2, max_tokens: int = 512) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        resp = requests.post(base_url, json=payload, stream=True, timeout=300)
        result = ""
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "message" in data:
                result += data["message"]["content"]
            if data.get("done"):
                break
        return result.strip()

    return chat


# ===========================
# Llama 3.x via HF Inference API
# ===========================
def _messages_to_llama3_prompt(messages: List[ChatMessage]) -> str:
    """
    Llama 3.x uses a simple chat format, similar to ChatML.
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            lines.append(f"<|start_header_id|>system<|end_header_id|>\n{msg['content']}<|eot_id|>")
        elif role == "user":
            lines.append(f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>")
        elif role == "assistant":
            lines.append(f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>")
    lines.append("<|start_header_id|>assistant<|end_header_id|>\n")  # start of generation
    return "".join(lines)


def llama3_hf_llm(model: str = None) -> ChatFn:
    """
    Call Llama 3.x models (e.g., meta-llama/Llama-3.2-3B-Instruct)
    via Hugging Face Inference API.
    Env: HF_TOKEN (or HUGGINGFACE_TOKEN)
    """
    import requests
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGINGFACE_TOKEN) not found in environment")

    model = model or "meta-llama/Llama-3.2-3B-Instruct"
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    def chat(messages: List[ChatMessage], temperature: float = 0.3, max_tokens: int = 512) -> str:
        prompt = _messages_to_llama3_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)

    return chat

# ===========================
# Vi-Qwen (Transformers local, Mac-friendly)
# ===========================
def viqwen_local_llm(model: str = None) -> ChatFn:
    """
    Chạy Vi-Qwen local (ví dụ: AITeamVN/Vi-Qwen2-3B-RAG) an toàn trên Mac M-series.
    Không cần API key, dùng transformers trực tiếp.
    """

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = model or "AITeamVN/Vi-Qwen2-3B-RAG"

    # --- Xác định device và dtype phù hợp cho Mac ---
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"🔹 Loading local model: {model_id} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    def chat(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Nhận list messages [{'role': 'system'|'user'|'assistant', 'content': str}], trả về text.
        """

        # --- Gộp các messages thành prompt theo format chat ---
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),
                do_sample=False,   # Greedy decoding (ổn định hơn sampling)
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode phần sinh thêm (bỏ phần input gốc)
        result = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        return result

    return chat

# ===========================
# 0wovv0/Qwen-3B (Transformers local, Mac-friendly)
# ===========================
def viqwen_3b_local_llm(model: str = None) -> ChatFn:
    """
    Chạy Vi-Qwen local (ví dụ: AITeamVN/Vi-Qwen2-3B-RAG) an toàn trên Mac M-series.
    Không cần API key, dùng transformers trực tiếp.
    """

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = model or "0wovv0/Qwen-3B"

    # --- Xác định device và dtype phù hợp cho Mac ---
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"🔹 Loading local model: {model_id} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    def chat(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Nhận list messages [{'role': 'system'|'user'|'assistant', 'content': str}], trả về text.
        """

        # --- Gộp các messages thành prompt theo format chat ---
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),
                do_sample=False,   # Greedy decoding (ổn định hơn sampling)
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode phần sinh thêm (bỏ phần input gốc)
        result = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        return result

    return chat



# ===========================
# LLM Factory
# ===========================
# def get_llm(provider: str, model: str | None = None) -> ChatFn:
#     p = (provider or "").lower()
#     if p in ("openai", "oai"):
#         return openai_llm(model)
#     if p in ("mistral", "mistralai"):
#         return mistral_llm(model)
#     if p in ("llama2", "huggingface", "hf"):
#         return llama2_llm(model)
#     if p in ("ollama", "llama3", "llama", "local"):
#         return llama3_ollama_llm(model)
#     raise ValueError(f"Unknown provider '{provider}'. Use: openai | mistral | llama2| ollama")

def get_llm(provider: str, model: str | None = None) -> ChatFn:
    p = (provider or "").lower()
    if p in ("openai", "oai"):
        return openai_llm(model)
    if p in ("mistral", "mistralai"):
        return mistral_llm(model)
    if p in ("llama2", "huggingface", "hf"):
        return llama2_llm(model)
    if p in ("llama3hf", "llama3.2", "meta-llama3", "hf-llama3"):
        return llama3_hf_llm(model)
    if p in ("viqwen_local", "viqwen", "qwen_local", "qwen"):
        return viqwen_local_llm(model)
    if p in ("viqwen_3b_local", "viqwen", "qwen_local", "qwen"):
        return viqwen_3b_local_llm(model)
    if p in ("ollama", "llama3", "llama", "local"):
        return llama3_ollama_llm(model)
    raise ValueError(
        f"Unknown provider '{provider}'. Use: openai | mistral | llama2 | llama3hf | viqwen_local | viqwen_3B_local| ollama"
    )
