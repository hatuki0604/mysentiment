
"""
User Selection & CLI Interface Utilities
Organized with section comments for clarity and maintainability.

This module provides interactive CLI functions for dataset selection, 
LLM provider/model selection, and sample size configuration across all pipeline modules.
"""

# ===========================
# Imports & Environment Setup
# ===========================
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ===========================
# Path Resolution
# ===========================
# This file lives at: methods/LLM_analysis/user_selection.py
REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()
ROOT = HERE
for _ in range(6):  # climb up to 6 levels just in case
    if (ROOT / "data" / "processed").exists():
        break
    ROOT = ROOT.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# ===========================
# Dataset Selection
# ===========================
def choose_dataset() -> tuple[str, str, str]:
    """
    Interactive dataset selection with automatic path resolution.
    
    Returns:
        tuple: (dataset_name, csv_file_path)
        
    Raises:
        FileNotFoundError: If the selected dataset CSV cannot be located
    """
    print("\nDatasets:\n  1) aware\n  2) spotify\n  3) google_play\n 4) UIT-ViSFD\n 5) UIT-ViSFD-All\n 6)TGDD Reviews")
    choice = (input("Choose 1-6 [2]: ").strip() or "2")
    ds_map = {"1": "aware", "2": "spotify", "3": "google_play", "4": "uit", "5": "uit_all", "6": "processed_tgdd_reviews"}
    topic_map = {
        "1": "Đây là các reviews về đánh giá ứng dụng", 
        "2": "Đây là các reviews về dịch vụ nghe nhạc trực tuyến Spotify", 
        "3": "Đây là các reviews về ứng dụng trên cửa hàng Google Play", 
        "4": "Đây là các reviews về điện thoại di động",
        "5": "Đây là các reviews về các sản phẩm điện tử tiêu dùng",
        "6": "Đây là các reviews về điện thoại di động của Thế Giới Di Động"
        }
    topic = topic_map.get(choice, "Đây là các reviews về sản phẩm điện thoại di động")
    ds = ds_map.get(choice, "spotify")
    primary   = PROCESSED_DIR / f"{ds}_clean.csv"
    fallback1 = PROCESSED_DIR / f"{ds}_processed.csv"
    fallback2 = PROCESSED_DIR / f"{ds}.csv"
    if primary.exists():
        resolved = primary
    elif fallback1.exists():
        print(f"(note) Using fallback file: {fallback1.name}")
        resolved = fallback1
    elif fallback2.exists():
        print(f"(note) Using fallback file: {fallback2.name}")
        resolved = fallback2
    else:
        candidates = list(REPO_ROOT.glob(f"**/{ds}_clean.csv")) + \
                     list(REPO_ROOT.glob(f"**/{ds}_processed.csv"))
        if candidates:
            candidates.sort(key=lambda p: ( "data\\processed" not in str(p).lower()
                                            and "data/processed" not in str(p).lower(), len(str(p)) ))
            resolved = candidates[0]
        else:
            existing = []
            if PROCESSED_DIR.exists():
                existing = [p.name for p in PROCESSED_DIR.glob("*.csv")]
            raise FileNotFoundError(
                "Could not locate the cleaned CSV for dataset '{ds}'.\n"
                f"Tried: {primary} and {fallback1}\n"
                f"data/processed listing: {existing}\n"
                f"Repo root: {REPO_ROOT}"
            )
    print(f"Using dataset: {ds} → {resolved}")
    return ds, str(resolved), topic

# ===========================
# Sample Size Selection (Optional sampling - lấy mẫu ngẫu nhiên)
# ===========================
def choose_sample_size(total: int) -> Optional[int]:
    s = input(f"\nSample size (type number or 'all') [all, max {total}]: ").strip().lower()
    if not s or s == "all":
        return None
    try:
        n = int(s)
        return max(1, min(n, total))
    except ValueError:
        print("Invalid number — using all rows.")
        return None
    
# ===========================
# Defaults & Suggestions
# ===========================
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODELS = {
    "openai":  os.getenv("OPENAI_MODEL",  "gpt-4o"),
    "mistral": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    "llama2":  os.getenv("LLAMA2_MODEL",  "meta-llama/Llama-2-7b-chat-hf"),
    "llama3hf": os.getenv("LLAMA3HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
    "viqwen": os.getenv("VIQWEN_MODEL", "AITeamVN/Vi-Qwen2-1.5B-RAG"),
    "viqwen-3B": os.getenv("VIQWEN_3b_MODEL", "0wovv0/Qwen-3B"),
    "ollama":  os.getenv("OLLAMA_MODEL",  "llama3"),
}
SUGGESTED_MODELS = {
    "openai":  ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "llama2":  ["meta-llama/Llama-2-7b-chat-hf"],
    "llama3hf": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    "viqwen": [
        "AITeamVN/Vi-Qwen2-1.5B-RAG",
        "Qwen/Qwen2-1.5B-Instruct",
    ],
    "viqwen-3b": ["VIQWEN_3b_MODEL", "0wovv0/Qwen-3B"],
    "ollama":  os.getenv("OLLAMA_MODEL",  "llama3"),
    "ollama":  ["llama3", "llama3:8b", "llama3:70b"]
}

# ===========================
# Provider & Model Selection
# ===========================
def choose_provider_and_model() -> tuple[str, str]:
    providers = ["openai", "mistral", "llama2", "llama3hf", "viqwen_local","viqwen_3b_local", "ollama"]

    def_idx = providers.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in providers else 0
    # print("\nProviders:")
    # for i, p in enumerate(providers, 1):
    #     star = " (default)" if (i - 1) == def_idx else ""
    #     print(f"  {i}) {p}{star}")
    # raw = input("Choose 1-4 [default]: ").strip()

    print("\nProviders:")
    for i, p in enumerate(providers, 1):
        star = " (default)" if (i - 1) == def_idx else ""
        print(f"  {i}) {p}{star}")
    raw = input(f"Choose 1-{len(providers)} [default]: ").strip()
##
    provider = (
        providers[int(raw) - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(providers)
        else providers[def_idx]
    )

    base = DEFAULT_MODELS.get(provider)

    def dedup_keep_order(seq):
        seen, out = set(), []
        for x in seq:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    # Model fallback map
    default_fallbacks = {
        "openai": "gpt-4o",
        "mistral": "mistral-small-latest",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "llama3hf": "meta-llama/Llama-3.2-3B-Instruct",
        "viqwen": "AITeamVN/Vi-Qwen2-7B-RAG",
        "viqwen_local": "AITeamVN/Vi-Qwen2-3B-RAG",
        "viqwen_3b_local": "0wovv0/Qwen-3B",
        "ollama": "llama3",
    }

    # Candidate list (combine base + suggested + fallback)
    cands = (
        dedup_keep_order([base] + SUGGESTED_MODELS.get(provider, []))
        or [default_fallbacks.get(provider, "gpt-4o")]
    )

    print("\nModels:")
    for i, m in enumerate(cands, 1):
        print(f"  {i}) {m}")
    print(f"  {len(cands) + 1}) <custom>")
    raw_m = input(f"Choose 1-{len(cands) + 1} [{cands[0]}]: ").strip()
    if raw_m.isdigit():
        idx = int(raw_m)
        if 1 <= idx <= len(cands):
            model = cands[idx - 1]
        elif idx == len(cands) + 1:
            typed = input("Type model name: ").strip()
            model = typed or cands[0]
        else:
            model = cands[0]
    elif raw_m:
        model = raw_m
    else:
        model = cands[0]
    return provider, model

# ===========================
# Prompt Tuning Selection
# ===========================
def choose_prompt_tuning(default: bool = False) -> bool:
    """
    Ask the user whether to enable prompt optimization.
    Returns True if enabled, False otherwise.
    """
    print("\nPrompt optimization adapts the prompt to this dataset for potentially better quality.")
    raw = input(f"Enable prompt optimization? 1=Yes, 0=No [{'1' if default else '0'}]: ").strip()
    if raw in {"1", "0"}:
        return raw == "1"
    return default
