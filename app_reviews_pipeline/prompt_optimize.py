"""
Prompt Optimization & Meta-Prompting Utilities
Organized with section comments for clarity and maintainability.

This module provides comprehensive prompt optimization and repair capabilities:

1. **Domain-Aware Prompt Optimization**: Automatically improve prompts using dataset-specific vocabulary
2. **Quality-Based Selection**: LLM-based evaluation to choose the best prompt version
3. **Self-Critique & Repair**: Meta-prompting for schema validation and grounding enforcement
4. **JSON Repair**: Robust parsing and repair of LLM-generated JSON responses
5. **Topic Label Validation**: Specialized validation for topic modeling labels

Key Functions:
- optimize_prompt(): Main optimization pipeline with caching and evaluation
- self_critique_and_fix(): RAG-style critic for answer validation
- repair_json_list/dict(): Robust JSON parsing with LLM fallback
- revise_topic_label_if_needed(): Topic label quality assurance

Configuration:
- Enable/disable optimization via environment: PROMPT_TUNE=1
- Caching per dataset/provider/model combination for efficiency
- Quality-based selection using QualityJudge evaluation system

Dependencies:
- quality_judge.py for automated prompt evaluation
- LLM provider (via chat_fn interface)
- Optional: scikit-learn for TF-IDF term extraction
"""

# ===========================
# Imports & Configuration
# ===========================
from __future__ import annotations
import os, re, json, hashlib, random
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence, Tuple

# Use absolute import instead of relative
from quality_judge import QUALITY_JUDGE_PROMPT, QualityJudge

# ===========================
# JSON Extraction Utilities
# ===========================

def _extract_json_from_response(response: str) -> str:
    """
    Extract JSON from LLM response that might be wrapped in markdown code blocks.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Clean JSON string
    """
    # Strip whitespace
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith('```json'):
        # Find the end of the code block
        end_marker = response.find('```', 7)  # Start looking after '```json'
        if end_marker != -1:
            response = response[7:end_marker].strip()
    elif response.startswith('```'):
        # Handle generic code blocks
        end_marker = response.find('```', 3)
        if end_marker != -1:
            response = response[3:end_marker].strip()
    
    return response

# ===========================
# Domain-Aware Prompt Optimization  
# ===========================

META_PROMPT = """
Bạn là một kỹ sư prompt (prompt engineer).  Hãy cải thiện BASE_PROMPT dành cho một trợ lý rằng cần tuân thủ nghiêm ngặt cấu trúc đầu ra 
và chỉ sử dụng thông tin trong CONTEXT được cung cấp.
Mục tiêu:
- Giữ NGUYÊN tất cả tiêu đề các phần và cấu trúc đầu ra (schema) giống hệt trong BASE_PROMPT.  
- Tăng cường các yếu tố sau:
  (a) Liên kết chặt chẽ với CONTEXT,  
  (b) Từ chối trả lời khi thiếu đủ bằng chứng,  
  (c) Giới hạn độ dài, phong cách ngắn gọn,  
  (d) Quy tắc trích dẫn nguyên văn có đánh chỉ mục.
- Thêm nhận thức về miền (domain-awareness): ưu tiên sử dụng từ vựng trong DOMAIN_TERMS để giảm mơ hồ.
- Viết gọn gàng, chính xác. Không thêm phần mới hoặc đổi tên phần có sẵn.

Chỉ trả về phần prompt đã được cải thiện (không kèm bình luận).

DOMAIN_TERMS: {terms_csv}

BASE_PROMPT:
{base_prompt}

""".strip()


def _cache_path(cache_dir: Path, dataset: str, provider: str, model: str) -> Path:
    """
    Generate cache file path for optimized prompts.
    
    Args:
        cache_dir: Directory for storing cached prompts
        dataset: Dataset name (e.g., 'spotify', 'aware')
        provider: LLM provider (e.g., 'openai', 'mistral')
        model: Model name (e.g., 'gpt-4o')
        
    Returns:
        Path to cache file for this dataset/provider/model combination
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dataset}__{provider}__{model}".encode("utf-8", "ignore")
    return cache_dir / (hashlib.sha1(key).hexdigest() + ".txt")


def _tfidf_terms(texts: Sequence[str], max_terms: int = 60) -> List[str]:
    """
    Extract domain-specific vocabulary using TF-IDF or frequency analysis.
    
    Args:
        texts: Collection of text samples from the dataset
        max_terms: Maximum number of terms to extract
        
    Returns:
        List of important domain terms for prompt optimization
        
    Note:
        Uses scikit-learn TF-IDF if available, falls back to simple frequency analysis
        for 1-grams and 2-grams with stop word filtering.
    """
    clean = [str(t or "") for t in texts]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=max(5, int(0.001 * max(1, len(clean)))),
            max_df=0.8,
        )
        X = vec.fit_transform(clean)
        terms = vec.get_feature_names_out()
        scores = X.mean(0).A1
        top = scores.argsort()[-max_terms:][::-1]
        return [terms[i] for i in top]
    except Exception:
        # simple regex tokenization; 1-grams and bigrams
        from collections import Counter
        toks = []
        for t in clean:
            words = re.findall(r"[a-z0-9][a-z0-9\-]+", t.lower())
            toks.extend(words)
            toks.extend([f"{w1} {w2}" for w1, w2 in zip(words, words[1:])])
        common = [w for w, _ in Counter(toks).most_common(max_terms * 2)]
        # filter stop-ish tokens
        bad = set("the a an and or for with to of is are be been was were on in at by from".split())
        out: List[str] = []
        for w in common:
            if any(p in bad for p in w.split()):
                continue
            if 2 <= len(w) <= 40:
                out.append(w)
            if len(out) >= max_terms:
                break
        return out


def optimize_prompt(
    dataset: str,
    sample_texts: Sequence[str],
    chat_fn,                 # your ChatFn(messages, temperature, max_tokens) -> str
    base_prompt: str,
    provider: str,
    model: str,
    cache_dir: str = "prompt_cache",
    enabled: Optional[bool] = None,
    must_have_tags: Optional[List[str]] = None,
    sample_size: int = 5000,
    max_terms: int = 60,
    prompt_type: str = "general",  # Added for judge evaluation context
    context: str = ""  # Added for evaluation
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimize a prompt using domain-specific vocabulary and quality-based selection.
    
    This is the main optimization pipeline that:
    1. Extracts domain vocabulary from sample texts using TF-IDF
    2. Uses META_PROMPT to generate an improved version
    3. Evaluates both prompts using QualityJudge
    4. Returns the best prompt with evaluation details
    5. Caches results per dataset/provider/model combination
    
    Args:
        dataset: Dataset name for cache key and domain awareness
        sample_texts: Text samples for vocabulary extraction
        chat_fn: LLM function interface
        base_prompt: Original prompt to optimize
        provider: LLM provider name
        model: LLM model name
        cache_dir: Directory for caching optimized prompts
        enabled: Enable/disable optimization (env PROMPT_TUNE controls default)
        must_have_tags: Required tags that must appear in outputs
        sample_size: Maximum samples for vocabulary extraction
        max_terms: Maximum domain terms to extract
        prompt_type: Type of prompt for evaluation context
        context: Additional context for evaluation
        
    Returns:
        Tuple of (selected_prompt, evaluation_details)
        
    Note:
        Results are cached per (dataset, provider, model) combination.
        Set PROMPT_TUNE=0 in environment to disable optimization.
    """
    if enabled is None:
        enabled = os.getenv("PROMPT_TUNE", "1") == "1"  # Default to enabled
    if not enabled:
        return base_prompt, {"explanation": "Prompt optimization disabled", "recommendation": "base"}

    # Initialize evaluation details
    evaluation = {
        "explanation": "",
        "recommendation": "base"
    }

    cache_file = _cache_path(Path(cache_dir), dataset, provider, model)
    optimized_prompt = None
    
    if cache_file.exists():
        optimized_prompt = cache_file.read_text(encoding="utf-8")
    else:
        # small, fast sample for vocabulary extraction
        texts = list(sample_texts)[: sample_size] if sample_texts else []
        if len(texts) > sample_size:
            random.Random(42).shuffle(texts)
            texts = texts[:sample_size]
        terms = _tfidf_terms(texts, max_terms=max_terms)
        prompt = META_PROMPT.format(terms_csv=", ".join(terms), base_prompt=base_prompt)

        messages = [
            {"role": "system", "content": "Return only the improved prompt text."},
            {"role": "user", "content": prompt},
        ]
        optimized_prompt = chat_fn(messages, temperature=0.2, max_tokens=1500).strip()

        # sanity: ensure the optimized prompt still contains critical anchors
        anchors = must_have_tags or []
        if anchors:
            ok = all(tag in optimized_prompt for tag in anchors)
            if not ok:
                return base_prompt, {"explanation": "Optimization failed anchor check", "recommendation": "base"}

        cache_file.write_text(optimized_prompt, encoding="utf-8")

    # Use the quality judge to evaluate and select the best prompt
    judge = QualityJudge(chat_fn=chat_fn, cache_dir=cache_dir)
    
    # Generate sample outputs from both prompts for evaluation
    sample_context = context or (sample_texts[0] if sample_texts else "")
    test_messages = [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": sample_context}
    ]
    base_output = chat_fn(test_messages, temperature=0.2, max_tokens=500)
    
    test_messages = [
        {"role": "system", "content": optimized_prompt},
        {"role": "user", "content": sample_context}
    ]
    optimized_output = chat_fn(test_messages, temperature=0.2, max_tokens=500)
    
    # Let the judge evaluate and select the best output
    selected_output, evaluation = judge.evaluate_outputs(
        base_output,
        optimized_output,
        dataset=dataset,
        prompt_type=prompt_type,
        context=sample_context
    )
    
    # Return the corresponding prompt based on the evaluation
    final_prompt = base_prompt if evaluation["recommendation"] == "a" else optimized_prompt
    return final_prompt, evaluation

# ===========================
# Self-Critique & Repair System
# ===========================

CRITIC_PROMPT = """
Bạn là một người đánh giá nghiêm khắc cho câu trả lời của trợ lý (assistant).

Đầu vào gồm:
- QUESTION (Câu hỏi)
- CONTEXT (các đoạn trích đánh giá có đánh số [idx])
- ANSWER (phần trả lời của trợ lý)

Yêu cầu kiểm tra:

1) Định dạng (Format) 
   - Đối với chế độ Hỏi & Đáp (Q&A), các phần được phép bao gồm chỉ những mục sau:  
     ANSWER:  
     EVIDENCE:  
     INSIGHTS: (tùy chọn)  
   - Đối với chế độ Tóm tắt (SUMMARY), các phần được phép bao gồm chỉ những mục sau:  
     SUMMARY:  
     TOP_EXAMPLES:

2) Căn cứ (Grounding)
   - Mọi nhận định trong câu trả lời phải được hỗ trợ bằng trích dẫn nguyên văn từ các đoạn CONTEXT,  
     và phải ghi rõ chỉ mục [idx] tương ứng.  
   - Không được thêm thông tin ngoài (no outside facts).

Trả về DUY NHẤT một đối tượng JSON với các trường:
{
  "format_ok": true|false,
  "grounding_ok": true|false,
  "fix_instructions": "Nếu một trong hai là false, hãy nêu chính xác cách chỉnh sửa để sửa lỗi, đồng thời vẫn tuân thủ các phần được phép."
}
""".strip()


def self_critique_and_fix(
    question: str,
    context_json: str,   # the same JSON string you pass to the prompt
    raw_answer: str,
    chat_fn,
    max_tokens_report: int = 400,
    max_tokens_fix: int = 700,
) -> str:
    """
    Apply self-critique and repair to LLM outputs for schema validation and grounding.
    
    This function implements a meta-prompting approach where the LLM critiques its own
    output against format requirements and evidence grounding, then rewrites if needed.
    
    Args:
        question: The original question/query
        context_json: JSON string of context/evidence snippets  
        raw_answer: The LLM's initial answer to validate
        chat_fn: LLM function interface
        max_tokens_report: Token limit for critic evaluation
        max_tokens_fix: Token limit for answer rewriting
        
    Returns:
        Improved answer that meets format and grounding requirements
        
    Note:
        Uses CRITIC_PROMPT to evaluate format compliance and evidence grounding.
        Only rewrites if critic identifies issues with the original answer.
    """
    messages = [
        {"role": "system", "content": "Return only JSON as specified."},
        {"role": "user",
         "content": f"{CRITIC_PROMPT}\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context_json}\n\nANSWER:\n{raw_answer}"}
    ]
    report = chat_fn(messages, temperature=0.0, max_tokens=max_tokens_report)
    try:
        clean_json = _extract_json_from_response(report)
        data = json.loads(clean_json)
    except Exception:
        return raw_answer

    if data.get("format_ok") and data.get("grounding_ok"):
        return raw_answer

    # Rewrite using the critic's fix instructions
    rewrite = f"""
You will revise the assistant's answer using the FIX INSTRUCTIONS.

Ràng buộc (Constraints):
- Chỉ giữ lại CÁC PHẦN /TIÊU ĐỀ được phép theo chế độ đã phát hiện (Q&A hoặc SUMMARY).  
- KHÔNG được thêm thông tin bên ngoài. Chỉ sử dụng CONTEXT đã cho.  
- Giữ các gạch đầu dòng (bullet) ngắn gọn, súc tích.

QUESTION:
{question}

CONTEXT:
{context_json}

CURRENT_ANSWER:
{raw_answer}

FIX INSTRUCTIONS:
{data.get('fix_instructions','')}

CHỈ trả lại văn bản câu trả lời đã sửa.
""".strip()
    messages = [{"role": "user", "content": rewrite}]
    return chat_fn(messages, temperature=0.0, max_tokens=max_tokens_fix).strip()

# ===========================
# JSON Repair & Validation  
# ===========================

_JSON_REPAIR_PROMPT = """
Bạn là một công cụ sửa chữa JSON.

Nhiệm vụ: Chuyển đổi INPUT thành JSON hợp lệ khớp với mô tả SCHEMA.
- Nếu đầu vào đã hợp lệ, hãy trả về giá trị không đổi.
- Không có văn xuôi, không có markdown, CHỈ trả về JSON.

SCHEMA:
{schema_desc}

INPUT:
{raw}
""".strip()


def _try_json_load(text: str) -> Any:
    """
    Attempt to parse JSON from text with markdown code block handling.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        clean_json = _extract_json_from_response(text)
        return json.loads(clean_json)
    except Exception:
        return None


def repair_json_list(raw_text: str, chat_fn, item_desc: str = "array of strings") -> List[Any]:
    """
    Repair and parse JSON list with LLM fallback for invalid JSON.
    
    Args:
        raw_text: Raw text that should contain a JSON array
        chat_fn: LLM function for repair if parsing fails
        item_desc: Description of array items for repair schema
        
    Returns:
        Python list parsed from JSON, empty list if repair fails
        
    Note:
        First attempts direct parsing, falls back to LLM repair if needed.
    """
    parsed = _try_json_load(raw_text)
    if isinstance(parsed, list):
        return parsed
    schema = f"A JSON array. Each item is a {item_desc}."
    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": _JSON_REPAIR_PROMPT.format(schema_desc=schema, raw=raw_text)},
    ]
    fixed = chat_fn(messages, temperature=0.0, max_tokens=400)
    parsed = _try_json_load(fixed)
    return parsed if isinstance(parsed, list) else []


def repair_json_dict(raw_text: str, chat_fn, value_desc: str = "one of: Positive, Negative, Neutral") -> Dict[str, Any]:
    """
    Repair and parse JSON dictionary with LLM fallback for invalid JSON.
    
    Args:
        raw_text: Raw text that should contain a JSON object
        chat_fn: LLM function for repair if parsing fails  
        value_desc: Description of dictionary values for repair schema
        
    Returns:
        Python dictionary parsed from JSON, empty dict if repair fails
        
    Note:
        Commonly used for sentiment dictionaries and structured ABSA outputs.
    """
    parsed = _try_json_load(raw_text)
    if isinstance(parsed, dict):
        return parsed
    schema = f"A JSON object mapping string keys to values ({value_desc})."
    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": _JSON_REPAIR_PROMPT.format(schema_desc=schema, raw=raw_text)},
    ]
    fixed = chat_fn(messages, temperature=0.0, max_tokens=400)
    parsed = _try_json_load(fixed)
    return parsed if isinstance(parsed, dict) else {}

# ===========================
# Topic Label Validation & Repair
# ===========================

LABEL_CRITIC_PROMPT = """
Given a topic label candidate and its top keywords, decide if the label is acceptable.

Rules for an acceptable label:
- 2–5 words, Title Case, no trailing punctuation or quotes
- Specific and human-friendly; avoid generic terms (e.g., "General Feedback")
- Avoid pure brand terms unless they convey a distinct theme
- Should plausibly summarize the keywords

Return ONLY JSON:
{
  "ok": true|false,
  "reason": "why if false",
  "rewrite": "a better 2–5 word Title Case label (if false)"
}
""".strip()


def revise_topic_label_if_needed(
    label: str,
    keywords: List[str],
    chat_fn,
    forbidden_terms: Optional[List[str]] = None,
    max_tokens: int = 120,
) -> str:
    """
    Validate and improve topic labels using LLM-based critique.
    
    This function ensures topic labels meet quality standards:
    - 2-5 words in Title Case
    - Specific and human-friendly (avoid generic terms)
    - Avoid pure brand terms unless meaningful
    - Should summarize the provided keywords
    
    Args:
        label: Current topic label to validate
        keywords: Top keywords associated with this topic
        chat_fn: LLM function for label critique and revision
        forbidden_terms: Terms to avoid in labels (e.g., brand names)
        max_tokens: Token limit for label revision
        
    Returns:
        Validated/improved label that meets quality standards
        
    Note:
        Used primarily in M3 topic modeling for ensuring high-quality topic labels.
        Labels are automatically normalized to Title Case with punctuation removal.
    """
    fk = ", ".join(keywords[:10])
    forbid = ", ".join((forbidden_terms or [])[:15])
    user = f"""{LABEL_CRITIC_PROMPT}

Current label: {label}
Top keywords: {fk}
Forbidden terms to avoid in label (optional): {forbid}
"""
    messages = [
        {"role": "system", "content": "Return only JSON as specified."},
        {"role": "user", "content": user},
    ]
    resp = chat_fn(messages, temperature=0.0, max_tokens=max_tokens)
    try:
        clean_json = _extract_json_from_response(resp)
        data = json.loads(clean_json)
    except Exception:
        return label

    if data.get("ok"):
        return label

    new_label = (data.get("rewrite") or "").strip()
    # Simple normalization: Title Case, strip quotes/punct
    new_label = re.sub(r'["“”\'`]+', "", new_label).strip()
    if not new_label:
        return label
    # enforce <= 5 words
    words = new_label.split()
    if len(words) > 5:
        new_label = " ".join(words[:5])
    # Title Case
    new_label = " ".join(w[:1].upper() + w[1:] for w in words[:5])
    return new_label
