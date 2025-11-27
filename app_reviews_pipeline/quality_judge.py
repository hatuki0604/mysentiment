"""
quality_judge.py - LLM-based quality evaluation for comparing different prompt outputs.
Used to automatically select the best output between base and optimized prompts.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

QUALITY_JUDGE_PROMPT = """
Bạn là chuyên gia đánh giá kết quả đầu ra của NLP. Hãy so sánh hai kết quả đầu ra khác nhau cho cùng một nhiệm vụ và xác định kết quả nào tốt hơn dựa trên các tiêu chí sau:

Evaluation Criteria (Tiêu chí đánh giá)
1. Accuracy & Grounding (Độ chính xác & Cơ sở): Mức độ hỗ trợ của bằng chứng cho kết quả đầu ra
2. Completeness (Tính đầy đủ): Bao quát các khía cạnh quan trọng mà không bỏ sót thông tin chính
3. Conciseness (Sự súc tích): Truyền đạt rõ ràng và hiệu quả, không chứa chi tiết thừa thãi
4. Format Adherence (Tuân thủ định dạng): Tuân thủ nghiêm ngặt định dạng đầu ra bắt buộc
5. Domain Relevance (Tính liên quan đến lĩnh vực): Sử dụng thuật ngữ chuyên ngành phù hợp

Với mỗi đầu ra, chấm điểm (từ 1-10) với mỗi tiêu chí và giải thích bằng tiếng việt tại sao lại có số điểm đó.
Sau đó đưa ra recommendation (khuyến nghị) cuối cùng về đầu ra nào tốt hơn.

OUTPUT A:
{output_a}

OUTPUT B:
{output_b}

CONTEXT (if available):
{context}

Đánh giá cả hai đầu ra và chỉ trả về một đối tượng JSON như ví dụ sau:

{{
    "output_a_scores": {{
        "accuracy": 8,
        "completeness": 7,
        "conciseness": 9,
        "format": 8,
        "domain_relevance": 7
    }},
    "output_b_scores": {{
        "accuracy": 9,
        "completeness": 8,
        "conciseness": 8,
        "format": 9,
        "domain_relevance": 8
    }},
    "explanation": "Output B cung cấp phân tích chi tiết hơn với bằng chứng hỗ trợ tốt hơn",
    "recommendation": "b",
    "confidence": 0.85
}}

Important (Quan trọng)
- Tất cả điểm số phải nằm trong đoạn từ 1-10
- Recommendation (khuyến nghị) phải chính xác là "a" hoặc "b"
- confidence phải là một con số nằm giữa 0.5 và 1.0
"""

class QualityJudge:
    def __init__(self, chat_fn, cache_dir: str = "prompt_cache"):
        """
        Initialize the quality judge with a chat function and cache directory.
        
        Args:
            chat_fn: Function that takes (messages, temperature, max_tokens) and returns str
            cache_dir: Directory to store evaluation cache
        """
        self.chat_fn = chat_fn
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _extract_json_from_response(self, response: str) -> str:
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

    def _get_cache_path(self, dataset: str, prompt_type: str) -> Path:
        """Get the cache file path for a specific dataset and prompt type."""
        return self.cache_dir / f"judge_{dataset}_{prompt_type}.json"

    def _cache_result(self, dataset: str, prompt_type: str, result: Dict[str, Any]):
        """Cache the evaluation result."""
        cache_path = self._get_cache_path(dataset, prompt_type)
        try:
            cache_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"Warning: Failed to cache evaluation result: {e}")

    def _get_cached_result(self, dataset: str, prompt_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached evaluation result if it exists."""
        cache_path = self._get_cache_path(dataset, prompt_type)
        try:
            if cache_path.exists():
                return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return None

    def evaluate_outputs(
        self, 
        output_a: str,
        output_b: str,
        dataset: str,
        prompt_type: str,
        context: str = "",
        use_cache: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluate two different outputs and determine which is better.
        
        Args:
            output_a: The first output (typically from base prompt)
            output_b: The second output (typically from optimized prompt)
            dataset: Name of the dataset being processed
            prompt_type: Type of prompt (e.g., "absa", "topic", "rag")
            context: Optional context to help evaluation
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (recommended_output, evaluation_details)
        """
        if use_cache:
            cached = self._get_cached_result(dataset, prompt_type)
            if cached:
                recommended = output_a if cached["recommendation"] == "a" else output_b
                return recommended, cached

        messages = [
            {"role": "system", "content": "Return only the JSON evaluation as specified."},
            {"role": "user", "content": QUALITY_JUDGE_PROMPT.format(
                output_a=output_a,
                output_b=output_b,
                context=context
            )}
        ]

        try:
            raw_response = self.chat_fn(messages, temperature=0.1, max_tokens=800)
            clean_json = self._extract_json_from_response(raw_response)
            result = json.loads(clean_json)
            
            if use_cache:
                self._cache_result(dataset, prompt_type, result)
            
            recommended = output_a if result["recommendation"] == "a" else output_b
            return recommended, result
        except Exception as e:
            print(f"Warning: Evaluation failed ({e}), defaulting to first output")
            return output_a, {
                "error": str(e),
                "recommendation": "a",
                "confidence": 0.5,
                "explanation": "Evaluation failed, defaulting to first output"
            }

    def get_average_scores(self, dataset: str, prompt_type: str) -> Optional[Dict[str, float]]:
        """
        Get average scores from cached evaluations for analysis.
        
        Returns:
            Dict with average scores or None if no cache exists
        """
        cached = self._get_cached_result(dataset, prompt_type)
        if not cached:
            return None
            
        try:
            a_scores = cached["output_a_scores"]
            b_scores = cached["output_b_scores"]
            
            return {
                "base_prompt_avg": sum(a_scores.values()) / len(a_scores),
                "optimized_prompt_avg": sum(b_scores.values()) / len(b_scores),
                "confidence": cached["confidence"]
            }
        except Exception:
            return None