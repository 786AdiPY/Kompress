"""AI-driven compression recommendation engine powered by Groq LLM.

Parses optional model context text or uploaded context files (.pdf, .docx, .md, .txt)
and queries Groq LLM to intelligently recommend optimal framework, task, compression
methods, and quality tolerance thresholds.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

# Known Groq model endpoints and models
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound")


def extract_context_from_bytes(filename: str, content: bytes) -> str:
    """Extract plain text from uploaded context files (.pdf, .docx, .md, .txt)."""
    if not content:
        return ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "docx":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [node.text for node in tree.iter() if node.tag.endswith("}t") and node.text]
                return "\n".join(texts)
        except Exception:
            return content.decode("utf-8", errors="ignore")

    elif ext == "pdf":
        try:
            raw = content.decode("latin-1", errors="ignore")
            matches = re.findall(r"\((.*?)\)\s*Tj", raw)
            if not matches:
                matches = re.findall(r"\[(.*?)\]\s*TJ", raw)
            extracted = " ".join(matches).strip()
            return extracted if len(extracted) > 20 else raw[:2000]
        except Exception:
            return content.decode("utf-8", errors="ignore")

    else:  # .md, .txt, .json, etc.
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")


def get_ai_recommendation(
    model_name: str | None = None,
    framework: str | None = None,
    target_hardware: str = "cpu-generic",
    context_text: str = "",
    context_file_name: str | None = None,
    context_file_bytes: bytes | None = None,
) -> dict:
    """Generate AI compression recommendation using Groq API."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Extract text from uploaded context file if provided
    file_context = ""
    if context_file_bytes and context_file_name:
        file_context = extract_context_from_bytes(context_file_name, context_file_bytes)

    full_context = "\n".join(filter(None, [context_text.strip(), file_context.strip()])).strip()

    # Fallback response if no Groq API key configured
    fallback = {
        "framework": framework or "pytorch",
        "task": "binary_classification",
        "recommended_methods": ["onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"],
        "quality_tolerance": "standard",
        "reasoning": f"Defaulting to standard quantization suite for {target_hardware}.",
        "context_summary": full_context[:300] if full_context else "No context provided.",
        "ai_powered": False,
    }

    if not api_key:
        return fallback

    prompt = f"""
System: You are an expert Machine Learning Compression and Quantization Architect at Kompress AI.
Given the model specs, target deployment hardware, and domain context below, output a valid JSON recommendation object.

Input Specs:
- Model Name: {model_name or 'Unknown'}
- Framework Hint: {framework or 'Auto-detect'}
- Target Deployment Hardware: {target_hardware}
- Domain / Task Context:
\"\"\"
{full_context or 'Standard deep learning / tabular model for high-throughput deployment.'}
\"\"\"

Requirements:
Return ONLY a valid JSON object matching this exact schema:
{{
  "framework": "pytorch" | "sklearn" | "xgboost" | "lightgbm" | "onnx",
  "task": "binary_classification" | "multiclass_classification" | "regression",
  "recommended_methods": ["onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static", "onnx_fp16", "trt_int8"],
  "quality_tolerance": "strict" | "standard" | "relaxed",
  "reasoning": "Concise 1-2 sentence engineering rationale explaining the recommended compression strategy for this model and target hardware.",
  "context_summary": "Brief 1-sentence summary of model purpose."
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Kompress/1.0",
    }
    payload = {
        "model": DEFAULT_GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    try:
        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            # Extract JSON from code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            parsed["ai_powered"] = True
            if not parsed.get("context_summary"):
                parsed["context_summary"] = full_context[:200] if full_context else "Analyzed model structure."
            return parsed
    except Exception as err:
        print(f"[recommender] Groq API warning: {err}. Using rule-based fallback.", flush=True)
        fallback["reasoning"] = f"Rule-based recommendation ({err})."
        return fallback
