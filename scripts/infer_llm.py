#!/usr/bin/env python
"""
Step 4 – Run LLM inference over prompts.

Reads:  results/variantbench_100_prompts.jsonl
Writes: results/{model_name}_raw.jsonl      (raw text)
        results/{model_name}_parsed.jsonl   (clean JSON dict per variant)

Usage examples:
    python scripts/infer_llm.py --provider openai --model gpt-4o-mini
    python scripts/infer_llm.py --provider anthropic --model claude-3-opus-20240229
    python scripts/infer_llm.py --provider hf --endpoint http://localhost:8080/v1/completions --model llama-3-70b-instruct
"""
import os
from dotenv import load_dotenv
load_dotenv()

try:
    import google.generativeai as genai
except Exception:
    genai = None

import argparse, json, pathlib, re, time
import orjson
from dataclasses import dataclass
from typing import Dict, Any, Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional imports guarded per provider
try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROMPTS_JSONL = RESULTS / "variantbench_100_prompts.jsonl"

CONTEXT_LIMIT = 4096  
OUTPUT_TOKEN_HARD_CAP = 4500 

def estimate_tokens(text: str) -> int:

    return int(len(text.split()) * 1.3)

def safe_max_output_tokens(prompt: str, context_limit=CONTEXT_LIMIT, min_output_tokens=300, max_output_tokens=OUTPUT_TOKEN_HARD_CAP) -> int:
    prompt_tokens = estimate_tokens(prompt)
    out_tokens = context_limit - prompt_tokens
    # Never allow more than the hard output cap
    capped = min(max_output_tokens, out_tokens)
    if capped < min_output_tokens:
        print(f"Prompt too long (tokens={prompt_tokens})! Forcing max_output_tokens={min_output_tokens}")
        return min_output_tokens
    return capped


# Utilities 

def iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(orjson.dumps(r).decode("utf-8") + "\n")

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(text: str) -> dict | None:
    if not text:
        print("WARNING: Empty response from LLM")
        return None

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text else text
    m = JSON_BLOCK_RE.search(text)
    if not m:
        print(f"WARNING: No JSON found in response:\n{text[:200]}")
        return None
    try:
        return json.loads(m.group(0))
    except Exception as e:
        print(f"WARNING: Error parsing JSON: {e}\n{text[:200]}")
        return None



# Clients 

@dataclass
class LLMClient:
    provider: str
    model: str
    endpoint: str | None = None   # Only used for HF/custom
    temperature: float = 0.0

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=8),
           retry=retry_if_exception_type(Exception))
    def generate(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._openai(prompt)
        elif self.provider == "anthropic":
            return self._anthropic(prompt)
        elif self.provider == "gemini":
            return self._gemini(prompt)
        elif self.provider == "hf":
            return self._hf(prompt)
        else:
            raise ValueError(f"Unknown provider {self.provider}")

    def _openai(self, prompt: str) -> str:
        if openai is None:
            raise RuntimeError("openai package not installed")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",    "content": prompt}
            ]
        )
        return resp.choices[0].message.content

    def _anthropic(self, prompt: str) -> str:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        # Claude returns a list of content blocks; join text blocks.
        return "".join([c.text for c in resp.content if c.type == "text"])

    def _hf(self, prompt: str) -> str:
        if not self.endpoint:
            raise RuntimeError("HF/custom provider requires --endpoint")
        
        if self.endpoint.endswith("/chat/completions"):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        else:  
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": 1500,
            }

        headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN','')}"}
        with httpx.Client(timeout=120) as client:
            r = client.post(self.endpoint, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        if "choices" in data:

            choice = data["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            return choice.get("text", "")
        return data.get("generated_text", "")
    
    def _gemini(self, prompt: str) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai package not installed")
        api_key = os.environ.get("GEMINI_API_KEY") 
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env")
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(self.model)
        # Dynamic token budgeting
        max_output_tokens = safe_max_output_tokens(prompt)
        print(f"Prompt tokens (est): {estimate_tokens(prompt)} | max_output_tokens: {max_output_tokens}")

        # ----------------------------------------------------
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": max_output_tokens
                }
            )

            try: 
                if hasattr(resp, "candidates") and resp.candidates:
                    candidate = resp.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    print(f"Gemini finish_reason: {finish_reason}")
            except Exception as e:
                print(f"Could not extract finish_reason: {e}")
        except Exception as e:
            print(f"Exception during Gemini call: {e}")
            return ""
        try:
            if hasattr(resp, "text") and resp.text:
                return resp.text
        except Exception as e:
            print(f"Error extracting Gemini text: {e}")
            return ""
        print("WARNING: Gemini returned empty response!")
        print("Prompt was:\n", prompt)
        print("Full response object:\n", resp)
        return ""

    



# Main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "hf", "gemini"],
                help="Which API to call")
    ap.add_argument("--model", required=True, help="Model name")
    ap.add_argument("--endpoint", default=None,
                    help="For HF/custom endpoints (RunPod/vLLM/TGI). Example: http://<pod-url>/v1/chat/completions")
    ap.add_argument("--input", default=str(PROMPTS_JSONL), help="Path to prompt jsonl")
    ap.add_argument("--out_prefix", default=None,
                    help="Prefix for output files (default = model name)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls (rate limit safety)")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.input)
    prefix   = args.out_prefix or args.model.replace("/", "_")

    # Extract Track A/B from input filename 
    match = re.search(r'track([AB])', str(in_path), re.IGNORECASE)
    if match:
        track = f"_track{match.group(1).upper()}"
    else:
        track = ""

    OUTDIR = RESULTS / f"{args.provider.capitalize()}Results"
    OUTDIR.mkdir(exist_ok=True)

    raw_out    = OUTDIR / f"{prefix}{track}_raw.jsonl"
    parsed_out = OUTDIR / f"{prefix}{track}_parsed.jsonl"

    client = LLMClient(provider=args.provider, model=args.model,
                       endpoint=args.endpoint, temperature=0.0)

    raw_rows    = []
    parsed_rows = []

    for rec in iter_jsonl(in_path):
        vid     = rec["vid"]
        prompt  = rec["prompt"]

        # Call model
        text = client.generate(prompt)
        print(f"Finished vid={vid}")  # 

        # Parse JSON
        parsed = extract_first_json(text)
        if parsed is None:
            parsed = {"_parse_error": True, "raw": text}

        raw_rows.append({"vid": vid, "prompt": prompt, "response": text})
        parsed_rows.append({"vid": vid, **parsed})

        if args.sleep > 0:
            time.sleep(args.sleep)

    write_jsonl(raw_out, raw_rows)
    write_jsonl(parsed_out, parsed_rows)

    print(f"✅ Raw outputs  -> {raw_out}")
    print(f"✅ Parsed JSON  -> {parsed_out}")
    print("Done.")


if __name__ == "__main__":
    import os
    main()
