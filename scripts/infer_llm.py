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


# ----------------------- Utilities -----------------------

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

def extract_first_json(text: str) -> Dict[str, Any] | None:
    """Grab the first {...} block and json.loads it."""
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ----------------------- Clients -------------------------

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
        # Two common APIs exist: (a) OpenAI-compatible /v1/chat/completions; (b) text-generation-inference /v1/completions
        # We'll support both; detect by path.
        if self.endpoint.endswith("/chat/completions"):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        else:  # /v1/completions style
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
        # Try to normalize return text
        if "choices" in data:
            # OpenAI/TGI like
            choice = data["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            return choice.get("text", "")
        return data.get("generated_text", "")


# ----------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "hf"],
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
    raw_out  = RESULTS / f"{prefix}_raw.jsonl"
    parsed_out = RESULTS / f"{prefix}_parsed.jsonl"

    client = LLMClient(provider=args.provider, model=args.model,
                       endpoint=args.endpoint, temperature=0.0)

    raw_rows    = []
    parsed_rows = []

    for rec in iter_jsonl(in_path):
        vid     = rec["vid"]
        prompt  = rec["prompt"]

        # Call model
        text = client.generate(prompt)

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
