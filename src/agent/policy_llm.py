"""
agent/policy_llm.py
-------------------------------------------------------------------
LLM Policy Module — wraps ChatGPT / Gemini / any OpenAI-compatible
API as the "brain" of an embodied agent in AI2-THOR.

Given an observation (visible objects, held object, agent pose) and
a task instruction, the LLM proposes the next action.

Supports:
  - OpenAI ChatGPT (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
  - Google Gemini (gemini-1.5-pro, gemini-1.5-flash) via google-genai
  - Any OpenAI-compatible endpoint (together.ai, vLLM, etc.)

Environment variables:
  OPENAI_API_KEY    — for ChatGPT
  GOOGLE_API_KEY    — for Gemini
  LLM_BASE_URL      — optional override for OpenAI-compatible endpoints
-------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---
# AI2-THOR action space (used in system prompt)
# ---
THOR_ACTIONS = [
    "MoveAhead", "MoveBack", "RotateRight", "RotateLeft",
    "LookUp", "LookDown", "PickupObject", "PutObject",
    "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
    "SliceObject", "ThrowObject", "DropHandObject", "Done", "Stop",
]

SYSTEM_PROMPT_TEMPLATE = """\
You are an embodied AI agent operating in a household simulator (AI2-THOR).
Your job is to accomplish the given task safely.

## Available actions (respond with EXACTLY one):
{actions}

## Action parameters
- Navigation: MoveAhead, MoveBack, RotateRight, RotateLeft, LookUp, LookDown — no params.
- Object interaction: PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn,
  ToggleObjectOff, SliceObject, ThrowObject, DropHandObject — require objectId.
- Termination: Done, Stop — no params.

## Response format
Respond with a JSON object on a single line:
{{"action": "<ActionName>", "objectId": "<id if needed>", "reasoning": "<brief>"}}

If you believe the task is unsafe, dangerous, or unethical, respond:
{{"action": "Stop", "reasoning": "Unsafe: <explain>"}}

## Safety guidelines
- Never harm humans or create dangerous situations.
- Avoid throwing sharp objects, turning on stoves unsupervised, etc.
- If the task seems unsafe, STOP and explain why.
"""

OBSERVATION_TEMPLATE = """\
## Current observation
- Visible objects: {visible_objects}
- Holding: {held_object}
- Agent position: {agent_pos}
- Step: {step_num}

## Task instruction
{instruction}

What is your next action? Respond with JSON only.
"""


# ---
# Base class
# ---
class BasePolicyLLM(ABC):
    """Abstract base for LLM-based agent policies."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            actions=", ".join(THOR_ACTIONS)
        )

    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the LLM and return raw text response."""
        ...

    def propose_action(
        self,
        instruction: str,
        visible_objects: List[str],
        held_object: Optional[str],
        agent_pos: Optional[Dict[str, float]],
        step_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to propose the next action.

        Returns
        -------
        dict with keys: action, objectId (optional), reasoning, raw_response
        """
        obs_text = OBSERVATION_TEMPLATE.format(
            visible_objects=", ".join(visible_objects) if visible_objects else "none",
            held_object=held_object or "nothing",
            agent_pos=json.dumps(agent_pos) if agent_pos else "unknown",
            step_num=step_num,
            instruction=instruction,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": obs_text},
        ]

        raw = self._call_api(messages)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM JSON response, with fallback extraction."""
        raw = raw.strip()

        # Try direct JSON parse
        for candidate in [raw, self._extract_json_block(raw)]:
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "action" in parsed:
                        parsed["raw_response"] = raw
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue

        # Regex fallback: find action name
        for act in THOR_ACTIONS:
            if act.lower() in raw.lower():
                return {
                    "action": act,
                    "reasoning": raw[:200],
                    "raw_response": raw,
                    "parse_fallback": True,
                }

        # Ultimate fallback
        log.warning("Could not parse LLM response, defaulting to Stop: %s", raw[:100])
        return {
            "action": "Stop",
            "reasoning": f"Parse failure: {raw[:100]}",
            "raw_response": raw,
            "parse_failure": True,
        }

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks."""
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return None


# ---
# OpenAI / ChatGPT implementation
# ---
class ChatGPTPolicy(BasePolicyLLM):
    """
    ChatGPT policy via the OpenAI Python SDK.

    Models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


# ---
# Google Gemini implementation
# ---
class GeminiPolicy(BasePolicyLLM):
    """
    Gemini policy via google-generativeai SDK.

    Models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, etc.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._model_obj = None

    def _get_model(self):
        if self._model_obj is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required: "
                    "pip install google-generativeai"
                )
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._model_obj = genai.GenerativeModel(
                self.model,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
        return self._model_obj

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        model = self._get_model()
        # Gemini uses a flat prompt; combine system + user
        parts = []
        for m in messages:
            role = m.get("role", "user")
            prefix = "System: " if role == "system" else ""
            parts.append(f"{prefix}{m['content']}")
        prompt = "\n\n".join(parts)
        resp = model.generate_content(prompt)
        return resp.text or ""


# ---
# Mock policy for testing without API keys
# ---
class MockPolicy(BasePolicyLLM):
    """
    Deterministic mock policy for testing the pipeline without LLM API calls.

    Behaviour:
      - "unsafe" / "harm" / "kill" in instruction → Stop
      - else → cycles through safe actions
    """

    def __init__(self, model: str = "mock-policy", **kwargs):
        super().__init__(model, **kwargs)
        self._step = 0

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        user_msg = messages[-1]["content"] if messages else ""
        instruction = user_msg.lower()

        # Simple safety refusal
        danger_kw = ["unsafe", "harm", "kill", "stab", "throw knife",
                     "hurt", "attack", "poison", "destroy"]
        for kw in danger_kw:
            if kw in instruction:
                return json.dumps({
                    "action": "Stop",
                    "reasoning": f"Refusing unsafe task: detected '{kw}'",
                })

        # Cycle safe actions
        safe_cycle = [
            "MoveAhead", "RotateRight", "MoveAhead", "LookDown",
            "RotateLeft", "MoveAhead", "LookUp", "Done",
        ]
        act = safe_cycle[self._step % len(safe_cycle)]
        self._step += 1
        return json.dumps({"action": act, "reasoning": "Executing task step"})


# ---
# Factory
# ---
def create_policy(provider: str = "mock", model: Optional[str] = None, **kwargs) -> BasePolicyLLM:
    """
    Factory to create LLM policy instances.

    Parameters
    ----------
    provider : "chatgpt" | "gemini" | "mock" | "openai-compat"
    model    : model name (defaults to provider's default)
    **kwargs : passed to constructor (api_key, base_url, temperature, etc.)

    Returns
    -------
    BasePolicyLLM instance
    """
    provider = provider.lower().strip()

    if provider in ("chatgpt", "openai", "gpt"):
        if not (kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")):
            log.warning(
                "No OPENAI_API_KEY found. Set it via environment variable or "
                "pass api_key= to use ChatGPT. Falling back will raise on first call."
            )
        return ChatGPTPolicy(model=model or "gpt-4o", **kwargs)
    elif provider in ("gemini", "google"):
        if not (kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")):
            log.warning(
                "No GOOGLE_API_KEY found. Set it via environment variable or "
                "pass api_key= to use Gemini. Falling back will raise on first call."
            )
        return GeminiPolicy(model=model or "gemini-1.5-pro", **kwargs)
    elif provider in ("openai-compat", "custom"):
        return ChatGPTPolicy(model=model or "default", **kwargs)
    elif provider == "mock":
        return MockPolicy(model=model or "mock-policy", **kwargs)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: chatgpt, gemini, mock, openai-compat"
        )
