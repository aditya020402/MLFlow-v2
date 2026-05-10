"""Shared LLM client — all agents import from here."""

import os
from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
