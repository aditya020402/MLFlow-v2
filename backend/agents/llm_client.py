"""Shared LLM client — all agents import from here.

Set OPENAI_API_KEY in backend/.env to authenticate.
"""

import os
from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
