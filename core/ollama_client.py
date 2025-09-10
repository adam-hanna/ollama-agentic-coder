import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator
import aiohttp
from .config import OllamaConfig


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        # Use longer timeout for large models and connection setup
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=30,  # 30 seconds for connection
            sock_read=self.config.timeout  # Full timeout for reading response
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 3,
    ) -> str:
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")

        payload = {
            "model": model or self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if system:
            payload["system"] = system

        # Retry logic for handling timeouts and connection issues
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries} for model {payload['model']}")
                
                async with self.session.post(
                    f"{self.config.host}/api/generate", json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    if stream:
                        return await self._handle_stream(response)
                    else:
                        result = await response.json()
                        return result.get("response", "")
                        
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to Ollama after {max_retries} attempts. "
                                  f"Last error: {str(e)}. Please check if Ollama is running and "
                                  f"the model '{payload['model']}' is available.")
                
                # Exponential backoff: wait longer between retries
                wait_time = 2 ** attempt
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    async def _handle_stream(self, response) -> str:
        full_response = ""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        full_response += data["response"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        return full_response

    async def chat(
        self,
        messages: list[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 3,
    ) -> str:
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        # Retry logic for handling timeouts and connection issues
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Chat attempt {attempt + 1}/{max_retries} for model {payload['model']}")
                
                async with self.session.post(
                    f"{self.config.host}/api/chat", json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    if stream:
                        return await self._handle_chat_stream(response)
                    else:
                        result = await response.json()
                        return result.get("message", {}).get("content", "")
                        
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning(f"Chat attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to Ollama after {max_retries} attempts. "
                                  f"Last error: {str(e)}. Please check if Ollama is running and "
                                  f"the model '{payload['model']}' is available.")
                
                # Exponential backoff: wait longer between retries
                wait_time = 2 ** attempt
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    async def _handle_chat_stream(self, response) -> str:
        full_response = ""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        return full_response

    async def list_models(self) -> list[str]:
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")

        async with self.session.get(f"{self.config.host}/api/tags") as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status}")

            result = await response.json()
            return [model["name"] for model in result.get("models", [])]
