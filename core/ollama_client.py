import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator
import aiohttp
from .config import OllamaConfig

class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False
    ) -> str:
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")
        
        payload = {
            "model": model or self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        async with self.session.post(
            f"{self.config.host}/api/generate",
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status} - {await response.text()}")
            
            if stream:
                return await self._handle_stream(response)
            else:
                result = await response.json()
                return result.get("response", "")
    
    async def _handle_stream(self, response) -> str:
        full_response = ""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        return full_response
    
    async def chat(
        self,
        messages: list[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False
    ) -> str:
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")
        
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        async with self.session.post(
            f"{self.config.host}/api/chat",
            json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status} - {await response.text()}")
            
            if stream:
                return await self._handle_chat_stream(response)
            else:
                result = await response.json()
                return result.get("message", {}).get("content", "")
    
    async def _handle_chat_stream(self, response) -> str:
        full_response = ""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'message' in data and 'content' in data['message']:
                        full_response += data['message']['content']
                    if data.get('done', False):
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