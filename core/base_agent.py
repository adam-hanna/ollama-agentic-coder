from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from .ollama_client import OllamaClient
from .config import Config

class AgentMessage(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class AgentState(BaseModel):
    messages: List[AgentMessage] = []
    current_task: Optional[str] = None
    context: Dict[str, Any] = {}
    next_agent: Optional[str] = None

class BaseAgent(ABC):
    def __init__(self, name: str, config: Config, system_prompt: Optional[str] = None):
        self.name = name
        self.config = config
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.ollama_client = None
    
    async def __aenter__(self):
        self.ollama_client = OllamaClient(self.config.ollama)
        await self.ollama_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ollama_client:
            await self.ollama_client.__aexit__(exc_type, exc_val, exc_tb)
    
    @abstractmethod
    def _default_system_prompt(self) -> str:
        pass
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        pass
    
    async def generate_response(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        if not self.ollama_client:
            raise RuntimeError("Agent must be used as async context manager")
        
        # Use agent-specific model if not explicitly provided
        if model is None:
            model = self._get_agent_model()
        
        return await self.ollama_client.generate(
            prompt=prompt,
            system=system or self.system_prompt,
            model=model
        )
    
    async def chat_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> str:
        if not self.ollama_client:
            raise RuntimeError("Agent must be used as async context manager")
        
        # Use agent-specific model if not explicitly provided
        if model is None:
            model = self._get_agent_model()
        
        chat_messages = [{"role": "system", "content": self.system_prompt}]
        chat_messages.extend(messages)
        
        return await self.ollama_client.chat(
            messages=chat_messages,
            model=model
        )
    
    def _get_agent_model(self) -> str:
        """Get the model assigned to this agent, fallback to default"""
        agent_models = self.config.agents.models
        return agent_models.get(self.name, self.config.ollama.model)
    
    def add_message(self, state: AgentState, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> AgentState:
        message = AgentMessage(role=role, content=content, metadata=metadata)
        state.messages.append(message)
        
        if self.config.agents.enable_memory and len(state.messages) > self.config.agents.memory_window:
            state.messages = state.messages[-self.config.agents.memory_window:]
        
        return state
    
    def get_conversation_history(self, state: AgentState, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        messages = state.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [{"role": msg.role, "content": msg.content} for msg in messages]