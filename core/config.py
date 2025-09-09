from typing import Optional, Dict
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class OllamaConfig(BaseModel):
    host: str = Field(default="http://localhost:11434", description="Ollama server host")
    model: str = Field(default="qwen2.5-coder:32b", description="Default model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=4096, description="Maximum tokens to generate")
    timeout: int = Field(default=300, description="Request timeout in seconds")

class AgentConfig(BaseModel):
    max_iterations: int = Field(default=10, description="Maximum iterations for agent loops")
    enable_memory: bool = Field(default=True, description="Enable conversation memory")
    memory_window: int = Field(default=20, description="Number of messages to keep in memory")
    
    # Per-agent model configurations
    models: Dict[str, str] = Field(default_factory=lambda: {
        "supervisor": "qwen2.5:32b",           # Reasoning and orchestration
        "websearch": "llama3.1:8b",           # General purpose, lightweight
        "code_review": "qwen2.5-coder:32b",   # Code analysis specialist
        "code_analyzer": "qwen2.5-coder:32b", # Code structure analysis
        "file_operations": "qwen2.5-coder:32b", # File manipulation
        "test_generator": "qwen2.5-coder:32b", # Test creation
        "refactoring": "qwen2.5-coder:32b",   # Code optimization
        "git": "llama3.1:8b",                 # Git commands and workflows
        "documentation": "llama3.1:8b",      # Documentation writing
        "debugging": "qwen2.5-coder:32b",     # Debug analysis
        "security": "qwen2.5-coder:32b",      # Security analysis
    }, description="Model assignments per agent")

class SearchConfig(BaseModel):
    search_engine: str = Field(default="duckduckgo", description="Search engine to use")
    max_results: int = Field(default=5, description="Maximum search results to return")
    
class CodeAnalyzerConfig(BaseModel):
    supported_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go", ".rs"],
        description="File extensions to analyze"
    )
    max_file_size: int = Field(default=1024*1024, description="Maximum file size to analyze in bytes")
    enable_ast_indexing: bool = Field(default=True, description="Enable AST-based code indexing")
    auto_background_indexing: bool = Field(default=True, description="Enable automatic background indexing on startup")
    watch_file_changes: bool = Field(default=True, description="Watch for file changes and auto-reindex")

class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    code_analyzer: CodeAnalyzerConfig = Field(default_factory=CodeAnalyzerConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama=OllamaConfig(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "4096")) if os.getenv("OLLAMA_MAX_TOKENS") else None,
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "300"))
            )
        )

def get_config() -> Config:
    return Config.from_env()