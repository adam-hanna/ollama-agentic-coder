from typing import Optional, Dict
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class OllamaConfig(BaseModel):
    host: str = Field(
        default="http://localhost:11434", description="Ollama server host"
    )
    model: str = Field(default="qwen2.5-coder:32b", description="Default model to use")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens to generate"
    )
    timeout: int = Field(default=600, description="Request timeout in seconds")


class AgentConfig(BaseModel):
    max_iterations: int = Field(
        default=10, description="Maximum iterations for agent loops"
    )
    enable_memory: bool = Field(default=True, description="Enable conversation memory")
    memory_window: int = Field(
        default=20, description="Number of messages to keep in memory"
    )

    # Per-agent model configurations - optimized for domain-specific tasks
    models: Dict[str, str] = Field(
        default_factory=lambda: {
            # Code-specialized tasks (need coding expertise)
            "code_review": "qwen2.5-coder:32b",  # Code analysis, bug detection
            "code_analyzer": "qwen2.5-coder:32b",  # AST analysis, code structure
            "refactoring": "qwen2.5-coder:32b",  # Code patterns, optimization
            "test_generator": "qwen2.5-coder:32b",  # Test code generation
            "file_operations": "qwen2.5-coder:32b",  # Code-aware file editing
            # Language/communication tasks (better with general language models)
            "documentation": "llama3.1:8b",  # Natural language, clear writing
            "git": "llama3.1:8b",  # Commit messages, workflows
            "websearch": "llama3.1:8b",  # Query formulation, summarization
            # Orchestration (pure reasoning, not code-specific)
            "supervisor": "qwen2.5:32b",  # Complex multi-step reasoning
            # System operations
            "command_line": "llama3.1:8b",  # Command explanation and safety
            # Context management
            "context_manager": "llama3.1:8b",  # Memory management and summarization
        },
        description="Domain-optimized model assignments per agent",
    )


class SearchConfig(BaseModel):
    search_engine: str = Field(default="duckduckgo", description="Search engine to use")
    max_results: int = Field(default=5, description="Maximum search results to return")


class CodeAnalyzerConfig(BaseModel):
    supported_extensions: list[str] = Field(
        default=[
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
        ],
        description="File extensions to analyze",
    )
    max_file_size: int = Field(
        default=1024 * 1024, description="Maximum file size to analyze in bytes"
    )
    enable_ast_indexing: bool = Field(
        default=True, description="Enable AST-based code indexing"
    )
    auto_background_indexing: bool = Field(
        default=True, description="Enable automatic background indexing on startup"
    )
    watch_file_changes: bool = Field(
        default=True, description="Watch for file changes and auto-reindex"
    )


class CommandLineConfig(BaseModel):
    safety_mode: str = Field(
        default="safe", description="Command safety mode: safe, whitelist, yolo"
    )
    safe_commands: list[str] = Field(
        default=[
            # File/directory operations (read-only)
            "ls",
            "dir",
            "pwd",
            "find",
            "locate",
            "which",
            "whereis",
            "cat",
            "head",
            "tail",
            "less",
            "more",
            "grep",
            "awk",
            "sed",
            "wc",
            "sort",
            "uniq",
            "cut",
            "tr",
            "file",
            "stat",
            "du",
            "df",
            # System info (read-only)
            "ps",
            "top",
            "htop",
            "free",
            "uptime",
            "whoami",
            "id",
            "groups",
            "uname",
            "hostname",
            "date",
            "cal",
            "env",
            "printenv",
            # Network (read-only)
            "ping",
            "curl",
            "wget",
            "nslookup",
            "dig",
            "whois",
            "netstat",
            # Development tools (mostly read-only)
            "node",
            "python",
            "python3",
            "pip",
            "npm",
            "cargo",
            "go",
            "javac",
            "java",
            "gcc",
            "g++",
            "make",
            "cmake",
            # Package managers (info only)
            "apt",
            "yum",
            "brew",
            "pacman",
            "zypper",
        ],
        description="Commands allowed in 'safe' mode",
    )
    user_whitelist: list[str] = Field(
        default=[],
        description="Additional commands approved by user in 'whitelist' mode",
    )
    max_output_length: int = Field(
        default=10000, description="Maximum command output length"
    )
    timeout_seconds: int = Field(default=30, description="Command execution timeout")


class ContextManagerConfig(BaseModel):
    auto_summarize: bool = Field(
        default=True, description="Automatically summarize old conversations"
    )
    max_context_age_hours: int = Field(
        default=168, description="Maximum age for stored context (1 week default)"
    )
    context_store_max_size: int = Field(
        default=1000, description="Maximum number of context entries to store"
    )
    auto_context_threshold: int = Field(
        default=15, description="Message count threshold for auto-context management"
    )
    enable_semantic_search: bool = Field(
        default=False,
        description="Enable semantic search for context retrieval (requires embeddings)",
    )
    relevance_threshold: float = Field(
        default=1.0, description="Minimum relevance score for context retrieval"
    )


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    code_analyzer: CodeAnalyzerConfig = Field(default_factory=CodeAnalyzerConfig)
    command_line: CommandLineConfig = Field(default_factory=CommandLineConfig)
    context_manager: ContextManagerConfig = Field(default_factory=ContextManagerConfig)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama=OllamaConfig(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
                if os.getenv("OLLAMA_MAX_TOKENS")
                else None,
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "600")),
            )
        )


def get_config() -> Config:
    return Config.from_env()
