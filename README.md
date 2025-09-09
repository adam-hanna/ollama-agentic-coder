# Multi-Agent Coding Assistant

A sophisticated multi-agent coding assistant powered by LangGraph and Ollama, designed to help with complex software development tasks through specialized AI agents.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated agents working together for complex tasks
- **Local AI Models**: Uses Ollama for privacy and control over your AI models  
- **Per-Agent Model Optimization**: Different models optimized for each agent's specialty
- **Automatic Model Management**: Download required models on first run
- **Automatic Background Indexing**: Real-time codebase analysis with file watching  
- **Flexible Indexing Control**: Enable/disable automatic indexing with CLI options
- **Command Safety Controls**: Three security modes for system command execution
- **Specialized Agents** (10 total):
  - 🔍 **WebSearchAgent** (`llama3.1:8b`): Searches for documentation, examples, best practices
  - 🔍 **CodeReviewAgent** (`qwen2.5-coder:32b`): Analyzes code quality, bugs, security issues
  - 📊 **CodeAnalyzerAgent** (`qwen2.5-coder:32b`): Indexes codebases, tracks dependencies, AST analysis
  - 📁 **FileOperationsAgent** (`qwen2.5-coder:32b`): Direct file manipulation - read, write, edit files
  - 🧪 **TestGeneratorAgent** (`qwen2.5-coder:32b`): Creates comprehensive unit and integration tests
  - 🔄 **RefactoringAgent** (`qwen2.5-coder:32b`): Code optimization, design patterns, modernization
  - 📝 **GitAgent** (`llama3.1:8b`): Git workflows, commit messages, merge conflict resolution
  - 📚 **DocumentationAgent** (`llama3.1:8b`): README files, API docs, docstrings, user guides
  - 🖥️ **CommandLineAgent** (`llama3.1:8b`): System command execution with safety controls
  - 🎯 **SupervisorAgent** (`qwen2.5:32b`): Orchestrates multi-agent workflows using LangGraph

## 📋 Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally
- qwen2.5-coder:32b model (or your preferred coding model)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd langgraph-ollama-agent
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source ./.venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Ollama** (if not already installed):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

5. **Pull the coding model**:
```bash
ollama pull qwen2.5-coder:32b
```

## ⚙️ Configuration

The system can be configured via environment variables or the default config:

### Environment Variables
```bash
# Ollama settings
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5-coder:32b"
export OLLAMA_TEMPERATURE="0.1"
export OLLAMA_MAX_TOKENS="4096"
export OLLAMA_TIMEOUT="300"
```

### Available Models
You can use any Ollama-compatible model:
```bash
# Other excellent coding models
ollama pull deepseek-coder:33b
ollama pull codellama:34b
ollama pull starcoder2:15b
```

### Command Line Safety Modes
The CommandLineAgent operates in three safety modes:

- **SAFE** (default): Only pre-approved read-only commands (ls, cat, grep, etc.)
- **WHITELIST**: Safe commands + user-approved commands (prompts for approval)  
- **YOLO**: All commands allowed (use with extreme caution)

Configure via environment variable:
```bash
export COMMAND_SAFETY_MODE="safe"  # or "whitelist" or "yolo"
```

## 🚀 Usage

### Basic Usage
```bash
python main.py
```

### CLI Options
```bash
python main.py --help
python main.py --host http://localhost:11434 --model qwen2.5-coder:32b --temperature 0.1

# Indexing control options
python main.py --no-auto-index        # Disable automatic background indexing
python main.py --no-file-watch        # Disable file change watching
python main.py --no-auto-index --no-file-watch  # Disable both
```

### Interactive Commands

Once running, you can use these commands:

- `/help` - Show available commands and indexing status
- `/index` - Manual indexing commands:
  - `/index` - Index current directory
  - `/index [path]` - Index specific directory
  - `/index auto` - Toggle automatic background indexing
  - `/index stop` - Stop background indexing
- `/config` - Show current configuration
- `/agents` - List available agents and their assigned models
- `/clear` - Clear conversation history
- `/exit` - Exit the application

### Example Interactions

**Code Review:**
```
You: Review this Python file for bugs and security issues: src/auth.py
```

**File Operations:**
```
You: Create a new config.py file with database settings
You: Edit the main.py file to add logging configuration
```

**Test Generation:**
```
You: Generate comprehensive unit tests for the User class
You: Create integration tests for the authentication workflow
```

**Refactoring:**
```
You: Analyze auth.py for refactoring opportunities and suggest improvements
You: Refactor this complex function to use better design patterns
```

**Git Operations:**
```
You: Generate a commit message for my current changes
You: Help me resolve this merge conflict in src/models.py
```

**Documentation:**
```
You: Create a README for this project
You: Add comprehensive docstrings to all functions in this file
```

**Web Search:**
```  
You: Search for best practices for async Python error handling
```

**Command Line Operations:**
```
You: List all Python files in the current directory
You: Check the status of the development server
You: Run the test suite and show me the results
```

**Complex Multi-Agent Tasks:**
```
You: I need to implement user authentication. Create the models, generate tests, add documentation, and suggest a Git workflow.
You: Refactor my authentication system, generate tests for the changes, and update the documentation.
```

## 🏗️ Architecture

### Agent Structure
```
SupervisorAgent (qwen2.5:32b) - LangGraph Orchestration
├── WebSearchAgent (llama3.1:8b) - DuckDuckGo + Web Search
├── CodeReviewAgent (qwen2.5-coder:32b) - AST + Static Analysis
├── CodeAnalyzerAgent (qwen2.5-coder:32b) - Tree-sitter + Indexing
├── FileOperationsAgent (qwen2.5-coder:32b) - File System Operations
├── TestGeneratorAgent (qwen2.5-coder:32b) - Test Creation & Coverage
├── RefactoringAgent (qwen2.5-coder:32b) - Code Optimization
├── GitAgent (llama3.1:8b) - Version Control Operations
├── DocumentationAgent (llama3.1:8b) - Documentation Generation
└── CommandLineAgent (llama3.1:8b) - System Command Execution
```

### Data Flow
1. **User Input** → SupervisorAgent analyzes request
2. **Task Analysis** → AI determines which agent(s) are needed
3. **Agent Selection** → Route to appropriate specialized agent(s)
4. **Background Processing** → Real-time codebase indexing if enabled
5. **Agent Execution** → Specialized agents work on their tasks
6. **Result Synthesis** → SupervisorAgent combines results into coherent response

### Model Optimization Strategy
- **Coding Tasks**: `qwen2.5-coder:32b` - Specialized for code analysis, generation, review
- **General Tasks**: `llama3.1:8b` - Lighter, faster for search, git, documentation
- **Orchestration**: `qwen2.5:32b` - Strong reasoning for task coordination

## 🔧 Customization

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`:
```python
from core.base_agent import BaseAgent, AgentState

class MyCustomAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return "Your specialized agent prompt here"
    
    async def process(self, state: AgentState) -> AgentState:
        # Your agent logic here
        pass
```

2. Register it in the SupervisorAgent workflow

### Custom Models

The system supports any Ollama model. You can customize per-agent models in `core/config.py`:

```python
models: Dict[str, str] = {
    "supervisor": "qwen2.5:32b",
    "websearch": "llama3.1:8b", 
    "code_review": "qwen2.5-coder:32b",
    "file_operations": "qwen2.5-coder:32b",
    # ... customize as needed
}
```

**Popular model choices:**
- `qwen2.5-coder:32b` - Excellent for coding tasks
- `deepseek-coder:33b` - Strong coding capabilities
- `codellama:34b` - Meta's coding model
- `starcoder2:15b` - Smaller but capable
- `llama3.1:8b` - Fast general purpose model
- `qwen2.5:32b` - Strong reasoning for coordination

## 🧪 Testing

```bash
# Run the test suite
python -m pytest tests/

# Test specific components
python -m pytest tests/test_agents.py
```

## 📁 Project Structure

```
langgraph-ollama-agent/
├── core/                   # Core components
│   ├── config.py          # Configuration management
│   ├── ollama_client.py   # Ollama API client
│   └── base_agent.py      # Base agent class
├── agents/                # Specialized agents
│   ├── websearch_agent.py
│   ├── code_review_agent.py
│   ├── code_analyzer_agent.py
│   ├── file_operations_agent.py
│   ├── test_generator_agent.py
│   ├── refactoring_agent.py
│   ├── git_agent.py
│   ├── documentation_agent.py
│   ├── command_line_agent.py
│   └── supervisor_agent.py
├── cli/                   # Command line interface
│   └── main.py
├── utils/                 # Utility functions
├── tests/                 # Test files
├── requirements.txt       # Dependencies
├── main.py               # Entry point
└── README.md             # This file
```

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve
```

**Model Not Found:**
```bash
# Pull the required model
ollama pull qwen2.5-coder:32b
```

**Memory Issues:**
- Use smaller models like `qwen2.5-coder:14b` or `starcoder2:7b`
- Reduce `max_tokens` in configuration

**Slow Performance:**
- Use GPU acceleration if available
- Reduce model size (use smaller variants like `qwen2.5-coder:14b`)
- Disable background indexing: `--no-auto-index`
- Disable file watching: `--no-file-watch`
- Adjust temperature settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the multi-agent orchestration
- [Ollama](https://ollama.ai/) for local LLM serving
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B) for the excellent coding model
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing

## 🔮 Future Enhancements

- [x] ~~File system operations (read, write, edit)~~ ✅ **Implemented**
- [x] ~~Git integration for version control~~ ✅ **Implemented** 
- [x] ~~Test generation and coverage analysis~~ ✅ **Implemented**
- [x] ~~Code refactoring and optimization~~ ✅ **Implemented**
- [x] ~~Documentation generation~~ ✅ **Implemented**
- [x] ~~Command line operations with safety controls~~ ✅ **Implemented**
- [x] ~~Automatic model downloading and management~~ ✅ **Implemented**
- [ ] Integration with popular IDEs (VS Code extension)
- [ ] Web interface option
- [ ] Custom workflow definitions via YAML/JSON
- [ ] Plugin system for extending agents
- [ ] Collaborative multi-user sessions
- [ ] Database integration agent
- [ ] Deployment and DevOps agent
- [ ] Security scanning and vulnerability assessment
- [ ] Performance profiling and optimization agent
