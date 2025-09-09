import asyncio
import os
import sys
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Config, get_config
from core.base_agent import AgentState
from agents import (SupervisorAgent, WebSearchAgent, CodeReviewAgent, CodeAnalyzerAgent,
                   FileOperationsAgent, TestGeneratorAgent, RefactoringAgent, 
                   GitAgent, DocumentationAgent)
from utils.file_watcher import BackgroundIndexer

class AgentCLI:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.console = Console()
        self.supervisor = None
        self.current_state = AgentState()
        self.background_indexer = None
    
    async def start(self):
        self.console.print(Panel(
            Text("🤖 Multi-Agent Coding Assistant", style="bold blue", justify="center"),
            subtitle="Powered by LangGraph + Ollama",
            border_style="blue"
        ))
        
        self.console.print(f"[green]✓[/green] Ollama Host: {self.config.ollama.host}")
        self.console.print(f"[green]✓[/green] Default Model: {self.config.ollama.model}")
        self.console.print()
        
        self.supervisor = SupervisorAgent("supervisor", self.config)
        
        # Start background indexing if enabled
        if self.config.code_analyzer.auto_background_indexing:
            await self._start_background_indexing()
        else:
            self.console.print("[yellow]⚠️ Automatic background indexing is disabled[/yellow]")
            self.console.print("[dim]Use /index to manually index directories[/dim]")
        
        self.console.print("[yellow]Available commands:[/yellow]")
        self.console.print("  /help     - Show this help")
        self.console.print("  /index    - Manually index a directory (or toggle auto-indexing)")
        self.console.print("  /config   - Show current configuration")
        self.console.print("  /agents   - List available agents")  
        self.console.print("  /clear    - Clear conversation history")
        self.console.print("  /exit     - Exit the application")
        self.console.print()
        
        await self._chat_loop()
    
    async def _chat_loop(self):
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue
                
                self.current_state.current_task = user_input
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Processing request...", total=None)
                    
                    async with self.supervisor:
                        result_state = await self.supervisor.process(self.current_state)
                    
                    progress.remove_task(task)
                
                last_message = result_state.messages[-1] if result_state.messages else None
                if last_message and last_message.role in ["assistant", "supervisor"]:
                    self.console.print()
                    self.console.print(Panel(
                        last_message.content,
                        title="🤖 Assistant",
                        border_style="green"
                    ))
                    self.console.print()
                
                self.current_state = result_state
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit gracefully[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def _start_background_indexing(self):
        """Start background indexing for the current directory"""
        try:
            current_dir = os.getcwd()
            
            # Create code analyzer instance for background indexing
            code_analyzer = CodeAnalyzerAgent("code_analyzer", self.config)
            
            # Start background indexer
            self.background_indexer = BackgroundIndexer(code_analyzer, self.console)
            await self.background_indexer.start(current_dir)
            
        except Exception as e:
            self.console.print(f"[red]Failed to start background indexing: {e}[/red]")
    
    async def _stop_background_indexing(self):
        """Stop background indexing"""
        if self.background_indexer:
            await self.background_indexer.stop()
            self.background_indexer = None
            self.console.print("[yellow]🔄 Background indexing stopped[/yellow]")
        else:
            self.console.print("[dim]Background indexing is not running[/dim]")
    
    async def _manual_index_directory(self, path: str):
        """Manually index a directory"""
        if not os.path.exists(path):
            self.console.print(f"[red]Directory not found: {path}[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Manually indexing: {path}...", total=None)
            
            analyzer = CodeAnalyzerAgent("code_analyzer", self.config)
            async with analyzer:
                index_state = AgentState(current_task=f"index_directory:{path}")
                result = await analyzer.process(index_state)
            
            progress.remove_task(task)
        
        if result.messages:
            last_message = result.messages[-1]
            self.console.print(Panel(
                last_message.content,
                title=f"📁 Directory Indexed: {path}",
                border_style="green"
            ))
    
    async def _toggle_auto_indexing(self):
        """Toggle automatic background indexing"""
        if self.background_indexer and self.background_indexer._running:
            # Stop current background indexing
            await self.background_indexer.stop()
            self.background_indexer = None
            self.config.code_analyzer.auto_background_indexing = False
            self.console.print("[yellow]🔴 Automatic background indexing disabled[/yellow]")
        else:
            # Start background indexing
            self.config.code_analyzer.auto_background_indexing = True
            await self._start_background_indexing()
            self.console.print("[green]🟢 Automatic background indexing enabled[/green]")
    
    async def _handle_command(self, command: str):
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower() if cmd_parts else ""
        
        if cmd == "help":
            self._show_help()
        
        elif cmd == "index":
            if len(cmd_parts) > 1:
                if cmd_parts[1].lower() == "auto":
                    await self._toggle_auto_indexing()
                elif cmd_parts[1].lower() == "stop":
                    await self._stop_background_indexing()
                else:
                    path = cmd_parts[1]
                    await self._manual_index_directory(path)
            else:
                # Index current directory
                await self._manual_index_directory(".")
        
        elif cmd == "config":
            self._show_config()
        
        elif cmd == "agents":
            self._show_agents()
        
        elif cmd == "clear":
            self.current_state = AgentState()
            self.console.print("[green]Conversation history cleared[/green]")
        
        elif cmd == "exit":
            await self._stop_background_indexing()
            self.console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
        
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self._show_help()
    
    def _show_help(self):
        indexing_status = "🟢 Enabled" if self.background_indexer and self.background_indexer._running else "🔴 Disabled"
        
        help_text = f"""[bold]Available Commands:[/bold]

[cyan]/help[/cyan]     - Show this help message
[cyan]/index[/cyan]    - Manual indexing commands:
              • [cyan]/index[/cyan] - Index current directory
              • [cyan]/index[/cyan] [path] - Index specific directory 
              • [cyan]/index auto[/cyan] - Toggle automatic background indexing
              • [cyan]/index stop[/cyan] - Stop background indexing
[cyan]/config[/cyan]   - Show current configuration
[cyan]/agents[/cyan]   - List available agents and their capabilities
[cyan]/clear[/cyan]    - Clear conversation history
[cyan]/exit[/cyan]     - Exit the application

[bold]Indexing Status:[/bold] {indexing_status}
• [green]🔄 Background Mode:[/green] {"Active" if self.config.code_analyzer.auto_background_indexing else "Disabled"}
• [green]📁 File Watching:[/green] {"Active" if self.config.code_analyzer.watch_file_changes else "Disabled"}
• [green]🤖 Multi-Agent:[/green] Tasks are routed to specialized agents automatically

[bold]Example Usage:[/bold]
• "Review this Python file for bugs: [file_path]"
• "Search for best practices for async Python programming"
• "Analyze the dependencies in my project"
• "Find all functions named 'process' in the codebase"
"""
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    
    def _show_config(self):
        config_text = f"""[bold]Current Configuration:[/bold]

[cyan]Ollama Settings:[/cyan]
  Host: {self.config.ollama.host}
  Model: {self.config.ollama.model}
  Temperature: {self.config.ollama.temperature}
  Max Tokens: {self.config.ollama.max_tokens}
  Timeout: {self.config.ollama.timeout}s

[cyan]Agent Settings:[/cyan]
  Max Iterations: {self.config.agents.max_iterations}
  Enable Memory: {self.config.agents.enable_memory}
  Memory Window: {self.config.agents.memory_window}

[cyan]Search Settings:[/cyan]
  Search Engine: {self.config.search.search_engine}
  Max Results: {self.config.search.max_results}

[cyan]Code Analyzer:[/cyan]
  Supported Extensions: {', '.join(self.config.code_analyzer.supported_extensions)}
  Max File Size: {self.config.code_analyzer.max_file_size / 1024 / 1024:.1f}MB
  AST Indexing: {self.config.code_analyzer.enable_ast_indexing}
  Auto Background Indexing: {self.config.code_analyzer.auto_background_indexing}
  Watch File Changes: {self.config.code_analyzer.watch_file_changes}
"""
        self.console.print(Panel(config_text, title="Configuration", border_style="blue"))
    
    def _show_agents(self):
        agents_text = """[bold]Available Agents & Their Models:[/bold]

[cyan]🔍 WebSearchAgent[/cyan] ([yellow]llama3.1:8b[/yellow])
• Searches web for documentation and examples
• Finds best practices and troubleshooting info
• Provides up-to-date technical information

[cyan]🔍 CodeReviewAgent[/cyan] ([yellow]qwen2.5-coder:32b[/yellow])
• Analyzes code for bugs and security issues
• Checks code quality and best practices
• Provides refactoring suggestions
• Identifies performance problems

[cyan]📊 CodeAnalyzerAgent[/cyan] ([yellow]qwen2.5-coder:32b[/yellow])
• Indexes codebase structure and dependencies
• Tracks functions, classes, and imports
• Analyzes code metrics and complexity
• Finds code patterns and duplications

[cyan]📁 FileOperationsAgent[/cyan] ([yellow]qwen2.5-coder:32b[/yellow])
• Reads, writes, and edits files directly
• Creates and manages directory structures
• Handles file operations safely
• Supports multiple file formats

[cyan]🧪 TestGeneratorAgent[/cyan] ([yellow]qwen2.5-coder:32b[/yellow])
• Generates comprehensive unit tests
• Creates integration and end-to-end tests
• Provides test data and mock objects
• Follows testing best practices

[cyan]🔄 RefactoringAgent[/cyan] ([yellow]qwen2.5-coder:32b[/yellow])
• Analyzes code for refactoring opportunities
• Suggests performance optimizations
• Applies design patterns
• Modernizes legacy code

[cyan]📝 GitAgent[/cyan] ([yellow]llama3.1:8b[/yellow])
• Manages Git operations and workflows
• Generates meaningful commit messages
• Helps resolve merge conflicts
• Suggests branching strategies

[cyan]📚 DocumentationAgent[/cyan] ([yellow]llama3.1:8b[/yellow])
• Creates comprehensive documentation
• Generates README files and API docs
• Adds docstrings and comments
• Writes user guides and tutorials

[cyan]🎯 SupervisorAgent[/cyan] ([yellow]qwen2.5:32b[/yellow])
• Coordinates multiple agents for complex tasks
• Routes requests to appropriate specialists
• Synthesizes results from multiple agents
• Manages overall task workflow
"""
        self.console.print(Panel(agents_text, title="Agent Capabilities & Models", border_style="green"))

@click.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--host', '-h', default=None, help='Ollama host URL')
@click.option('--model', '-m', default=None, help='Model to use')
@click.option('--temperature', '-t', type=float, default=None, help='Temperature setting')
@click.option('--no-auto-index', is_flag=True, help='Disable automatic background indexing')
@click.option('--no-file-watch', is_flag=True, help='Disable file change watching')
def main(config, host, model, temperature, no_auto_index, no_file_watch):
    """Multi-Agent Coding Assistant CLI"""
    
    # Load configuration
    app_config = get_config()
    
    # Override with CLI options
    if host:
        app_config.ollama.host = host
    if model:
        app_config.ollama.model = model  
    if temperature is not None:
        app_config.ollama.temperature = temperature
    if no_auto_index:
        app_config.code_analyzer.auto_background_indexing = False
    if no_file_watch:
        app_config.code_analyzer.watch_file_changes = False
    
    # Start the CLI
    cli = AgentCLI(app_config)
    
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()