import subprocess
import asyncio
from typing import Set, List
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from .config import Config


class ModelManager:
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console

    async def check_and_pull_models(self) -> bool:
        """Check for required models and pull missing ones"""
        required_models = self._get_required_models()
        available_models = await self._get_available_models()
        missing_models = required_models - available_models

        if not missing_models:
            self.console.print("[green]✅ All required models are available[/green]")
            return True

        self.console.print(
            f"[yellow]📦 {len(missing_models)} models need to be downloaded[/yellow]"
        )

        # Show what will be downloaded
        self._show_download_info(missing_models)

        # Ask for confirmation
        if not self._confirm_download():
            self.console.print(
                "[red]❌ Model download cancelled. Some agents may not work.[/red]"
            )
            return False

        # Download missing models
        return await self._download_models(missing_models)

    def _get_required_models(self) -> Set[str]:
        """Get all unique models required by agents"""
        return set(self.config.agents.models.values())

    async def _get_available_models(self) -> Set[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )

            # Parse ollama list output
            models = set()
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # First column is model name
                    models.add(model_name)

            return models

        except subprocess.CalledProcessError:
            self.console.print(
                "[red]❌ Failed to list Ollama models. Is Ollama running?[/red]"
            )
            return set()
        except FileNotFoundError:
            self.console.print(
                "[red]❌ Ollama not found. Please install Ollama first.[/red]"
            )
            return set()

    def _show_download_info(self, missing_models: Set[str]):
        """Show information about models to be downloaded"""
        model_info = {
            "llama3.1:8b": "~4.7GB - Fast general-purpose model for language tasks",
            "qwen2.5:32b": "~18GB - Large reasoning model for orchestration",
            "qwen2.5-coder:32b": "~18GB - Specialized coding model (already installed)",
        }

        info_text = "[bold]Models to download:[/bold]\n\n"
        total_size = 0

        for model in sorted(missing_models):
            if model in model_info:
                info_text += f"[cyan]• {model}[/cyan] - {model_info[model]}\n"
                # Extract size for rough estimate
                if "4.7GB" in model_info[model]:
                    total_size += 4.7
                elif "18GB" in model_info[model]:
                    total_size += 18

        info_text += f"\n[yellow]Estimated total download: ~{total_size:.1f}GB[/yellow]"
        info_text += "\n[dim]Download time depends on your internet connection.[/dim]"

        self.console.print(
            Panel(info_text, title="📦 Model Download Required", border_style="yellow")
        )

    def _confirm_download(self) -> bool:
        """Ask user to confirm model download"""
        try:
            from rich.prompt import Confirm

            return Confirm.ask(
                "\n[bold]Download required models now?[/bold] (Recommended for full functionality)",
                default=True,
            )
        except KeyboardInterrupt:
            return False

    async def _download_models(self, models: Set[str]) -> bool:
        """Download missing models with progress indication"""
        success = True

        for model in sorted(models):
            self.console.print(f"\n[yellow]📥 Downloading {model}...[/yellow]")

            # Use subprocess to run ollama pull with real-time output
            success_model = await self._pull_single_model(model)
            if not success_model:
                success = False
                self.console.print(f"[red]❌ Failed to download {model}[/red]")
            else:
                self.console.print(f"[green]✅ Downloaded {model}[/green]")

        if success:
            self.console.print(
                f"\n[green]🎉 All {len(models)} models downloaded successfully![/green]"
            )
        else:
            self.console.print(
                f"\n[yellow]⚠️ Some models failed to download. Check your connection and try again.[/yellow]"
            )

        return success

    async def _pull_single_model(self, model: str) -> bool:
        """Pull a single model with progress feedback"""
        try:
            # Create a progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Downloading {model}", total=None)

                # Run ollama pull in subprocess
                process = await asyncio.create_subprocess_exec(
                    "ollama",
                    "pull",
                    model,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                # Monitor progress
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break

                    line_str = line.decode().strip()
                    if line_str:
                        # Update progress description with latest status
                        progress.update(
                            task, description=f"Downloading {model}: {line_str[:50]}..."
                        )

                # Wait for completion
                await process.wait()
                progress.remove_task(task)

                return process.returncode == 0

        except Exception as e:
            self.console.print(f"[red]Error downloading {model}: {str(e)}[/red]")
            return False

    def get_model_usage_info(self) -> str:
        """Get information about which agents use which models"""
        usage = {}
        for agent, model in self.config.agents.models.items():
            if model not in usage:
                usage[model] = []
            usage[model].append(agent)

        info = "[bold]Model Usage by Agent:[/bold]\n\n"
        for model, agents in usage.items():
            info += f"[cyan]{model}[/cyan]\n"
            for agent in agents:
                info += f"  • {agent}\n"
            info += "\n"

        return info
