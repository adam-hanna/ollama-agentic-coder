import asyncio
import os
from pathlib import Path
from typing import Callable, Set, Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from rich.console import Console

class CodeFileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable, supported_extensions: Set[str], console: Optional[Console] = None):
        super().__init__()
        self.callback = callback
        self.supported_extensions = supported_extensions
        self.console = console or Console()
        self._pending_changes: Dict[str, float] = {}
        self._debounce_delay = 0.5  # seconds
    
    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self._is_supported_file(file_path):
            asyncio.create_task(self._debounced_callback(file_path, 'modified'))
    
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self._is_supported_file(file_path):
            asyncio.create_task(self._debounced_callback(file_path, 'created'))
    
    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self._is_supported_file(file_path):
            asyncio.create_task(self._debounced_callback(file_path, 'deleted'))
    
    def on_moved(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        # Handle both old and new paths
        if hasattr(event, 'dest_path'):
            old_path = event.src_path
            new_path = event.dest_path
            
            if self._is_supported_file(old_path):
                asyncio.create_task(self._debounced_callback(old_path, 'deleted'))
            
            if self._is_supported_file(new_path):
                asyncio.create_task(self._debounced_callback(new_path, 'created'))
    
    def _is_supported_file(self, file_path: str) -> bool:
        if file_path.startswith('.') or '/.git/' in file_path or '__pycache__' in file_path:
            return False
        
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions
    
    async def _debounced_callback(self, file_path: str, event_type: str):
        import time
        current_time = time.time()
        
        # Debounce: only process if enough time has passed since last change
        if file_path in self._pending_changes:
            if current_time - self._pending_changes[file_path] < self._debounce_delay:
                await asyncio.sleep(self._debounce_delay)
        
        self._pending_changes[file_path] = current_time
        
        try:
            await self.callback(file_path, event_type)
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Error processing file change: {e}[/red]")

class FileWatcher:
    def __init__(self, root_path: str, callback: Callable, supported_extensions: Set[str], console: Optional[Console] = None):
        self.root_path = Path(root_path).resolve()
        self.callback = callback
        self.supported_extensions = supported_extensions
        self.console = console or Console()
        self.observer = Observer()
        self.handler = CodeFileHandler(callback, supported_extensions, console)
        self._running = False
    
    def start(self):
        """Start watching for file changes"""
        if self._running:
            return
        
        if not self.root_path.exists():
            self.console.print(f"[red]Warning: Path does not exist: {self.root_path}[/red]")
            return
        
        self.observer.schedule(
            self.handler, 
            str(self.root_path), 
            recursive=True
        )
        
        self.observer.start()
        self._running = True
        
        if self.console:
            self.console.print(f"[green]🔍 Watching for code changes in: {self.root_path}[/green]")
    
    def stop(self):
        """Stop watching for file changes"""
        if not self._running:
            return
        
        self.observer.stop()
        self.observer.join()
        self._running = False
        
        if self.console:
            self.console.print("[yellow]📂 File watching stopped[/yellow]")
    
    def is_running(self) -> bool:
        return self._running

class BackgroundIndexer:
    def __init__(self, code_analyzer, console: Optional[Console] = None):
        self.code_analyzer = code_analyzer
        self.console = console or Console()
        self.file_watcher: Optional[FileWatcher] = None
        self._indexing_queue = asyncio.Queue()
        self._indexing_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self, root_path: str):
        """Start background indexing and file watching"""
        if self._running:
            return
        
        self._running = True
        
        # Initial indexing
        await self._initial_index(root_path)
        
        # Start file watcher
        supported_extensions = set(self.code_analyzer.config.code_analyzer.supported_extensions)
        self.file_watcher = FileWatcher(
            root_path, 
            self._on_file_change,
            supported_extensions,
            self.console
        )
        
        self.file_watcher.start()
        
        # Start background indexing worker
        self._indexing_task = asyncio.create_task(self._indexing_worker())
    
    async def stop(self):
        """Stop background indexing and file watching"""
        if not self._running:
            return
        
        self._running = False
        
        if self.file_watcher:
            self.file_watcher.stop()
        
        if self._indexing_task:
            self._indexing_task.cancel()
            try:
                await self._indexing_task
            except asyncio.CancelledError:
                pass
    
    async def _initial_index(self, root_path: str):
        """Perform initial directory indexing"""
        try:
            self.console.print("[yellow]🔄 Performing initial code indexing...[/yellow]")
            
            # Import AgentState here to avoid circular imports
            from core.base_agent import AgentState
            
            async with self.code_analyzer:
                index_state = AgentState(
                    current_task=f"index_directory:{root_path}"
                )
                await self.code_analyzer.process(index_state)
            
            self.console.print("[green]✅ Initial indexing completed[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Initial indexing failed: {e}[/red]")
    
    async def _on_file_change(self, file_path: str, event_type: str):
        """Handle file system events"""
        await self._indexing_queue.put((file_path, event_type))
    
    async def _indexing_worker(self):
        """Background worker that processes file changes"""
        while self._running:
            try:
                # Wait for file changes
                file_path, event_type = await asyncio.wait_for(
                    self._indexing_queue.get(), 
                    timeout=1.0
                )
                
                if event_type == 'deleted':
                    # Remove from index
                    if file_path in self.code_analyzer.code_index:
                        del self.code_analyzer.code_index[file_path]
                        self.console.print(f"[yellow]🗑️ Removed {file_path} from index[/yellow]")
                
                elif event_type in ['created', 'modified']:
                    # Re-analyze file
                    try:
                        async with self.code_analyzer:
                            analysis = await self.code_analyzer._analyze_file_content(file_path)
                            self.code_analyzer.code_index[file_path] = analysis
                            self.console.print(f"[green]🔄 Updated index for {file_path}[/green]")
                    
                    except Exception as e:
                        self.console.print(f"[red]❌ Failed to index {file_path}: {e}[/red]")
                
                self._indexing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No new files to process, continue loop
                continue
            except Exception as e:
                self.console.print(f"[red]Error in indexing worker: {e}[/red]")