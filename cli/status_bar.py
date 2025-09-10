"""Status bar component for displaying system status at the bottom of the CLI."""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from core.shared_context import shared_context


class StatusBar:
    """A status bar that shows system information at the bottom of the CLI."""
    
    def __init__(self, console: Console):
        self.console = console
        self.is_active = False
        self.live = None
        self._last_update = datetime.now()
        
    def start(self):
        """Start the status bar display."""
        self.is_active = True
        
    def stop(self):
        """Stop the status bar display."""
        self.is_active = False
        if self.live:
            self.live.stop()
    
    def _get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status from shared context."""
        code_index = shared_context.get_code_index()
        
        if not code_index:
            return {
                "status": "No indexing",
                "percentage": 0,
                "files_count": 0,
                "functions_count": 0,
                "last_indexed": "Never"
            }
        
        total_files = len(code_index)
        total_functions = sum(len(analysis.get('functions', [])) for analysis in code_index.values())
        
        # Get timestamp of last indexing
        timestamp = shared_context.get_timestamp("code_index")
        last_indexed = timestamp.strftime("%H:%M:%S") if timestamp else "Unknown"
        
        return {
            "status": "Indexed",
            "percentage": 100,  # Assume complete for now
            "files_count": total_files,
            "functions_count": total_functions,
            "last_indexed": last_indexed
        }
    
    def _get_agent_status(self) -> Dict[str, str]:
        """Get current agent activity status."""
        # This could be enhanced to track active agents
        return {
            "supervisor": "Ready",
            "code_analyzer": "Ready",
            "code_review": "Ready",
            "websearch": "Ready"
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system-level status information."""
        return {
            "memory_window": shared_context.get("memory_window", 20),
            "context_entries": len(shared_context.get("context_store", {})),
            "session_duration": self._get_session_duration()
        }
    
    def _get_session_duration(self) -> str:
        """Get current session duration."""
        session_start = shared_context.get("session_start")
        if session_start:
            duration = datetime.now() - session_start
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return "00:00:00"
    
    def create_status_display(self) -> Panel:
        """Create the status bar display panel."""
        indexing_status = self._get_indexing_status()
        agent_status = self._get_agent_status()
        system_status = self._get_system_status()
        
        # Create status table
        table = Table.grid(padding=1)
        table.add_column("Section", style="bold cyan", width=12)
        table.add_column("Details", style="white", width=40)
        table.add_column("Stats", style="green", width=25)
        
        # Indexing status
        if indexing_status["files_count"] > 0:
            indexing_text = f"✅ {indexing_status['files_count']} files, {indexing_status['functions_count']} functions"
            indexing_time = f"Last: {indexing_status['last_indexed']}"
        else:
            indexing_text = "⏳ No codebase indexed"
            indexing_time = "Run /index to start"
        
        table.add_row(
            "📁 Indexing",
            indexing_text,
            indexing_time
        )
        
        # Agent status (show key agents)
        active_agents = ["supervisor", "code_review", "websearch"]
        agent_text = " | ".join([f"{agent}: {status}" for agent, status in agent_status.items() if agent in active_agents])
        
        table.add_row(
            "🤖 Agents",
            agent_text[:40],
            f"Memory: {system_status['memory_window']} msgs"
        )
        
        # Session info
        table.add_row(
            "⏰ Session", 
            f"Duration: {system_status['session_duration']}",
            f"Context: {system_status['context_entries']} entries"
        )
        
        return Panel(
            table,
            title="🔄 System Status",
            border_style="blue",
            height=5
        )


class ProgressTracker:
    """Tracks progress of long-running operations like indexing."""
    
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.active_operation = None
    
    def start_operation(self, operation_id: str, description: str, total_items: Optional[int] = None):
        """Start tracking a new operation."""
        self.operations[operation_id] = {
            "description": description,
            "total_items": total_items,
            "completed_items": 0,
            "status": "running",
            "start_time": datetime.now(),
            "last_update": datetime.now()
        }
        self.active_operation = operation_id
        shared_context.set(f"operation_{operation_id}", self.operations[operation_id])
    
    def update_progress(self, operation_id: str, completed_items: int, status_message: str = ""):
        """Update progress for an operation."""
        if operation_id in self.operations:
            self.operations[operation_id]["completed_items"] = completed_items
            self.operations[operation_id]["last_update"] = datetime.now()
            if status_message:
                self.operations[operation_id]["status_message"] = status_message
            
            shared_context.set(f"operation_{operation_id}", self.operations[operation_id])
    
    def complete_operation(self, operation_id: str):
        """Mark an operation as completed."""
        if operation_id in self.operations:
            self.operations[operation_id]["status"] = "completed"
            self.operations[operation_id]["end_time"] = datetime.now()
            shared_context.set(f"operation_{operation_id}", self.operations[operation_id])
            
            if self.active_operation == operation_id:
                self.active_operation = None
    
    def get_active_operation(self) -> Optional[Dict[str, Any]]:
        """Get the currently active operation."""
        if self.active_operation and self.active_operation in self.operations:
            return self.operations[self.active_operation]
        return None
    
    def get_progress_percentage(self, operation_id: str) -> int:
        """Get progress percentage for an operation."""
        if operation_id not in self.operations:
            return 0
        
        op = self.operations[operation_id]
        if op["total_items"] and op["total_items"] > 0:
            return int((op["completed_items"] / op["total_items"]) * 100)
        return 0 if op["status"] == "running" else 100


# Global progress tracker instance
progress_tracker = ProgressTracker()