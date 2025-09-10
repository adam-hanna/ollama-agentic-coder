"""Shared context manager for agents to share data across the workflow."""
from typing import Dict, Any, Optional
import threading
from datetime import datetime


class SharedContext:
    """Thread-safe shared context for agents to exchange data."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._timestamps: Dict[str, datetime] = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in shared context."""
        with self._lock:
            self._data[key] = value
            self._timestamps[key] = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context."""
        with self._lock:
            return self._data.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if key exists in shared context."""
        with self._lock:
            return key in self._data
    
    def keys(self) -> list:
        """Get all keys in shared context."""
        with self._lock:
            return list(self._data.keys())
    
    def get_timestamp(self, key: str) -> Optional[datetime]:
        """Get timestamp when key was last updated."""
        with self._lock:
            return self._timestamps.get(key)
    
    def clear(self) -> None:
        """Clear all shared context."""
        with self._lock:
            self._data.clear()
            self._timestamps.clear()
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple keys at once."""
        with self._lock:
            current_time = datetime.now()
            for key, value in data.items():
                self._data[key] = value
                self._timestamps[key] = current_time
    
    def get_code_index(self) -> Dict[str, Any]:
        """Get the shared code index if available."""
        return self.get("code_index", {})
    
    def set_code_index(self, index: Dict[str, Any]) -> None:
        """Set the shared code index."""
        self.set("code_index", index)
    
    def get_indexed_files(self) -> list:
        """Get list of indexed file paths."""
        index = self.get_code_index()
        return list(index.keys())
    
    def get_file_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get analysis data for a specific file."""
        index = self.get_code_index()
        return index.get(file_path)


# Global shared context instance
shared_context = SharedContext()