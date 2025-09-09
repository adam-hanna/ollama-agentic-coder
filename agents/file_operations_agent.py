import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from core.base_agent import BaseAgent, AgentState

class FileOperationsAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a file operations specialist that can safely read, write, create, and modify files. Your role is to:

1. Read file contents and understand file structures
2. Create new files with appropriate content  
3. Edit existing files with precise modifications
4. Move, copy, and organize files and directories
5. Ensure all operations are safe and follow best practices

Safety guidelines:
- Always check if files exist before operations
- Create backups for destructive operations when requested
- Validate file paths and permissions
- Follow coding conventions and file formatting
- Never overwrite files without explicit confirmation
- Handle encoding issues gracefully

You can perform these operations:
- read_file: Read and return file contents
- write_file: Create or overwrite a file
- edit_file: Make specific edits to existing files
- create_directory: Create new directories
- copy_file: Copy files with optional renaming  
- move_file: Move or rename files
- delete_file: Remove files (with caution)
- list_directory: List directory contents"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No file operation specified")
        
        task = state.current_task
        
        try:
            if task.startswith("read_file:"):
                result = await self._read_file(task.replace("read_file:", "").strip())
            
            elif task.startswith("write_file:"):
                # Format: write_file:path|content
                parts = task.replace("write_file:", "").split("|", 1)
                if len(parts) != 2:
                    result = "Invalid write_file format. Use: write_file:path|content"
                else:
                    result = await self._write_file(parts[0].strip(), parts[1])
            
            elif task.startswith("edit_file:"):
                # Format: edit_file:path|old_text|new_text
                parts = task.replace("edit_file:", "").split("|", 2)
                if len(parts) != 3:
                    result = "Invalid edit_file format. Use: edit_file:path|old_text|new_text"
                else:
                    result = await self._edit_file(parts[0].strip(), parts[1], parts[2])
            
            elif task.startswith("create_directory:"):
                result = await self._create_directory(task.replace("create_directory:", "").strip())
            
            elif task.startswith("copy_file:"):
                # Format: copy_file:source|destination
                parts = task.replace("copy_file:", "").split("|", 1)
                if len(parts) != 2:
                    result = "Invalid copy_file format. Use: copy_file:source|destination"
                else:
                    result = await self._copy_file(parts[0].strip(), parts[1].strip())
            
            elif task.startswith("move_file:"):
                # Format: move_file:source|destination
                parts = task.replace("move_file:", "").split("|", 1)
                if len(parts) != 2:
                    result = "Invalid move_file format. Use: move_file:source|destination"
                else:
                    result = await self._move_file(parts[0].strip(), parts[1].strip())
            
            elif task.startswith("delete_file:"):
                result = await self._delete_file(task.replace("delete_file:", "").strip())
            
            elif task.startswith("list_directory:"):
                result = await self._list_directory(task.replace("list_directory:", "").strip())
            
            else:
                result = await self._intelligent_file_operation(task)
            
            return self.add_message(
                state,
                "assistant",
                result,
                metadata={"operation_type": "file_operation"}
            )
        
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"File operation failed: {str(e)}"
            )
    
    async def _read_file(self, file_path: str) -> str:
        """Read and return file contents"""
        path = Path(file_path)
        
        if not path.exists():
            return f"File not found: {file_path}"
        
        if not path.is_file():
            return f"Path is not a file: {file_path}"
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"File contents of {file_path}:\n\n{content}"
        
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return f"File contents of {file_path} (latin-1 encoding):\n\n{content}"
            except Exception as e:
                return f"Failed to read file {file_path}: {str(e)}"
        
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"
    
    async def _write_file(self, file_path: str, content: str) -> str:
        """Create or overwrite a file with content"""
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
        
        except Exception as e:
            return f"Failed to write file {file_path}: {str(e)}"
    
    async def _edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        """Edit existing file by replacing old_text with new_text"""
        path = Path(file_path)
        
        if not path.exists():
            return f"File not found: {file_path}"
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_text not in content:
                return f"Text to replace not found in {file_path}"
            
            updated_content = content.replace(old_text, new_text)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return f"Successfully edited {file_path}: replaced text"
        
        except Exception as e:
            return f"Failed to edit file {file_path}: {str(e)}"
    
    async def _create_directory(self, dir_path: str) -> str:
        """Create a new directory"""
        path = Path(dir_path)
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory: {dir_path}"
        
        except Exception as e:
            return f"Failed to create directory {dir_path}: {str(e)}"
    
    async def _copy_file(self, source: str, destination: str) -> str:
        """Copy a file from source to destination"""
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            return f"Source file not found: {source}"
        
        try:
            # Create parent directories if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dst_path)
            return f"Successfully copied {source} to {destination}"
        
        except Exception as e:
            return f"Failed to copy file: {str(e)}"
    
    async def _move_file(self, source: str, destination: str) -> str:
        """Move a file from source to destination"""
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            return f"Source file not found: {source}"
        
        try:
            # Create parent directories if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dst_path))
            return f"Successfully moved {source} to {destination}"
        
        except Exception as e:
            return f"Failed to move file: {str(e)}"
    
    async def _delete_file(self, file_path: str) -> str:
        """Delete a file (with caution)"""
        path = Path(file_path)
        
        if not path.exists():
            return f"File not found: {file_path}"
        
        try:
            if path.is_file():
                path.unlink()
                return f"Successfully deleted file: {file_path}"
            elif path.is_dir():
                path.rmdir()  # Only removes empty directories
                return f"Successfully deleted empty directory: {file_path}"
            else:
                return f"Path is neither file nor directory: {file_path}"
        
        except Exception as e:
            return f"Failed to delete {file_path}: {str(e)}"
    
    async def _list_directory(self, dir_path: str) -> str:
        """List directory contents"""
        if not dir_path:
            dir_path = "."
        
        path = Path(dir_path)
        
        if not path.exists():
            return f"Directory not found: {dir_path}"
        
        if not path.is_dir():
            return f"Path is not a directory: {dir_path}"
        
        try:
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            if not items:
                return f"Directory {dir_path} is empty"
            
            return f"Contents of {dir_path}:\n" + "\n".join(items)
        
        except Exception as e:
            return f"Failed to list directory {dir_path}: {str(e)}"
    
    async def _intelligent_file_operation(self, task: str) -> str:
        """Handle complex file operations using AI"""
        prompt = f"""Analyze this file operation request and determine the best approach:

Request: {task}

Based on this request, provide a structured response with:
1. The specific file operation(s) needed
2. Step-by-step plan to accomplish the task
3. Any safety considerations or warnings
4. Expected outcome

If the request involves reading files, modifying code, or creating new files, provide specific recommendations."""
        
        return await self.generate_response(prompt)