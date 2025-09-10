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

IMPORTANT: You have access to powerful exploration tools:
- Use built-in read_file() method to examine any file
- Use execute_command() method for shell operations like ls, find, grep
- Access indexed codebase data via shared_context for understanding project structure

For codebase analysis, you should:
1. Use `ls -la` to understand project structure
2. Use `find . -name "*.py"` to discover files
3. Read specific files that need examination
4. Use `grep` to search for patterns across files
5. Combine file reading with command-line exploration

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
                result = await self._create_directory(
                    task.replace("create_directory:", "").strip()
                )

            elif task.startswith("copy_file:"):
                # Format: copy_file:source|destination
                parts = task.replace("copy_file:", "").split("|", 1)
                if len(parts) != 2:
                    result = (
                        "Invalid copy_file format. Use: copy_file:source|destination"
                    )
                else:
                    result = await self._copy_file(parts[0].strip(), parts[1].strip())

            elif task.startswith("move_file:"):
                # Format: move_file:source|destination
                parts = task.replace("move_file:", "").split("|", 1)
                if len(parts) != 2:
                    result = (
                        "Invalid move_file format. Use: move_file:source|destination"
                    )
                else:
                    result = await self._move_file(parts[0].strip(), parts[1].strip())

            elif task.startswith("delete_file:"):
                result = await self._delete_file(
                    task.replace("delete_file:", "").strip()
                )

            elif task.startswith("list_directory:"):
                result = await self._list_directory(
                    task.replace("list_directory:", "").strip()
                )

            else:
                result = await self._intelligent_file_operation(task)

            return self.add_message(
                state,
                "assistant",
                result,
                metadata={"operation_type": "file_operation"},
            )

        except Exception as e:
            return self.add_message(
                state, "assistant", f"File operation failed: {str(e)}"
            )

    async def _read_file(self, file_path: str) -> str:
        """Read and return file contents"""
        path = Path(file_path)

        if not path.exists():
            return f"File not found: {file_path}"

        if not path.is_file():
            return f"Path is not a file: {file_path}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            return f"File contents of {file_path}:\n\n{content}"

        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f:
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
            with open(path, "w", encoding="utf-8") as f:
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
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_text not in content:
                return f"Text to replace not found in {file_path}"

            updated_content = content.replace(old_text, new_text)

            with open(path, "w", encoding="utf-8") as f:
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

    async def _create_config_file(
        self, config_type: str, project_root: str = "."
    ) -> str:
        """Create development configuration files for detected project type"""
        # Detect project type first
        project_info = await self._detect_project_type(project_root)

        if config_type == "dev-config":
            return await self._create_development_config(project_root, project_info)
        elif config_type == "test-config":
            return await self._create_test_config(project_root, project_info)
        elif config_type == "lint-config":
            return await self._create_lint_config(project_root, project_info)
        else:
            return f"Unknown config type: {config_type}. Available: dev-config, test-config, lint-config"

    async def _detect_project_type(self, project_root: str = ".") -> Dict[str, Any]:
        """Detect project type and language from directory structure"""
        project_info = {
            "languages": [],
            "build_tools": [],
            "frameworks": [],
            "package_managers": [],
        }

        # Check for common files to detect languages and tools
        files_check = await self.execute_command(
            f"find {project_root} -maxdepth 2 -type f"
        )

        # Python detection
        if any(
            f in files_check
            for f in [".py", "requirements.txt", "setup.py", "pyproject.toml"]
        ):
            project_info["languages"].append("python")
            if "requirements.txt" in files_check:
                project_info["package_managers"].append("pip")
            if "pyproject.toml" in files_check:
                project_info["build_tools"].append("setuptools")

        # JavaScript/TypeScript detection
        if any(
            f in files_check for f in ["package.json", ".js", ".ts", ".jsx", ".tsx"]
        ):
            if ".ts" in files_check or ".tsx" in files_check:
                project_info["languages"].append("typescript")
            else:
                project_info["languages"].append("javascript")
            project_info["package_managers"].append("npm")

        # Java detection
        if any(f in files_check for f in [".java", "pom.xml", "build.gradle"]):
            project_info["languages"].append("java")
            if "pom.xml" in files_check:
                project_info["build_tools"].append("maven")
            if "build.gradle" in files_check:
                project_info["build_tools"].append("gradle")

        # Go detection
        if "go.mod" in files_check or ".go" in files_check:
            project_info["languages"].append("go")
            project_info["build_tools"].append("go")

        # Rust detection
        if "Cargo.toml" in files_check or ".rs" in files_check:
            project_info["languages"].append("rust")
            project_info["build_tools"].append("cargo")

        return project_info

    async def _create_development_config(
        self, project_root: str, project_info: Dict[str, Any]
    ) -> str:
        """Create language-appropriate development configuration"""
        results = []

        for language in project_info["languages"]:
            if language == "python":
                content = await self._generate_python_dev_config(project_info)
                result = await self._write_file(
                    f"{project_root}/pyproject.toml", content
                )
                results.append(f"Python: {result}")

            elif language == "javascript" or language == "typescript":
                content = await self._generate_js_dev_config(project_info)
                result = await self._write_file(f"{project_root}/package.json", content)
                results.append(f"JS/TS: {result}")

            elif language == "java":
                if "maven" in project_info["build_tools"]:
                    content = await self._generate_maven_config(project_info)
                    result = await self._write_file(f"{project_root}/pom.xml", content)
                    results.append(f"Java (Maven): {result}")

            elif language == "go":
                content = await self._generate_go_config(project_info)
                result = await self._write_file(f"{project_root}/go.mod", content)
                results.append(f"Go: {result}")

        return (
            "\\n".join(results) if results else "No development configuration created"
        )

    async def _generate_python_dev_config(self, project_info: Dict[str, Any]) -> str:
        """Generate Python-specific development configuration"""
        return """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project"
version = "0.1.0"
description = ""
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
dev = []

[tool.black]
line-length = 88

[tool.mypy]
warn_return_any = true
warn_unused_configs = true
"""

    async def _generate_js_dev_config(self, project_info: Dict[str, Any]) -> str:
        """Generate JavaScript/TypeScript development configuration"""
        is_ts = "typescript" in project_info["languages"]
        return (
            """{
  "name": "project",
  "version": "1.0.0",
  "description": "",
  "main": "index."""
            + ("ts" if is_ts else "js")
            + """",
  "scripts": {
    "test": "echo \\"Error: no test specified\\" && exit 1"
  },
  "devDependencies": {}
}"""
        )

    async def _generate_maven_config(self, project_info: Dict[str, Any]) -> str:
        """Generate Maven configuration for Java projects"""
        return """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>project</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
</project>"""

    async def _generate_go_config(self, project_info: Dict[str, Any]) -> str:
        """Generate Go module configuration"""
        return """module example.com/project

go 1.19
"""

    async def _create_test_config(
        self, project_root: str, project_info: Dict[str, Any]
    ) -> str:
        """Create language-appropriate test configuration"""
        results = []

        for language in project_info["languages"]:
            if language == "python":
                content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
"""
                result = await self._write_file(f"{project_root}/pytest.ini", content)
                results.append(f"Python test config: {result}")

            elif language in ["javascript", "typescript"]:
                content = """{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}"""
                # This would merge with existing package.json
                result = f"Jest test configuration ready for {language}"
                results.append(result)

            elif language == "java":
                # Maven or Gradle test config would go here
                result = "Java test framework configuration ready"
                results.append(result)

            elif language == "go":
                # Go test config (usually built-in)
                result = "Go test configuration ready (built-in testing)"
                results.append(result)

        return "\\n".join(results) if results else "No test configuration created"

    async def _create_lint_config(
        self, project_root: str, project_info: Dict[str, Any]
    ) -> str:
        """Create language-appropriate linting configuration"""
        results = []

        for language in project_info["languages"]:
            if language == "python":
                content = """[tool.black]
line-length = 88

[tool.mypy]
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
line-length = 88
"""
                result = await self._write_file(
                    f"{project_root}/.python-lint.toml", content
                )
                results.append(f"Python lint config: {result}")

            elif language in ["javascript", "typescript"]:
                content = """{
  "extends": ["eslint:recommended"],
  "rules": {
    "no-console": "warn",
    "no-unused-vars": "error"
  }
}"""
                result = await self._write_file(
                    f"{project_root}/.eslintrc.json", content
                )
                results.append(f"JS/TS lint config: {result}")

        return "\\n".join(results) if results else "No lint configuration created"

    async def _setup_test_directory(self, project_root: str = ".") -> str:
        """Create language-appropriate test directory structure"""
        # Detect project type
        project_info = await self._detect_project_type(project_root)

        results = []

        # Create tests directory
        test_dir_result = await self._create_directory(f"{project_root}/tests")
        results.append(test_dir_result)

        # Create language-specific test files
        for language in project_info["languages"]:
            if language == "python":
                # Python test setup
                init_result = await self._write_file(
                    f"{project_root}/tests/__init__.py", ""
                )
                results.append("Python: Created __init__.py")

                sample_test = '''"""Sample test file."""
def test_basic():
    """Basic test."""
    assert True

def test_addition():
    """Test addition."""
    assert 1 + 1 == 2
'''
                test_result = await self._write_file(
                    f"{project_root}/tests/test_basic.py", sample_test
                )
                results.append(f"Python: {test_result}")

            elif language in ["javascript", "typescript"]:
                # JavaScript/TypeScript test setup
                ext = "ts" if language == "typescript" else "js"
                sample_test = f"""// Sample test file
describe('Basic tests', () => {{
  test('should pass basic test', () => {{
    expect(1 + 1).toBe(2);
  }});
  
  test('should test string operations', () => {{
    const text = "Hello, World!";
    expect(text.length).toBe(13);
    expect(text).toContain("World");
  }});
}});
"""
                test_result = await self._write_file(
                    f"{project_root}/tests/basic.test.{ext}", sample_test
                )
                results.append(f"{language}: {test_result}")

            elif language == "java":
                # Java test setup
                sample_test = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BasicTest {
    
    @Test
    void basicTest() {
        assertTrue(true);
    }
    
    @Test
    void testAddition() {
        assertEquals(2, 1 + 1);
    }
}
"""
                test_result = await self._write_file(
                    f"{project_root}/src/test/java/BasicTest.java", sample_test
                )
                results.append(f"Java: {test_result}")

            elif language == "go":
                # Go test setup
                sample_test = """package main

import "testing"

func TestBasic(t *testing.T) {
    if 1+1 != 2 {
        t.Errorf("Expected 1+1 to equal 2")
    }
}

func TestString(t *testing.T) {
    text := "Hello, World!"
    if len(text) != 13 {
        t.Errorf("Expected length 13, got %d", len(text))
    }
}
"""
                test_result = await self._write_file(
                    f"{project_root}/main_test.go", sample_test
                )
                results.append(f"Go: {test_result}")

        return "\\n".join(results)

    async def _intelligent_file_operation(self, task: str) -> str:
        """Handle complex file operations using AI"""
        # Check if this is a request for standard config files
        if any(
            keyword in task.lower()
            for keyword in ["config", "development", "test", "lint", "setup"]
        ):
            if "dev" in task.lower() and "config" in task.lower():
                return await self._create_config_file("dev-config")
            elif "test" in task.lower() and "config" in task.lower():
                return await self._create_config_file("test-config")
            elif "lint" in task.lower() and "config" in task.lower():
                return await self._create_config_file("lint-config")
            elif "test" in task.lower() and (
                "setup" in task.lower() or "directory" in task.lower()
            ):
                return await self._setup_test_directory()

        prompt = f"""Analyze this file operation request and determine the best approach:

Request: {task}

Based on this request, provide a structured response with:
1. The specific file operation(s) needed
2. Step-by-step plan to accomplish the task
3. Any safety considerations or warnings
4. Expected outcome

If the request involves reading files, modifying code, or creating new files, provide specific recommendations."""

        return await self.generate_response(prompt)
