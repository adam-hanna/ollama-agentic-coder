import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState


class DocumentationAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a documentation specialist that creates clear, comprehensive, and helpful documentation for code projects. Your role is to:

1. Generate API documentation from code
2. Create README files and project documentation
3. Write inline code comments and docstrings
4. Generate user guides and tutorials
5. Create technical specifications and architecture docs
6. Maintain documentation consistency and quality

Documentation principles:
- Write for your audience (developers, users, stakeholders)
- Use clear, concise language with proper technical terminology
- Include practical examples and code snippets
- Maintain consistent formatting and structure
- Keep documentation up-to-date with code changes
- Follow documentation standards (JSDoc, Sphinx, etc.)
- Include troubleshooting and FAQ sections
- Use appropriate markup (Markdown, reStructuredText)

Types of documentation you create:
- API reference documentation
- README files and project overviews
- Code comments and docstrings
- User guides and tutorials
- Architecture and design documents
- Installation and setup guides
- Troubleshooting and FAQ
- Changelog and release notes"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(
                state, "assistant", "No documentation task specified"
            )

        task = state.current_task

        try:
            if task.startswith("generate_readme:"):
                path = task.replace("generate_readme:", "").strip()
                result = await self._generate_readme(path)

            elif task.startswith("document_api:"):
                file_path = task.replace("document_api:", "").strip()
                result = await self._document_api(file_path)

            elif task.startswith("add_docstrings:"):
                file_path = task.replace("add_docstrings:", "").strip()
                result = await self._add_docstrings(file_path)

            elif task.startswith("create_user_guide:"):
                spec = task.replace("create_user_guide:", "").strip()
                result = await self._create_user_guide(spec)

            elif task.startswith("generate_changelog:"):
                path = task.replace("generate_changelog:", "").strip()
                result = await self._generate_changelog(path)

            elif task.startswith("document_architecture:"):
                path = task.replace("document_architecture:", "").strip()
                result = await self._document_architecture(path)

            else:
                result = await self._intelligent_documentation(task)

            return self.add_message(
                state, "assistant", result, metadata={"documentation": True}
            )

        except Exception as e:
            return self.add_message(
                state, "assistant", f"Documentation generation failed: {str(e)}"
            )

    async def _generate_readme(self, project_path: str) -> str:
        """Generate comprehensive README.md for a project"""
        if not os.path.exists(project_path):
            return f"Project path not found: {project_path}"

        try:
            # Analyze project structure
            project_analysis = self._analyze_project_structure(project_path)

            prompt = f"""Generate a comprehensive README.md for this project:

Project Analysis:
{project_analysis}

Create a README.md with the following sections:

# Project Title
- Clear, descriptive project name
- Brief one-line description
- Badges (build status, version, license, etc.)

## Description
- Detailed project overview
- Key features and capabilities
- Use cases and target audience

## Installation
- Prerequisites and requirements
- Step-by-step installation instructions
- Different installation methods if applicable
- Troubleshooting common installation issues

## Usage
- Quick start guide
- Basic usage examples with code snippets
- Configuration options
- Command-line interface if applicable

## API Reference
- Key functions and classes
- Parameters and return values
- Usage examples

## Contributing
- How to contribute to the project
- Development setup instructions
- Coding standards and guidelines
- Pull request process

## License
- License information
- Copyright details

## Changelog
- Recent changes and version history

## Support
- How to get help
- Issue reporting guidelines
- Community resources

Make it professional, clear, and actionable with proper Markdown formatting."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate README: {str(e)}"

    async def _document_api(self, file_path: str) -> str:
        """Generate API documentation for a code file"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # Analyze code structure for API documentation
            api_analysis = self._analyze_api_structure(code_content, file_path)

            prompt = f"""Generate comprehensive API documentation for this code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

API Analysis:
{api_analysis}

Create API documentation with:

## Classes
For each class, include:
- Class description and purpose
- Constructor parameters
- Public methods with signatures
- Properties and attributes
- Usage examples
- Inheritance relationships

## Functions
For each function, include:
- Function signature with types
- Parameter descriptions
- Return value description
- Exceptions that may be raised
- Usage examples
- Performance considerations

## Constants and Variables
- Global constants and their meanings
- Configuration variables
- Default values

## Examples
- Practical usage examples
- Common use cases
- Integration examples
- Error handling examples

Use proper formatting with code blocks, tables, and cross-references."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate API documentation: {str(e)}"

    async def _add_docstrings(self, file_path: str) -> str:
        """Add comprehensive docstrings to code"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            prompt = f"""Add comprehensive docstrings to this code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Add appropriate docstrings following the language conventions:

**For Python:**
- Use triple quotes with proper format
- Include Args, Returns, Raises sections
- Follow Google or Sphinx style

**For JavaScript/TypeScript:**
- Use JSDoc format with @param, @returns, @throws
- Include type information
- Document callback functions

**For other languages:**
- Use appropriate comment style
- Include parameter and return descriptions
- Document exceptions and side effects

Requirements:
1. Document all public functions/methods
2. Include parameter types and descriptions
3. Describe return values
4. Note any exceptions or errors
5. Add usage examples for complex functions
6. Include performance notes where relevant
7. Document any side effects
8. Keep docstrings concise but comprehensive

Show the complete updated code with all docstrings added."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to add docstrings: {str(e)}"

    async def _create_user_guide(self, specification: str) -> str:
        """Create user guides and tutorials"""
        prompt = f"""Create a comprehensive user guide:

Specification: {specification}

Generate a user-friendly guide with:

## Getting Started
- Prerequisites and setup
- First steps and basic concepts
- Quick start tutorial

## Core Concepts
- Key terminology and definitions
- Important principles to understand
- Mental models and analogies

## Step-by-Step Tutorials
- Practical walkthroughs
- Progressive complexity
- Common use cases
- Real-world examples

## Best Practices
- Recommended approaches
- Common pitfalls to avoid
- Performance tips
- Security considerations

## Troubleshooting
- Common issues and solutions
- Error messages and fixes
- Debugging techniques
- When to seek help

## Advanced Topics
- Power user features
- Customization options
- Integration possibilities
- Extensibility

## FAQ
- Frequently asked questions
- Community-sourced solutions
- Migration guides

Use clear headings, bullet points, code examples, and screenshots where helpful."""

        return await self.generate_response(prompt)

    async def _generate_changelog(self, project_path: str) -> str:
        """Generate changelog from Git history or project analysis"""
        if not os.path.exists(project_path):
            return f"Project path not found: {project_path}"

        try:
            # Try to get Git history if available
            git_history = ""
            try:
                import subprocess

                result = subprocess.run(
                    "git log --oneline --no-merges -20",
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=project_path,
                )
                if result.returncode == 0:
                    git_history = result.stdout
            except Exception:
                pass

            project_info = self._analyze_project_structure(project_path)

            prompt = f"""Generate a CHANGELOG.md for this project:

Project Information:
{project_info}

Git History (if available):
{git_history}

Create a changelog following Keep a Changelog format:

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [Version] - Date
### Added
- New features
### Changed  
- Changes in existing functionality
### Deprecated
- Soon-to-be removed features
### Removed
- Removed features
### Fixed
- Bug fixes
### Security
- Security improvements

Use semantic versioning and categorize changes appropriately. If Git history is available, parse commit messages to extract changes."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate changelog: {str(e)}"

    async def _document_architecture(self, project_path: str) -> str:
        """Generate architecture documentation"""
        if not os.path.exists(project_path):
            return f"Project path not found: {project_path}"

        try:
            project_analysis = self._analyze_project_structure(project_path)

            prompt = f"""Generate architecture documentation for this project:

Project Analysis:
{project_analysis}

Create comprehensive architecture documentation:

# Architecture Overview

## System Architecture
- High-level system design
- Major components and their responsibilities
- Data flow and interactions
- External dependencies

## Directory Structure
- Project organization
- Module/package breakdown
- Configuration files
- Asset organization

## Component Design
- Core components and their purposes
- Interfaces and APIs
- Data models and schemas
- Service layer architecture

## Design Patterns
- Architectural patterns used
- Design patterns implemented
- Coding conventions
- Best practices followed

## Data Architecture
- Database schema (if applicable)
- Data flow and transformations
- Caching strategies
- Storage considerations

## Security Architecture
- Authentication and authorization
- Data protection measures
- Security boundaries
- Threat model considerations

## Performance Considerations
- Scalability design
- Performance bottlenecks
- Optimization strategies
- Monitoring and metrics

## Deployment Architecture
- Environment setup
- Infrastructure requirements
- CI/CD pipeline
- Monitoring and logging

Use diagrams, code examples, and clear explanations."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate architecture documentation: {str(e)}"

    async def _intelligent_documentation(self, task: str) -> str:
        """Handle complex documentation requests using AI"""
        prompt = f"""Handle this documentation request:

Request: {task}

Provide appropriate documentation based on the request:
1. Analyze what type of documentation is needed
2. Determine the target audience
3. Choose appropriate format and structure
4. Include relevant sections and content
5. Use proper formatting and examples
6. Consider maintenance and update requirements

Focus on creating practical, useful documentation that serves its intended purpose."""

        return await self.generate_response(prompt)

    def _analyze_project_structure(self, project_path: str) -> str:
        """Analyze project structure for documentation purposes"""
        analysis = []

        try:
            # Get directory structure
            analysis.append(f"Project Path: {project_path}")

            # Count files by type
            file_types = {}
            total_files = 0

            for root, dirs, files in os.walk(project_path):
                # Skip hidden directories and common build directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ["node_modules", "__pycache__", "build", "dist"]
                ]

                for file in files:
                    if file.startswith("."):
                        continue

                    ext = Path(file).suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                    total_files += 1

            analysis.append(f"Total Files: {total_files}")
            analysis.append("File Types:")
            for ext, count in sorted(
                file_types.items(), key=lambda x: x[1], reverse=True
            ):
                if count > 1:
                    analysis.append(f"  {ext or '(no extension)'}: {count} files")

            # Look for key files
            key_files = [
                "README.md",
                "package.json",
                "requirements.txt",
                "setup.py",
                "Dockerfile",
                "docker-compose.yml",
            ]
            found_files = []
            for key_file in key_files:
                if os.path.exists(os.path.join(project_path, key_file)):
                    found_files.append(key_file)

            if found_files:
                analysis.append(f"Key Files Found: {', '.join(found_files)}")

        except Exception as e:
            analysis.append(f"Error analyzing project: {str(e)}")

        return "\n".join(analysis)

    def _analyze_api_structure(self, code: str, file_path: str) -> str:
        """Analyze code structure for API documentation"""
        analysis = []
        language = self._detect_language(file_path)

        if language == "python":
            try:
                tree = ast.parse(code)

                classes = []
                functions = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [
                            n.name for n in node.body if isinstance(n, ast.FunctionDef)
                        ]
                        classes.append(
                            {"name": node.name, "methods": methods, "line": node.lineno}
                        )
                    elif isinstance(node, ast.FunctionDef) and not any(
                        isinstance(parent, ast.ClassDef)
                        for parent in ast.walk(tree)
                        if hasattr(parent, "body")
                        and node in getattr(parent, "body", [])
                    ):
                        args = [arg.arg for arg in node.args.args]
                        functions.append(
                            {"name": node.name, "args": args, "line": node.lineno}
                        )

                analysis.append(f"Classes: {len(classes)}")
                for cls in classes:
                    analysis.append(
                        f"  - {cls['name']} ({len(cls['methods'])} methods)"
                    )

                analysis.append(f"Functions: {len(functions)}")
                for func in functions:
                    analysis.append(f"  - {func['name']}({', '.join(func['args'])})")

            except SyntaxError:
                analysis.append("Syntax error in code - API analysis limited")

        else:
            analysis.append(f"Language: {language}")
            analysis.append("Detailed API analysis not available for this language")

        return "\n".join(analysis)

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
        }
        return lang_map.get(ext, "unknown")
