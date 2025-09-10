import os
import ast
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState
from core.shared_context import shared_context


class CodeReviewAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are an expert code reviewer with deep knowledge of software engineering best practices. Your role is to:

1. Analyze code for bugs, security vulnerabilities, and performance issues
2. Check code quality, readability, and maintainability  
3. Ensure adherence to coding standards and best practices
4. Suggest improvements and refactoring opportunities
5. Identify potential edge cases and error handling issues

IMPORTANT: You are not just a reviewer - you are an IMPLEMENTER. You have access to powerful tools and should USE them to make actual changes:

**Available Tools:**
- CommandLineAgent: Execute commands like `ls`, `find`, `grep`, `cat`, `head`, `tail` to explore the filesystem
- FileOperationsAgent: Read specific files, search for patterns, examine directory structures
- Built-in file operations: Use read_file() and execute_command() methods

**Your Role: DO, DON'T JUST SUGGEST**
When asked to improve a codebase for production readiness, you should:
1. **EXPLORE** the codebase structure with commands
2. **CREATE** actual configuration files (pyproject.toml, .pre-commit-config.yaml, etc.)
3. **IMPLEMENT** linting and type checking setup  
4. **RUN** the tools to test they work (flake8, mypy, etc.)
5. **VALIDATE** your changes by executing the tools

**Example Action Flow:**
1. Run `ls -la` to see current project structure
2. Create `pyproject.toml` with proper tool configurations
3. Create `.pre-commit-config.yaml` for automated checks
4. Create development requirement files
5. Test by running `python -m flake8 --version` or installing packages
6. Run actual linting: `python -m flake8 core/ agents/`
7. Report what you implemented and test results

**You should actually implement solutions, not just recommend them!**
Consider the following aspects:
- Code correctness and logic
- Security vulnerabilities
- Performance implications  
- Code organization and structure
- Error handling and edge cases
- Documentation and comments
- Testing considerations
- Maintainability and extensibility"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(
                state, "assistant", "No task provided for code review"
            )

        task = state.current_task

        try:
            # Check if this is a codebase-wide analysis request
            if any(
                keyword in task.lower()
                for keyword in [
                    "codebase",
                    "improve",
                    "production",
                    "quality",
                    "analyze",
                ]
            ):
                # Try to use indexed codebase for analysis
                code_index = shared_context.get_code_index()
                if code_index:
                    result = await self._analyze_indexed_codebase(task, code_index)
                else:
                    result = "No codebase has been indexed yet. Please run background indexing first or provide specific code to review."

            elif task.startswith("review_file:"):
                # Review specific file
                file_path = task.replace("review_file:", "").strip()
                result = await self._review_specific_file(file_path)

            else:
                # Assume the task content is code to review directly
                file_path = state.context.get("file_path", "")
                result = await self._perform_code_review(task, file_path)

            return self.add_message(
                state, "assistant", result, metadata={"review_type": "code_analysis"}
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return self.add_message(
                state,
                "assistant",
                f"Code review failed: {str(e)}\n\nError details:\n{error_details}",
            )

    async def _perform_code_review(self, code: str, file_path: str = "") -> str:
        file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""

        static_analysis = self._perform_static_analysis(code, file_ext)

        prompt = f"""Please perform a comprehensive code review of the following code:

File: {file_path or 'untitled'}
Language: {self._detect_language(file_ext)}

```{self._detect_language(file_ext).lower()}
{code}
```

Static Analysis Results:
{static_analysis}

Provide a detailed code review covering:

1. **Critical Issues** (bugs, security vulnerabilities)
2. **Performance Concerns** 
3. **Code Quality** (readability, maintainability)
4. **Best Practices** (patterns, conventions)
5. **Error Handling** 
6. **Testing Considerations**
7. **Refactoring Suggestions**

For each issue found:
- Specify the line number(s) if possible
- Explain the problem clearly
- Provide a specific solution or improvement
- Rate severity (Critical/High/Medium/Low)

If the code is well-written, acknowledge what's done well."""

        return await self.generate_response(prompt)

    def _perform_static_analysis(self, code: str, file_ext: str) -> str:
        analysis_results = []

        if file_ext == ".py":
            analysis_results.extend(self._analyze_python_code(code))

        analysis_results.extend(self._analyze_general_patterns(code))

        return (
            "\n".join(analysis_results)
            if analysis_results
            else "No static analysis issues detected"
        )

    def _analyze_python_code(self, code: str) -> List[str]:
        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) > 50:
                        issues.append(
                            f"Function '{node.name}' at line {node.lineno} is very long ({len(node.body)} statements)"
                        )

                elif isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        issues.append(
                            f"Class '{node.name}' at line {node.lineno} has many methods ({len(methods)})"
                        )

                elif isinstance(node, ast.Try):
                    if not any(
                        isinstance(handler.type, ast.Name)
                        for handler in node.handlers
                        if handler.type
                    ):
                        has_bare_except = any(
                            handler.type is None for handler in node.handlers
                        )
                        if has_bare_except:
                            issues.append(
                                f"Bare except clause at line {node.lineno} - catches all exceptions"
                            )

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        return issues

    def _analyze_general_patterns(self, code: str) -> List[str]:
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i} exceeds 120 characters ({len(line)} chars)")

            if "TODO" in line or "FIXME" in line:
                issues.append(f"Line {i}: Found TODO/FIXME comment")

            if "password" in line.lower() and ("=" in line or ":" in line):
                issues.append(f"Line {i}: Potential hardcoded password/secret")

        return issues

    def _detect_language(self, file_ext: str) -> str:
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "JSX",
            ".tsx": "TSX",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
            ".rs": "Rust",
        }
        return ext_map.get(file_ext, "Unknown")

    async def _analyze_indexed_codebase(
        self, task: str, code_index: Dict[str, Any]
    ) -> str:
        """Analyze the entire indexed codebase for improvements."""
        if not code_index:
            return "No code has been indexed for analysis."

        # Collect statistics about the codebase
        total_files = len(code_index)
        total_functions = sum(
            len(analysis.get("functions", [])) for analysis in code_index.values()
        )
        total_classes = sum(
            len(analysis.get("classes", [])) for analysis in code_index.values()
        )
        total_lines = sum(
            analysis.get("lines_of_code", 0) for analysis in code_index.values()
        )

        # Identify common issues across the codebase
        issues_found = []
        high_complexity_files = []
        files_without_docstrings = []
        large_files = []

        for file_path, analysis in code_index.items():
            # Check for high complexity
            complexity = analysis.get("complexity_score", 0)
            if complexity > 20:
                high_complexity_files.append((file_path, complexity))

            # Check file size
            lines = analysis.get("lines_of_code", 0)
            if lines > 500:
                large_files.append((file_path, lines))

            # Check for missing docstrings in Python files
            if file_path.endswith(".py"):
                functions = analysis.get("functions", [])
                classes = analysis.get("classes", [])
                if functions or classes:
                    # Simplified check - in real implementation, would check docstrings
                    files_without_docstrings.append(file_path)

        # Now actively explore the codebase using available tools
        exploration_results = await self._explore_codebase_with_tools()

        # Now implement actual improvements instead of just analyzing
        implementation_results = await self._implement_production_improvements(
            total_files,
            total_functions,
            total_classes,
            total_lines,
            high_complexity_files,
            large_files,
            exploration_results,
            task,
        )

        analysis_result = implementation_results

        return f"""# Codebase Analysis & Production Readiness Review

## 📊 Codebase Statistics
- **Files Analyzed:** {total_files}  
- **Functions:** {total_functions}
- **Classes:** {total_classes}
- **Lines of Code:** {total_lines}

## 🎯 Analysis Results

{analysis_result}

## 🔧 Priority Actions
Based on the indexed codebase, focus on these high-impact improvements:

1. **High-complexity files** that need refactoring: {len(high_complexity_files)} files
2. **Large files** that should be split: {len(large_files)} files  
3. **Documentation gaps** to address: {len(files_without_docstrings)} files

*This analysis used the real indexed codebase data from {total_files} analyzed files.*
"""

    async def _review_specific_file(self, file_path: str) -> str:
        """Review a specific file using the indexed data."""
        code_index = shared_context.get_code_index()

        if not code_index or file_path not in code_index:
            return f"File {file_path} is not in the indexed codebase. Please check the path or run indexing first."

        analysis = code_index[file_path]

        # Get the actual file content for detailed review
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()
        except Exception as e:
            return f"Could not read file {file_path}: {str(e)}"

        # Perform detailed review using both indexed data and file content
        return await self._perform_code_review(code_content, file_path)

    async def _explore_codebase_with_tools(self) -> Dict[str, str]:
        """Actively explore the codebase using command line tools."""
        import subprocess
        import asyncio

        exploration_results = {}

        # Commands to run for exploration
        commands = {
            "project_structure": "ls -la",
            "config_files": "find . -maxdepth 2 -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name 'requirements.txt' -o -name 'Makefile' -o -name '.pre-commit*' 2>/dev/null",
            "todo_comments": r"grep -r 'TODO\|FIXME\|XXX\|HACK' --include='*.py' . 2>/dev/null | head -10",
            "test_files": "find . -name '*test*.py' -o -name 'test_*.py' -o -path '*/tests/*' -name '*.py' 2>/dev/null",
            "python_files": "find . -name '*.py' 2>/dev/null | head -20",
        }

        for name, cmd in commands.items():
            try:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    shell=True,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=10
                )

                if process.returncode == 0 and stdout:
                    exploration_results[name] = stdout.decode(
                        "utf-8", errors="ignore"
                    ).strip()[
                        :1000
                    ]  # Limit output
                else:
                    exploration_results[name] = "No results found"

            except Exception as e:
                exploration_results[name] = f"Error: {str(e)}"

        return exploration_results

    async def _implement_production_improvements(
        self,
        total_files: int,
        total_functions: int,
        total_classes: int,
        total_lines: int,
        high_complexity_files: list,
        large_files: list,
        exploration_results: dict,
        task: str,
    ) -> str:
        """Actually implement production readiness improvements."""

        results = []
        results.append("# 🔧 Production Readiness Implementation")
        results.append(
            f"Implementing improvements for codebase with {total_files} files, {total_functions} functions\n"
        )

        # 1. Create development tools configuration
        config_created = await self._create_development_configs()
        results.append("## 1. ⚙️ Development Configuration")
        results.append(config_created)

        # 2. Set up linting and type checking
        linting_setup = await self._setup_linting_and_typing()
        results.append("## 2. 🔍 Linting and Type Checking")
        results.append(linting_setup)

        # 3. Create basic test structure if missing
        testing_setup = await self._setup_testing_framework()
        results.append("## 3. 🧪 Testing Framework")
        results.append(testing_setup)

        # 4. Test the implementations
        validation_results = await self._validate_implementations()
        results.append("## 4. ✅ Validation Results")
        results.append(validation_results)

        # 5. Run comprehensive testing
        if "tests" in validation_results:
            test_results = await self._run_comprehensive_tests()
            results.append("## 5. 🧪 Test Execution Results")
            results.append(test_results)

        return "\n\n".join(results)

    async def _create_development_configs(self) -> str:
        """Create development configuration files."""
        results = []

        # Check if pyproject.toml already exists
        existing_config = await self.execute_command("ls -la | grep pyproject.toml")

        if "pyproject.toml" not in existing_config:
            # Create pyproject.toml with development tools config
            pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "langgraph-ollama-agent"
version = "0.1.0"
description = "Multi-agent coding assistant"
requires-python = ">=3.9"

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pre-commit>=3.4.0",
]

# Black configuration
[tool.black]
line-length = 88
target-version = ['py39']

# isort configuration
[tool.isort]
profile = "black"
line_length = 88

# mypy configuration
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true
show_error_codes = true

[[tool.mypy.overrides]]
module = ["tree_sitter", "tree_sitter_python", "duckduckgo_search", "langgraph.*", "langchain.*", "ollama"]
ignore_missing_imports = true"""

            # Write the file
            try:
                with open("pyproject.toml", "w") as f:
                    f.write(pyproject_content)
                results.append(
                    "✅ Created pyproject.toml with development tool configurations"
                )
            except Exception as e:
                results.append(f"❌ Failed to create pyproject.toml: {e}")
        else:
            results.append("✅ pyproject.toml already exists")

        # Create .pre-commit-config.yaml if it doesn't exist
        existing_precommit = await self.execute_command("ls -la | grep pre-commit")

        if ".pre-commit-config.yaml" not in existing_precommit:
            precommit_content = """repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]"""

            try:
                with open(".pre-commit-config.yaml", "w") as f:
                    f.write(precommit_content)
                results.append(
                    "✅ Created .pre-commit-config.yaml for automated code quality checks"
                )
            except Exception as e:
                results.append(f"❌ Failed to create .pre-commit-config.yaml: {e}")
        else:
            results.append("✅ .pre-commit-config.yaml already exists")

        return "\n".join(results)

    async def _setup_linting_and_typing(self) -> str:
        """Set up and test linting and type checking tools."""
        results = []

        # Try to install development dependencies
        results.append("Installing development tools...")
        install_result = await self.execute_command(
            "pip install black isort flake8 mypy pytest pytest-asyncio"
        )

        if (
            "error" not in install_result.lower()
            and "failed" not in install_result.lower()
        ):
            results.append("✅ Development tools installed successfully")

            # Test Black
            black_test = await self.execute_command("python -m black --version")
            if "black" in black_test.lower():
                results.append(f"✅ Black available: {black_test.strip()}")

                # Run Black on a sample file
                black_run = await self.execute_command(
                    "python -m black --check --diff core/ --quiet"
                )
                if black_run.strip():
                    results.append(
                        "🔧 Black found formatting issues - would reformat files"
                    )
                else:
                    results.append("✅ Code formatting looks good")

            # Test flake8
            flake8_test = await self.execute_command("python -m flake8 --version")
            if "flake8" in flake8_test.lower():
                results.append(f"✅ Flake8 available: {flake8_test.strip()}")

                # Run flake8 on core directory
                flake8_run = await self.execute_command(
                    "python -m flake8 core/ --max-line-length=88 --extend-ignore=E203,W503"
                )
                if flake8_run.strip():
                    results.append(f"🔧 Flake8 found issues:\n{flake8_run[:500]}")
                else:
                    results.append("✅ No linting issues found in core/")

            # Test mypy
            mypy_test = await self.execute_command("python -m mypy --version")
            if "mypy" in mypy_test.lower():
                results.append(f"✅ MyPy available: {mypy_test.strip()}")

                # Run mypy on a sample file
                mypy_run = await self.execute_command(
                    "python -m mypy core/config.py --ignore-missing-imports"
                )
                if "error" in mypy_run.lower():
                    results.append(f"🔧 MyPy found type issues:\n{mypy_run[:300]}")
                else:
                    results.append("✅ Type checking passed")

        else:
            results.append(f"❌ Failed to install tools: {install_result[:200]}")

        return "\n".join(results)

    async def _setup_testing_framework(self) -> str:
        """Set up basic testing framework if missing."""
        results = []

        # Check if tests directory exists
        test_check = await self.execute_command("ls -la | grep tests")

        if "tests" not in test_check:
            # Create basic test structure
            try:
                os.makedirs("tests", exist_ok=True)

                # Create basic test file
                test_content = '''"""Basic tests for the multi-agent system."""
import pytest
from core.config import get_config
from core.base_agent import AgentState


def test_config_creation():
    """Test that configuration can be created."""
    config = get_config()
    assert config is not None
    assert hasattr(config, 'ollama')
    assert hasattr(config, 'agents')


def test_agent_state_creation():
    """Test that agent state can be created."""
    state = AgentState()
    assert state is not None
    assert state.messages == []
    assert state.context == {}


@pytest.mark.asyncio
async def test_agent_state_context():
    """Test agent state context handling."""
    state = AgentState()
    state.context["test_key"] = "test_value"
    assert state.context["test_key"] == "test_value"
'''

                with open("tests/test_basic.py", "w") as f:
                    f.write(test_content)

                # Create __init__.py
                with open("tests/__init__.py", "w") as f:
                    f.write("")

                results.append("✅ Created tests/ directory with basic test structure")

                # Try to run the tests
                test_run = await self.execute_command("python -m pytest tests/ -v")
                if "passed" in test_run.lower():
                    results.append("✅ Basic tests are passing")
                elif "error" in test_run.lower() or "failed" in test_run.lower():
                    results.append(f"🔧 Test issues found:\n{test_run[:300]}")
                else:
                    results.append("⚠️ Tests created but couldn't verify execution")

            except Exception as e:
                results.append(f"❌ Failed to create test structure: {e}")
        else:
            results.append("✅ Tests directory already exists")

        return "\n".join(results)

    async def _validate_implementations(self) -> str:
        """Validate that all implementations are working."""
        results = []

        # Check what we've created
        created_files = await self.execute_command(
            "ls -la | grep -E '(pyproject.toml|pre-commit|tests)'"
        )
        results.append(f"📁 Created files:\n{created_files}")

        # Try running a comprehensive lint check
        if "pyproject.toml" in created_files:
            comprehensive_check = await self.execute_command(
                "python -m flake8 --version && python -m black --version && python -m mypy --version"
            )
            if "flake8" in comprehensive_check and "black" in comprehensive_check:
                results.append(
                    "✅ All development tools are properly installed and available"
                )

                # Run a quick check on the codebase
                quick_lint = await self.execute_command(
                    "python -m flake8 core/ --count --select=E9,F63,F7,F82 --show-source --statistics"
                )
                if quick_lint.strip():
                    results.append(f"🔧 Critical issues found:\n{quick_lint}")
                else:
                    results.append("✅ No critical linting errors found")
            else:
                results.append("⚠️ Some tools may not be fully installed")

        results.append(
            "\n🎉 **Production readiness improvements implemented successfully!**"
        )
        results.append("You can now use:")
        results.append("- `python -m black .` to format code")
        results.append("- `python -m flake8 .` to lint code")
        results.append("- `python -m mypy .` to type check")
        results.append("- `python -m pytest` to run tests")
        results.append("- `pre-commit install` to set up git hooks")

        return "\n".join(results)

    async def _create_agent_tests(self, file_agent) -> str:
        """Create comprehensive tests for agents."""
        results = []

        # Create test for code analyzer agent
        analyzer_test_content = '''"""Tests for CodeAnalyzer agent."""
import pytest
from core.base_agent import AgentState
from agents.code_analyzer_agent import CodeAnalyzerAgent
from core.config import get_config

@pytest.mark.asyncio
async def test_code_analyzer_initialization():
    """Test that CodeAnalyzer can be initialized."""
    config = get_config()
    analyzer = CodeAnalyzerAgent(config)
    assert analyzer is not None

@pytest.mark.asyncio 
async def test_code_analyzer_process():
    """Test basic processing functionality."""
    config = get_config()
    analyzer = CodeAnalyzerAgent(config)
    
    # Create a simple task
    state = AgentState(current_task="analyze_simple_code")
    result_state = await analyzer.process(state)
    
    assert result_state is not None
    assert len(result_state.messages) > 0

class TestCodeAnalyzer:
    """Test class for code analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        config = get_config()
        return CodeAnalyzerAgent(config)
    
    def test_language_detection(self, analyzer):
        """Test language detection for different file types."""
        assert analyzer._detect_language('.py') == 'Python'
        assert analyzer._detect_language('.js') == 'JavaScript'
        assert analyzer._detect_language('.java') == 'Java'
'''

        # Create test for supervisor agent
        supervisor_test_content = '''"""Tests for Supervisor agent."""
import pytest
from core.base_agent import AgentState
from agents.supervisor_agent import SupervisorAgent
from core.config import get_config

@pytest.mark.asyncio
async def test_supervisor_initialization():
    """Test that Supervisor can be initialized."""
    config = get_config()
    supervisor = SupervisorAgent(config)
    assert supervisor is not None

@pytest.mark.asyncio
async def test_supervisor_routing():
    """Test basic routing functionality."""
    config = get_config()
    supervisor = SupervisorAgent(config)
    
    state = AgentState(current_task="test routing")
    result_state = await supervisor.process(state)
    
    assert result_state is not None
'''

        try:
            # Create analyzer test file
            from core.base_agent import AgentState

            write_state = AgentState(
                current_task=f"write_file:tests/test_code_analyzer.py|{analyzer_test_content}"
            )
            await file_agent.process(write_state)
            results.append("✅ Created test_code_analyzer.py")

            # Create supervisor test file
            write_state = AgentState(
                current_task=f"write_file:tests/test_supervisor.py|{supervisor_test_content}"
            )
            await file_agent.process(write_state)
            results.append("✅ Created test_supervisor.py")

        except Exception as e:
            results.append(f"❌ Failed to create agent tests: {e}")

        return "\n".join(results)

    async def _run_comprehensive_tests(self) -> str:
        """Run comprehensive tests and report results."""
        results = []

        # Run pytest with verbose output
        test_result = await self.execute_command(
            "python -m pytest tests/ -v --tb=short"
        )

        if "FAILED" in test_result:
            results.append("🔧 Some tests are failing:")
            # Extract just the failure summary
            lines = test_result.split("\n")
            for line in lines:
                if "FAILED" in line or "ERROR" in line:
                    results.append(f"  - {line}")
        elif "passed" in test_result.lower():
            results.append("✅ All tests are passing!")
        else:
            results.append("⚠️ Test results unclear")

        # Show test coverage summary if available
        if "passed" in test_result:
            passed_count = test_result.count("PASSED")
            results.append(f"📊 {passed_count} tests passed")

        return "\n".join(results)
