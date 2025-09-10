import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState


class TestGeneratorAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a test generation specialist that creates comprehensive, high-quality tests for code. Your role is to:

1. Analyze code and generate appropriate unit tests
2. Create integration tests for complex workflows
3. Generate test data and mock objects
4. Follow testing best practices and frameworks
5. Ensure good test coverage and edge case handling

Test generation principles:
- Write tests that are clear, readable, and maintainable
- Cover happy path, edge cases, and error conditions
- Use appropriate test frameworks (pytest, unittest, jest, etc.)
- Generate meaningful test data and assertions
- Follow AAA pattern (Arrange, Act, Assert)
- Create isolated, independent tests
- Include performance tests when relevant
- Generate test documentation and comments

Supported test types:
- Unit tests for individual functions/methods
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance/load tests
- Security/penetration tests
- Property-based tests using hypothesis/quick-check
- Mock and fixture generation"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(
                state, "assistant", "No test generation task specified"
            )

        task = state.current_task

        try:
            if task.startswith("generate_unit_tests:"):
                file_path = task.replace("generate_unit_tests:", "").strip()
                result = await self._generate_unit_tests(file_path)

            elif task.startswith("generate_integration_tests:"):
                file_path = task.replace("generate_integration_tests:", "").strip()
                result = await self._generate_integration_tests(file_path)

            elif task.startswith("generate_test_data:"):
                spec = task.replace("generate_test_data:", "").strip()
                result = await self._generate_test_data(spec)

            elif task.startswith("generate_mocks:"):
                spec = task.replace("generate_mocks:", "").strip()
                result = await self._generate_mocks(spec)

            elif task.startswith("analyze_coverage:"):
                path = task.replace("analyze_coverage:", "").strip()
                result = await self._analyze_test_coverage(path)

            else:
                result = await self._intelligent_test_generation(task)

            return self.add_message(
                state, "assistant", result, metadata={"test_generation": True}
            )

        except Exception as e:
            return self.add_message(
                state, "assistant", f"Test generation failed: {str(e)}"
            )

    async def _generate_unit_tests(self, file_path: str) -> str:
        """Generate comprehensive unit tests for a given file"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # Analyze the code structure
            analysis = self._analyze_code_structure(code_content, file_path)

            prompt = f"""Generate comprehensive unit tests for this code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Code Analysis:
{analysis}

Generate unit tests that:
1. Test all public functions and methods
2. Cover edge cases and error conditions
3. Use appropriate test framework for the language
4. Include proper setup and teardown
5. Have clear, descriptive test names
6. Include docstrings explaining what each test does
7. Use mocks/stubs for external dependencies
8. Test both success and failure scenarios

Provide the complete test file with:
- Proper imports and setup
- Test class organization
- Helper methods and fixtures
- Comprehensive assertions
- Error handling tests"""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate unit tests: {str(e)}"

    async def _generate_integration_tests(self, file_path: str) -> str:
        """Generate integration tests for components"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            prompt = f"""Generate integration tests for this code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Create integration tests that:
1. Test component interactions and workflows
2. Verify data flow between different parts
3. Test external service integrations
4. Include database/API integration scenarios
5. Test configuration and environment handling
6. Verify end-to-end functionality
7. Include performance considerations
8. Test error propagation and handling

Focus on realistic usage scenarios and component boundaries."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to generate integration tests: {str(e)}"

    async def _generate_test_data(self, specification: str) -> str:
        """Generate test data based on specification"""
        prompt = f"""Generate comprehensive test data for:

Specification: {specification}

Create test data that includes:
1. Valid input examples (happy path)
2. Edge cases and boundary values
3. Invalid input examples
4. Empty/null/undefined values
5. Large datasets for performance testing
6. Special characters and encoding issues
7. Realistic mock data
8. Fixtures for complex objects

Provide the data in appropriate format (JSON, CSV, Python objects, etc.) with explanations for each dataset's purpose."""

        return await self.generate_response(prompt)

    async def _generate_mocks(self, specification: str) -> str:
        """Generate mock objects and stubs"""
        prompt = f"""Generate mocks and stubs for:

Specification: {specification}

Create mocking code that:
1. Mocks external dependencies and services
2. Provides realistic return values
3. Handles different scenarios (success, failure, timeout)
4. Uses appropriate mocking framework
5. Includes setup and teardown code
6. Provides both simple mocks and complex stubs
7. Handles async operations if needed
8. Includes error simulation capabilities

Include examples of how to use the mocks in tests."""

        return await self.generate_response(prompt)

    async def _analyze_test_coverage(self, path: str) -> str:
        """Analyze test coverage and suggest improvements"""
        if not os.path.exists(path):
            return f"Path not found: {path}"

        # Find test files
        test_files = []
        source_files = []

        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if "test" in file.lower() or file.startswith("test_"):
                        test_files.append(file_path)
                    elif file.endswith((".py", ".js", ".ts", ".java", ".cpp")):
                        source_files.append(file_path)
        else:
            if "test" in path.lower():
                test_files.append(path)
            else:
                source_files.append(path)

        analysis = f"""Test Coverage Analysis for: {path}

Found {len(test_files)} test files:
{chr(10).join(test_files)}

Found {len(source_files)} source files:
{chr(10).join(source_files[:10])}{'...' if len(source_files) > 10 else ''}
"""

        prompt = f"""Analyze test coverage and provide recommendations:

{analysis}

Provide analysis on:
1. Which source files lack corresponding tests
2. Potential gaps in test coverage
3. Recommendations for additional test types needed
4. Suggestions for improving existing tests
5. Test organization and structure improvements
6. Integration points that need testing
7. Performance testing opportunities
8. Security testing considerations

Give specific, actionable recommendations for improving test coverage."""

        return await self.generate_response(prompt)

    async def _intelligent_test_generation(self, task: str) -> str:
        """Handle complex test generation requests using AI"""
        prompt = f"""Analyze this test generation request:

Request: {task}

Determine the best approach and provide:
1. What type of tests are needed
2. Testing strategy and framework recommendations
3. Key scenarios to test
4. Test data requirements
5. Mock/stub needs
6. Performance and security considerations
7. Implementation approach
8. Expected outcomes

If code examples are needed, provide complete, runnable test code."""

        return await self.generate_response(prompt)

    def _analyze_code_structure(self, code: str, file_path: str) -> str:
        """Analyze code structure to inform test generation"""
        analysis = []
        language = self._detect_language(file_path)

        if language == "python":
            try:
                tree = ast.parse(code)

                functions = []
                classes = []
                imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        args = [arg.arg for arg in node.args.args]
                        functions.append(
                            {
                                "name": node.name,
                                "args": args,
                                "line": node.lineno,
                                "async": isinstance(node, ast.AsyncFunctionDef),
                            }
                        )

                    elif isinstance(node, ast.ClassDef):
                        methods = [
                            n.name for n in node.body if isinstance(n, ast.FunctionDef)
                        ]
                        classes.append(
                            {"name": node.name, "methods": methods, "line": node.lineno}
                        )

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            imports.extend([alias.name for alias in node.names])
                        else:
                            imports.append(node.module)

                analysis.append(f"Functions to test: {len(functions)}")
                for func in functions:
                    analysis.append(f"  - {func['name']}({', '.join(func['args'])})")

                analysis.append(f"Classes to test: {len(classes)}")
                for cls in classes:
                    analysis.append(
                        f"  - {cls['name']} with {len(cls['methods'])} methods"
                    )

                analysis.append(f"External dependencies: {len(set(imports))}")

            except SyntaxError:
                analysis.append(
                    "Note: Syntax error in code, will generate tests based on structure analysis"
                )

        else:
            lines = code.split("\n")
            analysis.append(f"Lines of code: {len(lines)}")
            analysis.append(f"Language: {language}")
            analysis.append("Structure analysis not available for this language")

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
