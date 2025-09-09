import os
import ast
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState

class CodeReviewAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are an expert code reviewer with deep knowledge of software engineering best practices. Your role is to:

1. Analyze code for bugs, security vulnerabilities, and performance issues
2. Check code quality, readability, and maintainability  
3. Ensure adherence to coding standards and best practices
4. Suggest improvements and refactoring opportunities
5. Identify potential edge cases and error handling issues

Provide constructive, actionable feedback with specific examples and suggestions.
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
            return self.add_message(state, "assistant", "No code provided for review")
        
        code_to_review = state.current_task
        file_path = state.context.get("file_path", "")
        
        try:
            review_result = await self._perform_code_review(code_to_review, file_path)
            
            return self.add_message(
                state,
                "assistant", 
                review_result,
                metadata={"review_type": "code_analysis", "file_path": file_path}
            )
        
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"Code review failed: {str(e)}"
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
        
        if file_ext == '.py':
            analysis_results.extend(self._analyze_python_code(code))
        
        analysis_results.extend(self._analyze_general_patterns(code))
        
        return "\n".join(analysis_results) if analysis_results else "No static analysis issues detected"
    
    def _analyze_python_code(self, code: str) -> List[str]:
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) > 50:
                        issues.append(f"Function '{node.name}' at line {node.lineno} is very long ({len(node.body)} statements)")
                
                elif isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        issues.append(f"Class '{node.name}' at line {node.lineno} has many methods ({len(methods)})")
                
                elif isinstance(node, ast.Try):
                    if not any(isinstance(handler.type, ast.Name) for handler in node.handlers if handler.type):
                        has_bare_except = any(handler.type is None for handler in node.handlers)
                        if has_bare_except:
                            issues.append(f"Bare except clause at line {node.lineno} - catches all exceptions")
        
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        return issues
    
    def _analyze_general_patterns(self, code: str) -> List[str]:
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i} exceeds 120 characters ({len(line)} chars)")
            
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"Line {i}: Found TODO/FIXME comment")
            
            if 'password' in line.lower() and ('=' in line or ':' in line):
                issues.append(f"Line {i}: Potential hardcoded password/secret")
        
        return issues
    
    def _detect_language(self, file_ext: str) -> str:
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.jsx': 'JSX',
            '.tsx': 'TSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        return ext_map.get(file_ext, 'Unknown')