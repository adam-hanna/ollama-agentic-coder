import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState

class RefactoringAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a code refactoring specialist with expertise in improving code quality, performance, and maintainability. Your role is to:

1. Analyze code for refactoring opportunities
2. Suggest performance optimizations
3. Improve code structure and design patterns
4. Extract reusable components and functions
5. Reduce code duplication and complexity
6. Modernize legacy code and adopt best practices

Refactoring principles:
- Maintain functionality while improving structure
- Follow SOLID principles and clean code practices
- Reduce cyclomatic complexity and code duplication
- Improve readability and maintainability
- Optimize performance without premature optimization
- Ensure backward compatibility when possible
- Use appropriate design patterns
- Follow language-specific best practices

Types of refactoring you can perform:
- Extract method/function
- Extract class/interface
- Rename variables, methods, classes
- Remove code duplication
- Simplify conditional expressions
- Optimize loops and data structures
- Improve error handling
- Modernize syntax and patterns
- Apply design patterns
- Performance optimizations"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No refactoring task specified")
        
        task = state.current_task
        
        try:
            if task.startswith("analyze_refactoring:"):
                file_path = task.replace("analyze_refactoring:", "").strip()
                result = await self._analyze_refactoring_opportunities(file_path)
            
            elif task.startswith("extract_method:"):
                spec = task.replace("extract_method:", "").strip()
                result = await self._extract_method(spec)
            
            elif task.startswith("remove_duplication:"):
                path = task.replace("remove_duplication:", "").strip()
                result = await self._remove_duplication(path)
            
            elif task.startswith("optimize_performance:"):
                file_path = task.replace("optimize_performance:", "").strip()
                result = await self._optimize_performance(file_path)
            
            elif task.startswith("modernize_code:"):
                file_path = task.replace("modernize_code:", "").strip()
                result = await self._modernize_code(file_path)
            
            elif task.startswith("apply_design_pattern:"):
                spec = task.replace("apply_design_pattern:", "").strip()
                result = await self._apply_design_pattern(spec)
            
            else:
                result = await self._intelligent_refactoring(task)
            
            return self.add_message(
                state,
                "assistant",
                result,
                metadata={"refactoring": True}
            )
        
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"Refactoring analysis failed: {str(e)}"
            )
    
    async def _analyze_refactoring_opportunities(self, file_path: str) -> str:
        """Analyze code and identify refactoring opportunities"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Perform static analysis
            analysis = self._perform_static_analysis(code_content, file_path)
            
            prompt = f"""Analyze this code for refactoring opportunities:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Static Analysis Results:
{analysis}

Provide a comprehensive refactoring analysis including:

1. **Code Quality Issues**:
   - Complex functions that should be split
   - Long parameter lists
   - Deep nesting levels
   - Poor naming conventions

2. **Design Improvements**:
   - Missing abstractions
   - Tight coupling issues
   - Single Responsibility Principle violations
   - Opportunities for design patterns

3. **Performance Optimizations**:
   - Inefficient algorithms or data structures
   - Unnecessary computations
   - Memory usage improvements
   - I/O optimizations

4. **Modernization Opportunities**:
   - Outdated syntax or patterns
   - Language-specific improvements
   - Library/framework updates

5. **Specific Refactoring Steps**:
   - Prioritized list of improvements
   - Step-by-step refactoring plan
   - Before/after code examples
   - Impact assessment

Rank issues by importance and provide concrete code examples."""
            
            return await self.generate_response(prompt)
        
        except Exception as e:
            return f"Failed to analyze refactoring opportunities: {str(e)}"
    
    async def _extract_method(self, specification: str) -> str:
        """Extract methods from complex functions"""
        prompt = f"""Help extract methods from complex code:

Specification: {specification}

Provide refactoring by:
1. Identifying code blocks that should be extracted
2. Suggesting meaningful method names
3. Determining appropriate parameters
4. Handling return values and side effects
5. Showing before/after code
6. Ensuring single responsibility principle

Include:
- Original code with extraction points marked
- New extracted methods with clear names
- Updated main method calling extracted methods
- Any necessary parameter passing
- Documentation for new methods"""
        
        return await self.generate_response(prompt)
    
    async def _remove_duplication(self, path: str) -> str:
        """Find and eliminate code duplication"""
        if not os.path.exists(path):
            return f"Path not found: {path}"
        
        duplication_analysis = []
        
        if os.path.isdir(path):
            # Analyze multiple files for duplication
            files_content = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                files_content[file_path] = f.read()
                        except Exception:
                            continue
            
            analysis_text = f"Analyzing {len(files_content)} files for duplication patterns"
        
        else:
            # Single file analysis
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            files_content = {path: content}
            analysis_text = f"Analyzing single file: {path}"
        
        prompt = f"""Find and eliminate code duplication:

{analysis_text}

File contents:
{chr(10).join([f"=== {file} ===" + chr(10) + content[:1000] + ("..." if len(content) > 1000 else "") for file, content in files_content.items()])}

Identify and provide solutions for:
1. **Exact Duplications**:
   - Identical code blocks across files
   - Copy-paste code patterns

2. **Similar Code Patterns**:
   - Functions with similar logic
   - Similar class structures

3. **Refactoring Solutions**:
   - Extract common functions/methods
   - Create utility classes or modules
   - Use inheritance or composition
   - Apply template method pattern

4. **Implementation Steps**:
   - Specific refactoring recommendations
   - New shared components to create
   - How to modify existing code
   - Testing considerations

Provide concrete code examples for the refactored solution."""
        
        return await self.generate_response(prompt)
    
    async def _optimize_performance(self, file_path: str) -> str:
        """Analyze and suggest performance optimizations"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Perform performance analysis
            perf_issues = self._analyze_performance_issues(code_content)
            
            prompt = f"""Analyze and optimize performance for this code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Potential Performance Issues Found:
{perf_issues}

Provide performance optimization recommendations:

1. **Algorithm Improvements**:
   - More efficient algorithms
   - Better data structures
   - Complexity analysis

2. **Memory Optimizations**:
   - Memory usage patterns
   - Object pooling opportunities
   - Garbage collection considerations

3. **I/O Optimizations**:
   - File/database access patterns
   - Caching strategies
   - Batch operations

4. **Language-Specific Optimizations**:
   - Built-in function usage
   - Library optimizations
   - Compiler/interpreter hints

5. **Concurrent Programming**:
   - Parallelization opportunities
   - Async/await patterns
   - Thread safety considerations

Include before/after performance comparisons and estimated improvements."""
            
            return await self.generate_response(prompt)
        
        except Exception as e:
            return f"Failed to analyze performance: {str(e)}"
    
    async def _modernize_code(self, file_path: str) -> str:
        """Modernize legacy code with current best practices"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            prompt = f"""Modernize this legacy code:

File: {file_path}
Code:
```{self._detect_language(file_path)}
{code_content}
```

Provide modernization recommendations:

1. **Syntax Modernization**:
   - Use latest language features
   - Replace deprecated patterns
   - Modern variable declarations

2. **Library Updates**:
   - Replace outdated libraries
   - Use modern alternatives
   - Update API calls

3. **Design Pattern Updates**:
   - Modern architectural patterns
   - Dependency injection
   - Configuration management

4. **Type Safety**:
   - Add type hints/annotations
   - Improve type checking
   - Generic programming

5. **Error Handling**:
   - Modern exception patterns
   - Result types or monads
   - Validation approaches

6. **Testing and Documentation**:
   - Update test patterns
   - Modern documentation formats
   - Code annotation standards

Show complete modernized code with explanations for each change."""
            
            return await self.generate_response(prompt)
        
        except Exception as e:
            return f"Failed to modernize code: {str(e)}"
    
    async def _apply_design_pattern(self, specification: str) -> str:
        """Apply appropriate design patterns to improve code structure"""
        prompt = f"""Apply design patterns to improve code structure:

Specification: {specification}

Analyze the code/requirements and recommend:

1. **Applicable Design Patterns**:
   - Which patterns would benefit this code
   - Why each pattern is appropriate
   - Trade-offs and considerations

2. **Implementation Examples**:
   - Complete code showing pattern application
   - Before/after comparisons
   - Integration with existing code

3. **Common Patterns to Consider**:
   - Creational: Factory, Builder, Singleton
   - Structural: Adapter, Decorator, Facade
   - Behavioral: Observer, Strategy, Command

4. **Best Practices**:
   - When to use each pattern
   - How to avoid over-engineering
   - Maintainability considerations

Provide complete, working code examples with clear explanations."""
        
        return await self.generate_response(prompt)
    
    async def _intelligent_refactoring(self, task: str) -> str:
        """Handle complex refactoring requests using AI"""
        prompt = f"""Analyze this refactoring request:

Request: {task}

Provide comprehensive refactoring guidance:
1. Understanding of the refactoring goal
2. Analysis approach and methodology
3. Specific refactoring techniques to apply
4. Step-by-step implementation plan
5. Code examples and transformations
6. Testing strategy for refactored code
7. Risk assessment and mitigation
8. Expected benefits and improvements

Focus on practical, actionable recommendations with code examples."""
        
        return await self.generate_response(prompt)
    
    def _perform_static_analysis(self, code: str, file_path: str) -> str:
        """Perform static analysis to identify refactoring opportunities"""
        issues = []
        language = self._detect_language(file_path)
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check function length
                        func_lines = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                        if func_lines > 20:
                            issues.append(f"Function '{node.name}' is long ({func_lines} statements) - consider extracting methods")
                        
                        # Check parameter count
                        if len(node.args.args) > 5:
                            issues.append(f"Function '{node.name}' has many parameters ({len(node.args.args)}) - consider parameter object")
                    
                    elif isinstance(node, ast.ClassDef):
                        # Check class size
                        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                        if len(methods) > 15:
                            issues.append(f"Class '{node.name}' has many methods ({len(methods)}) - consider splitting")
            
            except SyntaxError:
                issues.append("Syntax error in code - fix syntax before refactoring")
        
        # General code analysis
        lines = code.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
        if long_lines:
            issues.append(f"Long lines found: {long_lines[:5]}{'...' if len(long_lines) > 5 else ''}")
        
        return "\n".join(issues) if issues else "No obvious static analysis issues found"
    
    def _analyze_performance_issues(self, code: str) -> str:
        """Analyze potential performance issues"""
        issues = []
        lines = code.split('\n')
        
        # Look for common performance anti-patterns
        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()
            
            # Nested loops
            if 'for' in line_lower and any('for' in l.lower() for l in lines[i:i+10]):
                issues.append(f"Line {i}: Potential nested loops - consider optimization")
            
            # String concatenation in loops
            if '+=' in line and 'str' in line_lower:
                issues.append(f"Line {i}: String concatenation in loop - use join() or list")
            
            # Unnecessary computations
            if line_lower.count('.') > 3:
                issues.append(f"Line {i}: Multiple attribute access - consider caching")
        
        return "\n".join(issues) if issues else "No obvious performance issues detected"
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return lang_map.get(ext, 'unknown')