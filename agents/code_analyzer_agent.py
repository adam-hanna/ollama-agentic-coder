import os
import ast
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from core.base_agent import BaseAgent, AgentState
from core.shared_context import shared_context

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

class CodeAnalyzerAgent(BaseAgent):
    def __init__(self, name: str, config, system_prompt: Optional[str] = None):
        super().__init__(name, config, system_prompt)
        self.code_index: Dict[str, Any] = {}
        self.ast_cache: Dict[str, Any] = {}
        self._setup_tree_sitter()
    
    def _setup_tree_sitter(self):
        if TREE_SITTER_AVAILABLE:
            try:
                PY_LANGUAGE = Language(tspython.language())
                self.parser = Parser(PY_LANGUAGE)
            except Exception as e:
                print(f"Warning: Tree-sitter setup failed: {e}")
                self.parser = None
        else:
            print("Tree-sitter not available, using AST-only parsing")
            self.parser = None
    
    def _default_system_prompt(self) -> str:
        return """You are a code analysis specialist with expertise in understanding codebases. Your role is to:

1. Analyze code structure, dependencies, and relationships
2. Index functions, classes, variables, and imports across the codebase
3. Identify patterns, architectural issues, and code organization
4. Track code metrics and complexity
5. Find code duplications and similar patterns
6. Analyze code flow and call graphs

Provide insights about:
- Code architecture and organization
- Function and class relationships  
- Import dependencies
- Code complexity and metrics
- Potential refactoring opportunities
- Dead or unused code
- Code coverage analysis"""
    
    async def process(self, state: AgentState) -> AgentState:
        task = state.current_task
        
        if task.startswith("index_directory:"):
            directory_path = task.replace("index_directory:", "").strip()
            result = await self._index_directory(directory_path)
            
        elif task.startswith("analyze_file:"):
            file_path = task.replace("analyze_file:", "").strip()
            result = await self._analyze_file(file_path)
            
        elif task.startswith("find_function:"):
            function_name = task.replace("find_function:", "").strip()
            result = await self._find_function(function_name)
            
        elif task.startswith("analyze_dependencies:"):
            file_path = task.replace("analyze_dependencies:", "").strip()
            result = await self._analyze_dependencies(file_path)
            
        else:
            result = await self._general_analysis(task)
        
        return self.add_message(
            state,
            "assistant",
            result,
            metadata={"analysis_type": "code_analysis", "index_size": len(self.code_index)}
        )
    
    async def _index_directory(self, directory_path: str) -> str:
        if not os.path.exists(directory_path):
            return f"Directory not found: {directory_path}"
        
        indexed_files = []
        total_functions = 0
        total_classes = 0
        
        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(ext) for ext in self.config.code_analyzer.supported_extensions):
                    file_path = os.path.join(root, file)
                    
                    if os.path.getsize(file_path) > self.config.code_analyzer.max_file_size:
                        continue
                    
                    try:
                        analysis = await self._analyze_file_content(file_path)
                        self.code_index[file_path] = analysis
                        indexed_files.append(file_path)
                        total_functions += len(analysis.get('functions', []))
                        total_classes += len(analysis.get('classes', []))
                    
                    except Exception as e:
                        print(f"Failed to analyze {file_path}: {e}")
        
        # Share the code index with other agents
        shared_context.set_code_index(self.code_index)
        
        summary = f"""Directory indexing complete for: {directory_path}

Statistics:
- Files indexed: {len(indexed_files)}
- Total functions: {total_functions}  
- Total classes: {total_classes}
- Languages detected: {self._get_language_stats()}

The codebase index is now ready for analysis queries."""
        
        return summary
    
    async def _analyze_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        try:
            analysis = await self._analyze_file_content(file_path)
            self.code_index[file_path] = analysis
            
            prompt = f"""Analyze this code file in detail:

File: {file_path}
Analysis Results: {json.dumps(analysis, indent=2)}

Provide insights about:
1. Code structure and organization
2. Complexity analysis
3. Dependencies and imports
4. Potential issues or improvements
5. Architectural patterns used
6. Code quality assessment"""
            
            return await self.generate_response(prompt)
        
        except Exception as e:
            return f"Failed to analyze file: {str(e)}"
    
    async def _analyze_file_content(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        analysis = {
            'file_path': file_path,
            'language': self._detect_language(file_path),
            'lines_of_code': len(content.split('\n')),
            'file_size': len(content),
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity_score': 0
        }
        
        if file_path.endswith('.py'):
            analysis.update(self._analyze_python_ast(content))
        elif self.parser and file_path.endswith('.py'):
            analysis.update(self._analyze_with_tree_sitter(content))
        
        return analysis
    
    def _analyze_python_ast(self, content: str) -> Dict[str, Any]:
        analysis = {'functions': [], 'classes': [], 'imports': [], 'complexity_score': 0}
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [self._ast_to_string(d) for d in node.decorator_list],
                        'async': isinstance(node, ast.AsyncFunctionDef)
                    })
                    analysis['complexity_score'] += self._calculate_complexity(node)
                
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': methods,
                        'bases': [self._ast_to_string(base) for base in node.bases]
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append({
                                'module': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
                    else:
                        for alias in node.names:
                            analysis['imports'].append({
                                'module': node.module,
                                'name': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
        
        except SyntaxError:
            pass
        
        return analysis
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _ast_to_string(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        else:
            return str(node)
    
    async def _find_function(self, function_name: str) -> str:
        matches = []
        
        for file_path, analysis in self.code_index.items():
            for func in analysis.get('functions', []):
                if function_name.lower() in func['name'].lower():
                    matches.append({
                        'file': file_path,
                        'function': func['name'],
                        'line': func['line'],
                        'args': func['args']
                    })
        
        if not matches:
            return f"No functions found matching: {function_name}"
        
        result = f"Found {len(matches)} functions matching '{function_name}':\n\n"
        for match in matches:
            result += f"- {match['function']}({', '.join(match['args'])}) at {match['file']}:{match['line']}\n"
        
        return result
    
    async def _analyze_dependencies(self, file_path: str) -> str:
        if file_path not in self.code_index:
            return f"File not indexed: {file_path}"
        
        analysis = self.code_index[file_path]
        imports = analysis.get('imports', [])
        
        if not imports:
            return f"No imports found in {file_path}"
        
        dependency_analysis = {
            'stdlib': [],
            'third_party': [],
            'local': []
        }
        
        stdlib_modules = {'os', 'sys', 'json', 'ast', 'pathlib', 'typing', 'asyncio', 'collections'}
        
        for imp in imports:
            module = imp['module']
            if module in stdlib_modules or (module and module.split('.')[0] in stdlib_modules):
                dependency_analysis['stdlib'].append(module)
            elif module and ('.' not in module or not module.startswith('.')):
                dependency_analysis['third_party'].append(module)
            else:
                dependency_analysis['local'].append(module)
        
        result = f"Dependency analysis for {file_path}:\n\n"
        result += f"Standard Library: {len(dependency_analysis['stdlib'])} modules\n"
        result += f"Third Party: {len(dependency_analysis['third_party'])} packages\n"
        result += f"Local Imports: {len(dependency_analysis['local'])} modules\n\n"
        
        for category, deps in dependency_analysis.items():
            if deps:
                result += f"{category.title()} dependencies:\n"
                for dep in sorted(set(deps)):
                    result += f"  - {dep}\n"
                result += "\n"
        
        return result
    
    async def _general_analysis(self, task: str) -> str:
        if not self.code_index:
            return "No codebase has been indexed yet. Please index a directory first."
        
        context = f"Current code index contains {len(self.code_index)} files with analysis data."
        
        prompt = f"""Analyze the following request about the codebase:

Request: {task}

Context: {context}

Based on the indexed codebase, provide relevant analysis and insights."""
        
        return await self.generate_response(prompt)
    
    def _detect_language(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
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
    
    def _get_language_stats(self) -> str:
        languages = {}
        for file_path in self.code_index:
            lang = self._detect_language(file_path)
            languages[lang] = languages.get(lang, 0) + 1
        
        return ', '.join([f"{lang}: {count}" for lang, count in languages.items()])