from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState
from core.shared_context import shared_context

class CodingAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are an expert software engineer specializing in implementing code solutions. Your role is to:

1. **Implement actual code** based on specifications and requirements
2. **Write, edit, and refactor code** across multiple programming languages
3. **Follow best practices** for the detected programming language
4. **Create modular, maintainable code** with proper structure
5. **Handle edge cases and error conditions** appropriately

**Available Tools:**
- Use built-in read_file() method to examine existing code
- Use execute_command() method for language-specific operations (compile, run, test)
- Access indexed codebase data via shared_context for understanding project structure
- Coordinate with FileOperationsAgent for file management

**Language Support:**
- **Python**: Write Pythonic code, use type hints, follow PEP 8
- **JavaScript/TypeScript**: Modern ES6+, proper async/await, TypeScript types
- **Java**: Object-oriented design, proper exception handling, Maven/Gradle aware
- **Go**: Idiomatic Go, proper error handling, module-aware
- **Rust**: Memory safety, proper ownership, Cargo-aware
- **Other languages**: Detect and adapt to language conventions

**Your Implementation Process:**
1. **Analyze** the request and understand requirements
2. **Examine** existing codebase structure and patterns
3. **Design** the solution approach
4. **Implement** the code following language best practices
5. **Test** the implementation (create basic tests if needed)
6. **Document** the code appropriately

**Code Quality Standards:**
- Write clean, readable code with appropriate comments
- Handle errors gracefully with proper exception handling
- Follow language-specific naming conventions and patterns
- Ensure code is testable and modular
- Consider performance and security implications

You implement solutions, you don't just provide suggestions."""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No coding task specified")
        
        task = state.current_task
        
        try:
            if task.startswith("implement:"):
                result = await self._implement_feature(task.replace("implement:", "").strip())
            elif task.startswith("refactor:"):
                result = await self._refactor_code(task.replace("refactor:", "").strip())
            elif task.startswith("fix:"):
                result = await self._fix_bug(task.replace("fix:", "").strip())
            elif task.startswith("create:"):
                result = await self._create_new_code(task.replace("create:", "").strip())
            elif task.startswith("optimize:"):
                result = await self._optimize_code(task.replace("optimize:", "").strip())
            else:
                result = await self._analyze_and_implement(task)
            
            return self.add_message(
                state,
                "assistant", 
                result,
                metadata={"operation_type": "coding", "language": await self._detect_project_language()}
            )
        
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"Coding operation failed: {str(e)}"
            )
    
    async def _detect_project_language(self) -> str:
        """Detect the primary programming language of the project"""
        try:
            # Check common files to detect language
            file_check = await self.execute_command("find . -maxdepth 2 -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.java' -o -name '*.go' -o -name '*.rs' | head -10")
            
            if ".py" in file_check:
                return "python"
            elif ".ts" in file_check:
                return "typescript"
            elif ".js" in file_check:
                return "javascript" 
            elif ".java" in file_check:
                return "java"
            elif ".go" in file_check:
                return "go"
            elif ".rs" in file_check:
                return "rust"
            else:
                return "unknown"
        except:
            return "unknown"
    
    async def _implement_feature(self, specification: str) -> str:
        """Implement a new feature based on specification"""
        results = []
        results.append("# 🔧 Feature Implementation")
        results.append(f"**Specification**: {specification}")
        
        # 1. Analyze project structure
        project_analysis = await self._analyze_project_structure()
        results.append("## 📁 Project Analysis")
        results.append(project_analysis)
        
        # 2. Design the implementation approach
        design = await self._design_implementation(specification)
        results.append("## 🎯 Implementation Design")
        results.append(design)
        
        # 3. Implement the actual code
        implementation = await self._write_implementation_code(specification, design)
        results.append("## 💻 Code Implementation")
        results.append(implementation)
        
        # 4. Create basic tests
        testing = await self._create_implementation_tests(specification)
        results.append("## 🧪 Testing")
        results.append(testing)
        
        return "\n\n".join(results)
    
    async def _analyze_project_structure(self) -> str:
        """Analyze the current project structure"""
        try:
            # Get directory structure
            structure = await self.execute_command("find . -type f -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.java' -o -name '*.go' -o -name '*.rs' | head -20")
            
            # Get language-specific info
            language = await self._detect_project_language()
            
            # Check for existing patterns
            patterns_info = await self._analyze_code_patterns()
            
            return f"""**Primary Language**: {language}
**Project Structure**:
```
{structure}
```
**Detected Patterns**: {patterns_info}"""
        
        except Exception as e:
            return f"Could not analyze project structure: {e}"
    
    async def _analyze_code_patterns(self) -> str:
        """Analyze existing code patterns to maintain consistency"""
        try:
            language = await self._detect_project_language()
            
            if language == "python":
                # Check for Python patterns
                patterns = await self.execute_command("grep -r 'class \\|def \\|import ' . --include='*.py' | head -10")
                return f"Python patterns detected: classes, functions, imports found"
            
            elif language in ["javascript", "typescript"]:
                # Check for JS/TS patterns
                patterns = await self.execute_command("grep -r 'function\\|class\\|export\\|import' . --include='*.js' --include='*.ts' | head -10")
                return f"JS/TS patterns detected: functions, classes, modules found"
            
            elif language == "java":
                # Check for Java patterns
                patterns = await self.execute_command("grep -r 'public class\\|public interface\\|package' . --include='*.java' | head -10")
                return f"Java patterns detected: classes, interfaces, packages found"
            
            else:
                return f"Basic patterns for {language}"
                
        except Exception as e:
            return f"Pattern analysis limited: {e}"
    
    async def _design_implementation(self, specification: str) -> str:
        """Design the implementation approach"""
        prompt = f"""Based on this specification: "{specification}"

And the current project context, design an implementation approach that includes:

1. **Architecture decisions** - How will this fit into the existing codebase?
2. **File structure** - What files need to be created or modified?
3. **Key components** - What are the main classes/functions/modules needed?
4. **Dependencies** - What external libraries or internal modules are needed?
5. **Testing strategy** - How will this be tested?

Provide a clear, actionable implementation plan."""
        
        return await self.generate_response(prompt)
    
    async def _write_implementation_code(self, specification: str, design: str) -> str:
        """Write the actual implementation code"""
        prompt = f"""Now implement the actual code for:

**Specification**: {specification}

**Design Plan**: {design}

**Requirements**:
1. Write complete, working code (not pseudocode)
2. Follow best practices for the detected language
3. Include proper error handling
4. Add appropriate comments and documentation
5. Make the code modular and testable

**Output Format**: Provide the actual code files with their file paths and complete implementation."""
        
        code_response = await self.generate_response(prompt)
        
        # Extract code blocks and write files if possible
        return await self._process_code_response(code_response)
    
    async def _process_code_response(self, code_response: str) -> str:
        """Process the code response and create actual files if appropriate"""
        results = [code_response]
        
        # This would parse code blocks and use FileOperationsAgent to create files
        # For now, just return the code response with a note about file creation
        results.append("\n📝 **Note**: Code provided above can be implemented using FileOperationsAgent coordination")
        
        return "\n".join(results)
    
    async def _create_implementation_tests(self, specification: str) -> str:
        """Create basic tests for the implementation"""
        language = await self._detect_project_language()
        
        prompt = f"""Create comprehensive tests for the implementation:

**Specification**: {specification}
**Language**: {language}

Generate appropriate test cases including:
1. **Unit tests** for individual components
2. **Integration tests** if needed
3. **Edge case tests** for error conditions
4. **Happy path tests** for normal usage

Follow testing best practices for {language}."""
        
        return await self.generate_response(prompt)
    
    async def _refactor_code(self, refactor_request: str) -> str:
        """Refactor existing code"""
        results = []
        results.append("# 🔄 Code Refactoring")
        results.append(f"**Request**: {refactor_request}")
        
        # Analyze current code
        analysis = await self._analyze_code_for_refactoring(refactor_request)
        results.append("## 📊 Code Analysis")
        results.append(analysis)
        
        # Propose refactoring
        refactoring = await self._propose_refactoring(refactor_request, analysis)
        results.append("## ✨ Refactoring Implementation")
        results.append(refactoring)
        
        return "\n\n".join(results)
    
    async def _analyze_code_for_refactoring(self, request: str) -> str:
        """Analyze code that needs refactoring"""
        prompt = f"""Analyze the current codebase for refactoring request: "{request}"

Examine the code structure, identify:
1. **Code smells** or issues to address
2. **Complexity metrics** (if applicable)
3. **Dependencies** that might be affected
4. **Potential risks** of refactoring
5. **Benefits** of the proposed changes

Provide specific analysis with code examples where possible."""
        
        return await self.generate_response(prompt)
    
    async def _propose_refactoring(self, request: str, analysis: str) -> str:
        """Propose specific refactoring changes"""
        prompt = f"""Based on the refactoring request: "{request}"
And the analysis: "{analysis}"

Provide a detailed refactoring implementation:

1. **Before/After code examples**
2. **Step-by-step changes** needed
3. **Impact assessment** on other parts of the system
4. **Testing implications** - what tests need to be updated
5. **Migration strategy** if data/API changes are involved

Make the refactoring practical and actionable."""
        
        return await self.generate_response(prompt)
    
    async def _fix_bug(self, bug_description: str) -> str:
        """Fix a reported bug"""
        results = []
        results.append("# 🐛 Bug Fix")
        results.append(f"**Bug Description**: {bug_description}")
        
        # Investigate the bug
        investigation = await self._investigate_bug(bug_description)
        results.append("## 🔍 Investigation")
        results.append(investigation)
        
        # Implement fix
        fix = await self._implement_bug_fix(bug_description, investigation)
        results.append("## 🔧 Fix Implementation")
        results.append(fix)
        
        return "\n\n".join(results)
    
    async def _investigate_bug(self, bug_description: str) -> str:
        """Investigate the bug to understand root cause"""
        prompt = f"""Investigate this bug: "{bug_description}"

Steps to analyze:
1. **Reproduce** the bug scenario
2. **Identify root cause** through code analysis
3. **Trace the issue** through the codebase
4. **Assess impact** - what other areas might be affected
5. **Propose solution approach** - how to fix it properly

Use available tools to examine relevant code files and understand the bug's context."""
        
        return await self.generate_response(prompt)
    
    async def _implement_bug_fix(self, bug_description: str, investigation: str) -> str:
        """Implement the actual bug fix"""
        prompt = f"""Implement a fix for: "{bug_description}"

Based on investigation: "{investigation}"

Provide:
1. **Exact code changes** needed (with before/after)
2. **Files to modify** with specific line changes
3. **Additional tests** to prevent regression
4. **Validation steps** to verify the fix works
5. **Side effects** to watch out for

Make the fix minimal, targeted, and robust."""
        
        return await self.generate_response(prompt)
    
    async def _create_new_code(self, creation_request: str) -> str:
        """Create new code from scratch"""
        results = []
        results.append("# ✨ New Code Creation")
        results.append(f"**Request**: {creation_request}")
        
        # Plan the creation
        planning = await self._plan_new_code(creation_request)
        results.append("## 📋 Planning")
        results.append(planning)
        
        # Create the code
        creation = await self._execute_code_creation(creation_request, planning)
        results.append("## 💻 Implementation")
        results.append(creation)
        
        return "\n\n".join(results)
    
    async def _plan_new_code(self, request: str) -> str:
        """Plan new code creation"""
        language = await self._detect_project_language()
        
        prompt = f"""Plan the creation of new code: "{request}"

For language: {language}

Planning considerations:
1. **Purpose and scope** - What exactly needs to be built
2. **Architecture design** - How it fits with existing code  
3. **Interface design** - APIs, classes, functions needed
4. **Data structures** - What data will be processed
5. **Integration points** - How it connects to existing systems
6. **Testing approach** - How to validate it works

Provide a comprehensive plan before implementation."""
        
        return await self.generate_response(prompt)
    
    async def _execute_code_creation(self, request: str, planning: str) -> str:
        """Execute the actual code creation"""
        prompt = f"""Create new code for: "{request}"

Based on planning: "{planning}"

Requirements:
1. **Complete implementation** - full working code
2. **Language best practices** - follow conventions
3. **Proper structure** - organized and modular
4. **Documentation** - appropriate comments
5. **Error handling** - robust error management
6. **Testability** - designed for easy testing

Output the complete code with file organization."""
        
        return await self.generate_response(prompt)
    
    async def _optimize_code(self, optimization_request: str) -> str:
        """Optimize existing code for performance or other metrics"""
        results = []
        results.append("# ⚡ Code Optimization")
        results.append(f"**Optimization Request**: {optimization_request}")
        
        # Analyze current performance
        analysis = await self._analyze_performance(optimization_request)
        results.append("## 📊 Performance Analysis")
        results.append(analysis)
        
        # Implement optimizations
        optimization = await self._implement_optimizations(optimization_request, analysis)
        results.append("## 🚀 Optimization Implementation")
        results.append(optimization)
        
        return "\n\n".join(results)
    
    async def _analyze_performance(self, request: str) -> str:
        """Analyze current code performance"""
        prompt = f"""Analyze performance for: "{request}"

Performance analysis should cover:
1. **Current bottlenecks** - where is time/memory being spent
2. **Algorithmic complexity** - Big O analysis where relevant  
3. **Resource usage** - memory, CPU, I/O patterns
4. **Scalability concerns** - how it performs under load
5. **Profiling data** - if available, or suggest profiling approach

Identify specific optimization opportunities."""
        
        return await self.generate_response(prompt)
    
    async def _implement_optimizations(self, request: str, analysis: str) -> str:
        """Implement performance optimizations"""
        prompt = f"""Implement optimizations for: "{request}"

Based on analysis: "{analysis}"

Optimization implementation:
1. **Specific code changes** - before/after comparisons
2. **Algorithm improvements** - better approaches if applicable
3. **Data structure optimizations** - more efficient structures
4. **Caching strategies** - where appropriate
5. **Performance validation** - how to measure improvement

Ensure optimizations maintain correctness and readability."""
        
        return await self.generate_response(prompt)
    
    async def _analyze_and_implement(self, task: str) -> str:
        """Analyze task and implement appropriate solution"""
        analysis_prompt = f"""Analyze this coding task: "{task}"

Determine:
1. **Task type** - What kind of coding work is needed?
2. **Scope and complexity** - How extensive is this task?
3. **Dependencies** - What needs to be understood first?
4. **Approach** - What's the best way to tackle this?
5. **Deliverables** - What should be the final output?

Then proceed with implementation following the determined approach."""
        
        return await self.generate_response(analysis_prompt)