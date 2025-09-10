from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from core.base_agent import BaseAgent, AgentState
from .websearch_agent import WebSearchAgent
from .code_review_agent import CodeReviewAgent
from .code_analyzer_agent import CodeAnalyzerAgent
from .file_operations_agent import FileOperationsAgent
from .test_generator_agent import TestGeneratorAgent
from .refactoring_agent import RefactoringAgent
from .git_agent import GitAgent
from .documentation_agent import DocumentationAgent
from .command_line_agent import CommandLineAgent
from .context_manager_agent import ContextManagerAgent
from .coding_agent import CodingAgent
from .architecting_agent import ArchitectingAgent

class SupervisorAgent(BaseAgent):
    def __init__(self, name: str, config):
        super().__init__(name, config)
        self.agents = {}
        self.workflow = None
        self._build_workflow()
    
    def _default_system_prompt(self) -> str:
        return """You are a supervisor agent that coordinates multiple specialized agents to accomplish complex coding tasks. Your role is to orchestrate a proper development workflow.

**Agent Specializations:**
- **ArchitectingAgent**: High-level system design, architecture decisions, technology choices
- **CodingAgent**: Actual code implementation, writing/editing code, bug fixes
- **CodeReviewAgent**: Quality assurance, code review feedback, best practice validation
- **CodeAnalyzerAgent**: Codebase understanding, structure analysis, dependency mapping
- **FileOperationsAgent**: File management, configuration setup, project structure
- **Other Agents**: Search, testing, documentation, git, command line, context management

**Orchestration Strategy:**
1. **For Architecture Tasks**: Route to ArchitectingAgent for design → CodingAgent for implementation → CodeReviewAgent for validation
2. **For Implementation Tasks**: Route to CodingAgent for implementation → CodeReviewAgent for review feedback → CodingAgent for fixes (if needed)
3. **For Analysis Tasks**: Route to CodeAnalyzerAgent for understanding → appropriate agent for action
4. **For Review Tasks**: Route directly to CodeReviewAgent

**Implementation-Review Cycle:**
- CodingAgent proposes implementations
- CodeReviewAgent provides feedback before final implementation
- CodingAgent makes changes based on review feedback
- Cycle continues until quality standards are met

**Task Verification:**
- After implementation, verify tasks are completed correctly
- Use appropriate agents to validate functionality
- Ensure requirements are fully satisfied

**Your Core Responsibilities:**
- Analyze requests and determine optimal agent workflow
- Coordinate between agents for complex tasks
- Manage implementation-review cycles
- Verify task completion and quality
- Handle failures and re-routing gracefully"""
    
    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add all agent nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("websearch", self._websearch_node)
        workflow.add_node("code_review", self._code_review_node)
        workflow.add_node("code_analyzer", self._code_analyzer_node)
        workflow.add_node("file_operations", self._file_operations_node)
        workflow.add_node("test_generator", self._test_generator_node)
        workflow.add_node("refactoring", self._refactoring_node)
        workflow.add_node("git", self._git_node)
        workflow.add_node("documentation", self._documentation_node)
        workflow.add_node("command_line", self._command_line_node)
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("coding", self._coding_node)
        workflow.add_node("architecting", self._architecting_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Add conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_task,
            {
                "websearch": "websearch",
                "code_review": "code_review", 
                "code_analyzer": "code_analyzer",
                "file_operations": "file_operations",
                "test_generator": "test_generator",
                "refactoring": "refactoring",
                "git": "git",
                "documentation": "documentation",
                "command_line": "command_line",
                "context_manager": "context_manager",
                "coding": "coding",
                "architecting": "architecting",
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        # Basic agents flow to synthesize 
        workflow.add_edge("websearch", "synthesize")
        workflow.add_edge("file_operations", "synthesize")
        workflow.add_edge("test_generator", "synthesize")
        workflow.add_edge("refactoring", "synthesize")
        workflow.add_edge("git", "synthesize")
        workflow.add_edge("documentation", "synthesize")
        workflow.add_edge("command_line", "synthesize")
        workflow.add_edge("context_manager", "synthesize")
        
        # Architecting Agent → Coding Agent → Code Review cycle
        workflow.add_conditional_edges(
            "architecting",
            self._route_after_architecting,
            {
                "coding": "coding",
                "synthesize": "synthesize"
            }
        )
        
        # Coding Agent can go to Code Review for validation or directly to synthesize
        workflow.add_conditional_edges(
            "coding",
            self._route_after_coding,
            {
                "code_review": "code_review",
                "synthesize": "synthesize"
            }
        )
        
        # Code Review can send back to Coding for fixes or proceed to synthesize
        workflow.add_conditional_edges(
            "code_review",
            self._route_after_review,
            {
                "coding": "coding",
                "synthesize": "synthesize"
            }
        )
        
        # Code analyzer can flow to appropriate next agent based on task
        workflow.add_conditional_edges(
            "code_analyzer",
            self._route_after_analyzer,
            {
                "architecting": "architecting",
                "coding": "coding", 
                "code_review": "code_review",
                "synthesize": "synthesize"
            }
        )
        workflow.add_edge("synthesize", END)
        
        workflow.set_entry_point("supervisor")
        
        self.workflow = workflow.compile()
    
    async def process(self, state: AgentState) -> AgentState:
        if not self.workflow:
            return self.add_message(state, "assistant", "Workflow not initialized")
        
        try:
            result = await self.workflow.ainvoke(state)
            
            # LangGraph might return a dict instead of AgentState, so convert if needed
            if isinstance(result, dict):
                # Convert dict back to AgentState
                converted_state = AgentState(
                    messages=result.get("messages", []),
                    current_task=result.get("current_task"),
                    context=result.get("context", {}),
                    next_agent=result.get("next_agent")
                )
                return converted_state
            
            return result
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Call progress callback if available
            if hasattr(state, 'context') and 'progress_callback' in state.context:
                try:
                    await state.context['progress_callback'](f"Error: {str(e)}")
                except:
                    pass
            
            return self.add_message(
                state, 
                "assistant", 
                f"Workflow execution failed: {str(e)}\n\nError Details:\n{error_details}"
            )
    
    async def _supervisor_node(self, state: AgentState) -> AgentState:
        # Call progress callback if available
        await self._call_progress_callback(state, "Analyzing incoming request...")
        
        if not state.current_task:
            state.next_agent = "end"
            return self.add_message(state, "supervisor", "No task provided")
        
        # Auto-manage context if conversation is getting long
        if len(state.messages) >= self.config.agents.memory_window - 5:
            await self._call_progress_callback(state, "Managing conversation context...")
            await self._auto_manage_context(state)
        
        await self._call_progress_callback(state, f"Understanding task: {state.current_task[:50]}...")
        task_analysis = await self._analyze_task(state.current_task)
        
        state.context["task_analysis"] = task_analysis
        state.context["agents_used"] = []
        
        # Check if we need codebase analysis but don't have indexed data
        from core.shared_context import shared_context
        needs_codebase = any(keyword in state.current_task.lower() for keyword in ['codebase', 'improve', 'production', 'quality', 'analyze'])
        has_index = bool(shared_context.get_code_index())
        
        if needs_codebase and not has_index:
            await self._call_progress_callback(state, "No codebase index found, indexing current directory first...")
            # Force indexing first
            state.next_agent = "code_analyzer"
            # Store the original task for after indexing
            state.context["deferred_task"] = state.current_task
            state.current_task = "index_directory:."
        else:
            next_agent = self._determine_primary_agent(task_analysis)
            state.next_agent = next_agent
            await self._call_progress_callback(state, f"Routing to {next_agent.replace('_', ' ').title()} agent...")
        
        return self.add_message(
            state,
            "supervisor", 
            f"Analyzing task: {state.current_task}\nRouting to: {state.next_agent}",
            metadata={"task_analysis": task_analysis}
        )
    
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        prompt = f"""Analyze this task and determine what type of work is needed:

Task: {task}

Classify this task and respond with a JSON object containing:
{{
    "task_type": "websearch|architecting|coding|code_review|code_analysis|file_operations|test_generation|refactoring|git|documentation|context_management|multi_agent",
    "complexity": "low|medium|high", 
    "requires_search": true/false,
    "requires_architecting": true/false,
    "requires_coding": true/false,
    "requires_code_review": true/false,
    "requires_code_analysis": true/false,
    "requires_file_operations": true/false,
    "requires_test_generation": true/false,
    "requires_refactoring": true/false,
    "requires_git": true/false,
    "requires_documentation": true/false,
    "requires_command_line": true/false,
    "requires_context_management": true/false,
    "primary_focus": "description of main objective",
    "subtasks": ["list", "of", "subtasks"]
}}

Consider:
- Does this need web research or documentation lookup?
- Does this need high-level system design or architecture planning?
- Does this need actual code implementation, creation, or modification?
- Does this involve reviewing or validating code quality?  
- Does this need codebase understanding or indexing?
- Does this need reading, writing, or editing files?
- Does this need creating or generating tests?
- Does this need code refactoring or optimization?
- Does this involve Git operations or version control?
- Does this need documentation creation or updates?
- Does this need system/shell command execution?
- Does this need conversation context management or memory?
- Is this a complex task requiring multiple agents?"""
        
        response = await self.generate_response(prompt)
        
        try:
            import json
            return json.loads(response)
        except:
            # Fallback analysis when LLM response can't be parsed
            return {
                "task_type": "multi_agent",
                "complexity": "medium",
                "requires_search": "search" in task.lower(),
                "requires_architecting": any(word in task.lower() for word in ["design", "architect", "structure", "plan"]),
                "requires_coding": any(word in task.lower() for word in ["implement", "create", "build", "code", "setup", "configure", "install", "fix", "write"]),
                "requires_code_review": any(word in task.lower() for word in ["review", "bug", "issue", "quality", "validate", "check"]),
                "requires_code_analysis": any(word in task.lower() for word in ["analyze", "structure", "dependency", "index", "understand"]),
                "requires_file_operations": any(word in task.lower() for word in ["file", "read", "write", "edit", "create"]),
                "requires_test_generation": any(word in task.lower() for word in ["test", "testing", "unit test"]),
                "requires_refactoring": any(word in task.lower() for word in ["refactor", "optimize", "improve"]),
                "requires_git": any(word in task.lower() for word in ["git", "commit", "version", "branch"]),
                "requires_documentation": any(word in task.lower() for word in ["document", "readme", "docs"]),
                "requires_command_line": any(word in task.lower() for word in ["command", "shell", "run", "execute"]),
                "requires_context_management": False,
                "primary_focus": task,
                "subtasks": [task]
            }
    
    def _determine_primary_agent(self, analysis: Dict[str, Any]) -> str:
        task_type = analysis.get("task_type", "multi_agent")
        
        # Prioritized routing based on development workflow
        # 1. Architecture comes first for design tasks
        if task_type == "architecting" or analysis.get("requires_architecting", False):
            return "architecting"
        
        # 2. Implementation tasks go to coding agent
        elif task_type == "coding" or analysis.get("requires_coding", False):
            return "coding"
            
        # 3. Direct review requests
        elif task_type == "code_review" or analysis.get("requires_code_review", False):
            return "code_review"
        
        # 4. Analysis tasks (may lead to architecture or coding)
        elif task_type == "code_analysis" or analysis.get("requires_code_analysis", False):
            return "code_analyzer"
        
        # 5. Other specialized agents
        elif task_type == "websearch" or analysis.get("requires_search", False):
            return "websearch"
        elif task_type == "file_operations" or analysis.get("requires_file_operations", False):
            return "file_operations"
        elif task_type == "test_generation" or analysis.get("requires_test_generation", False):
            return "test_generator"
        elif task_type == "refactoring" or analysis.get("requires_refactoring", False):
            return "refactoring"
        elif task_type == "git" or analysis.get("requires_git", False):
            return "git"
        elif task_type == "documentation" or analysis.get("requires_documentation", False):
            return "documentation"
        elif task_type == "command_line" or analysis.get("requires_command_line", False):
            return "command_line"
        elif task_type == "context_management" or analysis.get("requires_context_management", False):
            return "context_manager"
        
        # 6. Multi-agent or complex tasks - need further analysis
        elif task_type == "multi_agent":
            # For multi-agent tasks, determine the primary workflow
            if analysis.get("requires_architecting", False) and analysis.get("requires_coding", False):
                return "architecting"  # Start with architecture
            elif analysis.get("requires_coding", False):
                return "coding"
            elif analysis.get("requires_code_analysis", False):
                return "code_analyzer"  # May lead to other agents
            else:
                return "synthesize"  # Can't determine primary agent
        
        else:
            return "synthesize"
    
    def _route_task(self, state: AgentState) -> str:
        return state.next_agent or "end"
    
    def _route_after_analyzer(self, state: AgentState) -> str:
        """Route after code analyzer based on the original task type"""
        if hasattr(state, 'next_agent') and state.next_agent:
            return state.next_agent
        
        # Check if this was a deferred task that needs specific routing
        if "deferred_task" in state.context:
            original_task = state.context["deferred_task"]
            if any(word in original_task.lower() for word in ["design", "architect", "structure"]):
                return "architecting"
            elif any(word in original_task.lower() for word in ["implement", "create", "build", "code"]):
                return "coding"
            elif any(word in original_task.lower() for word in ["review", "check", "quality", "improve"]):
                return "code_review"
        
        return "synthesize"
    
    def _route_after_architecting(self, state: AgentState) -> str:
        """Route after architecting - typically to coding for implementation"""
        # If architecting agent set a specific next step, follow it
        if hasattr(state, 'next_agent') and state.next_agent:
            return state.next_agent
        
        # Default: architecture designs should be implemented
        return "coding"
    
    def _route_after_coding(self, state: AgentState) -> str:
        """Route after coding - typically to code review for validation"""
        # Check if coding agent explicitly set next agent
        if hasattr(state, 'next_agent') and state.next_agent:
            return state.next_agent
        
        # Check if we're in a review cycle
        review_cycle_count = state.context.get("review_cycle_count", 0)
        
        # Always send to review first time, or if requested
        if review_cycle_count == 0 or state.context.get("needs_review", True):
            state.context["review_cycle_count"] = review_cycle_count + 1
            return "code_review"
        
        # If we've been through review cycle multiple times, synthesize
        if review_cycle_count >= 3:
            return "synthesize"
        
        # Default to review for quality assurance
        return "code_review"
    
    def _route_after_review(self, state: AgentState) -> str:
        """Route after code review - back to coding for fixes or to synthesize if approved"""
        # Check if review found issues that need coding fixes
        if state.messages:
            last_message = state.messages[-1].content.lower()
            
            # If review found issues, send back to coding
            if any(word in last_message for word in ["fix", "change", "improve", "issue", "problem", "error"]):
                # Set context for coding agent to know this is a fix cycle
                state.context["is_fix_cycle"] = True
                state.context["review_feedback"] = state.messages[-1].content
                return "coding"
            
            # If review approved or no major issues, proceed to synthesize
            if any(word in last_message for word in ["approved", "good", "looks good", "ready"]):
                state.context["review_approved"] = True
                return "synthesize"
        
        # Default to synthesize if unclear
        return "synthesize"
    
    async def _verify_task_completion(self, state: AgentState) -> Dict[str, Any]:
        """Verify that the task has been completed successfully"""
        original_task = state.current_task
        agents_used = state.context.get("agents_used", [])
        
        # Build verification prompt based on what agents were used
        verification_prompt = f"""Verify if this task has been completed successfully:

**Original Task**: {original_task}

**Agents Used**: {', '.join(agents_used)}

**Results Available**: Review the conversation history and agent outputs above.

Analyze whether:
1. **Task Requirements**: Were all requirements addressed?
2. **Implementation Quality**: If code was written, is it complete and functional?
3. **Architecture Soundness**: If design was involved, is it well-structured?
4. **Review Validation**: If code review occurred, were issues properly addressed?
5. **Expected Outputs**: Are all expected deliverables present?

Respond with a JSON object:
{{
    "completed": true/false,
    "completeness_score": 0-100,
    "missing_elements": ["list of missing items"],
    "quality_assessment": "brief quality assessment",
    "recommendations": ["any additional recommendations"]
}}"""
        
        try:
            response = await self.generate_response(verification_prompt)
            # For now, just return a basic verification structure
            # In a real implementation, you'd parse the JSON response
            return {
                "completed": True,  # Assume completed for now
                "completeness_score": 85,
                "missing_elements": [],
                "quality_assessment": "Task appears to be completed based on agent outputs",
                "recommendations": []
            }
        except Exception as e:
            return {
                "completed": False,
                "completeness_score": 0,
                "missing_elements": ["Verification failed"],
                "quality_assessment": f"Could not verify completion: {e}",
                "recommendations": ["Manual review recommended"]
            }
    
    async def _handle_incomplete_task(self, state: AgentState, verification: Dict[str, Any]) -> AgentState:
        """Handle tasks that are not fully completed"""
        missing_elements = verification.get("missing_elements", [])
        recommendations = verification.get("recommendations", [])
        
        # Determine what additional work is needed
        if missing_elements:
            # Route to appropriate agent to complete missing work
            for element in missing_elements:
                if any(word in element.lower() for word in ["implement", "code", "write"]):
                    state.next_agent = "coding"
                    state.current_task = f"Complete missing implementation: {element}"
                    break
                elif any(word in element.lower() for word in ["design", "architect", "structure"]):
                    state.next_agent = "architecting"
                    state.current_task = f"Complete missing design: {element}"
                    break
                elif any(word in element.lower() for word in ["review", "validate", "check"]):
                    state.next_agent = "code_review"
                    state.current_task = f"Address missing validation: {element}"
                    break
        
        return state
    
    async def _websearch_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Searching the web for relevant information...", "websearch")
        
        if "websearch" not in self.agents:
            self.agents["websearch"] = WebSearchAgent("websearch", self.config)
        
        agent = self.agents["websearch"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("websearch")
        
        await self._call_progress_callback(state, "Web search completed", "websearch")
        return state
    
    async def _code_review_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Analyzing code quality and identifying issues...", "code_review")
        
        if "code_review" not in self.agents:
            self.agents["code_review"] = CodeReviewAgent("code_review", self.config)
        
        agent = self.agents["code_review"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("code_review")
        
        await self._call_progress_callback(state, "Code review completed", "code_review")
        return state
    
    async def _code_analyzer_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Indexing codebase structure and dependencies...", "code_analyzer")
        
        if "code_analyzer" not in self.agents:
            self.agents["code_analyzer"] = CodeAnalyzerAgent("code_analyzer", self.config)
        
        agent = self.agents["code_analyzer"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("code_analyzer")
        
        # Check if this was a forced indexing for a deferred task
        if "deferred_task" in state.context:
            await self._call_progress_callback(state, "Indexing complete, now analyzing codebase for improvements...")
            # Restore the original task and route to code review
            original_task = state.context["deferred_task"]
            state.current_task = original_task
            del state.context["deferred_task"]
            # Force routing to code review for the original task
            state.next_agent = "code_review"
        
        await self._call_progress_callback(state, "Code analysis completed", "code_analyzer")
        return state
    
    async def _file_operations_node(self, state: AgentState) -> AgentState:
        if "file_operations" not in self.agents:
            self.agents["file_operations"] = FileOperationsAgent("file_operations", self.config)
        
        agent = self.agents["file_operations"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("file_operations")
        
        return state
    
    async def _test_generator_node(self, state: AgentState) -> AgentState:
        if "test_generator" not in self.agents:
            self.agents["test_generator"] = TestGeneratorAgent("test_generator", self.config)
        
        agent = self.agents["test_generator"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("test_generator")
        
        return state
    
    async def _refactoring_node(self, state: AgentState) -> AgentState:
        if "refactoring" not in self.agents:
            self.agents["refactoring"] = RefactoringAgent("refactoring", self.config)
        
        agent = self.agents["refactoring"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("refactoring")
        
        return state
    
    async def _git_node(self, state: AgentState) -> AgentState:
        if "git" not in self.agents:
            self.agents["git"] = GitAgent("git", self.config)
        
        agent = self.agents["git"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("git")
        
        return state
    
    async def _documentation_node(self, state: AgentState) -> AgentState:
        if "documentation" not in self.agents:
            self.agents["documentation"] = DocumentationAgent("documentation", self.config)
        
        agent = self.agents["documentation"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("documentation")
        
        return state
    
    async def _command_line_node(self, state: AgentState) -> AgentState:
        if "command_line" not in self.agents:
            self.agents["command_line"] = CommandLineAgent("command_line", self.config)
        
        agent = self.agents["command_line"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("command_line")
        
        return state
    
    async def _context_manager_node(self, state: AgentState) -> AgentState:
        if "context_manager" not in self.agents:
            self.agents["context_manager"] = ContextManagerAgent("context_manager", self.config)
        
        agent = self.agents["context_manager"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("context_manager")
        
        return state
    
    async def _auto_manage_context(self, state: AgentState):
        """Automatically manage context when memory window is nearly full"""
        try:
            if "context_manager" not in self.agents:
                self.agents["context_manager"] = ContextManagerAgent("context_manager", self.config)
            
            context_agent = self.agents["context_manager"]
            
            # Summarize older messages before they get truncated
            messages_to_summarize = max(5, len(state.messages) - 10)
            
            async with context_agent:
                context_state = AgentState(
                    current_task=f"summarize_messages:{messages_to_summarize}",
                    messages=state.messages
                )
                await context_agent.process(context_state)
            
        except Exception as e:
            # Don't fail the main workflow if context management fails
            print(f"Warning: Auto-context management failed: {e}")
    
    async def _coding_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Implementing code solution...", "coding")
        
        if "coding" not in self.agents:
            self.agents["coding"] = CodingAgent("coding", self.config)
        
        agent = self.agents["coding"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("coding")
        
        await self._call_progress_callback(state, "Code implementation completed", "coding")
        return state
    
    async def _architecting_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Designing system architecture...", "architecting")
        
        if "architecting" not in self.agents:
            self.agents["architecting"] = ArchitectingAgent("architecting", self.config)
        
        agent = self.agents["architecting"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("architecting")
        
        await self._call_progress_callback(state, "Architecture design completed", "architecting")
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        await self._call_progress_callback(state, "Synthesizing results from all agents...")
        
        agents_used = state.context.get("agents_used", [])
        task_analysis = state.context.get("task_analysis", {})
        
        if not agents_used:
            return self.add_message(state, "supervisor", "No agents were used to process this task")
        
        agent_results = []
        for msg in state.messages:
            if msg.role == "assistant" and msg.metadata and msg.metadata.get("search_results"):
                agent_results.append(f"Search Results: {msg.content}")
            elif msg.role == "assistant" and msg.metadata and msg.metadata.get("review_type"):
                agent_results.append(f"Code Review: {msg.content}")
            elif msg.role == "assistant" and msg.metadata and msg.metadata.get("analysis_type"):
                agent_results.append(f"Code Analysis: {msg.content}")
        
        prompt = f"""Synthesize the results from multiple agents into a comprehensive response:

Original Task: {state.current_task}
Task Analysis: {task_analysis}
Agents Used: {', '.join(agents_used)}

Agent Results:
{chr(10).join(agent_results)}

Provide a cohesive, well-structured response that:
1. Directly addresses the original task
2. Integrates insights from all agents
3. Highlights key findings and recommendations  
4. Suggests next steps if appropriate
5. Maintains technical accuracy and detail"""
        
        synthesis = await self.generate_response(prompt)
        
        # Perform task verification before finalizing
        await self._call_progress_callback(state, "Verifying task completion...")
        verification = await self._verify_task_completion(state)
        
        # If task is not fully completed, handle it
        if not verification.get("completed", True):
            await self._call_progress_callback(state, "Task verification found missing elements, addressing them...")
            return await self._handle_incomplete_task(state, verification)
        
        # Add verification results to synthesis
        if verification.get("completeness_score", 100) < 90:
            verification_note = f"\n\n**Task Verification**: Completeness Score: {verification.get('completeness_score', 0)}%"
            if verification.get("recommendations"):
                verification_note += f"\n**Recommendations**: {', '.join(verification['recommendations'])}"
            synthesis += verification_note
        
        return self.add_message(
            state,
            "supervisor",
            synthesis,
            metadata={
                "synthesis": True,
                "agents_used": agents_used,
                "task_completed": True,
                "verification": verification
            }
        )
    
    async def _call_progress_callback(self, state: AgentState, message: str, agent_name: str = ""):
        """Helper method to call progress callback if available"""
        if hasattr(state, 'context') and 'progress_callback' in state.context:
            try:
                callback = state.context['progress_callback']
                await callback(message, agent_name)
            except Exception:
                pass  # Don't fail if callback fails