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

class SupervisorAgent(BaseAgent):
    def __init__(self, name: str, config):
        super().__init__(name, config)
        self.agents = {}
        self.workflow = None
        self._build_workflow()
    
    def _default_system_prompt(self) -> str:
        return """You are a supervisor agent that coordinates multiple specialized agents to accomplish complex coding tasks. Your role is to:

1. Analyze incoming requests and break them down into subtasks
2. Route tasks to the appropriate specialized agents:
   - WebSearchAgent: Finding information, documentation, examples
   - CodeReviewAgent: Analyzing code quality, bugs, security issues
   - CodeAnalyzerAgent: Understanding codebase structure, dependencies, metrics
   - FileOperationsAgent: Reading, writing, editing files directly
   - TestGeneratorAgent: Creating unit tests, integration tests, test data
   - RefactoringAgent: Code optimization, restructuring, pattern application
   - GitAgent: Version control operations, commit messages, workflows
   - DocumentationAgent: Creating docs, README files, API references
3. Coordinate the workflow between agents
4. Synthesize results from multiple agents into coherent responses
5. Manage the overall task progress and ensure completion

You should:
- Determine which agents are needed for each task
- Provide clear instructions to each agent
- Handle agent failures gracefully
- Combine results meaningfully
- Track task completion status"""
    
    async def _build_workflow(self):
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
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        # All agents flow to synthesize (except direct end)
        workflow.add_edge("websearch", "synthesize")
        workflow.add_edge("code_review", "synthesize")
        workflow.add_edge("code_analyzer", "synthesize")
        workflow.add_edge("file_operations", "synthesize")
        workflow.add_edge("test_generator", "synthesize")
        workflow.add_edge("refactoring", "synthesize")
        workflow.add_edge("git", "synthesize")
        workflow.add_edge("documentation", "synthesize")
        workflow.add_edge("synthesize", END)
        
        workflow.set_entry_point("supervisor")
        
        self.workflow = workflow.compile()
    
    async def process(self, state: AgentState) -> AgentState:
        if not self.workflow:
            return self.add_message(state, "assistant", "Workflow not initialized")
        
        try:
            result = await self.workflow.ainvoke(state)
            return result
        except Exception as e:
            return self.add_message(state, "assistant", f"Workflow execution failed: {str(e)}")
    
    async def _supervisor_node(self, state: AgentState) -> AgentState:
        if not state.current_task:
            state.next_agent = "end"
            return self.add_message(state, "supervisor", "No task provided")
        
        task_analysis = await self._analyze_task(state.current_task)
        
        state.context["task_analysis"] = task_analysis
        state.context["agents_used"] = []
        
        next_agent = self._determine_primary_agent(task_analysis)
        state.next_agent = next_agent
        
        return self.add_message(
            state,
            "supervisor", 
            f"Analyzing task: {state.current_task}\nRouting to: {next_agent}",
            metadata={"task_analysis": task_analysis}
        )
    
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        prompt = f"""Analyze this task and determine what type of work is needed:

Task: {task}

Classify this task and respond with a JSON object containing:
{{
    "task_type": "websearch|code_review|code_analysis|file_operations|test_generation|refactoring|git|documentation|multi_agent",
    "complexity": "low|medium|high", 
    "requires_search": true/false,
    "requires_code_review": true/false,
    "requires_code_analysis": true/false,
    "requires_file_operations": true/false,
    "requires_test_generation": true/false,
    "requires_refactoring": true/false,
    "requires_git": true/false,
    "requires_documentation": true/false,
    "primary_focus": "description of main objective",
    "subtasks": ["list", "of", "subtasks"]
}}

Consider:
- Does this need web research or documentation lookup?
- Does this involve reviewing or analyzing existing code?  
- Does this need codebase understanding or indexing?
- Does this need reading, writing, or editing files?
- Does this need creating or generating tests?
- Does this need code refactoring or optimization?
- Does this involve Git operations or version control?
- Does this need documentation creation or updates?
- Is this a complex task requiring multiple agents?"""
        
        response = await self.generate_response(prompt)
        
        try:
            import json
            return json.loads(response)
        except:
            return {
                "task_type": "multi_agent",
                "complexity": "medium",
                "requires_search": "search" in task.lower(),
                "requires_code_review": any(word in task.lower() for word in ["review", "bug", "issue", "quality"]),
                "requires_code_analysis": any(word in task.lower() for word in ["analyze", "structure", "dependency", "index"]),
                "primary_focus": task,
                "subtasks": [task]
            }
    
    def _determine_primary_agent(self, analysis: Dict[str, Any]) -> str:
        task_type = analysis.get("task_type", "multi_agent")
        
        # Direct task type routing
        if task_type == "websearch" or analysis.get("requires_search", False):
            return "websearch"
        elif task_type == "code_review" or analysis.get("requires_code_review", False):
            return "code_review"
        elif task_type == "code_analysis" or analysis.get("requires_code_analysis", False):
            return "code_analyzer"
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
        else:
            return "synthesize"
    
    def _route_task(self, state: AgentState) -> str:
        return state.next_agent or "end"
    
    async def _websearch_node(self, state: AgentState) -> AgentState:
        if "websearch" not in self.agents:
            self.agents["websearch"] = WebSearchAgent("websearch", self.config)
        
        agent = self.agents["websearch"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("websearch")
        
        return state
    
    async def _code_review_node(self, state: AgentState) -> AgentState:
        if "code_review" not in self.agents:
            self.agents["code_review"] = CodeReviewAgent("code_review", self.config)
        
        agent = self.agents["code_review"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("code_review")
        
        return state
    
    async def _code_analyzer_node(self, state: AgentState) -> AgentState:
        if "code_analyzer" not in self.agents:
            self.agents["code_analyzer"] = CodeAnalyzerAgent("code_analyzer", self.config)
        
        agent = self.agents["code_analyzer"]
        
        async with agent:
            result_state = await agent.process(state)
        
        state.messages.extend(result_state.messages[len(state.messages):])
        state.context["agents_used"].append("code_analyzer")
        
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
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
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
        
        return self.add_message(
            state,
            "supervisor",
            synthesis,
            metadata={
                "synthesis": True,
                "agents_used": agents_used,
                "task_completed": True
            }
        )