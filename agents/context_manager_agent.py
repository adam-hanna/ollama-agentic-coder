import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from core.base_agent import BaseAgent, AgentState, AgentMessage

class ContextManagerAgent(BaseAgent):
    """Intelligent context management agent that handles conversation memory, 
    summarization, and semantic retrieval of relevant past context."""
    
    def __init__(self, name: str, config, system_prompt: Optional[str] = None):
        super().__init__(name, config, system_prompt)
        # Long-term context storage
        self.context_store: Dict[str, Dict[str, Any]] = {}
        # Task-based context summaries
        self.task_summaries: Dict[str, str] = {}
        # Conversation segments
        self.conversation_segments: List[Dict[str, Any]] = []
        # Current session context
        self.session_context = {
            "start_time": datetime.now(),
            "active_tasks": [],
            "key_insights": [],
            "files_referenced": set(),
            "agents_used": set()
        }
    
    def _default_system_prompt(self) -> str:
        return """You are a context management specialist responsible for maintaining intelligent conversation memory and context. Your role is to:

1. Summarize conversation history to preserve important information
2. Extract key insights, decisions, and outcomes from agent interactions
3. Identify and store relevant context for future reference
4. Retrieve contextually relevant information for current tasks
5. Maintain relationships between tasks, files, and agent actions
6. Track project progress and decision patterns

You focus on:
- Key technical decisions and their rationale
- Important code changes and their impact
- Recurring patterns or issues
- Cross-agent learnings and insights
- File and function references
- Task completion status and outcomes
- User preferences and workflow patterns

Provide concise, actionable summaries that help maintain context across long conversations."""
    
    async def process(self, state: AgentState) -> AgentState:
        """Process context management tasks"""
        task = state.current_task
        
        try:
            if task.startswith("summarize_messages:"):
                # Extract message count from task
                count_str = task.replace("summarize_messages:", "").strip()
                count = int(count_str) if count_str.isdigit() else len(state.messages)
                result = await self._summarize_messages(state.messages[-count:])
                
            elif task.startswith("store_task_result:"):
                # Store task completion result
                task_data = task.replace("store_task_result:", "").strip()
                result = await self._store_task_result(task_data, state)
                
            elif task.startswith("retrieve_context:"):
                # Retrieve relevant context for current task
                query = task.replace("retrieve_context:", "").strip()
                result = await self._retrieve_relevant_context(query)
                
            elif task.startswith("analyze_session:"):
                # Analyze current session patterns
                result = await self._analyze_session_patterns()
                
            elif task.startswith("update_context:"):
                # Update context with current state
                result = await self._update_session_context(state)
                
            else:
                # General context analysis
                result = await self._general_context_analysis(task, state)
            
            return self.add_message(
                state,
                "assistant", 
                result,
                metadata={
                    "context_operation": True,
                    "context_store_size": len(self.context_store),
                    "session_insights": len(self.session_context["key_insights"])
                }
            )
            
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"Context management error: {str(e)}"
            )
    
    async def _summarize_messages(self, messages: List[AgentMessage]) -> str:
        """Intelligently summarize a list of messages"""
        if not messages:
            return "No messages to summarize."
        
        # Group messages by conversation segments
        segments = self._group_messages_by_task(messages)
        summaries = []
        
        for segment in segments:
            if len(segment["messages"]) < 2:
                continue
                
            # Create context for summarization
            message_text = "\n".join([
                f"{msg.role}: {msg.content[:200]}..." if len(msg.content) > 200 else f"{msg.role}: {msg.content}"
                for msg in segment["messages"]
            ])
            
            prompt = f"""Summarize this conversation segment focusing on key information that should be preserved:

Task Context: {segment.get('task', 'General conversation')}
Messages:
{message_text}

Create a concise summary that captures:
1. Main objective or question
2. Key findings or decisions made  
3. Important technical details (file names, functions, configurations)
4. Outcomes or next steps
5. Any issues or blockers encountered

Keep the summary under 150 words but preserve all critical information."""

            summary = await self.generate_response(prompt)
            
            # Store summary with metadata
            summary_data = {
                "task": segment.get("task", "unknown"),
                "timestamp": segment.get("timestamp", datetime.now().isoformat()),
                "message_count": len(segment["messages"]),
                "summary": summary,
                "agents_involved": segment.get("agents", []),
                "files_referenced": segment.get("files", [])
            }
            
            summaries.append(summary_data)
            
            # Store in context store
            context_key = self._generate_context_key(segment.get("task", "conversation"), summary_data["timestamp"])
            self.context_store[context_key] = summary_data
        
        # Store conversation segment
        self.conversation_segments.extend(summaries)
        
        return f"Summarized {len(summaries)} conversation segments. Key context preserved in long-term storage."
    
    def _group_messages_by_task(self, messages: List[AgentMessage]) -> List[Dict[str, Any]]:
        """Group messages into logical conversation segments"""
        segments = []
        current_segment = None
        
        for msg in messages:
            # Detect task boundaries
            is_new_task = (
                msg.role == "user" or 
                (msg.role == "supervisor" and "Analyzing task:" in msg.content) or
                (msg.metadata and msg.metadata.get("task_analysis"))
            )
            
            if is_new_task or current_segment is None:
                if current_segment:
                    segments.append(current_segment)
                
                current_segment = {
                    "task": self._extract_task_from_message(msg),
                    "timestamp": datetime.now().isoformat(),
                    "messages": [msg],
                    "agents": set([msg.role]),
                    "files": set()
                }
            else:
                current_segment["messages"].append(msg)
                current_segment["agents"].add(msg.role)
            
            # Extract file references
            files = self._extract_file_references(msg.content)
            current_segment["files"].update(files)
        
        if current_segment:
            segments.append(current_segment)
        
        # Convert sets to lists for JSON serialization
        for segment in segments:
            segment["agents"] = list(segment["agents"])
            segment["files"] = list(segment["files"])
        
        return segments
    
    def _extract_task_from_message(self, msg: AgentMessage) -> str:
        """Extract task description from message"""
        content = msg.content[:100]  # First 100 chars
        
        if msg.role == "user":
            return content
        elif "Analyzing task:" in msg.content:
            # Extract task from supervisor message
            lines = msg.content.split('\n')
            for line in lines:
                if "Analyzing task:" in line:
                    return line.replace("Analyzing task:", "").strip()
        
        return "General conversation"
    
    def _extract_file_references(self, content: str) -> set:
        """Extract file paths and references from message content"""
        files = set()
        
        # Common file patterns
        import re
        
        # File paths (basic pattern)
        file_patterns = [
            r'[\w/.-]+\.py\b',
            r'[\w/.-]+\.js\b', 
            r'[\w/.-]+\.ts\b',
            r'[\w/.-]+\.java\b',
            r'[\w/.-]+\.cpp\b',
            r'[\w/.-]+\.c\b',
            r'[\w/.-]+\.go\b',
            r'[\w/.-]+\.rs\b'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, content)
            files.update(matches)
        
        return files
    
    async def _store_task_result(self, task_data: str, state: AgentState) -> str:
        """Store task completion results for future reference"""
        try:
            # Parse task data (could be JSON or simple string)
            if task_data.startswith('{'):
                data = json.loads(task_data)
                task_name = data.get("task", "unknown")
                result = data.get("result", "")
                metadata = data.get("metadata", {})
            else:
                task_name = task_data
                result = "Task completed"
                metadata = {}
            
            # Generate unique key for this task result
            context_key = self._generate_context_key(task_name, datetime.now().isoformat())
            
            # Store task result with rich context
            self.context_store[context_key] = {
                "task": task_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "agents_used": list(self.session_context.get("agents_used", [])),
                "files_referenced": list(self.session_context.get("files_referenced", [])),
                "message_count": len(state.messages)
            }
            
            # Add to session context
            self.session_context["active_tasks"].append({
                "task": task_name,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            
            return f"Task result stored: {task_name}. Context database now contains {len(self.context_store)} entries."
            
        except Exception as e:
            return f"Failed to store task result: {str(e)}"
    
    async def _retrieve_relevant_context(self, query: str) -> str:
        """Retrieve contextually relevant information for the current query"""
        if not self.context_store:
            return "No stored context available."
        
        # Simple relevance scoring (could be enhanced with embeddings)
        relevant_contexts = []
        query_lower = query.lower()
        
        for context_key, context_data in self.context_store.items():
            relevance_score = 0
            
            # Check task similarity
            task = context_data.get("task", "").lower()
            if any(word in task for word in query_lower.split()):
                relevance_score += 2
            
            # Check content similarity
            content = str(context_data.get("result", "")).lower()
            if any(word in content for word in query_lower.split()):
                relevance_score += 1
            
            # Check file references
            files = context_data.get("files_referenced", [])
            for word in query_lower.split():
                if any(word in str(file).lower() for file in files):
                    relevance_score += 1
            
            # Boost recent contexts
            timestamp = context_data.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hours_ago = (datetime.now() - dt).total_seconds() / 3600
                    if hours_ago < 2:  # Within 2 hours
                        relevance_score += 1
                except:
                    pass
            
            if relevance_score > 0:
                relevant_contexts.append((relevance_score, context_key, context_data))
        
        if not relevant_contexts:
            return f"No relevant context found for: {query}"
        
        # Sort by relevance and take top 3
        relevant_contexts.sort(key=lambda x: x[0], reverse=True)
        top_contexts = relevant_contexts[:3]
        
        result = f"Found {len(top_contexts)} relevant context entries for '{query}':\n\n"
        
        for score, key, context in top_contexts:
            task = context.get("task", "Unknown task")
            timestamp = context.get("timestamp", "")
            summary = context.get("summary", context.get("result", ""))[:200]
            
            result += f"**{task}** (Score: {score})\n"
            result += f"Time: {timestamp}\n"
            result += f"Context: {summary}...\n\n"
        
        return result
    
    async def _analyze_session_patterns(self) -> str:
        """Analyze patterns in the current session"""
        session = self.session_context
        
        # Calculate session duration
        duration = datetime.now() - session["start_time"]
        
        # Analyze patterns
        analysis = f"""Session Analysis ({duration.total_seconds() / 3600:.1f} hours):

**Activity Summary:**
- Active tasks: {len(session.get('active_tasks', []))}
- Key insights captured: {len(session.get('key_insights', []))}
- Files referenced: {len(session.get('files_referenced', []))}
- Agents used: {len(session.get('agents_used', []))}

**Recent Tasks:**"""
        
        recent_tasks = session.get("active_tasks", [])[-5:]  # Last 5 tasks
        for task in recent_tasks:
            analysis += f"\n- {task.get('task', 'Unknown')}: {task.get('status', 'unknown')}"
        
        # Most referenced files
        files = session.get("files_referenced", [])
        if files:
            analysis += f"\n\n**Most Referenced Files:**"
            file_counts = {}
            for file in files:
                file_counts[file] = file_counts.get(file, 0) + 1
            
            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for file, count in top_files:
                analysis += f"\n- {file} ({count} times)"
        
        return analysis
    
    async def _update_session_context(self, state: AgentState) -> str:
        """Update session context with current state information"""
        # Update agents used
        for msg in state.messages[-5:]:  # Last 5 messages
            self.session_context["agents_used"].add(msg.role)
        
        # Extract file references from recent messages  
        for msg in state.messages[-3:]:  # Last 3 messages
            files = self._extract_file_references(msg.content)
            self.session_context["files_referenced"].update(files)
        
        # Update current task
        if state.current_task:
            task_exists = any(
                task["task"] == state.current_task 
                for task in self.session_context["active_tasks"]
            )
            
            if not task_exists:
                self.session_context["active_tasks"].append({
                    "task": state.current_task,
                    "status": "active",
                    "timestamp": datetime.now().isoformat()
                })
        
        return f"Session context updated. Tracking {len(self.session_context['agents_used'])} agents and {len(self.session_context['files_referenced'])} files."
    
    async def _general_context_analysis(self, task: str, state: AgentState) -> str:
        """Provide general context analysis and insights"""
        
        # Check if we have relevant stored context
        relevant_context = await self._retrieve_relevant_context(task)
        
        # Analyze current conversation
        recent_summary = ""
        if len(state.messages) > 0:
            recent_messages = state.messages[-5:]  # Last 5 messages
            recent_summary = await self._summarize_messages(recent_messages)
        
        prompt = f"""Analyze the current context and provide insights:

Current Task: {task}

Recent Conversation Summary: {recent_summary}

Relevant Stored Context: {relevant_context}

Session Context: 
- Duration: {datetime.now() - self.session_context['start_time']}
- Active Tasks: {len(self.session_context.get('active_tasks', []))}
- Files Referenced: {len(self.session_context.get('files_referenced', []))}

Provide insights about:
1. How current task relates to previous work
2. Relevant context that might help
3. Potential patterns or recurring themes
4. Suggestions for maintaining context continuity"""
        
        return await self.generate_response(prompt)
    
    def _generate_context_key(self, task: str, timestamp: str) -> str:
        """Generate a unique key for context storage"""
        combined = f"{task}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    async def get_context_summary(self) -> str:
        """Get a summary of all stored context"""
        total_contexts = len(self.context_store)
        recent_contexts = sum(1 for context in self.context_store.values() 
                            if self._is_recent_context(context.get("timestamp", "")))
        
        return f"Context Store: {total_contexts} total entries, {recent_contexts} recent (last 24h)"
    
    def _is_recent_context(self, timestamp: str) -> bool:
        """Check if context is recent (within 24 hours)"""
        if not timestamp:
            return False
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return (datetime.now() - dt) < timedelta(hours=24)
        except:
            return False
    
    def clear_old_context(self, max_age_hours: Optional[int] = None):
        """Clear context older than specified hours"""
        max_age = max_age_hours or self.config.context_manager.max_context_age_hours
        cutoff = datetime.now() - timedelta(hours=max_age)
        
        keys_to_remove = []
        for key, context in self.context_store.items():
            timestamp = context.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if dt < cutoff:
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)  # Remove invalid timestamps
        
        for key in keys_to_remove:
            del self.context_store[key]
        
        return len(keys_to_remove)