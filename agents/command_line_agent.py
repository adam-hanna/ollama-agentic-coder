import os
import subprocess
import asyncio
import shlex
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState

class CommandLineAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a command line specialist that helps users execute system commands safely and effectively. Your role is to:

1. Analyze command requests and suggest appropriate shell commands
2. Execute commands within safety constraints based on the current mode
3. Explain command syntax, options, and expected behavior
4. Provide safety warnings for potentially dangerous operations
5. Suggest alternatives for unsafe commands in restricted modes

Safety Modes:
- SAFE: Only pre-approved, read-only commands are allowed
- WHITELIST: Safe commands + user-approved commands (with prompts for new ones)
- YOLO: All commands allowed (use with extreme caution)

Best practices you follow:
- Always explain what a command does before executing
- Warn about potentially destructive operations
- Suggest safer alternatives when possible
- Use appropriate command options for better output
- Handle errors gracefully and explain what went wrong
- Provide examples and usage tips
- Consider cross-platform compatibility when relevant"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No command specified")
        
        task = state.current_task
        
        try:
            if task.startswith("execute:"):
                command = task.replace("execute:", "").strip()
                result = await self._execute_command(command)
            
            elif task.startswith("explain:"):
                command = task.replace("explain:", "").strip()
                result = await self._explain_command(command)
            
            elif task.startswith("suggest:"):
                goal = task.replace("suggest:", "").strip()
                result = await self._suggest_command(goal)
            
            elif task.startswith("safety_check:"):
                command = task.replace("safety_check:", "").strip()
                result = await self._safety_check_command(command)
            
            else:
                # Default: try to understand and execute the request
                result = await self._intelligent_command_handling(task)
            
            return self.add_message(
                state,
                "assistant",
                result,
                metadata={"command_execution": True}
            )
        
        except Exception as e:
            return self.add_message(
                state,
                "assistant",
                f"Command execution failed: {str(e)}"
            )
    
    async def _execute_command(self, command: str) -> str:
        """Execute a command with safety checks"""
        # Safety check first
        safety_result = self._check_command_safety(command)
        
        if not safety_result["allowed"]:
            if safety_result["requires_approval"]:
                # In whitelist mode, ask for user approval
                approval = await self._request_user_approval(command, safety_result["reason"])
                if not approval:
                    return f"❌ Command execution cancelled: {safety_result['reason']}"
                
                # Add to user whitelist if approved
                await self._add_to_whitelist(command)
            else:
                return f"❌ Command not allowed: {safety_result['reason']}"
        
        # Execute the command
        return await self._run_command(command)
    
    def _check_command_safety(self, command: str) -> Dict[str, Any]:
        """Check if a command is safe to execute based on current mode"""
        mode = self.config.command_line.safety_mode.lower()
        
        # Parse command to get the base command
        try:
            parts = shlex.split(command)
            if not parts:
                return {"allowed": False, "reason": "Empty command"}
            base_command = parts[0]
        except ValueError:
            return {"allowed": False, "reason": "Invalid command syntax"}
        
        # YOLO mode - allow everything (with warning)
        if mode == "yolo":
            return {
                "allowed": True,
                "reason": "YOLO mode - all commands allowed",
                "warning": "⚠️ YOLO mode: No safety restrictions applied!"
            }
        
        # Check against dangerous commands
        dangerous_commands = {
            "rm", "del", "rmdir", "format", "fdisk", "mkfs",
            "dd", "shred", "wipe", "chmod", "chown", "su", "sudo",
            "passwd", "useradd", "userdel", "groupadd", "groupdel",
            "iptables", "ufw", "firewall-cmd", "systemctl", "service",
            "reboot", "shutdown", "halt", "poweroff", "kill", "killall",
            "pkill", "crontab", "at", ">", ">>", "|"
        }
        
        if any(dangerous in command for dangerous in dangerous_commands):
            if mode == "whitelist":
                return {
                    "allowed": False,
                    "requires_approval": True,
                    "reason": f"Potentially dangerous command '{base_command}' requires approval"
                }
            else:  # safe mode
                return {
                    "allowed": False,
                    "reason": f"Dangerous command '{base_command}' not allowed in safe mode"
                }
        
        # SAFE mode - only allow pre-approved commands
        if mode == "safe":
            safe_commands = self.config.command_line.safe_commands
            if base_command not in safe_commands:
                return {
                    "allowed": False,
                    "reason": f"Command '{base_command}' not in safe command list"
                }
        
        # WHITELIST mode - allow safe commands + user whitelist
        elif mode == "whitelist":
            safe_commands = self.config.command_line.safe_commands
            user_whitelist = self.config.command_line.user_whitelist
            
            if base_command not in safe_commands and base_command not in user_whitelist:
                return {
                    "allowed": False,
                    "requires_approval": True,
                    "reason": f"Command '{base_command}' not in whitelist - approval required"
                }
        
        return {"allowed": True, "reason": "Command approved"}
    
    async def _request_user_approval(self, command: str, reason: str) -> bool:
        """Request user approval for a command (in whitelist mode)"""
        # In a real implementation, this would show a prompt to the user
        # For now, we'll return False to be safe
        # TODO: Implement actual user prompting mechanism
        return False
    
    async def _add_to_whitelist(self, command: str):
        """Add a command to the user whitelist"""
        parts = shlex.split(command)
        base_command = parts[0] if parts else command
        
        if base_command not in self.config.command_line.user_whitelist:
            self.config.command_line.user_whitelist.append(base_command)
            # TODO: Persist to config file
    
    async def _run_command(self, command: str) -> str:
        """Actually execute the command"""
        try:
            # Use shell=True but with safety measures
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.command_line.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"❌ Command timed out after {self.config.command_line.timeout_seconds} seconds"
            
            # Prepare output
            output_parts = []
            
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='replace')
                if len(stdout_text) > self.config.command_line.max_output_length:
                    stdout_text = stdout_text[:self.config.command_line.max_output_length] + "\n... (output truncated)"
                output_parts.append(f"📤 Output:\n{stdout_text}")
            
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='replace')
                if len(stderr_text) > self.config.command_line.max_output_length:
                    stderr_text = stderr_text[:self.config.command_line.max_output_length] + "\n... (output truncated)"
                output_parts.append(f"⚠️ Errors:\n{stderr_text}")
            
            if process.returncode != 0:
                output_parts.append(f"❌ Exit code: {process.returncode}")
            else:
                output_parts.append("✅ Command completed successfully")
            
            result = f"🖥️ Executed: `{command}`\n\n" + "\n\n".join(output_parts)
            
            # Add safety mode indicator
            mode = self.config.command_line.safety_mode.upper()
            result += f"\n\n🔒 Safety Mode: {mode}"
            
            return result
            
        except Exception as e:
            return f"❌ Failed to execute command: {str(e)}"
    
    async def _explain_command(self, command: str) -> str:
        """Explain what a command does without executing it"""
        prompt = f"""Explain this command in detail:

Command: {command}

Provide a comprehensive explanation including:
1. What the command does (primary function)
2. Breakdown of each part/option if complex
3. What the expected output would be
4. Any potential risks or side effects
5. Common use cases and examples
6. Alternative commands that do similar things
7. Platform compatibility notes (Linux/macOS/Windows)

Be educational and include practical tips."""
        
        explanation = await self.generate_response(prompt)
        
        # Add safety assessment
        safety_result = self._check_command_safety(command)
        safety_info = f"\n\n🔒 Safety Assessment:\n"
        
        if safety_result["allowed"]:
            safety_info += f"✅ This command is allowed in {self.config.command_line.safety_mode.upper()} mode"
        else:
            safety_info += f"❌ This command is NOT allowed in {self.config.command_line.safety_mode.upper()} mode"
            safety_info += f"\nReason: {safety_result['reason']}"
        
        return explanation + safety_info
    
    async def _suggest_command(self, goal: str) -> str:
        """Suggest commands to accomplish a goal"""
        prompt = f"""Suggest shell commands to accomplish this goal:

Goal: {goal}

Provide suggestions that include:
1. The exact command(s) to run
2. Explanation of what each command does
3. Expected output or results
4. Any prerequisites or setup needed
5. Safety considerations
6. Alternative approaches
7. Cross-platform considerations

Current safety mode: {self.config.command_line.safety_mode.upper()}

Focus on commands that would be appropriate for the current safety level."""
        
        return await self.generate_response(prompt)
    
    async def _safety_check_command(self, command: str) -> str:
        """Perform a detailed safety check on a command"""
        safety_result = self._check_command_safety(command)
        
        result = f"🔍 Safety Check for: `{command}`\n\n"
        result += f"🔒 Current Safety Mode: {self.config.command_line.safety_mode.upper()}\n\n"
        
        if safety_result["allowed"]:
            result += "✅ **ALLOWED** - This command can be executed\n"
            if "warning" in safety_result:
                result += f"⚠️ {safety_result['warning']}\n"
        else:
            result += "❌ **NOT ALLOWED** - This command cannot be executed\n"
            result += f"Reason: {safety_result['reason']}\n"
        
        # Add detailed analysis
        prompt = f"""Analyze this command for potential security and safety risks:

Command: {command}

Assess:
1. What files/directories it might access or modify
2. Network operations it might perform
3. System changes it could make
4. Data it might read, write, or delete
5. Privilege escalation concerns
6. Potential for unintended consequences
7. Reversibility of its actions

Provide a risk assessment (LOW/MEDIUM/HIGH) and explain why."""
        
        analysis = await self.generate_response(prompt)
        result += f"\n🔍 **Detailed Risk Analysis:**\n{analysis}"
        
        return result
    
    async def _intelligent_command_handling(self, task: str) -> str:
        """Handle general command requests using AI"""
        prompt = f"""The user wants to accomplish this task using command line:

Task: {task}

Current safety mode: {self.config.command_line.safety_mode.upper()}

Analyze the request and provide:
1. Understanding of what the user wants to do
2. Appropriate command(s) to accomplish this
3. Step-by-step explanation
4. Safety considerations for the current mode
5. Whether the commands would be allowed or not
6. Alternative approaches if needed

If the task seems to require dangerous commands in safe mode, suggest safer alternatives or explain why the operation requires elevated permissions."""
        
        return await self.generate_response(prompt)