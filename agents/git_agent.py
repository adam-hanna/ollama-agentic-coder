import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState


class GitAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a Git version control specialist that helps with repository management and workflow automation. Your role is to:

1. Analyze Git repository status and history
2. Generate meaningful commit messages
3. Suggest branching strategies and workflows
4. Help with merge conflicts and rebasing
5. Manage branches and tags
6. Analyze code changes and diffs

Git expertise areas:
- Repository initialization and configuration
- Staging, committing, and pushing changes
- Branch management and merging strategies
- Conflict resolution and rebasing
- Remote repository management
- Git hooks and automation
- Collaborative workflows (GitFlow, GitHub Flow)
- Code review processes
- Release management and tagging

Best practices you follow:
- Write clear, descriptive commit messages
- Use atomic commits (one logical change per commit)
- Maintain clean commit history
- Use appropriate branching strategies
- Follow conventional commit formats
- Ensure proper .gitignore configuration
- Handle sensitive data appropriately
- Maintain repository hygiene"""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(state, "assistant", "No Git operation specified")

        task = state.current_task

        try:
            if task.startswith("git_status"):
                result = await self._git_status()

            elif task.startswith("generate_commit_message:"):
                changes = task.replace("generate_commit_message:", "").strip()
                result = await self._generate_commit_message(changes)

            elif task.startswith("analyze_changes"):
                result = await self._analyze_changes()

            elif task.startswith("suggest_branch:"):
                feature = task.replace("suggest_branch:", "").strip()
                result = await self._suggest_branch_name(feature)

            elif task.startswith("resolve_conflict:"):
                file_path = task.replace("resolve_conflict:", "").strip()
                result = await self._help_resolve_conflict(file_path)

            elif task.startswith("analyze_history:"):
                params = task.replace("analyze_history:", "").strip()
                result = await self._analyze_git_history(params)

            elif task.startswith("suggest_workflow"):
                result = await self._suggest_workflow()

            else:
                result = await self._intelligent_git_assistance(task)

            return self.add_message(
                state, "assistant", result, metadata={"git_operation": True}
            )

        except Exception as e:
            return self.add_message(
                state, "assistant", f"Git operation failed: {str(e)}"
            )

    async def _git_status(self) -> str:
        """Get comprehensive Git repository status"""
        if not self._is_git_repo():
            return "Not a Git repository. Use 'git init' to initialize a repository."

        try:
            # Get various Git status information
            status = self._run_git_command("status --porcelain")
            branch = self._run_git_command("branch --show-current").strip()
            unpushed = self._run_git_command(
                "log --oneline @{u}..HEAD", allow_error=True
            )
            unpulled = self._run_git_command(
                "log --oneline HEAD..@{u}", allow_error=True
            )

            # Parse status
            modified = []
            staged = []
            untracked = []

            for line in status.split("\n"):
                if not line.strip():
                    continue

                status_code = line[:2]
                file_path = line[3:]

                if status_code[0] in ["M", "A", "D", "R", "C"]:
                    staged.append(f"{status_code[0]} {file_path}")
                if status_code[1] in ["M", "D"]:
                    modified.append(f"{status_code[1]} {file_path}")
                if status_code == "??":
                    untracked.append(file_path)

            # Build comprehensive status report
            report = f"""📊 Git Repository Status

Current Branch: {branch or 'HEAD (detached)'}

📁 Staged Changes ({len(staged)}):
{chr(10).join(f"  ✅ {change}" for change in staged) if staged else "  (none)"}

📝 Modified Files ({len(modified)}):
{chr(10).join(f"  📝 {change}" for change in modified) if modified else "  (none)"}

❓ Untracked Files ({len(untracked)}):
{chr(10).join(f"  ❓ {file}" for file in untracked) if untracked else "  (none)"}

🔄 Commits ahead of remote: {len(unpushed.split(chr(10))) - 1 if unpushed.strip() else 0}
⬇️ Commits behind remote: {len(unpulled.split(chr(10))) - 1 if unpulled.strip() else 0}

💡 Suggestions:
"""

            # Add contextual suggestions
            if staged:
                report += "\n  • Ready to commit staged changes"
            if modified:
                report += "\n  • Stage modified files with 'git add'"
            if untracked:
                report += "\n  • Add untracked files or update .gitignore"
            if unpushed.strip():
                report += "\n  • Push commits to remote repository"
            if unpulled.strip():
                report += "\n  • Pull latest changes from remote"

            return report

        except Exception as e:
            return f"Failed to get Git status: {str(e)}"

    async def _generate_commit_message(self, changes: str) -> str:
        """Generate meaningful commit messages based on changes"""
        if not changes:
            # Get current diff if no changes provided
            try:
                diff = self._run_git_command("diff --cached --name-status")
                if not diff:
                    diff = self._run_git_command("diff --name-status")
                changes = diff
            except Exception:
                changes = "No changes detected"

        prompt = f"""Generate a clear, meaningful commit message for these changes:

Changes:
{changes}

Follow conventional commit format when appropriate:
- feat: new features
- fix: bug fixes  
- docs: documentation changes
- style: formatting changes
- refactor: code refactoring
- test: adding or updating tests
- chore: maintenance tasks

Guidelines:
1. Use imperative mood ("Add feature" not "Added feature")
2. Keep first line under 72 characters
3. Add detailed description if needed
4. Reference issues/tickets when applicable
5. Be descriptive but concise

Provide multiple commit message options with explanations."""

        return await self.generate_response(prompt)

    async def _analyze_changes(self) -> str:
        """Analyze current changes in the repository"""
        if not self._is_git_repo():
            return "Not a Git repository."

        try:
            # Get diff information
            staged_diff = self._run_git_command("diff --cached --stat")
            unstaged_diff = self._run_git_command("diff --stat")

            # Get detailed changes
            staged_changes = self._run_git_command("diff --cached --name-only")
            unstaged_changes = self._run_git_command("diff --name-only")

            analysis = f"""🔍 Git Changes Analysis

📊 Staged Changes Statistics:
{staged_diff if staged_diff.strip() else '  (no staged changes)'}

📊 Unstaged Changes Statistics:
{unstaged_diff if unstaged_diff.strip() else '  (no unstaged changes)'}

📁 Files Changed:
Staged: {', '.join(staged_changes.split()) if staged_changes.strip() else 'none'}
Unstaged: {', '.join(unstaged_changes.split()) if unstaged_changes.strip() else 'none'}
"""

            # Add AI analysis of the changes
            if staged_changes.strip() or unstaged_changes.strip():
                all_changes = (staged_changes + "\n" + unstaged_changes).strip()

                prompt = f"""Analyze these code changes and provide insights:

Changed files:
{all_changes}

Provide analysis on:
1. Type of changes (features, fixes, refactoring, etc.)
2. Potential impact and risk assessment
3. Suggestions for commit organization
4. Testing recommendations
5. Documentation needs
6. Code review considerations"""

                ai_analysis = await self.generate_response(prompt)
                analysis += f"\n🤖 AI Analysis:\n{ai_analysis}"

            return analysis

        except Exception as e:
            return f"Failed to analyze changes: {str(e)}"

    async def _suggest_branch_name(self, feature_description: str) -> str:
        """Suggest appropriate branch names for features"""
        prompt = f"""Suggest appropriate Git branch names for this feature:

Feature Description: {feature_description}

Provide branch name suggestions following these conventions:
1. **Feature branches**: feature/description or feat/description
2. **Bug fixes**: bugfix/description or fix/description  
3. **Hotfixes**: hotfix/description
4. **Documentation**: docs/description
5. **Chores**: chore/description

Guidelines:
- Use lowercase with hyphens (kebab-case)
- Be descriptive but concise
- Include issue numbers if applicable
- Avoid special characters
- Keep under 50 characters

Provide multiple options with rationale for each."""

        return await self.generate_response(prompt)

    async def _help_resolve_conflict(self, file_path: str) -> str:
        """Help resolve merge conflicts"""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        try:
            # Check if file has conflict markers
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "<<<<<<<" not in content:
                return f"No merge conflicts detected in {file_path}"

            prompt = f"""Help resolve merge conflicts in this file:

File: {file_path}
Content:
```
{content}
```

Provide conflict resolution guidance:
1. **Conflict Analysis**:
   - Identify the conflicting sections
   - Understand what each side is trying to accomplish
   - Assess the nature of the conflict

2. **Resolution Strategy**:
   - Recommend which changes to keep
   - Suggest how to merge both changes if needed
   - Identify potential issues with each approach

3. **Resolved Code**:
   - Show the final merged version
   - Explain the resolution decisions
   - Highlight any additional changes needed

4. **Testing Recommendations**:
   - What to test after resolution
   - Potential edge cases to consider"""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to analyze conflict: {str(e)}"

    async def _analyze_git_history(self, parameters: str) -> str:
        """Analyze Git commit history and patterns"""
        if not self._is_git_repo():
            return "Not a Git repository."

        try:
            # Get commit history
            log_command = "log --oneline --graph -20"
            if parameters.strip():
                log_command += f" {parameters}"

            history = self._run_git_command(log_command)
            contributors = self._run_git_command("shortlog -sn")
            recent_activity = self._run_git_command(
                "log --oneline --since='1 month ago'"
            )

            analysis = f"""📈 Git History Analysis

Recent Commits:
{history}

👥 Contributors:
{contributors}

📅 Recent Activity (last month):
{len(recent_activity.split(chr(10)))} commits

"""

            # Add AI insights
            prompt = f"""Analyze this Git repository history and provide insights:

Commit History:
{history}

Contributors:
{contributors}

Provide analysis on:
1. Commit patterns and frequency
2. Code quality indicators from commit messages
3. Collaboration patterns
4. Potential workflow improvements
5. Branch management observations
6. Release patterns if visible"""

            ai_insights = await self.generate_response(prompt)
            analysis += f"🤖 Insights:\n{ai_insights}"

            return analysis

        except Exception as e:
            return f"Failed to analyze Git history: {str(e)}"

    async def _suggest_workflow(self) -> str:
        """Suggest appropriate Git workflows"""
        if not self._is_git_repo():
            return "Not a Git repository."

        try:
            # Analyze repository characteristics
            branches = self._run_git_command("branch -a")
            remotes = self._run_git_command("remote -v")
            contributors = self._run_git_command("shortlog -sn")

            analysis = f"""Repository Analysis:
- Branches: {len(branches.split())}
- Remotes: {len(remotes.split(chr(10)))}  
- Contributors: {len(contributors.split(chr(10)))}"""

            prompt = f"""Suggest appropriate Git workflows for this repository:

{analysis}

Consider and recommend:
1. **Workflow Types**:
   - Git Flow (feature/develop/main branches)
   - GitHub Flow (feature branches + main)
   - GitLab Flow (environment branches)
   - Trunk-based development

2. **Branching Strategy**:
   - Branch naming conventions
   - Integration patterns
   - Release management

3. **Collaboration Practices**:
   - Code review processes
   - Pull/merge request guidelines
   - Conflict resolution strategies

4. **Automation Opportunities**:
   - Git hooks
   - CI/CD integration
   - Automated testing

Provide specific recommendations based on team size and project characteristics."""

            return await self.generate_response(prompt)

        except Exception as e:
            return f"Failed to suggest workflow: {str(e)}"

    async def _intelligent_git_assistance(self, task: str) -> str:
        """Handle complex Git requests using AI"""
        prompt = f"""Provide Git assistance for this request:

Request: {task}

Analyze the request and provide:
1. Understanding of the Git operation needed
2. Step-by-step instructions
3. Command examples with explanations
4. Best practices and warnings
5. Alternative approaches if applicable
6. Troubleshooting tips
7. Related workflows or processes

Focus on practical, safe Git operations with clear explanations."""

        return await self.generate_response(prompt)

    def _is_git_repo(self) -> bool:
        """Check if current directory is a Git repository"""
        return (
            os.path.exists(".git")
            or self._run_git_command("rev-parse --git-dir", allow_error=True) != ""
        )

    def _run_git_command(self, command: str, allow_error: bool = False) -> str:
        """Run a Git command and return the output"""
        try:
            result = subprocess.run(
                f"git {command}",
                shell=True,
                capture_output=True,
                text=True,
                check=not allow_error,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if allow_error:
                return ""
            raise Exception(f"Git command failed: {e.stderr}")
