from typing import Dict, Any, List, Optional
from core.base_agent import BaseAgent, AgentState
from core.shared_context import shared_context


class ArchitectingAgent(BaseAgent):
    def _default_system_prompt(self) -> str:
        return """You are a senior software architect with expertise in designing scalable, maintainable systems. Your role is to:

1. **Design high-level system architecture** for new features and applications
2. **Analyze existing architecture** and propose improvements
3. **Make technology and design decisions** based on requirements
4. **Create architectural documentation** and design specifications
5. **Consider scalability, maintainability, and performance** in all designs

**Available Tools:**
- Use built-in read_file() method to examine existing architecture
- Use execute_command() method to analyze system structure
- Access indexed codebase data via shared_context for understanding current architecture
- Coordinate with other agents for implementation details

**Architecture Domains:**
- **System Design**: Overall application architecture, microservices vs monolith decisions
- **Database Design**: Data modeling, database selection, schema design
- **API Design**: RESTful APIs, GraphQL, event-driven architectures
- **Frontend Architecture**: Component structure, state management, routing
- **Backend Architecture**: Service layers, dependency injection, messaging patterns
- **DevOps Architecture**: Deployment strategies, CI/CD, containerization
- **Security Architecture**: Authentication, authorization, data protection
- **Performance Architecture**: Caching, load balancing, optimization strategies

**Your Design Process:**
1. **Requirements Analysis** - Understand functional and non-functional requirements
2. **Current State Assessment** - Analyze existing system architecture
3. **Design Alternatives** - Propose multiple architectural approaches
4. **Trade-off Analysis** - Compare alternatives with pros/cons
5. **Detailed Design** - Create comprehensive architectural specifications
6. **Implementation Guidance** - Provide clear direction for developers

**Architecture Principles:**
- Design for scalability and future growth
- Maintain separation of concerns and modularity
- Consider performance, security, and maintainability
- Document decisions and rationale clearly
- Follow industry best practices and patterns
- Consider team capabilities and constraints

You provide architectural guidance, not implementation details."""

    async def process(self, state: AgentState) -> AgentState:
        if not state.current_task:
            return self.add_message(
                state, "assistant", "No architecture task specified"
            )

        task = state.current_task

        try:
            if task.startswith("design:"):
                result = await self._design_architecture(
                    task.replace("design:", "").strip()
                )
            elif task.startswith("analyze:"):
                result = await self._analyze_existing_architecture(
                    task.replace("analyze:", "").strip()
                )
            elif task.startswith("improve:"):
                result = await self._improve_architecture(
                    task.replace("improve:", "").strip()
                )
            elif task.startswith("migrate:"):
                result = await self._design_migration(
                    task.replace("migrate:", "").strip()
                )
            elif task.startswith("pattern:"):
                result = await self._recommend_patterns(
                    task.replace("pattern:", "").strip()
                )
            else:
                result = await self._architectural_analysis(task)

            return self.add_message(
                state,
                "assistant",
                result,
                metadata={"operation_type": "architecture", "scope": "system_design"},
            )

        except Exception as e:
            return self.add_message(
                state, "assistant", f"Architecture analysis failed: {str(e)}"
            )

    async def _design_architecture(self, requirements: str) -> str:
        """Design new system architecture based on requirements"""
        results = []
        results.append("# 🏗️ System Architecture Design")
        results.append(f"**Requirements**: {requirements}")

        # 1. Requirements analysis
        analysis = await self._analyze_requirements(requirements)
        results.append("## 📋 Requirements Analysis")
        results.append(analysis)

        # 2. System context analysis
        context = await self._analyze_system_context()
        results.append("## 🌐 System Context")
        results.append(context)

        # 3. Design alternatives
        alternatives = await self._design_alternatives(requirements, analysis)
        results.append("## 🎯 Design Alternatives")
        results.append(alternatives)

        # 4. Recommended architecture
        recommendation = await self._recommend_architecture(requirements, alternatives)
        results.append("## ⭐ Recommended Architecture")
        results.append(recommendation)

        # 5. Implementation roadmap
        roadmap = await self._create_implementation_roadmap(recommendation)
        results.append("## 🗺️ Implementation Roadmap")
        results.append(roadmap)

        return "\n\n".join(results)

    async def _analyze_requirements(self, requirements: str) -> str:
        """Analyze and categorize requirements"""
        prompt = f"""Analyze these requirements: "{requirements}"

Categorize and elaborate on:

**Functional Requirements:**
- Core features and capabilities needed
- User interactions and workflows
- Data processing requirements
- Integration needs

**Non-Functional Requirements:**
- Performance expectations (latency, throughput)
- Scalability needs (concurrent users, data volume)  
- Security requirements (authentication, data protection)
- Availability and reliability needs
- Maintainability and extensibility requirements

**Constraints:**
- Technology stack limitations
- Budget and timeline constraints
- Team capabilities and experience
- Legacy system compatibility
- Regulatory compliance needs

Provide specific, measurable requirements where possible."""

        return await self.generate_response(prompt)

    async def _analyze_system_context(self) -> str:
        """Analyze the current system context and environment"""
        try:
            # Get project structure
            structure = await self.execute_command(
                "find . -type f -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.java' -o -name '*.go' | head -20"
            )

            # Check for existing architecture indicators
            config_files = await self.execute_command(
                "find . -name '*.json' -o -name '*.yaml' -o -name '*.toml' -o -name 'Dockerfile' -o -name 'docker-compose.*' | head -10"
            )

            # Analyze dependencies
            deps_info = await self._analyze_dependencies()

            return f"""**Current Project Structure:**
```
{structure}
```

**Configuration & Infrastructure:**
```
{config_files}
```

**Dependencies Analysis:**
{deps_info}

**Architecture Implications:**
- Existing technology stack influences new design decisions
- Current patterns should be considered for consistency
- Integration points are already established
- Deployment and configuration patterns are in place"""

        except Exception as e:
            return f"System context analysis limited: {e}"

    async def _analyze_dependencies(self) -> str:
        """Analyze project dependencies and technology stack"""
        try:
            # Python dependencies
            python_deps = await self.execute_command(
                "find . -name 'requirements*.txt' -o -name 'pyproject.toml' -o -name 'Pipfile' | head -5"
            )

            # JavaScript dependencies
            js_deps = await self.execute_command(
                "find . -name 'package.json' -o -name 'yarn.lock' -o -name 'package-lock.json' | head -5"
            )

            # Java dependencies
            java_deps = await self.execute_command(
                "find . -name 'pom.xml' -o -name 'build.gradle' | head -5"
            )

            # Other build files
            other_deps = await self.execute_command(
                "find . -name 'go.mod' -o -name 'Cargo.toml' -o -name 'composer.json' | head -5"
            )

            deps_summary = []
            if python_deps.strip():
                deps_summary.append(f"Python: {python_deps}")
            if js_deps.strip():
                deps_summary.append(f"JavaScript: {js_deps}")
            if java_deps.strip():
                deps_summary.append(f"Java: {java_deps}")
            if other_deps.strip():
                deps_summary.append(f"Other: {other_deps}")

            return (
                "\n".join(deps_summary)
                if deps_summary
                else "No dependency files detected"
            )

        except Exception as e:
            return f"Dependency analysis limited: {e}"

    async def _design_alternatives(self, requirements: str, analysis: str) -> str:
        """Design multiple architectural alternatives"""
        prompt = f"""Based on requirements: "{requirements}"
And analysis: "{analysis}"

Design 3 different architectural approaches:

**Alternative 1: Monolithic Architecture**
- When appropriate: Simple applications, small teams, rapid prototyping
- Architecture overview
- Key components and layers
- Data flow and communication
- Pros and cons
- Technology recommendations

**Alternative 2: Microservices Architecture**  
- When appropriate: Large scale, multiple teams, complex domains
- Service breakdown and boundaries
- Communication patterns (REST, messaging, events)
- Data management strategy
- Pros and cons
- Technology recommendations

**Alternative 3: Hybrid/Modular Architecture**
- When appropriate: Medium complexity, growth flexibility
- Module organization and boundaries  
- Integration patterns
- Deployment options
- Pros and cons
- Technology recommendations

For each alternative, consider scalability, maintainability, complexity, and team fit."""

        return await self.generate_response(prompt)

    async def _recommend_architecture(
        self, requirements: str, alternatives: str
    ) -> str:
        """Recommend the best architecture based on analysis"""
        prompt = f"""Given requirements: "{requirements}"
And alternatives: "{alternatives}"

Provide a detailed recommendation:

**Recommended Architecture**: [Selected approach]

**Rationale**: Why this approach is best for the given requirements

**Detailed Design**:
1. **System Overview** - High-level architecture diagram description
2. **Component Breakdown** - Major components and their responsibilities
3. **Data Architecture** - Database design, data flow, storage strategy
4. **API Design** - Interface definitions and communication patterns
5. **Security Architecture** - Authentication, authorization, data protection
6. **Deployment Architecture** - Infrastructure, scalability, monitoring

**Technology Stack Recommendations**:
- Backend technologies and frameworks
- Frontend technologies (if applicable)
- Database and storage solutions
- Infrastructure and DevOps tools
- Monitoring and logging tools

**Risk Mitigation**:
- Identified architectural risks
- Mitigation strategies
- Contingency plans"""

        return await self.generate_response(prompt)

    async def _create_implementation_roadmap(self, architecture: str) -> str:
        """Create implementation roadmap for the architecture"""
        prompt = f"""Based on the recommended architecture: "{architecture}"

Create a practical implementation roadmap:

**Phase 1: Foundation (Weeks 1-2)**
- Core infrastructure setup
- Basic project structure
- Essential configurations
- Initial development environment

**Phase 2: Core Components (Weeks 3-6)**  
- Implement primary system components
- Basic data layer
- Core business logic
- Initial API endpoints

**Phase 3: Integration (Weeks 7-8)**
- Component integration
- End-to-end workflows
- Initial testing framework
- Basic security implementation

**Phase 4: Advanced Features (Weeks 9-12)**
- Advanced functionality
- Performance optimization
- Comprehensive testing
- Documentation

**Phase 5: Production Readiness (Weeks 13-14)**
- Deployment setup
- Monitoring and logging
- Security hardening
- Performance tuning

**Dependencies and Prerequisites**:
- What needs to be in place before starting
- Team skill requirements
- Infrastructure needs
- Third-party service setup

**Success Criteria**:
- Measurable milestones for each phase
- Quality gates and reviews
- Performance benchmarks"""

        return await self.generate_response(prompt)

    async def _analyze_existing_architecture(self, focus_area: str) -> str:
        """Analyze existing system architecture"""
        results = []
        results.append("# 🔍 Architecture Analysis")
        results.append(f"**Focus Area**: {focus_area}")

        # System overview
        overview = await self._get_system_overview()
        results.append("## 🏗️ System Overview")
        results.append(overview)

        # Architecture patterns
        patterns = await self._identify_architecture_patterns()
        results.append("## 🔄 Architecture Patterns")
        results.append(patterns)

        # Quality assessment
        quality = await self._assess_architecture_quality()
        results.append("## ⚡ Quality Assessment")
        results.append(quality)

        # Recommendations
        recommendations = await self._architecture_recommendations(focus_area)
        results.append("## 💡 Recommendations")
        results.append(recommendations)

        return "\n\n".join(results)

    async def _get_system_overview(self) -> str:
        """Get high-level system overview"""
        try:
            # Count different types of files
            file_counts = await self.execute_command(
                "find . -type f | grep -E '\\.(py|js|ts|java|go|rs)$' | sed 's/.*\\.//' | sort | uniq -c"
            )

            # Check for architecture indicators
            services = await self.execute_command(
                "find . -name 'service*' -o -name '*Service*' | wc -l"
            )
            controllers = await self.execute_command(
                "find . -name '*controller*' -o -name '*Controller*' | wc -l"
            )
            models = await self.execute_command(
                "find . -name '*model*' -o -name '*Model*' | wc -l"
            )

            return f"""**File Distribution:**
```
{file_counts}
```

**Architecture Indicators:**
- Services/Service Layer: {services.strip()} files
- Controllers/Handlers: {controllers.strip()} files  
- Models/Entities: {models.strip()} files

**System Complexity**: Based on file distribution and organization patterns"""

        except Exception as e:
            return f"System overview limited: {e}"

    async def _identify_architecture_patterns(self) -> str:
        """Identify architectural patterns in use"""
        prompt = """Analyze the current codebase to identify architectural patterns:

**Structural Patterns:**
- Layered architecture (presentation, business, data)
- Model-View-Controller (MVC)
- Component-based architecture
- Module organization patterns

**Integration Patterns:**
- API patterns (REST, GraphQL, RPC)
- Messaging patterns
- Database access patterns
- External service integration

**Design Patterns:**
- Dependency injection
- Factory patterns
- Observer/Event patterns
- Repository patterns

Examine the code structure and identify which patterns are being used and how well they're implemented."""

        return await self.generate_response(prompt)

    async def _assess_architecture_quality(self) -> str:
        """Assess the quality of current architecture"""
        prompt = """Assess the architecture quality across these dimensions:

**Maintainability:**
- Code organization and modularity
- Separation of concerns
- Documentation quality
- Test coverage and testability

**Scalability:**
- Performance bottlenecks
- Resource utilization
- Horizontal scaling capabilities
- Database scalability

**Reliability:**
- Error handling patterns
- Fault tolerance
- Recovery mechanisms
- System resilience

**Security:**
- Security architecture
- Data protection measures
- Authentication/authorization
- Vulnerability exposure

**Flexibility:**
- Extension points
- Configuration management
- Technology upgradability
- Business logic adaptability

Provide specific observations and metrics where possible."""

        return await self.generate_response(prompt)

    async def _architecture_recommendations(self, focus_area: str) -> str:
        """Provide architecture improvement recommendations"""
        prompt = f"""Based on the architecture analysis, with focus on: "{focus_area}"

Provide specific, actionable recommendations:

**High Priority Issues:**
- Critical architectural problems that need immediate attention
- Impact on system reliability, performance, or security
- Specific solutions and implementation approaches

**Medium Priority Improvements:**
- Architecture enhancements that would provide significant value
- Modernization opportunities
- Pattern improvements and refactoring suggestions

**Long-term Architectural Goals:**
- Strategic direction for system evolution
- Technology upgrade paths
- Scalability preparation

**Implementation Strategy:**
- Prioritization based on impact vs effort
- Risk assessment for changes
- Migration strategies for major changes
- Resource requirements and timeline estimates

Focus recommendations on practical, implementable changes."""

        return await self.generate_response(prompt)

    async def _improve_architecture(self, improvement_area: str) -> str:
        """Design architecture improvements"""
        results = []
        results.append("# 🚀 Architecture Improvement")
        results.append(f"**Improvement Area**: {improvement_area}")

        # Current state analysis
        current_state = await self._analyze_current_state(improvement_area)
        results.append("## 📊 Current State")
        results.append(current_state)

        # Improvement design
        improvements = await self._design_improvements(improvement_area, current_state)
        results.append("## ✨ Improvement Design")
        results.append(improvements)

        # Migration strategy
        migration = await self._design_migration_strategy(
            improvement_area, improvements
        )
        results.append("## 🔄 Migration Strategy")
        results.append(migration)

        return "\n\n".join(results)

    async def _analyze_current_state(self, area: str) -> str:
        """Analyze current state for specific improvement area"""
        prompt = f"""Analyze the current state for improvement area: "{area}"

**Current Architecture Assessment:**
- How is this area currently implemented?
- What are the specific problems or limitations?
- What architectural debts exist?
- How does it impact other system components?

**Pain Points:**
- Performance issues
- Maintainability challenges  
- Scalability limitations
- Security concerns
- Developer experience issues

**Dependencies and Constraints:**
- What other components depend on this area?
- What would be impacted by changes?
- Technical constraints to consider
- Business constraints and requirements

Provide a clear baseline for improvement planning."""

        return await self.generate_response(prompt)

    async def _design_improvements(self, area: str, current_state: str) -> str:
        """Design specific improvements"""
        prompt = f"""Design improvements for: "{area}"
Current state: "{current_state}"

**Improvement Strategy:**
1. **Target Architecture** - What should the improved architecture look like?
2. **Key Changes** - Specific architectural changes needed
3. **Technology Decisions** - New technologies or patterns to adopt
4. **Component Design** - How components should be restructured
5. **Interface Design** - New APIs or contracts needed

**Benefits Analysis:**
- Performance improvements expected
- Maintainability gains
- Scalability enhancements
- Developer productivity improvements
- Risk reductions

**Implementation Approach:**
- Incremental vs big-bang changes
- Backward compatibility considerations
- Testing strategy for changes
- Rollback plans if needed

Design practical, implementable improvements."""

        return await self.generate_response(prompt)

    async def _design_migration_strategy(self, area: str, improvements: str) -> str:
        """Design migration strategy for improvements"""
        prompt = f"""Design migration strategy for: "{area}"
Improvements: "{improvements}"

**Migration Plan:**
1. **Pre-migration Setup** - Preparation steps needed
2. **Phase 1**: Initial changes and groundwork
3. **Phase 2**: Core migration activities  
4. **Phase 3**: Final migration and cleanup
5. **Post-migration**: Validation and optimization

**Risk Mitigation:**
- Identified migration risks
- Contingency plans
- Rollback procedures
- Monitoring during migration

**Testing Strategy:**
- How to validate migration success
- Performance testing approach
- User acceptance criteria
- Rollback testing

**Communication Plan:**
- Stakeholder communication
- Developer team coordination
- User impact communication
- Timeline and milestone updates

Make the migration as safe and smooth as possible."""

        return await self.generate_response(prompt)

    async def _design_migration(self, migration_request: str) -> str:
        """Design system migration architecture"""
        results = []
        results.append("# 🔄 System Migration Design")
        results.append(f"**Migration Request**: {migration_request}")

        # Migration analysis
        analysis = await self._analyze_migration_requirements(migration_request)
        results.append("## 📋 Migration Analysis")
        results.append(analysis)

        # Migration architecture
        architecture = await self._design_migration_architecture(
            migration_request, analysis
        )
        results.append("## 🏗️ Migration Architecture")
        results.append(architecture)

        return "\n\n".join(results)

    async def _analyze_migration_requirements(self, request: str) -> str:
        """Analyze migration requirements and constraints"""
        prompt = f"""Analyze migration requirements: "{request}"

**Migration Scope:**
- What systems/components are being migrated?
- What's the source and target architecture?
- Data migration requirements
- Integration points affected

**Business Constraints:**
- Downtime limitations
- Performance requirements during migration
- Rollback requirements
- Compliance and regulatory needs

**Technical Constraints:**
- Data volume and complexity
- System dependencies
- Resource availability
- Timeline constraints

**Risk Assessment:**
- High-risk areas of the migration
- Data integrity concerns
- Performance impact risks
- Business continuity risks

Provide comprehensive migration requirement analysis."""

        return await self.generate_response(prompt)

    async def _design_migration_architecture(self, request: str, analysis: str) -> str:
        """Design the migration architecture and process"""
        prompt = f"""Design migration architecture for: "{request}"
Based on analysis: "{analysis}"

**Migration Architecture:**
1. **Data Migration Strategy** - How data will be moved and transformed
2. **System Integration** - How old and new systems will coexist
3. **Cutover Strategy** - How to switch from old to new system
4. **Rollback Architecture** - How to reverse migration if needed

**Migration Phases:**
1. **Preparation Phase** - Setup, testing, validation
2. **Pilot Migration** - Limited scope test migration
3. **Full Migration** - Complete system migration
4. **Optimization Phase** - Performance tuning, cleanup

**Technical Implementation:**
- Migration tools and scripts needed
- Infrastructure requirements
- Monitoring and validation approaches
- Performance optimization strategies

Design a robust, safe migration approach."""

        return await self.generate_response(prompt)

    async def _recommend_patterns(self, pattern_request: str) -> str:
        """Recommend architectural patterns for specific needs"""
        prompt = f"""Recommend architectural patterns for: "{pattern_request}"

**Pattern Analysis:**
1. **Problem Context** - What architectural challenge needs to be solved?
2. **Applicable Patterns** - Which patterns could address this need?
3. **Pattern Comparison** - Pros and cons of each applicable pattern
4. **Recommended Pattern** - Best pattern for this specific context

**Implementation Guidance:**
- How to implement the recommended pattern
- Integration with existing architecture
- Code structure and organization
- Testing strategies for the pattern

**Examples and Best Practices:**
- Concrete examples in appropriate language
- Common pitfalls to avoid
- Performance considerations
- Maintenance implications

Provide practical, implementable pattern recommendations."""

        return await self.generate_response(prompt)

    async def _architectural_analysis(self, task: str) -> str:
        """General architectural analysis for unspecified tasks"""
        prompt = f"""Provide architectural analysis for: "{task}"

**Analysis Approach:**
1. **Problem Understanding** - What architectural question or challenge is being addressed?
2. **Context Assessment** - What's the current system context and constraints?
3. **Solution Options** - What are the possible architectural approaches?
4. **Recommendation** - What's the best architectural direction?

**Deliverables:**
- Clear architectural guidance
- Specific recommendations with rationale
- Implementation considerations
- Risk assessment and mitigation

Provide comprehensive architectural insight for this request."""

        return await self.generate_response(prompt)
