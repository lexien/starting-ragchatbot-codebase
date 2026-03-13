import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Outline questions** (what lessons does a course have, what is the course structure, what does the course cover): Use `get_course_outline` — it returns the course title, course link, and the number and title of each lesson.
- **Content questions** (explain a concept, what does lesson X say about Y): Use `search_course_content`
- **Up to 2 sequential tool calls per query** — use a second call only when the first result reveals a need to search a different concept, course, or lesson to fully answer the question. For most queries, one call is sufficient.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline questions**: Call `get_course_outline`, then present the course title, course link, and full lesson list
- **Course content questions**: Call `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Multi-round tool loop (up to MAX_TOOL_ROUNDS)
        rounds_remaining = self.MAX_TOOL_ROUNDS
        while response.stop_reason == "tool_use" and rounds_remaining > 0 and tool_manager:
            rounds_remaining -= 1

            tool_results = self._execute_tool_calls(response, tool_manager)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            next_params = {**self.base_params, "messages": messages, "system": system_content}
            if rounds_remaining > 0:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**next_params)

        return response.content[0].text

    def _execute_tool_calls(self, response, tool_manager) -> list:
        """Execute all tool_use blocks in response, returning tool_result dicts."""
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool error: {str(e)}",
                        "is_error": True,
                    })
        return tool_results