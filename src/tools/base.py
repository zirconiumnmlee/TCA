from enum import Enum
from typing import Callable, Dict, Any, Type, List
from abc import ABC, abstractmethod
from functools import wraps


class ToolType(str, Enum):
    CALCULATION = "calculation"
    CONTEXT = "context"
    RETRIEVING = "retrieving"
    TEXT_PROCESSING = "text_processing"


class Tool(ABC):
    name: str
    description: str
    tool_type: ToolType
    execute_input_description: Dict[str, str]
    execute_output_description: Dict[str, str]

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses have required attributes"""
        required_attrs = [
            "name",
            "description",
            "tool_type",
            "execute_input_description",
            "execute_output_description",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise TypeError(
                    f"Tool subclass {cls.__name__} must define '{attr}' attribute"
                )
        super().__init_subclass__(**kwargs)

    @classmethod
    @abstractmethod
    async def execute(cls, agent, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool's function."""
        pass

    @classmethod
    def get_tool_description(cls):
        """Get the description of the tool."""
        description = f"""
        Input：{cls.execute_input_description}
        Description: {cls.description}
        Output：{cls.execute_output_description}
        """

        return description


class ToolsRegistry:
    """Registry for all available tools with selective registration support."""

    _all_tools = {}  # All defined tools
    _active_tools = {}  # Currently active/registered tools
    _enabled_tools = None  # Specific tools to enable (None = all)

    @classmethod
    def register(cls, tool: Type[Tool]):
        """Register a tool to the all_tools registry."""
        if tool.name in cls._all_tools:
            print(f"Tool {tool.name} is already registered. Overwriting.")
        cls._all_tools[tool.name] = tool

        # Auto-activate if no specific tools are enabled or if this tool is enabled
        if cls._enabled_tools is None or tool.name in cls._enabled_tools:
            cls._active_tools[tool.name] = tool

        return tool

    @classmethod
    def set_enabled_tools(cls, tool_names: List[str] = None):
        """
        Set which tools should be enabled. This controls which tools are active.

        Args:
            tool_names: List of tool names to enable. If None, enables all tools.
        """
        cls._enabled_tools = tool_names
        cls._active_tools = {}

        if tool_names is None:
            # Enable all tools
            cls._active_tools = cls._all_tools.copy()
        else:
            # Enable only specified tools
            for tool_name in tool_names:
                if tool_name in cls._all_tools:
                    cls._active_tools[tool_name] = cls._all_tools[tool_name]
                else:
                    print(f"Warning: Tool '{tool_name}' not found in registry.")

    @classmethod
    def get_enabled_tools(cls) -> List[str]:
        """Get list of currently enabled tool names."""
        return list(cls._active_tools.keys())

    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of all available tool names."""
        return list(cls._all_tools.keys())

    @classmethod
    def get_tools(cls, tool_names: List[str] = None) -> Dict[str, Tool]:
        """
        Get tools by name from active tools only.

        Args:
            tool_names: List of tool names to retrieve. If None, returns all active tools.

        Returns:
            Dict[str, Tool]: Dictionary mapping tool names to tool instances

        Raises:
            ValueError: If a requested tool name is not found in active tools
        """
        if not tool_names:
            return cls._active_tools

        tools = {}
        for tool_name in tool_names:
            if tool_name not in cls._active_tools:
                raise ValueError(
                    f"Tool {tool_name} is not in active tools. Active tools: {list(cls._active_tools.keys())}"
                )
            tools[tool_name] = cls._active_tools[tool_name]
        return tools


#def setup_filtering_tools():
    """
    Set up only the three essential filtering tools:
    - exact_string_filter: for exact matching
    - fuzzy_match_filter: for fuzzy matching
    - inverse_filter: for inverse filtering
    """
#    essential_tools = ["exact_string_filter", "fuzzy_match_filter", "inverse_filter"]
#    ToolsRegistry.set_enabled_tools(essential_tools)
#    return ToolsRegistry.get_enabled_tools()
