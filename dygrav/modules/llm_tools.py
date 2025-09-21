"""
LLM-tools expert for symbolic reasoning with budget constraints.
Provides structured API for navigation-specific queries.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np

from ..core.types import Region


class ToolType(Enum):
    """Available tool types for LLM."""
    COUNT = "count"
    IS_ABOVE = "is_above"
    IS_BELOW = "is_below"
    SHORTEST_PATH = "shortest_path"
    DISAMBIGUATE = "disambiguate"
    VERIFY = "verify"
    SPATIAL_RELATION = "spatial_relation"


@dataclass
class ToolCall:
    """Structured tool call."""
    tool: ToolType
    args: Dict[str, Any]
    max_tokens: int = 16


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    value: Any
    tokens_used: int
    latency_ms: float
    fallback_used: bool = False


class LLMToolsExpert:
    """
    LLM-tools expert with structured API and budget constraints.
    Uses deterministic fallbacks when LLM is unavailable or times out.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 48,
        timeout_ms: float = 50.0,
        use_llm: bool = False  # Set to True when LLM is available
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout_ms = timeout_ms
        self.use_llm = use_llm
        
        # Initialize LLM client if available
        self.llm_client = None
        if use_llm:
            try:
                # Placeholder for actual LLM initialization
                # from openai import OpenAI
                # self.llm_client = OpenAI()
                pass
            except Exception:
                self.use_llm = False
        
        # Tool schemas for structured output
        self.tool_schemas = self._init_tool_schemas()
    
    def _init_tool_schemas(self) -> Dict[ToolType, Dict]:
        """Initialize tool schemas for structured API."""
        return {
            ToolType.COUNT: {
                "description": "Count objects in a specific direction",
                "parameters": {
                    "object_type": "string",
                    "direction": "string (left/right/forward/all)"
                },
                "returns": "integer"
            },
            ToolType.IS_ABOVE: {
                "description": "Check if object A is above object B",
                "parameters": {
                    "object_a": "string",
                    "object_b": "string"
                },
                "returns": "boolean"
            },
            ToolType.IS_BELOW: {
                "description": "Check if object A is below object B",
                "parameters": {
                    "object_a": "string",
                    "object_b": "string"
                },
                "returns": "boolean"
            },
            ToolType.SHORTEST_PATH: {
                "description": "Find shortest path to target",
                "parameters": {
                    "target": "string",
                    "avoid": "list[string] (optional)"
                },
                "returns": "list[action]"
            },
            ToolType.DISAMBIGUATE: {
                "description": "Disambiguate between similar objects",
                "parameters": {
                    "objects": "list[object]",
                    "criteria": "string"
                },
                "returns": "object_id"
            },
            ToolType.VERIFY: {
                "description": "Verify an attribute of an object",
                "parameters": {
                    "object": "string",
                    "attribute": "string"
                },
                "returns": "boolean"
            },
            ToolType.SPATIAL_RELATION: {
                "description": "Check spatial relation between objects",
                "parameters": {
                    "object_a": "string",
                    "object_b": "string",
                    "relation": "string"
                },
                "returns": "boolean"
            }
        }
    
    def parse_instruction(self, instruction: str, context: Dict) -> List[ToolCall]:
        """
        Parse instruction to identify needed tool calls.
        
        Args:
            instruction: Navigation instruction
            context: Current context (regions, scene info)
            
        Returns:
            List of tool calls to execute
        """
        tool_calls = []
        instruction_lower = instruction.lower()
        
        # Pattern matching for tool identification
        if "count" in instruction_lower or "how many" in instruction_lower:
            # Extract what to count
            tokens = instruction_lower.split()
            object_type = self._extract_object_type(tokens)
            direction = self._extract_direction(tokens)
            
            tool_calls.append(ToolCall(
                tool=ToolType.COUNT,
                args={"object_type": object_type, "direction": direction},
                max_tokens=16
            ))
        
        if "above" in instruction_lower:
            objects = self._extract_objects(instruction_lower)
            if len(objects) >= 2:
                tool_calls.append(ToolCall(
                    tool=ToolType.IS_ABOVE,
                    args={"object_a": objects[0], "object_b": objects[1]},
                    max_tokens=8
                ))
        
        if "below" in instruction_lower or "under" in instruction_lower:
            objects = self._extract_objects(instruction_lower)
            if len(objects) >= 2:
                tool_calls.append(ToolCall(
                    tool=ToolType.IS_BELOW,
                    args={"object_a": objects[0], "object_b": objects[1]},
                    max_tokens=8
                ))
        
        if "shortest" in instruction_lower or "quickest" in instruction_lower:
            target = self._extract_target(instruction_lower)
            tool_calls.append(ToolCall(
                tool=ToolType.SHORTEST_PATH,
                args={"target": target},
                max_tokens=32
            ))
        
        # Check for disambiguation needs from context
        if "regions" in context and len(context["regions"]) > 1:
            similar_regions = self._find_similar_regions(context["regions"])
            if similar_regions:
                tool_calls.append(ToolCall(
                    tool=ToolType.DISAMBIGUATE,
                    args={"objects": similar_regions, "criteria": instruction},
                    max_tokens=16
                ))
        
        return tool_calls
    
    def execute_tool(
        self,
        tool_call: ToolCall,
        context: Dict,
        regions: List[Region]
    ) -> ToolResult:
        """
        Execute a tool call with budget constraints.
        
        Args:
            tool_call: Tool to execute
            context: Current context
            regions: Detected regions
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        
        # Check token budget
        if tool_call.max_tokens > self.max_tokens:
            tool_call.max_tokens = self.max_tokens
        
        # Try LLM execution first if available
        if self.use_llm and self.llm_client:
            result = self._execute_with_llm(tool_call, context)
            if result and (time.time() - start_time) * 1000 < self.timeout_ms:
                latency_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    success=True,
                    value=result,
                    tokens_used=tool_call.max_tokens,
                    latency_ms=latency_ms,
                    fallback_used=False
                )
        
        # Fallback to deterministic implementation
        result = self._execute_fallback(tool_call, context, regions)
        latency_ms = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            value=result,
            tokens_used=0,  # No tokens for fallback
            latency_ms=latency_ms,
            fallback_used=True
        )
    
    def _execute_with_llm(self, tool_call: ToolCall, context: Dict) -> Optional[Any]:
        """Execute tool call using LLM (placeholder)."""
        # This would contain actual LLM API calls
        # For now, return None to trigger fallback
        return None
    
    def _execute_fallback(
        self,
        tool_call: ToolCall,
        context: Dict,
        regions: List[Region]
    ) -> Any:
        """Deterministic fallback implementations."""
        
        if tool_call.tool == ToolType.COUNT:
            return self._count_fallback(
                regions,
                tool_call.args.get("object_type"),
                tool_call.args.get("direction")
            )
        
        elif tool_call.tool == ToolType.IS_ABOVE:
            return self._is_above_fallback(
                regions,
                tool_call.args.get("object_a"),
                tool_call.args.get("object_b")
            )
        
        elif tool_call.tool == ToolType.IS_BELOW:
            return self._is_below_fallback(
                regions,
                tool_call.args.get("object_a"),
                tool_call.args.get("object_b")
            )
        
        elif tool_call.tool == ToolType.SHORTEST_PATH:
            return self._shortest_path_fallback(
                context,
                tool_call.args.get("target")
            )
        
        elif tool_call.tool == ToolType.DISAMBIGUATE:
            return self._disambiguate_fallback(
                tool_call.args.get("objects"),
                tool_call.args.get("criteria")
            )
        
        elif tool_call.tool == ToolType.VERIFY:
            return self._verify_fallback(
                regions,
                tool_call.args.get("object"),
                tool_call.args.get("attribute")
            )
        
        elif tool_call.tool == ToolType.SPATIAL_RELATION:
            return self._spatial_relation_fallback(
                regions,
                tool_call.args.get("object_a"),
                tool_call.args.get("object_b"),
                tool_call.args.get("relation")
            )
        
        return None
    
    def _count_fallback(
        self,
        regions: List[Region],
        object_type: str,
        direction: str
    ) -> int:
        """Count objects using geometry."""
        count = 0
        
        for region in regions:
            # Check if region matches object type
            if object_type and object_type in region.label.lower():
                # Check direction constraint
                if direction == "all":
                    count += 1
                elif direction == "left" and region.x1 < 320:  # Assuming 640 width
                    count += 1
                elif direction == "right" and region.x2 > 320:
                    count += 1
                elif direction == "forward" and region.y1 < 240:  # Assuming 480 height
                    count += 1
        
        return count
    
    def _is_above_fallback(
        self,
        regions: List[Region],
        object_a: str,
        object_b: str
    ) -> bool:
        """Check if A is above B using bounding boxes."""
        region_a = self._find_region(regions, object_a)
        region_b = self._find_region(regions, object_b)
        
        if region_a and region_b:
            # A is above B if A's bottom is higher than B's top
            return region_a.y2 < region_b.y1
        
        return False
    
    def _is_below_fallback(
        self,
        regions: List[Region],
        object_a: str,
        object_b: str
    ) -> bool:
        """Check if A is below B using bounding boxes."""
        return self._is_above_fallback(regions, object_b, object_a)
    
    def _shortest_path_fallback(
        self,
        context: Dict,
        target: str
    ) -> List[str]:
        """Simple shortest path heuristic."""
        # Very simplified - just return basic actions
        return ["forward", "turn_right", "forward"]
    
    def _disambiguate_fallback(
        self,
        objects: List[Any],
        criteria: str
    ) -> int:
        """Disambiguate based on simple heuristics."""
        if not objects:
            return -1
        
        # Simple heuristic: choose the first one
        # In practice, would use more sophisticated logic
        return 0
    
    def _verify_fallback(
        self,
        regions: List[Region],
        object_name: str,
        attribute: str
    ) -> bool:
        """Verify object attribute using simple checks."""
        region = self._find_region(regions, object_name)
        
        if not region:
            return False
        
        # Simple attribute checks
        if attribute == "large":
            area = (region.x2 - region.x1) * (region.y2 - region.y1)
            return area > 10000  # Threshold for "large"
        elif attribute == "small":
            area = (region.x2 - region.x1) * (region.y2 - region.y1)
            return area < 5000
        elif attribute == "centered":
            center_x = (region.x1 + region.x2) / 2
            return abs(center_x - 320) < 50  # Near center
        
        return False
    
    def _spatial_relation_fallback(
        self,
        regions: List[Region],
        object_a: str,
        object_b: str,
        relation: str
    ) -> bool:
        """Check spatial relation using geometry."""
        region_a = self._find_region(regions, object_a)
        region_b = self._find_region(regions, object_b)
        
        if not region_a or not region_b:
            return False
        
        if relation == "left_of":
            return region_a.x2 < region_b.x1
        elif relation == "right_of":
            return region_a.x1 > region_b.x2
        elif relation == "near":
            dist = self._compute_distance(region_a, region_b)
            return dist < 100  # Pixel threshold
        elif relation == "far":
            dist = self._compute_distance(region_a, region_b)
            return dist > 200
        
        return False
    
    def _find_region(self, regions: List[Region], object_name: str) -> Optional[Region]:
        """Find region by object name."""
        for region in regions:
            if object_name.lower() in region.label.lower():
                return region
        return None
    
    def _compute_distance(self, region_a: Region, region_b: Region) -> float:
        """Compute distance between region centers."""
        center_a = ((region_a.x1 + region_a.x2) / 2, (region_a.y1 + region_a.y2) / 2)
        center_b = ((region_b.x1 + region_b.x2) / 2, (region_b.y1 + region_b.y2) / 2)
        
        dist = np.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)
        return dist
    
    def _extract_object_type(self, tokens: List[str]) -> str:
        """Extract object type from tokens."""
        objects = ["door", "chair", "table", "window", "stairs", "painting"]
        for token in tokens:
            if token in objects:
                return token
        return "object"
    
    def _extract_direction(self, tokens: List[str]) -> str:
        """Extract direction from tokens."""
        directions = ["left", "right", "forward", "ahead"]
        for token in tokens:
            if token in directions:
                return token
        return "all"
    
    def _extract_target(self, instruction: str) -> str:
        """Extract navigation target from instruction."""
        # Simple extraction - look for nouns after "to"
        tokens = instruction.lower().split()
        if "to" in tokens:
            idx = tokens.index("to")
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
        return "goal"
    
    def _extract_objects(self, instruction: str) -> List[str]:
        """Extract object names from instruction."""
        objects = []
        known_objects = ["door", "chair", "table", "window", "stairs", "painting", "lamp"]
        
        for obj in known_objects:
            if obj in instruction:
                objects.append(obj)
        
        return objects
    
    def _find_similar_regions(self, regions: List[Region]) -> List[Region]:
        """Find regions with similar labels."""
        if len(regions) < 2:
            return []
        
        # Group by label
        label_groups = {}
        for region in regions:
            label = region.label.lower()
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(region)
        
        # Find groups with multiple items
        similar = []
        for label, group in label_groups.items():
            if len(group) > 1:
                similar.extend(group)
        
        return similar
    
    def generate_action_bias(
        self,
        tool_results: List[ToolResult],
        num_actions: int = 6
    ) -> torch.Tensor:
        """
        Generate action bias based on tool results.
        
        Args:
            tool_results: Results from executed tools
            num_actions: Number of actions in action space
            
        Returns:
            Action bias tensor
        """
        bias = torch.zeros(num_actions)
        
        for result in tool_results:
            if not result.success:
                continue
            
            # Map tool results to action biases
            if isinstance(result.value, list) and "forward" in result.value:
                bias[0] += 0.5  # Bias forward action
            
            if isinstance(result.value, bool):
                if result.value:
                    bias[1] += 0.3  # Positive result biases turn
            
            if isinstance(result.value, int) and result.value > 2:
                bias[2] += 0.4  # Multiple objects bias stop
        
        return bias


def create_llm_tools_expert(token_budget: int = 48) -> LLMToolsExpert:
    """
    Factory function to create LLM-tools expert with specified budget.
    
    Args:
        token_budget: Maximum tokens (16 or 48)
        
    Returns:
        Configured LLMToolsExpert
    """
    # Adjust timeout based on budget
    if token_budget <= 16:
        timeout_ms = 20.0
    else:
        timeout_ms = 50.0
    
    return LLMToolsExpert(
        max_tokens=token_budget,
        timeout_ms=timeout_ms
    )
