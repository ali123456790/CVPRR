from typing import List, Dict, Optional, Set, Tuple
import math
import time
from dataclasses import dataclass
from collections import defaultdict

from ..core.types import Region, RelationEdge


@dataclass
class GraphNode:
    """Node in the micro scene graph."""
    id: int
    region: Region
    label: str
    clip_score: float
    features: Optional[List[float]] = None
    is_hypothesis: bool = False
    hypothesis_group: Optional[int] = None
    creation_time: float = 0.0


@dataclass 
class MicroGraph:
    """Transient micro scene graph with TTL and K-cap."""
    nodes: List[GraphNode]
    edges: List[RelationEdge]
    k_cap: int
    ttl: float
    creation_time: float


class SceneGraphBuilder:
    """
    Enhanced micro scene graph builder with K-cap, TTL, and hypothesis pruning.
    Implements the Micro-Graph expert (π_G) from the specification.
    """
    
    def __init__(
        self,
        next_to_thresh: float = 0.15,
        k_cap: int = 12,
        ttl_steps: int = 5,
        enable_hypothesis: bool = True
    ):
        self.next_to_thresh = next_to_thresh
        self.k_cap = k_cap
        self.ttl_steps = ttl_steps
        self.enable_hypothesis = enable_hypothesis
        
        # Persistent graph storage with TTL
        self.current_graph: Optional[MicroGraph] = None
        self.steps_since_creation = 0
    
    def _center(self, r: Region) -> Tuple[float, float]:
        """Compute region center."""
        return ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0)
    
    def _compute_area(self, r: Region) -> float:
        """Compute region area."""
        return (r.x2 - r.x1) * (r.y2 - r.y1)
    
    def build(
        self,
        regions: List[Region],
        clip_scores: Optional[List[float]] = None,
        instruction_spans: Optional[List[str]] = None,
        force_rebuild: bool = False
    ) -> List[RelationEdge]:
        """
        Build or update micro scene graph with advanced features.
        
        Args:
            regions: Detected regions
            clip_scores: CLIP scores for each region
            instruction_spans: Relevant spans from instruction
            force_rebuild: Force graph rebuild even if TTL hasn't expired
            
        Returns:
            List of relation edges
        """
        # Check if we should use existing graph
        if (self.current_graph is not None and 
            self.steps_since_creation < self.ttl_steps and
            not force_rebuild):
            self.steps_since_creation += 1
            return self.current_graph.edges
        
        # Build new graph
        current_time = time.time()
        
        # Apply K-cap: select top-K regions by relevance
        selected_regions, selected_scores = self._apply_k_cap(
            regions, clip_scores
        )
        
        # Create nodes
        nodes = []
        for i, (region, score) in enumerate(zip(selected_regions, selected_scores)):
            node = GraphNode(
                id=i,
                region=region,
                label=region.label,
                clip_score=score,
                creation_time=current_time
            )
            nodes.append(node)
        
        # Add hypothesis nodes if enabled
        if self.enable_hypothesis and instruction_spans:
            hypothesis_nodes = self._create_hypothesis_nodes(
                nodes, instruction_spans
            )
            nodes.extend(hypothesis_nodes)
        
        # Build edges with enhanced relations
        edges = self._build_enhanced_edges(nodes)
        
        # Prune inconsistent hypotheses
        if self.enable_hypothesis:
            nodes, edges = self._prune_hypotheses(nodes, edges)
        
        # Store graph
        self.current_graph = MicroGraph(
            nodes=nodes,
            edges=edges,
            k_cap=self.k_cap,
            ttl=self.ttl_steps,
            creation_time=current_time
        )
        self.steps_since_creation = 0
        
        return edges
    
    def _apply_k_cap(
        self,
        regions: List[Region],
        clip_scores: Optional[List[float]] = None
    ) -> Tuple[List[Region], List[float]]:
        """
        Apply K-cap to limit number of nodes.
        Select top-K by CLIP score or area if scores unavailable.
        """
        if not regions:
            return [], []
        
        if clip_scores and len(clip_scores) == len(regions):
            # Sort by CLIP score
            paired = list(zip(regions, clip_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
        else:
            # Fallback: sort by area (larger objects more relevant)
            paired = [(r, self._compute_area(r)) for r in regions]
            paired.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K
        k = min(self.k_cap, len(paired))
        selected = paired[:k]
        
        selected_regions = [r for r, _ in selected]
        selected_scores = [s for _, s in selected]
        
        return selected_regions, selected_scores
    
    def _create_hypothesis_nodes(
        self,
        nodes: List[GraphNode],
        instruction_spans: List[str]
    ) -> List[GraphNode]:
        """
        Create hypothesis nodes for ambiguous referents.
        Multiple nodes can match the same referent phrase.
        """
        hypothesis_nodes = []
        hypothesis_id = len(nodes)
        
        # Group nodes by label similarity
        label_groups = defaultdict(list)
        for node in nodes:
            base_label = node.label.split('_')[0].lower()
            label_groups[base_label].append(node)
        
        # Check instruction spans for ambiguous references
        for span in instruction_spans:
            span_lower = span.lower()
            
            # Find matching groups
            for label, group_nodes in label_groups.items():
                if label in span_lower and len(group_nodes) > 1:
                    # Create hypothesis node for this ambiguous group
                    for i, node in enumerate(group_nodes):
                        hyp_node = GraphNode(
                            id=hypothesis_id,
                            region=node.region,
                            label=f"{node.label}_hyp_{i}",
                            clip_score=node.clip_score,
                            is_hypothesis=True,
                            hypothesis_group=hash(span) % 1000,
                            creation_time=node.creation_time
                        )
                        hypothesis_nodes.append(hyp_node)
                        hypothesis_id += 1
        
        return hypothesis_nodes
    
    def _build_enhanced_edges(self, nodes: List[GraphNode]) -> List[RelationEdge]:
        """
        Build edges with multiple relation types.
        Includes: left_of, right_of, above, below, in, near, next_to
        """
        edges = []
        
        for i, node_a in enumerate(nodes):
            a = node_a.region
            ax, ay = self._center(a)
            a_area = self._compute_area(a)
            
            for j, node_b in enumerate(nodes):
                if i == j:
                    continue
                
                b = node_b.region
                bx, by = self._center(b)
                b_area = self._compute_area(b)
                
                dx, dy = bx - ax, by - ay
                dist = math.hypot(dx, dy)
                
                # Horizontal relations
                if abs(dx) > abs(dy) * 0.5:  # Significantly horizontal
                    if bx < ax:
                        conf = min(1.0, abs(dx) / max(1.0, dist))
                        edges.append(RelationEdge(i, j, "left_of", conf))
                    else:
                        conf = min(1.0, abs(dx) / max(1.0, dist))
                        edges.append(RelationEdge(i, j, "right_of", conf))
                
                # Vertical relations
                if abs(dy) > abs(dx) * 0.5:  # Significantly vertical
                    if by < ay:
                        conf = min(1.0, abs(dy) / max(1.0, dist))
                        edges.append(RelationEdge(i, j, "above", conf))
                    else:
                        conf = min(1.0, abs(dy) / max(1.0, dist))
                        edges.append(RelationEdge(i, j, "below", conf))
                
                # Containment relation
                if self._is_contained(a, b):
                    edges.append(RelationEdge(i, j, "in", 0.9))
                elif self._is_contained(b, a):
                    edges.append(RelationEdge(j, i, "in", 0.9))
                
                # Proximity relations
                diag_a = math.hypot(a.x2 - a.x1, a.y2 - a.y1)
                normalized_dist = dist / max(1.0, diag_a)
                
                if normalized_dist < self.next_to_thresh:
                    conf = 1.0 - normalized_dist / self.next_to_thresh
                    edges.append(RelationEdge(i, j, "next_to", conf))
                
                if normalized_dist < 0.5:
                    conf = 1.0 - normalized_dist * 2
                    edges.append(RelationEdge(i, j, "near", conf))
        
        return edges
    
    def _is_contained(self, inner: Region, outer: Region) -> bool:
        """Check if inner region is contained in outer region."""
        return (inner.x1 >= outer.x1 and inner.x2 <= outer.x2 and
                inner.y1 >= outer.y1 and inner.y2 <= outer.y2)
    
    def _prune_hypotheses(
        self,
        nodes: List[GraphNode],
        edges: List[RelationEdge]
    ) -> Tuple[List[GraphNode], List[RelationEdge]]:
        """
        Prune inconsistent hypothesis nodes based on relations and CLIP scores.
        """
        if not any(n.is_hypothesis for n in nodes):
            return nodes, edges
        
        # Group hypotheses by their group ID
        hypothesis_groups = defaultdict(list)
        for i, node in enumerate(nodes):
            if node.is_hypothesis:
                hypothesis_groups[node.hypothesis_group].append(i)
        
        nodes_to_remove = set()
        
        for group_id, node_indices in hypothesis_groups.items():
            if len(node_indices) <= 1:
                continue
            
            # Score each hypothesis based on consistency
            scores = []
            for idx in node_indices:
                score = self._score_hypothesis(idx, nodes, edges)
                scores.append((idx, score))
            
            # Keep only the best hypothesis
            scores.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in scores[1:]:
                nodes_to_remove.add(idx)
        
        # Remove pruned nodes and their edges
        filtered_nodes = [n for i, n in enumerate(nodes) if i not in nodes_to_remove]
        
        # Remap indices
        index_map = {}
        new_idx = 0
        for old_idx in range(len(nodes)):
            if old_idx not in nodes_to_remove:
                index_map[old_idx] = new_idx
                new_idx += 1
        
        # Filter and remap edges
        filtered_edges = []
        for edge in edges:
            if edge.from_idx not in nodes_to_remove and edge.to_idx not in nodes_to_remove:
                new_edge = RelationEdge(
                    index_map[edge.from_idx],
                    index_map[edge.to_idx],
                    edge.relation,
                    edge.confidence
                )
                filtered_edges.append(new_edge)
        
        return filtered_nodes, filtered_edges
    
    def _score_hypothesis(
        self,
        node_idx: int,
        nodes: List[GraphNode],
        edges: List[RelationEdge]
    ) -> float:
        """
        Score a hypothesis node based on consistency with relations and CLIP score.
        """
        node = nodes[node_idx]
        score = node.clip_score  # Start with CLIP score
        
        # Check relation consistency
        for edge in edges:
            if edge.from_idx == node_idx or edge.to_idx == node_idx:
                # Boost score for consistent relations
                score += edge.confidence * 0.1
        
        return score
    
    def get_action_override(
        self,
        chosen_node: Optional[GraphNode],
        instruction: str
    ) -> Optional[str]:
        """
        Generate action override based on chosen node.
        
        Args:
            chosen_node: Selected node from graph
            instruction: Current instruction
            
        Returns:
            Action string or None
        """
        if not chosen_node:
            return None
        
        # Simple heuristic for action generation
        cx, cy = self._center(chosen_node.region)
        
        # Image assumed to be 640x480
        if cx < 213:  # Left third
            return "turn_left"
        elif cx > 427:  # Right third
            return "turn_right"
        else:
            return "forward"
    
    def reset(self):
        """Reset the graph (call at episode boundaries)."""
        self.current_graph = None
        self.steps_since_creation = 0
