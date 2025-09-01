from dygrav.modules.scene_graph import SceneGraphBuilder
from dygrav.core.types import Region

def test_scene_graph_edges_present():
    sg = SceneGraphBuilder(next_to_thresh=10.0)  # huge to force next_to
    regs = [
        Region(0,0,10,10,0.9),   # center ~ (5,5)
        Region(12,0,22,10,0.9),  # center ~ (17,5)
        Region(30,0,40,10,0.9),  # center ~ (35,5)
    ]
    edges = sg.build(regs)
    rel_types = {e.rel_type for e in edges}
    assert "left_of" in rel_types or "right_of" in rel_types
    assert "next_to" in rel_types
    assert all(e.src_idx != e.dst_idx for e in edges)
