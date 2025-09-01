from dygrav.modules.ambiguity import AmbiguityDetector
from dygrav.core.types import Region

def test_ambiguity_triggers_on_low_conf_and_multi_candidates():
    amb = AmbiguityDetector(tau_conf=0.6, tau_entropy=1.0, max_candidates=2)
    regions = [Region(0,0,10,10,0.9), Region(20,20,30,30,0.8)]
    dec = amb(policy_confidence=0.5, attention_entropy=0.5, textual_attribute_present=True, candidate_regions=regions)
    assert dec.trigger
    assert "low_policy_conf" in dec.reason
    assert "multi_attr_candidates" in dec.reason
    assert len(dec.candidates) == 2
