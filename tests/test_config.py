from hr_assistant.config import AGENTS, SENSITIVITY_LEVELS, SENSITIVE_FIELDS


def test_sensitivity_levels_are_ordered():
    assert SENSITIVITY_LEVELS == ["public", "internal", "sensitive", "clinical"]


def test_all_agents_have_required_keys():
    required = {"sensitivity", "accepts", "type"}
    for name, spec in AGENTS.items():
        assert required.issubset(spec.keys()), f"{name} missing keys"
        assert spec["sensitivity"] in SENSITIVITY_LEVELS


def test_sensitive_fields_declared():
    assert "patient_context" in SENSITIVE_FIELDS
    assert "cpr_number" in SENSITIVE_FIELDS


def test_hr_feed_agents_exist():
    assert "feed_policy" in AGENTS
    assert "feed_benefits" in AGENTS
    assert "feed_collective_agreement" in AGENTS
