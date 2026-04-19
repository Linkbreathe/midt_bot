from hr_assistant.guards import classify_query_sensitivity


def test_clean_query_is_public():
    assert classify_query_sensitivity("What are the rules for parental leave?") == "public"


def test_danish_cpr_elevates_to_clinical():
    # Danish CPR: 6 digits + dash + 4 digits, or 10 straight digits
    assert classify_query_sensitivity("Patient CPR 0102031234 needs info") == "clinical"
    assert classify_query_sensitivity("CPR 010203-1234 in file") == "clinical"


def test_patient_id_elevates_to_clinical():
    assert classify_query_sensitivity("See patient P-4567 chart") == "clinical"


def test_reason_returned_alongside():
    from hr_assistant.guards import classify_with_reason
    level, reason = classify_with_reason("CPR 0102031234")
    assert level == "clinical"
    assert "CPR" in reason or "cpr" in reason
