"""Constants shared by the API tests."""
SECRET_KEY = "sk-test-secret-123456"
EMP = {"X-Demo-Role": "employee", "X-Demo-User": "alice"}
EMP2 = {"X-Demo-Role": "employee", "X-Demo-User": "bob"}
ADM = {"X-Demo-Role": "admin", "X-Demo-User": "root"}

CLS_INTERNAL = dict(intent="internal_it", confidence=0.9)
GRADE_OK = dict(sufficient=True, confidence=0.9, supporting_refs=[1], recommended_action="answer")
DRAFT = dict(can_answer=True, answer="Open the SecureLink client and connect [1].")
OK_CHECK = dict(grounded=True)
