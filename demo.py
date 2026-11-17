from score_pair import PLAN, score_pair

# Synthetic example records only; no customer data is included in this repository.
record_a = {
    "name": "S. Example",
    "email": "sam.example80@example.com",
    "phone": "+91 98765 43210",
    "city": "Bengaluru",
    "pin_code": "560001",
    "date_of_birth": "23/11/1994",
}
record_b = {
    "name": "Sam Example",
    "email": "sam.examples81@example.com",
    "phone": "9123456789",
    "city": "Mysuru",
    "pin_code": "570001",
    "date_of_birth": "10-Feb-1978",
}

result = score_pair(record_a, record_b, PLAN)
print(f"Confidence: {result.confidence:.3f}")
print(f"Algorithm: {result.algorithm}")
print(f"Field scores: {result.field_scores}")
