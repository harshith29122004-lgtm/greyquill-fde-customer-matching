import unittest
from score_pair import FieldComparison, PLAN, score_pair


class ScorePairTests(unittest.TestCase):
    def test_missing_fields_are_excluded_and_weights_renormalized(self):
        plan = [FieldComparison("name", "exact", 0.25), FieldComparison("email", "exact", 0.75)]
        result = score_pair({"name": "Asha", "email": ""}, {"name": "asha", "email": "x"}, plan)
        self.assertEqual(result.field_scores, {"name": 1.0})
        self.assertEqual(result.confidence, 1.0)

    def test_no_comparable_fields(self):
        result = score_pair({"name": ""}, {"name": None}, PLAN)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.field_scores, {})

    def test_phone_normalization(self):
        plan = [FieldComparison("phone", "normalized_phone", 1.0)]
        result = score_pair({"phone": "+91 98765-43210"}, {"phone": "09876543210"}, plan)
        self.assertEqual(result.confidence, 1.0)

    def test_date_formats_represent_same_date(self):
        plan = [FieldComparison("dob", "date", 1.0)]
        result = score_pair({"dob": "19/01/1965"}, {"dob": "1965-01-19"}, plan)
        self.assertEqual(result.confidence, 1.0)

    def test_exact_ignores_case_and_outer_spaces(self):
        plan = [FieldComparison("city", "exact", 1.0)]
        result = score_pair({"city": " Bengaluru "}, {"city": "bengaluru"}, plan)
        self.assertEqual(result.confidence, 1.0)

    def test_name_order_and_initial(self):
        plan = [FieldComparison("name", "phonetic", 1.0)]
        reordered = score_pair({"name": "Patel, Fatima"}, {"name": "Fathima Patel"}, plan)
        initial = score_pair({"name": "S. Chatterjee"}, {"name": "Swathi Chatterjee"}, plan)
        self.assertGreaterEqual(reordered.confidence, 0.8)
        self.assertGreaterEqual(initial.confidence, 0.8)

    def test_result_is_bounded(self):
        result = score_pair(
            {"name": "Asha Rao", "email": "asha@example.com"},
            {"name": "Asha R", "email": "asha2@example.com"},
            PLAN,
        )
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
