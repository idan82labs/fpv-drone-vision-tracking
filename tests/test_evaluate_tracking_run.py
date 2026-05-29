import unittest

from scripts.evaluate_tracking_run import row_is_selected


class EvaluateTrackingRunTests(unittest.TestCase):
    def test_row_is_selected_filters_explicit_unselected_blank_rows(self):
        self.assertFalse(row_is_selected({"frame": "1", "selected": "0", "x": "", "y": ""}))
        self.assertFalse(row_is_selected({"frame": "1", "selected": "false", "x": "4", "y": "5"}))

    def test_row_is_selected_keeps_detector_rows_without_selected_column(self):
        self.assertTrue(row_is_selected({"frame": "1", "x": "4", "y": "5"}))


if __name__ == "__main__":
    unittest.main()
