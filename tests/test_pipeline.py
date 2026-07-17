"""tests/test_pipeline.py — ingest_single quality-gate behavior (force bypass)."""
from unittest.mock import patch, mock_open

import pipeline


class TestIngestSingleGate:
    def test_below_threshold_skipped_without_force(self):
        # vote_average/vote_count below MIN_INGEST_* → skipped, no I/O.
        assert pipeline.ingest_single(1, vote_average=0.0, vote_count=0) is False

    def test_force_bypasses_quality_gate(self):
        # An explicitly-referenced (obscure) title should ingest despite low votes.
        with patch("pipeline.os.path.exists", return_value=False), \
             patch("pipeline.ingest_movie", return_value={"title": "Obscure", "id": 1}) as mock_im, \
             patch("pipeline.build_richtext", return_value="rich"), \
             patch("pipeline.upsert_movie"), \
             patch("pipeline.update_index"), \
             patch("builtins.open", mock_open()), \
             patch("pipeline.json.dump"):
            result = pipeline.ingest_single(1, vote_average=0.0, vote_count=0, force=True)
        assert result is True
        mock_im.assert_called_once_with(1)
