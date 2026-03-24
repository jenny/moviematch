import pytest
from claude import _filter_results, _extract_result_objects


class TestFilterResults:
    def test_all_valid_titles_pass_through(self):
        results = [
            {"title": "Inception", "explanation": "Mind-bending"},
            {"title": "The Dark Knight", "explanation": "Gritty superhero"},
        ]
        valid = {"Inception", "The Dark Knight"}
        assert _filter_results(results, valid) == results

    def test_hallucinated_title_is_rejected(self):
        results = [
            {"title": "Real Movie", "explanation": "Genuine"},
            {"title": "Fabricated Title", "explanation": "Made up"},
        ]
        assert _filter_results(results, {"Real Movie"}) == [results[0]]

    def test_all_invalid_returns_empty(self):
        results = [{"title": "Fake 1", "explanation": "..."}, {"title": "Fake 2", "explanation": "..."}]
        assert _filter_results(results, {"Real Movie"}) == []

    def test_empty_results_returns_empty(self):
        assert _filter_results([], {"Real Movie"}) == []

    def test_empty_valid_titles_rejects_all(self):
        results = [{"title": "Any Movie", "explanation": "..."}]
        assert _filter_results(results, set()) == []

    def test_none_title_is_rejected(self):
        results = [
            {"title": None, "explanation": "No title"},
            {"title": "Real Movie", "explanation": "Has title"},
        ]
        assert _filter_results(results, {"Real Movie"}) == [results[1]]

    def test_preserves_order(self):
        results = [
            {"title": "B", "explanation": "second"},
            {"title": "A", "explanation": "first"},
        ]
        valid = {"A", "B"}
        assert _filter_results(results, valid) == results


class TestExtractResultObjects:
    def test_single_complete_object(self):
        json_str = '{"results": [{"title": "Inception", "explanation": "Great"}]}'
        result = _extract_result_objects(json_str)
        assert result == ['{"title": "Inception", "explanation": "Great"}']

    def test_multiple_complete_objects(self):
        json_str = '{"results": [{"title": "A", "explanation": "x"}, {"title": "B", "explanation": "y"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 2
        assert '{"title": "A", "explanation": "x"}' in result
        assert '{"title": "B", "explanation": "y"}' in result

    def test_partial_last_object_not_included(self):
        json_str = '{"results": [{"title": "A", "explanation": "x"}, {"title": "B"'
        result = _extract_result_objects(json_str)
        assert len(result) == 1
        assert result[0] == '{"title": "A", "explanation": "x"}'

    def test_empty_string_returns_empty(self):
        assert _extract_result_objects("") == []

    def test_no_complete_objects_returns_empty(self):
        assert _extract_result_objects('{"results": [') == []

    def test_escaped_quote_in_string_not_treated_as_delimiter(self):
        json_str = r'{"results": [{"title": "It\"s Alive", "explanation": "Classic"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 1
        assert r'"It\"s Alive"' in result[0]

    def test_escaped_backslash_handled_correctly(self):
        json_str = r'{"results": [{"title": "AC\\DC", "explanation": "Rock"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 1

    def test_unicode_escape_sequence_handled_correctly(self):
        json_str = r'{"results": [{"title": "Caf\u00e9", "explanation": "French film"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 1

    def test_braces_inside_string_not_counted_as_depth(self):
        json_str = '{"results": [{"title": "A {nested} title", "explanation": "ok"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 1

    def test_extracted_objects_are_valid_json(self):
        import json
        json_str = '{"results": [{"title": "Parasite", "explanation": "Bong Joon-ho masterpiece"}]}'
        result = _extract_result_objects(json_str)
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert parsed["title"] == "Parasite"
        assert parsed["explanation"] == "Bong Joon-ho masterpiece"
