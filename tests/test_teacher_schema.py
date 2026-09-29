from llm.schema import teacher_to_soft_labels, validate_teacher_output


class TinyTokenizer:
    def tokenize(self, word):
        return [word.lower()]


def test_teacher_schema_normalises_scores_and_labels():
    output = validate_teacher_output(
        {
            "entities": [
                {"start": 0, "end": 1, "label": "MISC", "image_ids": [0], "confidence": 2.0}
            ],
            "image_scores": [2.0, 0.0],
        },
        token_count=2,
    )
    assert output.entities[0].label == "MIS"
    assert sum(output.image_scores) == 1.0
    assert output.entities[0].confidence == 1.0


def test_teacher_to_soft_labels_marks_bio_positions():
    labels = {
        "O": 0,
        "B-MIS": 1,
        "I-MIS": 2,
        "B-PER": 3,
        "I-PER": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-LOC": 7,
        "I-LOC": 8,
        "X": 9,
    }
    teacher = validate_teacher_output(
        {"entities": [{"start": 0, "end": 2, "label": "PER"}], "image_scores": [1]},
        token_count=2,
    )
    probs, scores = teacher_to_soft_labels(
        teacher, ["Alice", "Smith"], TinyTokenizer(), 8, labels
    )
    assert probs[1][labels["B-PER"]] > probs[1][labels["O"]]
    assert probs[2][labels["I-PER"]] > probs[2][labels["O"]]
    assert scores[0] == 1.0


