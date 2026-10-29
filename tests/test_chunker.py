from docflow.document_service import ingest


def test_short_text_is_one_chunk():
    assert ingest.split("Hello world.") == ["Hello world."]


def test_blank_text_gives_no_chunks():
    assert ingest.split("  \n ") == []


def test_chunks_fit_and_nothing_is_lost():
    text = " ".join(f"Sentence number {i} says something." for i in range(1, 201))
    chunks = ingest.split(text, max_chars=300, overlap=40)
    assert len(chunks) > 5 and all(len(c) <= 300 for c in chunks)
    for i in range(1, 201):
        assert any(f"Sentence number {i} says" in c for c in chunks)


def test_paragraphs_stay_together_when_they_fit():
    text = "First paragraph.\n\nSecond paragraph.\n\n" + "word " * 100
    assert "First paragraph.\n\nSecond paragraph." in ingest.split(text, max_chars=100, overlap=0)[0]


def test_overlap_repeats_the_end_of_the_previous_chunk():
    text = " ".join(f"w{i}" for i in range(1, 61))
    a, b, *_ = ingest.split(text, max_chars=80, overlap=20)
    assert a.split(" ")[-1] in b


def test_text_without_separators_is_cut_by_characters():
    assert [len(c) for c in ingest.split("a" * 250, max_chars=100, overlap=0)] == [100, 100, 50]


def test_a_canary_sentence_survives_chunking_whole():
    text = ("Filler sentence. " * 80) + "\n\nReference code CANARY-doc-0001-v2-working.\n\n" + ("More filler. " * 80)
    assert sum("CANARY-doc-0001-v2-working" in c for c in ingest.split(text)) >= 1
