"""Prompt construction: what the model is told when context is missing.

The model's actual wording needs a live LLM (see the fallback checks in the
README); these cover the deterministic half — that an empty retrieval is stated
plainly rather than passed off as an empty context block.
"""

from rag_app.rag import SYSTEM_PROMPT, RAGPipeline
from rag_app.vectorstore import RetrievedChunk


def _build(question, chunks, turns=()):
    # _build_prompt only reads its arguments, so it needs no configured pipeline.
    return RAGPipeline._build_prompt(None, question, list(chunks), list(turns))


def _chunk(text, source):
    return RetrievedChunk(text=text, score=0.9, source=source, metadata={})


def test_retrieved_chunks_are_labelled_with_their_source():
    prompt = _build("revenue?", [_chunk("revenue grew 12%", "report.pdf")])
    assert "Context:" in prompt
    assert "[Source: report.pdf]" in prompt
    assert "revenue grew 12%" in prompt
    assert "Question: revenue?" in prompt


def test_empty_retrieval_states_the_absence_instead_of_an_empty_context():
    """An empty context block reads as 'context exists and is silent'.

    The model has to know retrieval came back with nothing, or it can't tell
    that this is the fall-back-to-general-knowledge case.
    """
    prompt = _build("what is a transformer?", [])
    assert "No context was retrieved" in prompt
    assert "Context:" not in prompt
    assert "(no context found)" not in prompt  # the old placeholder
    assert "Question: what is a transformer?" in prompt


def test_history_is_included_for_follow_ups_with_or_without_context():
    turns = [("user", "who wrote it?"), ("assistant", "Kim")]
    with_context = _build("and when?", [_chunk("text", "a.txt")], turns)
    without_context = _build("and when?", [], turns)
    for prompt in (with_context, without_context):
        assert "Conversation so far:" in prompt
        assert "user: who wrote it?" in prompt


def test_system_prompt_asks_for_a_fallback_answer_and_an_upload():
    lowered = SYSTEM_PROMPT.lower()
    assert "general knowledge" in lowered
    assert "upload" in lowered
    # It must still refuse to dress general knowledge up as a citation.
    assert "never cite a filename" in lowered
