"""The replay corpus, its two transports, and the config that selects them (#510).

``tests/integration/test_shipped_cli_hermetic_forecast.py`` proves the whole
composition through ``windbreak run``. This module is the seam beneath it: what
a corpus refuses to load, what each transport does on a hit and on a miss, and
what ``forecast.replay_corpus`` resolves to -- because an end-to-end test that
passes tells you the happy path works and nothing at all about the branches it
never takes.

The two facts most worth stating up front:

* **A miss degrades; it does not raise out of the tick.** An unrecorded
  subquestion returns no candidates (so the forecast abstains on zero verified
  citations), an unrecorded URL raises :class:`OSError` (which the gather and
  the citation re-check both already treat as an unreachable page), and an
  unrecorded ensemble member raises a
  :class:`~windbreak.forecast.providers.base.ProviderVoteError` (which the
  pipeline discards per-vote). Failing closed on the *capability*, never on the
  process, is PR #487's standard and each of the three is pinned below.
* **A malformed corpus is different in kind and refuses at load.** An operator
  who pointed at the wrong directory must not get a corpus that reads as
  legitimately knowing nothing.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from windbreak.config.schema import (
    PROVIDER_TRANSPORT_LIVE,
    REPLAY_CORPUS_DISABLED,
    REPLAY_CORPUS_REPLAY,
    UNCONFIGURED_PLACEHOLDER,
    ForecastConfig,
    ProviderTransportConfig,
    ReplayCorpusConfig,
    WindbreakConfig,
)
from windbreak.forecast.cassettes import LlmRequest
from windbreak.forecast.corpus import (
    CORPUS_RESEARCH_FILENAME,
    CORPUS_VOTES_FILENAME,
    PROVIDER_FAILURE_CORPUS_MISS,
    CorpusFormatError,
    CorpusResearchTransport,
    CorpusVoteTransport,
    ProviderCorpusMissError,
    load_replay_corpus,
    vote_key,
)
from windbreak.forecast.providers.base import (
    NO_RESPONSE_FINGERPRINT,
    ProviderVoteError,
)
from windbreak.forecast.sandbox import EgressDeniedError
from windbreak.scheduler.loop import _resolve_replay_corpus
from windbreak.scheduler.provider_wiring import (
    REPLAY_CORPUS_SOURCE_CONFIGURED,
    REPLAY_CORPUS_SOURCE_DEFAULT,
    LiveProviderHttp,
    build_corpus_research_tools,
    build_corpus_vote_transport,
    replay_corpus_directory,
    replay_corpus_source,
)

#: The repository root, resolved from this file rather than the process's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed corpus the shipped hermetic demonstration replays.
DEMO_CORPUS = REPO_ROOT / "tests" / "fixtures" / "forecast" / "hermetic_corpus"

#: The demonstration market's title, and therefore the tail of every recorded
#: subquestion `decompose_subquestions` builds for it.
DEMO_TITLE = "Hermetic demonstration market"

#: One recorded subquestion, spelled as the pipeline spells it.
RECORDED_QUERY = f"Base rate for {DEMO_TITLE}"

#: A subquestion no corpus here records.
UNRECORDED_QUERY = "Base rate for some entirely different market"

#: A URL no corpus here records, on a host it holds nothing for.
UNRECORDED_URL = "https://elsewhere.example/article"

#: An ensemble member the committed corpus records no vote for.
UNRECORDED_MEMBER = ("openai", "gpt-4o-2024-11-20")


def _corpus_copy(tmp_path: Path) -> Path:
    """Copy the committed corpus so a test can spoil one leaf of it.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The private copy.
    """
    corpus = tmp_path / "corpus"
    shutil.copytree(DEMO_CORPUS, corpus)
    return corpus


def _rewrite_research(corpus: Path, mutate: object) -> None:
    """Apply ``mutate`` to a corpus copy's parsed ``research.json`` in place.

    Args:
        corpus: The corpus copy to rewrite.
        mutate: A callable taking the parsed document and mutating it.
    """
    path = corpus / CORPUS_RESEARCH_FILENAME
    research = json.loads(path.read_text(encoding="utf-8"))
    mutate(research)  # type: ignore[operator]
    path.write_text(json.dumps(research, indent=2), encoding="utf-8")


def _replaying_config(corpus: Path) -> WindbreakConfig:
    """Return a configuration selecting ``corpus`` and nothing else.

    Args:
        corpus: The corpus directory to name.

    Returns:
        The configuration.
    """
    return WindbreakConfig(
        forecast=ForecastConfig(
            replay_corpus=ReplayCorpusConfig(
                mode=REPLAY_CORPUS_REPLAY, corpus_dir=str(corpus)
            )
        )
    )


def test_the_committed_corpus_loads_with_every_recording_reachable() -> None:
    """The committed corpus loads, and its results name only documents it holds.

    The cross-reference is the load-time invariant worth stating: a result
    pointing at a URL the corpus cannot serve would degrade at fetch time into
    an unreachable citation, and a run that abstained for what is really a typo
    in a fixture is the least debuggable failure this module can produce.
    """
    corpus = load_replay_corpus(DEMO_CORPUS)

    assert set(corpus.results) == {
        f"{prefix} {DEMO_TITLE}"
        for prefix in ("Base rate for", "Recent evidence on", "Counter-signal for")
    }
    referenced = {url for urls in corpus.results.values() for url in urls}
    assert referenced != set()
    assert referenced <= set(corpus.documents)
    assert corpus.hosts() == frozenset({"corpus.local"})
    assert len(corpus.documents) == 3
    assert len(corpus.votes) == 3


def test_each_ensemble_member_gets_its_own_recorded_vote() -> None:
    """The vote transport keys on ``provider:model_version``, not on the prompt.

    Asserted by handing the transport three requests that differ *only* in
    provider and model version -- one identical prompt for all three -- and
    comparing each returned completion against the committed file. If the key
    dropped the model version the two OpenAI members would collide; if it
    dropped the provider, nothing would resolve at all; if the prompt mattered,
    all three would return the same thing.
    """
    corpus = load_replay_corpus(DEMO_CORPUS)
    transport = CorpusVoteTransport(corpus)
    recorded = json.loads(
        (DEMO_CORPUS / CORPUS_VOTES_FILENAME).read_text(encoding="utf-8")
    )

    served = {
        vote_key(provider, model): transport.complete(
            LlmRequest(provider=provider, model_version=model, prompt="one prompt")
        )
        for provider, model in (
            ("openai", "gpt-5-2025-08-07"),
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("openai", "gpt-5-mini-2025-08-07"),
        )
    }

    assert sorted(served) == sorted(recorded)
    assert {key: completion.text for key, completion in served.items()} == {
        key: entry["response"] for key, entry in recorded.items()
    }
    assert len({completion.text for completion in served.values()}) == 3
    usage = served["openai:gpt-5-2025-08-07"].usage
    assert usage is not None
    assert (
        usage.input_tokens
        == recorded["openai:gpt-5-2025-08-07"]["usage"]["input_tokens"]
    )


def test_an_unrecorded_member_is_discarded_rather_than_crashing_the_run() -> None:
    """A member the corpus does not cover raises a discardable vote failure.

    The error's *type* is what makes the degradation happen: the pipeline
    catches :class:`~windbreak.forecast.providers.base.ProviderVoteError` per
    vote and carries on with the survivors. The failure code and fingerprint are
    pinned too, because a discard is ledgered by both.
    """
    transport = CorpusVoteTransport(load_replay_corpus(DEMO_CORPUS))
    provider, model = UNRECORDED_MEMBER

    with pytest.raises(ProviderCorpusMissError) as caught:
        transport.complete(
            LlmRequest(provider=provider, model_version=model, prompt="p")
        )

    assert isinstance(caught.value, ProviderVoteError)
    assert str(caught.value) == (
        f"replay corpus holds no recorded vote for ensemble member '{provider}:{model}'"
    )
    assert caught.value.failure_code == PROVIDER_FAILURE_CORPUS_MISS
    assert caught.value.response_fingerprint == NO_RESPONSE_FINGERPRINT
    assert caught.value.cost_micros == 0


def test_an_unrecorded_subquestion_finds_nothing_rather_than_raising() -> None:
    """A question the corpus never recorded returns no candidates.

    Which is the honest answer, and the one that lets the pipeline abstain on
    zero verified citations instead of the tick dying. The positive control is
    the whole point: a recorded question comes back with its recorded URL
    through the same call.
    """
    transport = CorpusResearchTransport(load_replay_corpus(DEMO_CORPUS))

    found = transport.search(RECORDED_QUERY)

    assert found == ("https://corpus.local/base-rate",)
    assert transport.search(UNRECORDED_QUERY) == ()


def test_a_recorded_page_is_byte_identical_on_every_fetch() -> None:
    """Two fetches of one recorded URL return the same bytes.

    Not a restatement of "it reads a dict": ``verify_citation`` re-fetches every
    citation and fails it on any byte drift against the hash taken at gather
    time, so a transport that returned a re-rendered body would abstain on
    ``content_hash_mismatch`` and look exactly like a corpus with no coverage.
    """
    transport = CorpusResearchTransport(load_replay_corpus(DEMO_CORPUS))
    url = "https://corpus.local/base-rate"

    first = transport.fetch(url)

    assert first == transport.fetch(url)
    assert "datePublished" in first


def test_an_unrecorded_url_reads_as_an_unreachable_page() -> None:
    """Fetching a URL the corpus lacks raises ``OSError``, not a bespoke error.

    The type is load-bearing: ``bounded_web_research`` and ``verify_citation``
    both catch ``OSError`` already, so a gap in a corpus costs one citation
    rather than the run.
    """
    transport = CorpusResearchTransport(load_replay_corpus(DEMO_CORPUS))

    with pytest.raises(OSError) as caught:
        transport.fetch(UNRECORDED_URL)

    assert str(caught.value) == (
        f"replay corpus holds no recorded document for '{UNRECORDED_URL}'"
    )


def test_the_sandbox_allowlist_is_the_corpus_own_host_set(tmp_path: Path) -> None:
    """The research sandbox admits exactly the hosts the corpus holds pages for.

    The allowlist is *derived* rather than configured beside the corpus, so this
    asserts both directions: the recorded host is reachable through the sandbox,
    and a host the corpus knows nothing about is refused by the sandbox's own
    egress guard before the transport is consulted at all.
    """
    tools = build_corpus_research_tools(
        load_replay_corpus(DEMO_CORPUS), tmp_path / "cache"
    )

    body = tools.fetch("https://corpus.local/base-rate")

    assert "datePublished" in body
    with pytest.raises(EgressDeniedError):
        tools.fetch(UNRECORDED_URL)


@pytest.mark.parametrize(
    ("spoil", "expected"),
    [
        pytest.param(
            lambda doc: doc.pop("documents"),
            "needs a non-empty 'documents' list",
            id="no_documents",
        ),
        pytest.param(
            lambda doc: doc.__setitem__("documents", []),
            "needs a non-empty 'documents' list",
            id="empty_documents",
        ),
        pytest.param(
            lambda doc: doc.__setitem__("documents", ["not an object"]),
            "every 'documents' entry must be an object",
            id="scalar_document",
        ),
        pytest.param(
            lambda doc: doc["documents"][0].__setitem__("body", 7),
            "needs string 'url' and 'body' leaves",
            id="non_string_body",
        ),
        pytest.param(
            lambda doc: doc["documents"][0].__setitem__("url", "file:///etc/passwd"),
            "must be an http(s) URL with a host",
            id="non_http_url",
        ),
        pytest.param(
            lambda doc: doc.pop("results"),
            "needs a non-empty 'results' object",
            id="no_results",
        ),
        pytest.param(
            lambda doc: doc["results"].__setitem__("q", "not-a-list"),
            "must be a list of URL strings",
            id="scalar_result",
        ),
        pytest.param(
            lambda doc: doc["results"].__setitem__("q", ["https://corpus.local/gone"]),
            "which this corpus holds no recorded document for",
            id="dangling_result",
        ),
    ],
)
def test_a_malformed_corpus_refuses_to_load(
    tmp_path: Path, spoil: object, expected: str
) -> None:
    """Every malformed shape refuses at load, naming what is wrong.

    A corpus an operator pointed at and got wrong must not read as a corpus that
    legitimately knows nothing: the two produce identical ledgers -- an
    abstention on zero verified citations -- and only one of them is a fixture
    bug the operator can fix.

    Args:
        tmp_path: pytest's per-test temporary directory.
        spoil: The single mutation applied to a copy of the committed corpus.
        expected: The substring the refusal must name.
    """
    corpus = _corpus_copy(tmp_path)
    _rewrite_research(corpus, spoil)

    with pytest.raises(CorpusFormatError) as caught:
        load_replay_corpus(corpus)

    assert type(caught.value) is CorpusFormatError
    assert expected in str(caught.value)


def test_a_missing_corpus_file_refuses_by_name(tmp_path: Path) -> None:
    """A directory with no ``votes.json`` refuses, naming the missing file."""
    corpus = _corpus_copy(tmp_path)
    (corpus / CORPUS_VOTES_FILENAME).unlink()

    with pytest.raises(CorpusFormatError) as caught:
        load_replay_corpus(corpus)

    assert str(caught.value) == (
        f"replay corpus is missing its {CORPUS_VOTES_FILENAME!r} file"
    )


def test_a_corpus_recording_no_vote_at_all_refuses(tmp_path: Path) -> None:
    """An empty ``votes.json`` refuses rather than discarding every vote.

    Distinct from a corpus missing *one* member, which is a per-vote discard: a
    corpus that records nothing would abstain on a vote shortfall on every
    market forever, which is a configuration error wearing a forecast's clothes.
    """
    corpus = _corpus_copy(tmp_path)
    votes = corpus / CORPUS_VOTES_FILENAME
    votes.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CorpusFormatError) as caught:
        load_replay_corpus(corpus)

    assert str(caught.value) == f"{votes} records no vote for any ensemble member"


def test_the_default_configuration_selects_no_corpus() -> None:
    """The shipped default resolves to no corpus and reports the default source.

    The composition trap in its exact form: a config leaf nothing reads is a
    leaf that always looks correct. This is the branch an operator gets by
    omission, so it is asserted rather than assumed.
    """
    shipped = WindbreakConfig()

    assert replay_corpus_directory(shipped) is None
    assert replay_corpus_source(shipped) == REPLAY_CORPUS_SOURCE_DEFAULT
    assert shipped.forecast.replay_corpus.mode == REPLAY_CORPUS_DISABLED


def test_a_configured_corpus_resolves_to_its_directory_and_names_its_source() -> None:
    """A written-down corpus resolves to its path and reports the config source."""
    configured = _replaying_config(DEMO_CORPUS)

    assert replay_corpus_directory(configured) == DEMO_CORPUS
    assert replay_corpus_source(configured) == REPLAY_CORPUS_SOURCE_CONFIGURED


def test_an_unknown_corpus_mode_refuses_to_start() -> None:
    """An unrecognized mode aborts rather than defaulting to either branch.

    Silently choosing a default would discard a stated intent, which is the same
    fail-closed reading :func:`~windbreak.scheduler.provider_wiring.is_live_mode`
    already gives an unknown transport mode.
    """
    config = WindbreakConfig(
        forecast=ForecastConfig(replay_corpus=ReplayCorpusConfig(mode="maybe"))
    )

    with pytest.raises(ValueError) as caught:
        replay_corpus_directory(config)

    assert str(caught.value) == (
        "unknown forecast.replay_corpus.mode 'maybe'; expected 'disabled' or 'replay'"
    )


def test_the_disabled_token_is_not_a_yaml_boolean() -> None:
    """``disabled`` survives a YAML round trip as a string.

    YAML 1.1 reads a bare ``off`` as ``False``, so an operator writing the
    obvious token would have met "expected a string, got bool" from the loader
    rather than the mode they asked for. This pins the token against that
    reading -- and against a future rename back into it.
    """
    import yaml

    loaded = yaml.safe_load(f"mode: {REPLAY_CORPUS_DISABLED}\n")

    assert loaded == {"mode": REPLAY_CORPUS_DISABLED}
    assert isinstance(loaded["mode"], str)
    assert yaml.safe_load("mode: off\n") == {"mode": False}


def test_a_replay_mode_naming_no_directory_refuses_to_start() -> None:
    """Selecting the corpus without naming one aborts, naming both leaves."""
    config = WindbreakConfig(
        forecast=ForecastConfig(
            replay_corpus=ReplayCorpusConfig(mode=REPLAY_CORPUS_REPLAY)
        )
    )

    with pytest.raises(ValueError) as caught:
        replay_corpus_directory(config)

    assert str(caught.value) == (
        "forecast.replay_corpus.mode is 'replay' but "
        "forecast.replay_corpus.corpus_dir is still "
        f"'{UNCONFIGURED_PLACEHOLDER}'; name the committed corpus directory or "
        "select 'disabled'"
    )


def test_a_corpus_beside_the_live_transport_refuses_to_start() -> None:
    """Recorded material and the live world cannot both be a run's evidence.

    Refused explicitly rather than resolved by precedence: silently preferring
    either would hand an operator a paper tape while they believed they were
    reading the market, or the reverse.
    """
    live = dataclasses.replace(
        _replaying_config(DEMO_CORPUS),
        forecast=dataclasses.replace(
            _replaying_config(DEMO_CORPUS).forecast,
            provider_transport=ProviderTransportConfig(mode=PROVIDER_TRANSPORT_LIVE),
        ),
    )
    seam = LiveProviderHttp(llm={}, search=None, fetch=None)

    with pytest.raises(ValueError) as caught:
        _resolve_replay_corpus(live, seam)

    assert str(caught.value) == (
        "forecast.replay_corpus selects a recorded corpus while "
        "forecast.provider_transport.mode is 'live'; a run reads recorded "
        "material or the world, never both -- select one"
    )


def test_the_resolved_corpus_backs_both_transports(tmp_path: Path) -> None:
    """One selection builds both the research bundle and the vote transport.

    The two halves are a single token because closing research alone converts a
    graceful abstention into a ``CassetteMissError`` out of the tick. This
    asserts that they really are one decision: the same resolved corpus answers
    a recorded subquestion *and* a recorded ensemble member.
    """
    corpus = _resolve_replay_corpus(_replaying_config(DEMO_CORPUS), None)

    assert corpus is not None
    tools = build_corpus_research_tools(corpus, tmp_path / "cache")
    votes = build_corpus_vote_transport(corpus)

    assert tools.search(RECORDED_QUERY) != ()
    assert (
        votes.complete(
            LlmRequest(provider="openai", model_version="gpt-5-2025-08-07", prompt="p")
        ).text
        != ""
    )
