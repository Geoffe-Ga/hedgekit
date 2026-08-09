"""Fail-closed projection of held positions onto the risk caps' inputs (#407, RED).

`windbreak.scheduler.exposure` does not exist yet, so every test here fails at
import with `ModuleNotFoundError` -- the intended Gate 1 RED for issue #407.

Why this module exists at all. Two risk caps in this codebase run on every tick,
report success, and prove nothing:

* `concentration_limits` (`windbreak/riskkernel/checks.py:611`) compares four
  exposure dimensions against their equity shares, but
  `windbreak/scheduler/loop.py::_account_from_verification` hands it four zeros.
  Zero is *permissive*: `0 + cost > share` is false for any sane cost, so the
  check has never vetoed anything.
* The selector's per-bucket notional cap (`windbreak/selector/sizing.py:265`)
  aggregates `bucket_peers` against the target's `correlation_tags`, but nothing
  in `windbreak/` has ever constructed a `CorrelationTag`, so the loop passes
  `correlation_tags=()`, `effective_buckets(()) == ()`, and
  `aggregate_bucket_exposure` returns `(0, None)`.

This module is the seam that feeds both from evidence. It is deliberately
**pure**: it takes already-resolved `HeldMarket` triples rather than connector
`Position` rows or a `PaperExchange`, so the fail-closed contract can be
exercised without a venue and the scheduler keeps the only import of both sides.

The fail-closed rule under test is the sharp one. A market whose correlation
bucket the operator has not declared is *unbucketable*, and an unbucketable
market must not size as though its bucket were empty -- so `project_exposure`
returns `None` (refuse) rather than a projection carrying a fabricated zero.
That applies to the target market and to every held peer alike: a holding whose
bucket is unknown could be in the target's bucket, so the target's bucket
exposure is unprovable while it is held.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from windbreak.config.schema import CorrelationConfig, CorrelationTagConfig
from windbreak.numeric import MoneyMicros
from windbreak.scheduler.exposure import (
    HeldMarket,
    declared_correlation_tags,
    project_exposure,
)
from windbreak.selector.correlation import BUCKET_FED_POLICY, BUCKET_WEATHER

#: A fixed, offset-aware instant for every declared tag in this module. Real,
#: not `utcnow()`: `CorrelationTag.tagged_at` is provenance, and a test that
#: read the wall clock could not pin it.
_TAGGED_AT = "2026-03-01T00:00:00+00:00"

#: The same instant as a `datetime`, for asserting the parsed tag carries it.
_TAGGED_AT_DT = datetime(2026, 3, 1, tzinfo=UTC)


def _correlation(*entries: tuple[str, tuple[str, ...]]) -> CorrelationConfig:
    """Build a `CorrelationConfig` from `(ticker, bucket_ids)` pairs.

    Args:
        entries: The operator-declared bucket assignments to include.

    Returns:
        The assembled configuration, every entry stamped `_TAGGED_AT`.
    """
    return CorrelationConfig(
        tags=tuple(
            CorrelationTagConfig(
                ticker=ticker, bucket_ids=bucket_ids, tagged_at=_TAGGED_AT
            )
            for ticker, bucket_ids in entries
        )
    )


def _held(ticker: str, event_ticker: str, value_micros: int) -> HeldMarket:
    """Build one `HeldMarket` with an integer value in micros.

    Args:
        ticker: The held market's ticker.
        event_ticker: The parent event's ticker, as the venue reports it.
        value_micros: The position's mark value, in micros.

    Returns:
        The assembled `HeldMarket`.
    """
    return HeldMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        value_micros=MoneyMicros(value_micros),
    )


class TestDeclaredCorrelationTags:
    """The operator-declared tag producer -- the only honest one available."""

    def test_declared_ticker_yields_a_human_sourced_tag(self) -> None:
        """A configured assignment becomes a `source="human"` tag.

        `source` is `Literal["llm", "human"]` and SPEC S9.9 calls tagging
        "LLM-assisted ... human-overridable". An operator writing a bucket into
        the configuration *is* the human, so `"human"` is the literal truth
        here, not a convenient label -- no third source value is invented.
        """
        tags = declared_correlation_tags(
            _correlation(("KXWEATHER-A", (BUCKET_WEATHER,))), "KXWEATHER-A"
        )
        assert len(tags) == 1
        assert tags[0].bucket_id == BUCKET_WEATHER
        assert tags[0].source == "human"
        assert tags[0].tagged_at == _TAGGED_AT_DT

    def test_undeclared_ticker_yields_no_tags(self) -> None:
        """A ticker absent from the configuration produces an empty tuple."""
        tags = declared_correlation_tags(
            _correlation(("KXWEATHER-A", (BUCKET_WEATHER,))), "KXOTHER-B"
        )
        assert tags == ()

    def test_multiple_buckets_on_one_ticker_all_survive(self) -> None:
        """A market declared in two buckets yields both tags."""
        tags = declared_correlation_tags(
            _correlation(("KXA", (BUCKET_WEATHER, BUCKET_FED_POLICY))), "KXA"
        )
        assert {tag.bucket_id for tag in tags} == {BUCKET_WEATHER, BUCKET_FED_POLICY}

    def test_naive_tagged_at_is_refused(self) -> None:
        """An offsetless `tagged_at` raises rather than reading as host-local."""
        config = CorrelationConfig(
            tags=(
                CorrelationTagConfig(
                    ticker="KXA",
                    bucket_ids=(BUCKET_WEATHER,),
                    tagged_at="2026-03-01T00:00:00",
                ),
            )
        )
        with pytest.raises(ValueError, match="tagged_at"):
            declared_correlation_tags(config, "KXA")

    def test_unknown_bucket_id_is_refused(self) -> None:
        """A bucket id outside the seed taxonomy raises at construction."""
        config = CorrelationConfig(
            tags=(
                CorrelationTagConfig(
                    ticker="KXA", bucket_ids=("not-a-bucket",), tagged_at=_TAGGED_AT
                ),
            )
        )
        with pytest.raises(ValueError, match="not a seed-taxonomy id"):
            declared_correlation_tags(config, "KXA")


class TestProjectExposureFailsClosed:
    """Absent evidence must refuse, never read as an empty bucket."""

    def test_undeclared_target_refuses(self) -> None:
        """A target with no declared bucket projects to `None`, not zero."""
        projection = project_exposure(
            (),
            target_ticker="KXTARGET",
            target_event_ticker="KXEVENT",
            correlation=_correlation(),
        )
        assert projection is None

    def test_undeclared_peer_refuses(self) -> None:
        """A held peer with no declared bucket makes the target unprovable.

        The peer could be in the target's bucket. While it is held and
        unclassified, the target's bucket exposure cannot be established, so
        sizing must refuse rather than aggregate over a peer set it knows to be
        incomplete.
        """
        projection = project_exposure(
            (_held("KXPEER", "KXEVENT2", 5_000_000),),
            target_ticker="KXTARGET",
            target_event_ticker="KXEVENT",
            correlation=_correlation(("KXTARGET", (BUCKET_WEATHER,))),
        )
        assert projection is None

    def test_all_declared_projects(self) -> None:
        """With every held ticker declared, the projection is produced."""
        projection = project_exposure(
            (_held("KXPEER", "KXEVENT2", 5_000_000),),
            target_ticker="KXTARGET",
            target_event_ticker="KXEVENT",
            correlation=_correlation(
                ("KXTARGET", (BUCKET_WEATHER,)), ("KXPEER", (BUCKET_WEATHER,))
            ),
        )
        assert projection is not None


class TestProjectExposureArithmetic:
    """The four exposure terms, pinned to exact micros."""

    #: Three holdings: the target itself, an event sibling, and a bucket-only
    #: peer in an unrelated event. Every term below is a different subset sum,
    #: so a mutant collapsing any two dimensions is caught.
    _HELD = (
        _held("KXTARGET", "KXEVENT", 7_000_000),
        _held("KXSIBLING", "KXEVENT", 3_000_000),
        _held("KXFAR", "KXOTHER", 11_000_000),
    )

    _CORRELATION = _correlation(
        ("KXTARGET", (BUCKET_WEATHER,)),
        ("KXSIBLING", (BUCKET_WEATHER,)),
        ("KXFAR", (BUCKET_FED_POLICY,)),
    )

    def _projected(self) -> object:
        """Project the shared fixture, asserting it was provable.

        Returns:
            The `ExposureProjection` under test.
        """
        projection = project_exposure(
            self._HELD,
            target_ticker="KXTARGET",
            target_event_ticker="KXEVENT",
            correlation=self._CORRELATION,
        )
        assert projection is not None
        return projection

    def test_market_exposure_is_the_target_ticker_only(self) -> None:
        """Only the holding in the target's own ticker counts: 7_000_000."""
        assert self._projected().market_exposure == MoneyMicros(7_000_000)

    def test_event_exposure_sums_the_event_siblings(self) -> None:
        """Target plus its event sibling: 7_000_000 + 3_000_000."""
        assert self._projected().event_exposure == MoneyMicros(10_000_000)

    def test_bucket_exposure_sums_the_bucket_not_the_event(self) -> None:
        """The weather bucket holds the target and its sibling: 10_000_000.

        `KXFAR` is 11_000_000 and in `fed-policy`, so a projection that summed
        every peer would report 21_000_000 and a projection that reused the
        event term would coincide with it -- both are excluded by this value.
        """
        assert self._projected().bucket_exposure == MoneyMicros(10_000_000)

    def test_total_exposure_sums_every_holding(self) -> None:
        """All three holdings: 7_000_000 + 3_000_000 + 11_000_000."""
        assert self._projected().total_exposure == MoneyMicros(21_000_000)

    def test_bucket_peers_carry_every_holding_with_its_tags(self) -> None:
        """Each holding becomes one peer entry the selector can aggregate."""
        peers = self._projected().bucket_peers
        by_ticker = {peer.market_ticker: peer for peer in peers}
        assert set(by_ticker) == {"KXTARGET", "KXSIBLING", "KXFAR"}
        assert by_ticker["KXFAR"].exposure_micros == MoneyMicros(11_000_000)
        assert by_ticker["KXFAR"].tags[0].bucket_id == BUCKET_FED_POLICY

    def test_target_tags_are_the_targets_own_declaration(self) -> None:
        """The projection carries the tags the selector must resolve."""
        tags = self._projected().target_tags
        assert tuple(tag.bucket_id for tag in tags) == (BUCKET_WEATHER,)

    def test_no_holdings_projects_all_zero_for_a_declared_target(self) -> None:
        """A flat account is evidence of zero exposure, not absent evidence.

        The venue answered and reported no rows. That is an observed absence,
        which the connector's contract defines as genuinely flat -- unlike an
        *undeclared* market, where nothing was observed at all.
        """
        projection = project_exposure(
            (),
            target_ticker="KXTARGET",
            target_event_ticker="KXEVENT",
            correlation=_correlation(("KXTARGET", (BUCKET_WEATHER,))),
        )
        assert projection is not None
        assert projection.market_exposure == MoneyMicros(0)
        assert projection.event_exposure == MoneyMicros(0)
        assert projection.bucket_exposure == MoneyMicros(0)
        assert projection.total_exposure == MoneyMicros(0)
        assert projection.bucket_peers == ()
