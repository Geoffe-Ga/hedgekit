"""End-to-end tier: tests that run the shipped artifact, not an in-process graph.

Epic #465 diagnosed the gap this package fills. Every other suite in this
repository assembles its object graph inside the test process -- which is why
a compose file whose build context could not resolve (#445), a kernel wired
with ``kill_integration=None`` (#441) and an ``AlertDispatcher(sinks=[])``
(#444) all shipped with green tests. Nothing started the real thing.

Tests here start real OS processes, real images and real stacks, and assert on
durable evidence (ledger rows, exit codes, sockets, mounts) rather than on an
object a fixture handed them.
"""
