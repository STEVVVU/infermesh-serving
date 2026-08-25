# Paged KV Cache

The cache models LLM KV memory as fixed-size token blocks. It is deterministic
CPU-side accounting, not CUDA memory management.

## Objects

- A block has a numeric ID and an owner count.
- A prefix entry stores token IDs, ordered block IDs, active lease count, and
  an LRU tick.
- A request lease reports the cache key, reused token count, and blocks.

## Prefix reuse

Exact prefixes increment the entry's active lease count. For a new prefix, the
cache finds the longest cached token sequence that is a true prefix. Only
complete blocks are shared; a partial trailing block is recomputed. Shared
blocks increment their owner counts, while missing suffix blocks come from the
free set.

After successful dispatch, the selected worker records the prefix key as
affinity metadata. Later requests prefer that worker before comparing load.

## Eviction

Allocation evicts least-recently-used entries with zero active leases. Evicting
an entry decrements each block's owner count; a block returns to the free set
only when the count reaches zero. The source prefix selected for reuse is
protected during allocation.

If enough blocks cannot be freed because entries are pinned, admission fails
with resource exhaustion rather than overcommitting accounting.

## Invariants tested

- Derived prefixes share full block IDs with their source.
- Shared owners do not double-count allocated physical blocks.
- Active leases cannot be evicted.
- LRU unpinned entries are reclaimed first.
- Release rejects missing leases and refcount underflow.
- Repeated LLM scheduling produces one miss followed by a cache hit.

The implementation does not allocate device memory, store attention tensors,
or coordinate cache blocks across processes. Those are explicit extension
points for a CUDA worker implementation.
