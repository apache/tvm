---
name: tirx-ptx-dialect
description: Register, extend, or audit instructions in the table-driven T.ptx dialect (python/tvm/backend/cuda/ptx/table.py), and move the table to a newer PTX ISA version (9.3, 9.4, ...). Use when adding a PTX instruction or qualifier, widening an operand domain, fixing a ptxas certification failure, or checking the table's comments against the PTX ISA document.
---

# TIRx PTX dialect (`T.ptx`)

`T.ptx.<mnemonic>.<qualifiers>(operands...)` is not hand-written per
instruction. One data table describes every instruction family; one engine
resolves calls, renders an inline-asm helper, and registers the TVM op; three
generators derive the IDE stub, a coverage table and a helper dump from the
same table. Adding an instruction means adding **data**, then proving it
against ptxas.

| file | role |
|---|---|
| `python/tvm/backend/cuda/ptx/table.py` | the table: `InstructionEntry` / `ModifierSlot` / `OperandSlot`, check functions, variant enumeration. tvm-free. **Read its three dataclass docstrings before editing.** |
| `python/tvm/backend/cuda/ptx/render.py` | renders one variant to a `__forceinline__ __device__` helper (`asm` or `asm volatile`, according to `asm_volatile`). `CBinding` = how each TVM dtype binds a register; `BRIDGE` = three scoped cases covering the two register classes inline asm cannot bind directly (`.pred`, and the block-local `.b8` registers behind the `e2m1x2` and `st_async_b8reg` tokens) — not a general `.b8` mechanism. tvm-free. |
| `python/tvm/backend/cuda/ptx/engine.py` | `register_table()` makes each entry a TVM op `tirx.ptx.<name>` with a generic codegen; `PTXNamespace` resolves `T.ptx` attribute chains and the string form `T.ptx["..."]`; trace-time coercion and `check`/`imm_check`. |
| `python/tvm/backend/cuda/ptx/gen_stubs.py` | `python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi` — the checked-in stub; a test diffs it against the generator. |
| `python/tvm/backend/cuda/ptx/gen_helpers.py` | `python -m tvm.backend.cuda.ptx.gen_helpers <family>` — dump the exact helper source of every variant without compiling a kernel (it still imports `tvm`, so it needs the usual built worktree on `PYTHONPATH`). |
| `python/tvm/backend/cuda/ptx/gen_coverage.py` | markdown coverage table of the whole dialect. |
| `tests/python/tirx/codegen/test_ptx_dialect.py` | per-ISA-section dispatch tests, structural invariants, ptxas certification. |
| `tests/python/tirx/codegen/test_ptx_cvt.py` | one case per registered `cvt` syntax line (`_FORM_CASES`); coverage of every cvt entry is asserted. |
| `tests/python/tirx/codegen/test_ptx_addr.py` | `T.ptx.addr(base, byte_offset)` immediates. |

The namespace itself is installed by `python/tvm/backend/cuda/__init__.py::script_namespaces()`;
nothing there changes when the table grows.

> **DO NOT introduce a new mechanism in an `InstructionEntry` without the
> user's explicit approval.** Stop, explain why the existing entry model is
> insufficient and what codegen or validation behavior the mechanism would
> add, then wait for approval before implementing it.

## 1. Ground rules the table enforces

The first four are tested mechanically — a new entry that breaks one fails
the suite. The last one (what the comments claim) is audited by hand, §4.

- **One entry = one syntax shape.** Split into sibling entries sharing the
  `mnemonic` when the register-group shape or the result structure changes
  (scalar vs vector destination, `mul` vs `mul.wide`), or when an optional
  operand is told apart only by call arity (`{, count}`, `state` vs the `_`
  sink); the engine then selects by arity and by which tokens are written.
  An operand whose presence is fixed by one written qualifier stays in the
  entry with a 0-or-1-register `lanes` function — `{, cache_policy}` with
  `.L2::cache_hint`, `{, ctaMask}` with `.multicast::cluster` use
  `lanes=_present_lanes("<slot>")`. Variants that only add or remove dotted
  qualifiers are optional slots of one entry.
- **Single-instruction invariant**: every helper emits exactly one native PTX
  instruction. The only extra statements allowed are the framework `@p`
  wrapper and the `BRIDGE` boundary conversions. `raw_render` is the escape
  hatch and every user of it must be listed in `RAW_ENTRIES` inside
  `test_ptx_single_instruction_invariant`.
- **Every legal closed variant assembles**: `renderings()` enumerates the
  product of modifiers × operand dtypes × closed immediates × `@p` (for
  entries without a destination) and full certification assembles all of it
  through ptxas at `cert_arch`. Four things get *representative* coverage
  instead of the full product, and each says so in its docstring: an OPEN
  immediate is certified at `imm_combos`'s samples (default `"0"`), which
  proves the shape, not the caller's constant; sink masks are walked once at
  one representative modifier combination per sink domain (`renderings`);
  address immediates are a small separate axis (`_addr_offset_samples`);
  and the destination-predicated `pred=` helpers (`*_pred_undef` /
  `*_pred_keep`) are not enumerated at all — cover those with a
  production-shaped test when a call site needs them. Conversely a `check()`
  must never reject a legal form to make certification pass — narrow with
  evidence.
- **Dispatch must be unambiguous**: no two entries may accept the same call
  (`test_ptx_dispatch_unambiguous`); helper names must be unique
  (`test_ptx_all_variants_render_unique`).
- **Comments cite the ISA**: the section number, the syntax line reproduced
  verbatim, the type/qualifier domains, `sm_XX` floors, and a `NOT REGISTERED:`
  note with the reason for every syntax line or token deliberately left out.
  A fact that comes from ptxas rather than the document is marked `MEASURED`
  and quotes the ptxas message. Section and table numbers follow the ISA
  version named in the module docstring (currently **PTX ISA 9.2**, see §5).

## 2. Designing the entry

Open the instruction's section in the ISA document and transcribe, in this
order:

1. **Name / mnemonic / family.** `name` is the table key and must be a Python
   identifier. Dotted mnemonics are a family plus single-choice slots
   (`cvta.to.shared` → `cvta` + slots), or `mnemonic="st.bulk"` with
   `name="st_bulk"` when the dot is part of the instruction's identity.
   Several entries sharing a mnemonic (`mov`, `mbarrier`, `mma`) use
   distinct `name`s (`mov`, `mov_pack_2`, ...) and the same `mnemonic`.
   Entries that collide with an existing family on the same shape need a
   suffix (`add_int` vs `add`/`add_half`) and are told apart by their tokens.
2. **Slots** (`ModifierSlot(name, choices, optional)`), in asm render order —
   exactly the `{.qual}` positions of the syntax line, each with the ISA's
   `.qual = { ... }` list as `choices`. Tokens with `::` are written as-is
   (`"shared::cta"`); users type `shared__cta`. Python keywords get a
   trailing underscore only at the call site (`global_`).
3. **Operands** (`OperandSlot`), in PTX operand order, destinations first
   because PTX writes them first:
   - `rw`: `"w"` destination, `"rw"` accumulator (`"+"`), `"r"` input. A
     `"w"` operand binds `"="`; it does **not** block `pred=` — under a false
     predicate the value is undefined unless the caller passes
     `preserve_dst=True`, which switches the binding to `"+"` (see
     `test_ptx_predicated_destination_*`). Those two helpers are rendered on
     demand, not by `renderings()`.
   - `kind`: `"reg"` (default), `"addr"` (`[%k]`, space from `space=` or the
     entry's `space` slot; set `allow_imm_offset=True` only for an independent
     byte address), `"ptr"` (raw pointer value), `"imm"` (text operand:
     `literal=` fixed by the ISA, `choices=` closed caller set, neither = open
     immediate certified at samples; validate open ones with `imm_check`).
   - `dtype`: a `PTX_TYPE_DTYPES` key, a slot name, or a module-level pure
     function of the modifier map (`_wide_dtype`). `None` means the entry's
     `type` slot. `dtypes=` narrows/widens the TVM dtype domain independently
     (see the relaxed-carrier note in §6 before widening anything).
   - `lanes` (int or function of the modifier map) for brace-enclosed register
     groups; `vector=`, `bracket=`, `pipe=` for the few composite spellings;
     `sinkable=` only where the ISA lets a caller write `_` for a lane.
4. **`check`**: one pure module-level function `mod_map -> error | None` per
   entry with a one-line docstring (it is surfaced by the stub and the coverage
   table). Encode only what the ISA syntax block and notes state; every
   restriction that comes from ptxas instead is a separate `MEASURED` clause.
   Never a lambda: frozen dataclasses hash callables by identity.
5. **`cert_arch`**: the **maximum** `sm_XX` floor over the entry's variants
   (default is `PTX_ARCH`, `sm_90`). Certifying below a variant's floor makes
   ptxas report legal forms as illegal, and that verdict would then get baked
   into a `check()`.
6. `orders_memory=True` for fences/barriers/waits (no address operand but the
   `"memory"` clobber is needed); `asm_volatile` stays at its default.

Worked example — the whole registration of `movmatrix` (commit a29a5e97ba,
4 files, 40 lines):

```python
    # movmatrix per PTX ISA 9.7.14.5.17 -- transpose one distributed m8n8
    # matrix whose 16-bit elements are carried by one b32 register per lane.
    #
    #   movmatrix.sync.aligned.m8n8.trans.b16 d, a;
    InstructionEntry(
        name="movmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("type", ("b16",)),
        ),
        cert_arch="sm_75",
        operands=(
            OperandSlot("d", rw="w", dtype="b32"),
            OperandSlot("a", dtype="b32"),
        ),
    ),
```

Put the entry under its ISA chapter banner in `_ENTRIES` (`# PTX ISA 9.7.x —
...`), next to the instructions it shares a mnemonic with.

## 3. Verify — every gate, in this order

Run from the TVM repo root with the workspace `PYTHONPATH` (see `tir-test`).

1. **Table loads and enumerates.** A `check()` that filters out every
   combination, a duplicate name, or a bad `allow_imm_offset` slot fails at
   import.
   ```bash
   python -c "from tvm.backend.cuda.ptx.table import TABLE, variants; e=TABLE['<name>']; print(len(variants(e)))"
   python -m tvm.backend.cuda.ptx.gen_helpers <name> | head -60   # eyeball the asm
   ```
2. **Add tests for the new surface** (before running the suite, so one run
   settles everything):
   - a call in the dispatch test of its ISA section
     (`test_ptx_<section>_dispatch`) asserting the exact emitted line, e.g.
     `assert "mul.wide.s32 %0, %1, %2;" in src`;
   - a new `cvt` line → a `_FORM_CASES` row in `test_ptx_cvt.py`
     (`test_cvt_cases_cover_every_registered_entry` fails otherwise);
   - `raw_render` → `RAW_ENTRIES`;
   - a destination-predicated (`pred=`) call site → a production-shaped
     codegen test, since certification does not enumerate those helpers;
   - only when the instruction belongs to the MegaMoE extracted-intrinsics
     surface → its kernel and mnemonic list in
     `test_codegen_cuda.py::test_megamoe_extracted_intrinsics_codegen`
     (that test covers one workload, not the dialect).
3. **Regenerate the stub** — any change to a family's tokens or shapes makes
   `test_ptx_stub_up_to_date` fail until you do:
   ```bash
   python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi
   ```
4. **Fast suite** (no `PTX_CERT`): structural invariants + dispatch tests +
   a seeded ptxas sample.
   ```bash
   python -m pytest tests/python/tirx/codegen/test_ptx_dialect.py tests/python/tirx/codegen/test_ptx_cvt.py tests/python/tirx/codegen/test_ptx_addr.py -q
   ```
   `test_ptx_all_variants_render_unique` ends with `assert total == <N>`;
   its failure message prints the new total — update the constant and
   re-run.
5. **Full certification** — mandatory after any table change, and the only
   gate that proves the domain. It assembles every closed variant of every
   entry, split into 32 shards; pick 8, 16 or 32 workers to taste (each one
   drives its own nvcc; `-n auto` on a many-core box mostly burns memory).
   ~3.5 min at `-n 16`, ~6.5 min at `-n 8` on the current B200 host for the
   ~800k-variant PTX ISA 9.2 table.
   ```bash
   PTX_CERT=1 python -m pytest -n 16 -q \
     tests/python/tirx/codegen/test_ptx_dialect.py::test_ptx_all_helpers_certify
   ```
   A failure names `<arch> batch <k>` and the ptxas message. To find the
   variant: render the failing family with `gen_helpers`, or assemble the
   batch yourself (`nvcc -arch=<arch> -ptx` then `ptxas`, map the error line
   back to the preceding `.entry`). Fix by narrowing the domain with a
   `MEASURED` clause, never by loosening the invariant.
6. `pre-commit run --files <changed files>` and commit as
   `feat(lower-tirx): support PTX <instruction>` (Conventional Commits, see
   the workspace `CLAUDE.md`).

## 4. Auditing the comments against the ISA

The comments are load-bearing (they are what a reviewer checks the data
against), so audit them the way the data is certified: download the ISA HTML
for the version the module docstring names, convert to text, and verify every
section number, quoted sentence, reproduced syntax line, type list and
`sm_XX` note. Things that have gone wrong before and are worth a targeted
pass: a `NOT REGISTERED` bullet naming something the table registers a few
hundred lines later; a section number copied from a neighbouring
instruction; a version number attached to the wrong syntax line; "the N-th
syntax line" ordinals; `Table NN` numbers, which are global and shift between
ISA versions.

## 5. Toolchain and ISA-version model

- The dialect targets the **ptxas of the installed CUDA toolkit**, not the
  latest ISA document. CUDA 13.2's ptxas implements PTX ISA 9.2. Establish
  what yours implements from ptxas itself — `nvcc -ptx` only shows the
  version the front end chose to emit, and `-ptx` never validates inline
  asm:
  ```bash
  command -v nvcc ptxas && nvcc --version | tail -1 && ptxas --version | tail -1
  printf '.version 9.3\n.target sm_90\n.address_size 64\n.visible .entry k() { ret; }\n' \
    | ptxas -arch=sm_90 - -o /dev/null  # CUDA 13.2: "Unsupported .version 9.3; current version is '9.2'"
  ```
  Then let the certification suite (which compiles real inline-asm helpers
  to cubin) be the final word.
- The table's citations follow that same version (module docstring of
  `table.py`: "Section and table numbers cite PTX ISA 9.2", archive URL
  `docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/`). The
  live `docs.nvidia.com/cuda/parallel-thread-execution/` page is the newest
  version and its numbering differs; every ISA version stays available
  under `docs.nvidia.com/cuda/archive/<cuda version>/`.
- Certification needs only `nvcc`/`ptxas`; the on-GPU round-trip tests need
  a driver that can load a cubin from that toolkit. A newer toolkit with an
  older driver can therefore certify a table but not run it.
- `PTX_ARCH` (env, default `sm_90`) is the certification arch for entries
  without `cert_arch`.

## 6. Moving the table to PTX ISA 9.3 (and later)

Do this as one commit series, in this order:

1. **Toolchain first.** Install the CUDA toolkit whose ptxas implements the
   target ISA and confirm via `.version`. Until then, a 9.3 form cannot be
   registered: certification would fail, and a `check()` written against an
   older ptxas would silently delete the coverage later.
2. **Retarget the citations.** Update the module docstring's version
   statement and archive URL, then renumber. Between 9.2 and 9.3 the shifts
   are: chapter `9.7.10 Fabric Instructions` inserted (everything from
   Texture onward moves +1: sync `9.7.13→9.7.14`, mma `9.7.14→9.7.15`, wgmma
   `9.7.15→9.7.16`, tcgen05 `9.7.16→9.7.17`, misc `9.7.19→9.7.20`);
   `9.7.1.5 clmad` inserted (integer instructions from `mul24` on move +1);
   `9.7.9.13 multimem.st.async` inserted (data movement from `st.bulk` on
   move +1); `9.7.14.8 multimem.red.async` inserted (sync instructions from
   `vote` on move +1); the mbarrier chapter gains three descriptive
   subsections (Layouts, Tracking successful completion, Report-on), so its
   instruction subsections move +3 (`mbarrier.arrive` `9.7.13.15.13 →
   9.7.14.16.16`) and `mbarrier.check_layout` is appended; global table
   numbers shift (the tcgen05.ld/st register-count tables are 49/50 in 9.2
   and 52/53 in 9.3).
   Renumber intra-chapter subsections too, not just chapters, and re-run the
   audit of §4 afterwards — a chapter-only shift was the exact mistake made
   the first time this table changed versions.
3. **Register the newly legal forms.** Per the 9.3 release notes (ISA
   chapter 13.1) and the per-section "introduced in PTX ISA version 9.3"
   notes, the forms this table currently leaves out because they are 9.3 are:
   - `ld.mmio` with `.acquire`, `st.mmio` with `.release` (`_check_ld` /
     `_check_st`: "PTX ISA 9.2 spells only relaxed");
   - `clmad` (new integer family, 9.7.1 banner note);
   - `multimem.st.async`, `multimem.red.async` (new families, 9.7.9 / sync
     chapter);
   - `.sem`/`.scope` on `cp.async.bulk`, `cp.reduce.async.bulk`,
     `multimem.cp.async.bulk`, `multimem.cp.reduce.async.bulk` (new slots
     on the cp.async.bulk entries);
   - mbarrier: `.phase_type::*` on `test_wait`/`try_wait`, the
     `waitComplete|reportPredicate{, reportValue}` report forms (new
     entries — a second destination is a new shape), the `.layout` qualifier
     on `mbarrier.init`/`pending_count`, and `mbarrier.check_layout` (new
     family);
   - `fence.proxy.to_proxykind::from_proxykind_fabric.alias.sem_fabric.sys`
     (two new slots on the fence family; sm_100);
   - the `fabric.*` instructions (a whole new chapter; new families).
   Each one: read the section, apply §2, then §3 including full
   certification at the right `cert_arch`.
4. **Do not widen operand carriers from the ISA's relaxed-typing tables
   without certifying.** ISA §9.4.1 Tables 27/28 are an upper bound and say
   so ("some combinations may still be invalid for a particular
   instruction"); ptxas rejects, for example, any `cvt` that pairs a `.bf16`
   operand with a wider register on the other side, and 128-bit carriers on
   several `.ftz` conversions. The `_cvt_dst_dtypes` / `_cvt_src_dtypes`
   docstrings in `table.py` record the measured gaps; a version bump must
   re-run certification and re-measure them, because gaps can close.
5. Regenerate the stub, update the variant count, run the full suite, and
   run the §4 audit against the new document before merging.
