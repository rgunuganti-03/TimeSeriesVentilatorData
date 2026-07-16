# Brief — Pediatric / Neonatal Ventilation + A Labeling & Validation Exercise

**From:** Prashanth (Mentor) + Atlas (web-app / builder seat)
**To:** Riya + Claude Code (her agent)
**Date:** 2026-07-09
**Purpose:** Give Riya a physiological domain of her own to own (pediatric & neonatal ventilation),
and use it to run a real data-labeling + validation exercise.
**Scope:** ~2 weeks. Two phases, each ends in a concrete artifact.
**Expected output:** CRs Riya creates + implements, a small labeled dataset, and a short
label-agreement report.

---

## The Big Picture — why this, why now

Your simulator now covers VCV and PCV for adults across seven conditions. The adult single-
compartment engine is in good shape. This brief points you at two things that are **genuinely
useful to AiDen, deliberately off the team's immediate critical path, and rich to learn from** —
which is exactly what makes them good for you to own end-to-end at your own pace.

1. **Pediatric & neonatal ventilation** — a physiological world the whole platform currently does
   *not* model. Everything demo-facing at AiDen today is adult-only. Kids are not small adults:
   the physics rescales, new signatures appear (leak around the tube being the big one), and the
   diseases are different. This is a bounded domain you can become **our expert in.**
2. **A labeling & validation exercise** on top of it — because a dataset is only as good as the
   labels on it, and the discipline of labeling well is a core data-science skill. Your synthetic
   data has a superpower here: because *you* set the parameters, you already know the ground truth.
   That makes it the perfect, safe place to practice labeling for real.

You keep your stack (Python / NumPy / SciPy / Pandas / Plotly / Streamlit) and your architecture.
Nothing here asks you to rebuild what you have — it asks you to extend it into new physiology and
then be rigorous about the labels.

---

# PHASE 1 (Week 1) — Pediatric & Neonatal Physiology

## Kids are not small adults — what actually changes

The equation of motion is the same: `P(t) = V(t)/C + R·Flow(t) + PEEP`. What changes is the
*scale* of every term and the *addition of leak*. Your job in Phase 1 is to research the real
numbers (don't take mine as gospel — see "Ground the numbers" below) and extend the generator.

The shifts to expect, roughly, from adult → pediatric → neonatal:

| Property | Adult | Child (~1–8 yr) | Neonate (term, ~3 kg) | Why |
|---|---|---|---|---|
| Tidal volume | 400–600 mL | 4–6 mL/kg → tens of mL | 4–6 mL/kg → **~12–24 mL** | weight-scaled; neonatal VT is tiny |
| Respiratory rate | 12–16 | 20–30 | **30–60** | small lungs, high metabolic rate |
| Compliance (absolute) | 50–100 mL/cmH₂O | single-digit to low tens | **~2–5 mL/cmH₂O** | scales with lung size |
| Resistance (absolute) | 2–10 cmH₂O/L/s | tens | **50–150 cmH₂O/L/s** | tiny airways; R ∝ 1/radius⁴ |
| Time constant τ = R·C | ~0.5 s | shorter | **short** | permits fast rates |
| ETT | usually cuffed | often uncuffed | **usually uncuffed** | → leak, a defining feature |

**The headline new signature: LEAK.** With an uncuffed tube, gas escapes around it during the
breath, so **inspired volume exceeds expired volume (VTi > VTe)** and the volume waveform does not
return to baseline. This is *the* thing that distinguishes a lot of neonatal/peds waveforms from
adult ones — and it is a distinct, model-able feature (a leak term proportional to airway
pressure). Model it as an explicit, tunable leak fraction; make "leak on/off" a scenario knob.

## Pediatric / neonatal conditions to model

Start with the ones whose physics is closest to what you already have, then branch:

| Condition | Population | Mechanics signature | Closest adult analog you've built |
|---|---|---|---|
| Normal neonate | neonate | tiny C, high R, short τ, leak | Normal (rescaled) |
| **RDS** (surfactant deficiency) | preterm | very low compliance ("stiff baby lung") | Severe ARDS (rescaled) |
| **Meconium aspiration** | term neonate | high resistance + air trapping | COPD / Bronchospasm (rescaled) |
| **BPD** (chronic lung disease) | ex-preterm infant | mixed: low C *and* high R | ARDS + COPD blend |
| **Bronchiolitis / RSV** | infant | obstructive, air trapping | COPD (rescaled) |
| Pediatric asthma | child | acute high resistance | Bronchospasm (rescaled) |

You do not need all six. **RDS + meconium aspiration + a normal neonate** is a strong Week-1 target:
they exercise the two ends (restrictive vs obstructive) plus baseline, all at neonatal scale, plus
leak. Add more if time allows.

## Ground the numbers (this is the real work, and it's yours)

The table above is a *starting sketch from your mentor's builder — treat it as a hypothesis, not
truth.* Pediatric parameters are genuinely different from adult and easy to get wrong, and this is
medical modeling, so:

- **Look them up.** Pull compliance/resistance/VT/RR ranges per condition from the pediatric
  ventilation literature and note your source per number (the export schema already has a
  `source_assumptions` block — use it).
- **Flag uncertainty rather than guessing.** If you can't find a solid range for, say, BPD
  resistance, say so in your experiment log and pick a defensible placeholder — don't invent a
  confident number. (This is the house rule across AiDen: honest "I'm not sure" beats confident
  wrongness. It matters more here than anywhere.)
- Ask your mentor for access to the AiRA Literature Crawler's extractions if you want a head start
  on published thresholds.

## Phase 1 deliverables

1. **CR: pediatric/neonatal physiology document** — 1–2 pages in your own words: how peds/neonatal
   ventilation differs, the leak mechanism, and a sourced parameter table per condition.
2. **CR: generator extension** — add a weight-indexed parameter path + a leak term to your existing
   engine. A neonatal scenario should produce tiny volumes, fast breaths, and (with leak on) a
   visible VTi > VTe gap.
3. **A small generated dataset** — the 3+ conditions across a modest weight/parameter grid, exported
   in your existing JSON+CSV schema, with `population: neonatal|pediatric|adult` added to metadata.
4. **Validation plots** — representative waveforms next to what the literature says they should look
   like; note any adjustments. (This flows straight into Phase 2.)

---

# PHASE 2 (Week 2) — A Labeling & Validation Exercise

This phase is about the *labels*, and it's designed to be done **with your mentor** — he wants to
learn the labeling process hands-on, and this is the vehicle. You'll design the label scheme, then
you and he will each label a set of waveforms *blind* and compare. Because the data is synthetic,
we have the ground truth to grade both of you against.

## Step 1 — Design a label schema (the vocabulary)

A "label" is a tag that states what's clinically true about a waveform. The schema is the
*controlled vocabulary* of allowed tags plus the rule for when each applies. Draft it as a table —
label name, plain-English definition, and the **objective trigger** (the measurable condition that
makes it true). For example:

| Label | Definition | Objective trigger (example) |
|---|---|---|
| `low_compliance_signature` | Stiff lung; high pressure for delivered volume | driving pressure ÷ VT above threshold |
| `high_resistance_signature` | Obstructed airways | Ppeak − Pplat above threshold |
| `air_trapping` | Incomplete exhalation | end-expiratory flow ≠ 0 at next breath |
| `leak_around_ett` | Uncuffed-tube leak | VTi − VTe above X% of VTi |
| `size_appropriate_low_volume` | Correctly tiny neonatal VT | VT within weight-scaled band |

The hard part isn't listing labels — it's making each trigger **unambiguous**. That ambiguity is
the whole lesson (see Step 4).

## Step 2 — Auto-label from ground truth

Because you set the parameters, you can compute the *true* labels automatically from the trigger
rules. Generate ~30–50 scenarios (mix of conditions/populations) and attach the auto-computed
labels. **This is your gold set** — the answer key.

## Step 3 — Blind human labeling

Now the exercise. Take the same ~30–50 scenarios, **strip the labels**, and render just the
waveform plots. Then, independently and without conferring:

- **You** label each waveform by eye, using only the schema.
- **Your mentor** labels the same set (his first labeling task — that's the point).

Neither of you sees the ground-truth answer key while labeling.

## Step 4 — Compare, measure, and learn

Now the interesting part. Compute three things:

1. **You vs mentor** — where did two humans disagree? This is *inter-annotator agreement*
   (report it as raw % agreement, and if you want, Cohen's kappa). Disagreement usually means the
   *schema* was ambiguous, not that someone was wrong.
2. **Each of you vs the gold set** — accuracy against ground truth. Where a waveform genuinely
   shows the feature but a human missed it (or vice versa), why?
3. **The edge cases** — pull out every scenario where anything disagreed and write up *why*. These
   are gold: they reveal either a fuzzy trigger rule (fix the schema) or a genuine physiological
   grey zone (document it).

## Phase 2 deliverables

1. **CR: label schema** — the vocabulary table with objective triggers, versioned.
2. **The gold-labeled dataset** — scenarios + auto-computed labels.
3. **A short label-agreement report** — the three comparisons above, the disagreement table, and
   what you'd change about the schema as a result. **This report is the real product** — it's what
   makes the dataset trustworthy, and it's the artifact a data scientist would actually keep.

---

## Why this matters (the honest framing)

The team can generate waveforms fast. What's scarce — and what a motivated person with your
training does better than an AI running alone — is **checking whether the data is actually right,
and defining what "right" even means** via a clean label schema. That's the judgment work. Owning
the pediatric domain *and* the labeling discipline on it makes you the person who knows this cold.

There's no race here. This is deliberately not on the critical path, so take the time to understand
the physiology and get the schema clean. Depth beats speed on this one.

---

## How to work (same as before, worth repeating)

- **Break work into small CRs** — problem, current state, proposed change, acceptance criteria,
  files touched. (Continue your `Docs/crs/` numbering.)
- **Keep the experiment log alive** — when a neonatal waveform looks wrong, write down what looked
  wrong, what you checked, what you changed. That log is how we review your *understanding*, which
  matters more than the code.
- **Use your Claude Code agent for the mechanical parts** — literature lookups, the leak-term math,
  the sweep code, the agreement stats. Direct it, then *check* its output against your own
  physiological judgment — especially the pediatric numbers, where a confident-but-wrong value is
  the exact failure mode we care about. Learning to drive the agent well *and* catch when it's
  wrong is itself a goal of this internship.
- **Ask when a waveform or a number doesn't make sense.** Struggling silently isn't the goal;
  building understanding efficiently is.

## References to start with

- Any neonatal/pediatric mechanical ventilation text (e.g. Goldsmith, *Assisted Ventilation of the
  Neonate*) — parameter ranges, RDS/BPD/meconium physiology.
- Pediatric critical care references for weight-scaled VT and RR.
- Your existing adult conditions in `generator/conditions.py` — the peds presets are rescalings +
  leak, not a rewrite.
- The AiRA Literature Crawler extractions (ask your mentor) for published pediatric thresholds.

## Summary

Phase 1: extend the simulator into pediatric/neonatal physiology — rescaled mechanics, tiny fast
breaths, and leak — for a few key conditions, with sourced numbers. Phase 2: design a clean label
schema, auto-label from ground truth to build a gold set, then you and your mentor label blind and
measure agreement — and learn where the schema is fuzzy. The output is a labeled pediatric dataset
you own and a label-agreement report that makes it trustworthy.

Start with the physiology document. Create your first CR. Go.
