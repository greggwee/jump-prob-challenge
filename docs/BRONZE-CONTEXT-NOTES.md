# Bronze context incorporation — operator dossier → model channels

*2026-07-18, from the operator's third-place findings dossier (historical base rates
n=20/n=8, rotation + mentality mechanisms, FRA/ENG T3 context). Locks 07-19 01:00Z.*

## Verified sensitivity (offline reprice, FRA/ENG ratio held, total scaled)

The dossier's core claim — third-place games run ~+1 goal over normal knockouts
(modern era 3.62/game vs our anchored 2.77) — maps onto the slate like this:

| question | @2.77 (now) | @3.2 | @3.6 | exposure |
|---|---|---|---|---|
| 4+ total goals | .30 | .40 | .49 | **HIGH** |
| BTTS | .55 | .63 | .68 | **HIGH** |
| any player 2+ goals | .24 | .30 | .36 | **HIGH** |
| sub scores/assists | .48 | .53 | .58 | med (handler scales with λ) |
| goal before HB1 | .38 | .43 | .46 | med |
| tied at HT | .47 | .43 | .41 | med (down) |
| H2 > H1 goals | .50 | .53 | .54 | low |
| France scores first | .55 | .57 | .57 | low |
| France wins Bronze (advance) | .624 | .633 | .640 | **~none** |
| each team 4+ SoT | .198 | .194 | .191 | ~none (pinned 24) |
| 20+ total shots | .68 | .67 | .67 | ~none |

Draw-rate check (dossier: 19/20 decided in 90'): grid draw falls .258 → .218 at
total 3.6; historical ~.05–.10 is lower still, but the **outright is insensitive**
(.624→.640) because `advance` redistributes draw mass ~proportionally. The
already-noted optional book pin (0.63) covers the residual.

## Incorporation map (finding → channel → action)

1. **Openness/totals (T1)** → **the live book anchor** (primary, automatic). The
   book already prices the third-place premium (O2.5 −250 vs a typical knockout
   ~−120) and will move further as lineups confirm; the T-2 recompute carries
   every HIGH-exposure question with it. Do NOT manually override anchored goal
   markets off n=8 history — the market-first rule stands.
   **At-lock check (new)**: print the fitted bronze λ total next to the live
   O2.5 devig. The opening fit (2.77) sits below the book's over-implied total
   (~3.2+) because the 1X2 draw (+280 ≈ .25) and O2.5 (−250) are in tension and
   the fit compromises. The dossier's decided-in-90 evidence (19/20; P(record |
   draw=.25) ≈ 2%) says the truth is nearer the HIGH-total/LOW-draw corner. If
   the live fit still lands ≤2.9 while O2.5 devig ≥.65, the 4+goals/BTTS/brace
   trio inherits the gap — use the sensitivity row above to bound a manual
   override, and prefer the smallest defensible move.
2. **Rotation → both defences non-first-choice (T3)** → already priced: Saliba
   out is in the line; Kane/Dembélé have T-60 branch rules; Rice/Saka/James
   resting flows through the book + player-prop anchors at lock. No new action.
3. **Rotation does NOT depress team volume (T1 §5 empirics)** → consistent with
   the each-team-SoT pin at 24 and the 20+ shots keep at 68 — both verified
   tempo-INSENSITIVE in the model, so the pins carry the view regardless of the
   anchor's total.
4. **Golden Boot race (Mbappé 8 / Kane 6 / Bellingham 6) (T3)** → supports the
   scorer props staying priced (book +115 Kane agrees) and adds a concentration
   argument on the brace market ABOVE the tempo effect. Judgment: no manual
   nudge — brace realized 0/4 at crowd 27.5 this WC, and the tempo channel
   already lifts it at lock. Revisit only if the live anchor total ≥3.2 AND both
   stars start (then the .30+ row applies naturally anyway).
5. **Decided-in-90 / margins (T1)** → no ET or margin question on the bronze
   slate; outright verified insensitive. No action.
6. **Mentality asymmetry (Deschamps farewell, Tuchel criticism) (T3)** → no
   priceable channel beyond the book's 1X2; explicitly NOT hand-priced (the
   handoff's standing rule: motivation is real but the book is the best proxy).

## What fires when lineups confirm

- WebPanel writes bench/out → player-prop caps + Dembélé/Oyarzabal branch rules
  + lineup-aware race handler (final) — all automatic at T-2.
- Book anchor + `--refresh-props` reprice everything anchored — automatic.
- Manual (only if the fit/book tension above materializes): bounded override on
  4+ goals per the sensitivity table; B-team check on the each-team SoT pin
  (≥8 changes on either side → revert 24 → 21, per the pushback review).
