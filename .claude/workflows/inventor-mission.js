export const meta = {
  name: 'inventor-mission',
  description: 'Upskilled inventor (TRIZ/ARIZ, IFR) invents research methods, non-standard trade geometry and data-grounded setups; skeptic + supertrader check; ranked implementation plan',
  whenToUse: 'After the inventor knowledge base exists (.claude/agents/knowledge/inventor.md) and new data facts are available (screening maps, reverse-engineering rules, regimes, cohorts).',
  phases: [
    { title: 'Invent', detail: '3 streams: research methods, non-standard geometry, data-grounded setups' },
    { title: 'Check', detail: 'skeptic (testability, look-ahead, duplicates) and supertrader (live-trading practice) in parallel' },
    { title: 'Plan', detail: 'ranked implementation plan with code specs' },
  ],
}

const REPO = '/home/user/111'
const INVENTOR = `You are the project's SUPER-INVENTOR. First read ${REPO}/.claude/agents/inventor.md and your knowledge base ${REPO}/.claude/agents/knowledge/inventor.md, then PLAN.md, INSIGHTS.md, HYPOTHESES.md. Work every problem through the IDEAL FINAL RESULT (ИКР-1, ИКР-2), explicit physical contradictions, resources of our system and TRIZ principles; finish with your self-check checklist. Every output must be implementable with our data and tools (tp/engine.py, tp/screen.py, tp/placebo.py, tp/reverse.py, tp/hyp/) and testable against placebo substitutions; no look-ahead; a real payer. Do not modify repo files; you may run read-only python on data/raw via tp.data.load (distributions, not PnL). Russian.`

const ITEM = { type: 'object', properties: {
  id: { type: 'string' }, stream: { type: 'string' }, title: { type: 'string' },
  ikr: { type: 'string', description: 'ideal final result' }, contradiction: { type: 'string' },
  triz: { type: 'string' }, payer: { type: 'string' },
  spec: { type: 'string', description: 'exact algorithm / rule / geometry, ready to code; for engine extensions name the change in tp/engine.py' },
  test: { type: 'string', description: 'how to validate (which placebo, cohorts, periods), what falsifies it' },
  expected: { type: 'string' }, effort: { enum: ['S', 'M', 'L'] } },
  required: ['id', 'stream', 'title', 'ikr', 'contradiction', 'triz', 'payer', 'spec', 'test', 'expected', 'effort'] }
const ITEMS = { type: 'object', properties: { items: { type: 'array', items: ITEM } }, required: ['items'] }

const STREAMS = [
  { key: 'method', prompt: `STREAM "method": invent 5–8 new RESEARCH METHODS for finding edges, at the level of our reverse engineering (tp/reverse.py: label strong moves -> precursors with lift vs base rate -> asymmetry -> forward validation with placebos). Examples of the level we want (do not just repeat): failure mining (study where our best setups lose and invert), conditional event studies by regime/cohort, counterfactual trader, cross-sectional ranking discovery, anomaly transfer between cohorts. Each: the algorithm step by step, what it outputs, how to guard it against overfitting (discovery/validation split, placebo).` },
  { key: 'geometry', prompt: `STREAM "geometry": invent 6–10 NON-STANDARD trade geometries (entry, stop, targets, management, exit logic) that our grid (tp/research.py SCHEMES/exit_grid, tp/screen.py E/X/Y/H grid) does not cover — for the setups that passed screening (liq_flush, capitulation, stop_hunter, absorption; see results/screen/100/*.json plateau cells and INSIGHTS.md) and in general. Think: exits on OI/delta/funding state change, exits timed to funding settlement or session, entries at levels of past liquidation clusters or swept highs/lows, stop beyond the signal's own extreme with time-conditional tightening, reverse-on-stop, scale-in on confirmation, volatility- or regime-adaptive targets, basket execution at market-wide events (which coin at the moment of a market-wide cascade). Specify the exact engine change if one is needed.` },
  { key: 'setup', prompt: `STREAM "setup": invent 8–12 new setups GROUNDED IN OUR DATA FACTS: reverse-engineering rules (results/reverse/rules_100.json and logs/reverse_validate.log: e.g. momentum continuation on volume, near-weekly-low EU/US session drops), screening maps (results/screen/100/*.json: which geometry plateaus, which edge types), the market-regime table, the 5 liquidity cohorts, the failures recorded in HYPOTHESES.md. Turn precursors into mechanisms (who pays) and mechanisms into setups with entry type, stop, 3 variants; resolve the contradictions the facts reveal.` },
]

phase('Invent')
const inv = await parallel(STREAMS.map(s => () => agent(`${INVENTOR}\n\n${s.prompt}\nids prefixed "${s.key}-".`, { label: `invent:${s.key}`, phase: 'Invent', schema: ITEMS })))
const items = inv.filter(Boolean).flatMap(r => r.items)
log(`Inventor: ${items.length} items`)

phase('Check')
const V = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, verdict: { enum: ['implement', 'rework', 'reject'] }, reason: { type: 'string' },
  fix: { type: 'string' }, priority: { type: 'integer', minimum: 1, maximum: 5 } },
  required: ['id', 'verdict', 'reason', 'fix', 'priority'] } } }, required: ['verdicts'] }
const [sk, tr] = await parallel([
  () => agent(`You are a sceptical quant (read ${REPO}/PLAN.md, INSIGHTS.md, HYPOTHESES.md). For EACH item: implement / rework (how) / reject (no payer, look-ahead, untestable with our data, duplicate of existing or rejected work). Priority 1 = first. Russian.\nITEMS:\n${JSON.stringify(items)}`, { label: 'check:skeptic', phase: 'Check', schema: V, model: 'sonnet' }),
  () => agent(`You are the project's SUPERTRADER: read ${REPO}/.claude/agents/supertrader.md and your knowledge base ${REPO}/.claude/agents/knowledge/supertrader.md. For EACH item judge as a live trader on Binance futures with small size: would it work in practice or is there a trap (execution, slippage on small caps, fake signals, crowd adaptation, stops inside noise, fees)? implement / rework (how) / reject; priority 1 = first. Russian.\nITEMS:\n${JSON.stringify(items)}`, { label: 'check:trader', phase: 'Check', schema: V }),
])
const m = r => Object.fromEntries(((r && r.verdicts) || []).map(v => [v.id, v]))
const S = m(sk), T = m(tr)
const judged = items.map(i => ({ ...i, skeptic: S[i.id] || null, trader: T[i.id] || null }))

phase('Plan')
const plan = await agent(`Build a ranked implementation plan (Russian, markdown) from these inventions and their two reviews (skeptic = testability, trader = live practice). Keep items that neither reviewer rejected, apply their fixes, order by expected value / effort. For each kept item give: what to code where (tp/engine.py change, new module, tp/hyp/binv_*.py setup file), the validation protocol (tp.screen with placebos, cohorts, periods), and the falsification criterion. End with a list of rejected items with one-line reasons. Project: ${REPO} (PLAN.md, INSIGHTS.md).\nITEMS:\n${JSON.stringify(judged)}`, { label: 'plan', phase: 'Plan', model: 'sonnet' })
return { items: judged, plan }
