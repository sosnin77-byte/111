export const meta = {
  name: 'duo-cycle',
  description: 'Supertrader + inventor cycle on a setup: post-mortem of wins/losses -> TRIZ rules -> practice check -> code + screening with placebos',
  whenToUse: 'A setup passed screening (tp.screen) and we want to strengthen it: args = {setups: [{name, variant}], notes: "prior findings"}',
  phases: [
    { title: 'Postmortem', detail: 'supertrader reads 15 winners and 15 losers with the path after the signal' },
    { title: 'Invent', detail: 'inventor turns the trader patterns and test results into TRIZ-resolved rules' },
    { title: 'Practice', detail: 'supertrader checks the rules against real trading practice' },
    { title: 'Test', detail: 'implement rules as setups, run checker and screening with placebos, compare with base' },
  ],
}

const REPO = '/home/user/111'
const items = (args && args.setups) || [{ name: 'liq_flush', variant: 2 }]
const notes = (args && args.notes) || ''

const TRADER = `You are the project's SUPERTRADER (read ${REPO}/.claude/agents/supertrader.md for your role). This time it is a POST-MORTEM, not a blind test: you see finished trades of the optimisation period with what happened after the signal.`
const INVENTOR = `You are the project's SUPER-INVENTOR (read ${REPO}/.claude/agents/inventor.md for your role and method: TRIZ/ARIZ, contradictions, resources, ideal final result; every solution is a testable rule on our columns, no look-ahead).`
const COMMON = `Project context: read ${REPO}/PLAN.md, INSIGHTS.md, HYPOTHESES.md first. Write in Russian. Do not call the TRADER.PRO API. Do not change files other than those you are told to write.`

const PM = { type: 'object', properties: {
  patterns: { type: 'array', items: { type: 'object', properties: {
    pattern: { type: 'string' }, seen_in: { enum: ['winners', 'losers', 'both'] },
    how_to_see_before_entry: { type: 'string', description: 'what in the pre-signal snapshot reveals it' },
    management_fix: { type: 'string', description: 'entry/stop/targets/management change that would help, or empty' },
    cases: { type: 'array', items: { type: 'string' } } },
    required: ['pattern', 'seen_in', 'how_to_see_before_entry', 'management_fix', 'cases'] } },
  untradable_losers: { type: 'string' }, summary: { type: 'string' } },
  required: ['patterns', 'untradable_losers', 'summary'] }

const RULES = { type: 'object', properties: {
  rules: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, kind: { enum: ['filter', 'entry', 'stop', 'targets', 'management', 'new_setup'] },
    contradiction: { type: 'string' }, triz: { type: 'string' }, rule: { type: 'string' },
    formula: { type: 'string', description: 'exact condition known at signal close, on tp.data.load columns / F helpers / ref(BTC)' },
    variants: { type: 'string' }, expected_effect: { type: 'string' } },
    required: ['id', 'kind', 'contradiction', 'triz', 'rule', 'formula', 'variants', 'expected_effect'] } } },
  required: ['rules'] }

const PRACTICE = { type: 'object', properties: {
  verdicts: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, verdict: { enum: ['keep', 'modify', 'drop'] },
    trap: { type: 'string', description: 'what goes wrong in live trading, or empty' }, change: { type: 'string' } },
    required: ['id', 'verdict', 'trap', 'change'] } } },
  required: ['verdicts'] }

const TEST = { type: 'object', properties: {
  file: { type: 'string' },
  results: { type: 'array', items: { type: 'object', properties: {
    name: { type: 'string' }, rule_ids: { type: 'array', items: { type: 'string' } },
    signals_is: { type: 'integer' }, plateau_t: { type: 'number' },
    p_coin: { type: 'number' }, p_time: { type: 'number' }, p_dir: { type: 'number' },
    edge_type: { type: 'string' }, oos_plateau_pct: { type: ['number', 'null'] }, vs_base: { type: 'string' } },
    required: ['name', 'rule_ids', 'signals_is', 'plateau_t', 'p_coin', 'p_time', 'p_dir', 'edge_type', 'oos_plateau_pct', 'vs_base'] } },
  conclusion: { type: 'string' } },
  required: ['file', 'results', 'conclusion'] }

const out = await pipeline(items,
  (it) => agent(`${TRADER}\n${COMMON}\nSetup: ${it.name}, variant index ${it.variant}. Run  cd ${REPO} && python -m tp.trader postmortem ${it.name} --variant ${it.variant}  then read results/trader/pm_${it.name}.json (meta: geometry and IS stats; cases: 15 winners and 15 losers, shuffled, each with the pre-signal snapshot and the bars after). Study every case like a trading journal review: what distinguished winners from losers in terms a trader uses, what could be seen BEFORE entry, which losers a professional would never have taken, and which entry/stop/target/management would have saved them. ${notes ? 'Prior findings: ' + notes : ''}`,
    { label: `pm:${it.name}`, phase: 'Postmortem', schema: PM }).then(pm => ({ it, pm })),
  (s) => agent(`${INVENTOR}\n${COMMON}\nSetup: ${s.it.name} (variant ${s.it.variant}); its code is in tp/setups.py or tp/hyp/, its screening result in results/screen/100/${s.it.name}.json (plateau geometry, p-values vs coin/time/direction placebos). The supertrader's post-mortem of its trades:\n${JSON.stringify(s.pm)}\nFormulate the contradictions behind the losing patterns and the missed opportunities, resolve them with TRIZ, and give 3–6 rules (filters, entry, stop, targets, management, or a derived new setup), each with an exact formula known at the signal bar close and 3 variants. ${notes ? 'Prior findings: ' + notes : ''}`,
    { label: `invent:${s.it.name}`, phase: 'Invent', schema: RULES }).then(r => ({ ...s, rules: r })),
  (s) => agent(`${TRADER.replace('This time it is a POST-MORTEN, not a blind test: you see finished trades of the optimisation period with what happened after the signal.', '')}\n${COMMON}\nThe inventor proposed these rules for setup ${s.it.name}, based on your post-mortem. Judge each as a practitioner: would it work in live trading on Binance futures with small size, or is there a trap (execution, slippage on small caps, fake signals, the crowd adapting, stops too tight for the noise)? keep / modify (say how) / drop.\nRULES:\n${JSON.stringify(s.rules)}`,
    { label: `practice:${s.it.name}`, phase: 'Practice', schema: PRACTICE }).then(v => ({ ...s, practice: v })),
  (s) => agent(`You are the implementer. ${COMMON}\nImplement the kept/modified rules for setup ${s.it.name} as new setups in ONE file ${REPO}/tp/hyp/bduo_${s.it.name}.py (see tp/hyp/common.py and existing tp/hyp/b*.py or tp/setups.py for the format: SETUPS = [Setup(...)] with names duo_${s.it.name}_<rule>, 3 variants each, same TF as the base setup unless a rule says otherwise). One setup per rule applied on top of the base signal, plus one combining the best-supported rules. Then run  python -m tp.hyp.check bduo_${s.it.name}  until all OK (no look-ahead, enough signals), then  python -m tp.screen --setups <your setup names> --runs 50  and read results/screen/100/<name>.json. Compare each with the base (results/screen/100/${s.it.name}.json): plateau t, p-values, edge type, signals, OOS at plateau. Do not tune thresholds on screening results — use the variants as proposed.\nRULES:\n${JSON.stringify(s.rules)}\nPRACTICE VERDICTS:\n${JSON.stringify(s.practice)}`,
    { label: `test:${s.it.name}`, phase: 'Test', schema: TEST }).then(t => ({ ...s, test: t })),
)
return out.filter(Boolean)
