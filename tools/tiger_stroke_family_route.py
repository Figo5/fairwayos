"""Compose a declared stroke-family research route; never select from outcomes."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ghostcaddie.demo import target_benchmark as bench


def compose(classic, learned, cohort):
    """Only IDs and operator stroke types enter selection, never target labels."""
    result=copy.deepcopy(classic)
    result['variant']='stroke_family_head_research'
    old={c['id']:c for c in result['clips']};new={c['id']:c for c in learned['clips']}
    if set(old)!=set(new) or set(old)!={c['id'] for c in cohort}:
        raise ValueError('Routes must cover the same complete collection')
    engines={}
    for clip in cohort:
        cid=clip['id'];stroke=clip['stroke_type']
        if stroke not in {'putt','full_swing'}:raise ValueError('Unsupported stroke family')
        if stroke=='putt':
            old[cid]['targets']['clubhead']=copy.deepcopy(new[cid]['targets']['clubhead'])
            engines[cid]='seed-supported SAM2 head'
        else:engines[cid]='classic head baseline replay'
    return result,engines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','baseline','learned','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();dataset,sha=bench.load_dataset(args.dataset,verify_hashes=True)
    paths=[args.out,Path(str(args.out)+'.report.json'),Path(str(args.out)+'.report.md')]
    if any(path.resolve() in {x.resolve() for x in [args.dataset,args.baseline,args.learned]}|{Path(c['path']).resolve() for c in dataset['clips']} for path in paths):
        raise ValueError('Output must preserve all inputs')
    if any(path.exists() for path in paths):raise ValueError('Choose a fresh output name')
    classic=json.loads(args.baseline.read_text());learned=json.loads(args.learned.read_text())
    bench.validate_predictions(classic,dataset);bench.validate_predictions(learned,dataset)
    payload,engines=compose(classic,learned,[{'id':c['id'],'stroke_type':c['stroke_type']} for c in dataset['clips']])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(bench.canonical_json(payload))
    runtime={'research_only':True,'production_eligible':False,'model_rerun':False,
        'selection_rule':'putt: learned mask; full_swing: classic baseline, by declared operator stroke type only',
        'selection_uses_target_labels':False,'actual_engine_by_clip':engines,
        'baseline_predictions_sha256':bench.sha256_file(args.baseline),'learned_predictions_sha256':bench.sha256_file(args.learned),
        'tool_sha256':bench.sha256_file(Path(__file__)),
        'note':'Development choice on already inspected AI-reference footage, not independent holdout success. Ball blocks remain classic. No per-clip outcome-based switch.'}
    report=bench.evaluate(dataset,bench.validate_predictions(payload,dataset),dataset_path=str(args.dataset),dataset_sha256=sha,predictions_path=str(args.out),predictions_sha256=bench.sha256_file(args.out),runtime=runtime)
    Path(str(args.out)+'.report.json').write_text(bench.canonical_json(report));Path(str(args.out)+'.report.md').write_text(bench.render_markdown(report))

if __name__=='__main__':main()
