"""Log pooled expanded-food separation results to the configured W&B project."""
import json
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parents[2]
def main():
    settings=json.loads((ROOT/'weights_and_biases/settings.json').read_text())
    path=ROOT/'data/cache/statespace/food_expanded_results.json'
    results=json.loads(path.read_text())
    with wandb.init(entity=settings['entity'],project=settings['projects']['experiments'],
                    name='ryan/state-separation-food-two-way-bootstrap',job_type='evaluation',
                    tags=['state-space','food','text-only'],
                    config={'bootstrap_resamples':2000,'bootstrap_unit':'food name x recorded template family',
                            'authors':['Claude (original)','GPT-6 Codex (additions)'],
                            'moscato_extensions':'synthetic short labels, not MOSCATO annotations',
                            'aggregation':'pooled within each set; equal weight per anchor'}) as run:
        rows=[]
        for name,result in results.items():
            run.summary[f'{name}/states']=result['states']
            run.summary[f'{name}/food_names']=result['foods']
            run.summary[f'{name}/bootstrap']=result.get('bootstrap',{})
            run.summary[f'{name}/pair_accuracy_comparisons']=result['comparisons']
            run.summary[f'{name}/pair_accuracy_comparisons_food_only']=result.get('comparisons_food_only',{})
            for model,metrics in result['metrics'].items():
                rows.append([name,model,metrics['pair_accuracy'],metrics['strict']])
                for k,v in metrics.items():run.summary[f'{name}/{model}/{k}']=v
        run.log({'results':wandb.Table(columns=['set','model','pair_accuracy','strict'],data=rows)})
        print(run.url,flush=True)

if __name__=='__main__':main()
