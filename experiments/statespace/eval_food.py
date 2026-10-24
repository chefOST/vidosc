"""Reproduce food-only separation results using cached, unchanged phrase scores.

Run: python experiments/statespace/eval_food.py
Classification is an explicit reviewed allowlist of object names, independent of scores.
Foods, ingredients and drinks are included; packaging, tools, waste and ambiguous
contents are excluded. Original mixed sets/scores remain intact.
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/cache/statespace'
FOODS = set('''almond;almond drink;apple;avocado;bacon;bagel;balsamic reduction;balsamic vinegar;banana;bao bun;bean black sauce;bean salted curd;bean spicy sauce;beer;belly pork chunk;biscuit;blackberry;boil water;bouquet garni;bread;bread bit;bread crust;bread half;bread loaf;bread piece;bread stick;bread wrap;broccoli;broccoli piece;butter;butter slice;cabbage;candy;caper;carrot;carrot bit;carrot choppings;cass erole;celery;cereal;cereal bar;cheese;cheese curd;cheese mixture;cherry tomato;chicken;chicken breast;chicken thigh;chilli;chinese vinegar;chop tomato;chopped lettuce;chopped tomato;cocktail quorn sausage;cocoa powder;coffee;coffee bean;coffee ground;cold water;condiment;cooked sprout;cookie;coriander;coriander stalk;corn piece;courgette;courgette bit;courgette dice;cracker;cream;crumbles;cucumber;cucumber end;cucumber piece;cumin seed;curry leaf;curry only leaf;curry powder;doritos;dough ball;dough side;dried sun tomato;egg;egg scotch piece;excess butter;fish;fish finger;fish piece;fish sauce;fish skin;flour ball;food;fourth tomato;garlic;garlic piece;gin;ginger;gochujang;granola;grated cheese;half orange;ham;ham slice;hamburger;honey;hot dog;hot water;hummus;ice cube;jam;juice;ketchup;kitkat;kiwi;leek;leftover food;lemon;lemon grass;lemon zest;lentil;lettuce leaf;lime juice;lime piece;loaf;mango;maple syrup;mayonnaise;meat;meat bit;milk;milk cream;minced meat;mixture;mocha;more beer;more cheese;more flour;more water;mozzarella slice;mushroom;mushroom piece;new onion;noodle;nut;nutella;oat;oat pancake;oil;olive oil;omelette;omelette edge;onion;onion piece;onion ring;onion spring piece;open meat;oregano;paella;pak choi;pan ingredient;pancake;paneer;paprika;parmesan;pasta;pastry;patty;pea;peach;peanut;peeled tomato;pepper piece;pepper red paste;pesto;pesto sauce;pine nut;pineapple;pineapple slice;pistachio;pitta bread;pitta piece;pizza base;pizza top;pork;potato;potato bit;potato slice;quorn sausage;raisin;raspberry;red chilli;remaining onion;remaining water;rice;roll;rosemary;saag paneer;salad;salad dressing;salad leaf;salad remnant;salami;salami piece;salmon;salmon skin;salt;sausage;scotch egg;sesame oil;shreddies;smoothie;soda;soup;soy milk;soy sauce;soya milk;soya sauce;spaghetti;spice;spring onion;steamed bun;stir fry;stock cube;strawberry;sugar;sweet potato;sweetcorn;taco;tap water;tea;tea leaf;tenderstem broccoli;thyme;tiramisu;tofu;tofu piece;tomato;tomato end;tomato heart;tomato piece;topping;tortilla;tuna;tuna patty;vegetable;vinegar;whey;wrap;yoghurt;yogurt;batter;bread slices;brownie batter;cooked omelette;egg mixture;eggs;peanut butter;pepperoni;pizza;pizza sauce;vegetables;water'''.split(';'))
FOODS.discard('cass erole')  # casserole is ambiguous between dishware and food; excluded
MODELS = ['bge-large', 'qwen3-emb-4b', 'qwen3-emb-8b', 'siglip-centred', 'word-overlap']

def overlap(a,b):
    a,b = (set(x.lower().replace(',', ' ').split()) for x in (a,b))
    return len(a & b)/max(len(a | b),1)

def main(expanded=False):
    scores=json.loads((OUT/('food_expanded_scores.json' if expanded else 'sep_scores.json')).read_text())
    result={}
    for dataset in ['moscato','epic']:
        source=dataset+'-big'
        filename=f'{dataset}_food_expanded_triplets.json' if expanded else f'{dataset}_big_triplets.json'
        data=json.loads((OUT/filename).read_text())
        nouns=sorted({k.split('|')[0] for k in data})
        selected=data if expanded else {k:v for k,v in data.items() if k.split('|')[0] in FOODS}
        (OUT/f'{dataset}_food{"_expanded" if expanded else ""}_triplets.json').write_text(json.dumps(selected,indent=2)+'\n')
        (OUT/f'{dataset}_food{"_expanded" if expanded else ""}_classification.json').write_text(json.dumps({n:('food' if expanded or n in FOODS else 'excluded') for n in nouns},indent=2)+'\n')
        pairs=scores['pairs'][source]; idx={tuple(p):j for j,p in enumerate(pairs)}
        raw=scores['scores'][source]
        raw['word-overlap']=[overlap(a,b) for _,a,b in pairs]
        per={}
        anchor_records=[]
        for key,item in selected.items():
            obj,anchor=key.split('|',1)
            vals={}
            for model in MODELS:
                pos=np.array([raw[model][idx[obj,anchor,p]] for p in item['same']])
                neg=np.array([raw[model][idx[obj,anchor,c]] for _,c in item['contra']])
                vals[model]=[float((pos[:,None]>neg).mean()),float(pos.min()>neg.max())]
            per.setdefault(obj,[]).append(vals)
            anchor_records.append((key,obj,item,vals))
        objects=sorted(per)
        sums=np.array([[np.sum([r[m][0] for r in per[o]]) for m in MODELS] for o in objects])
        counts=np.array([len(per[o]) for o in objects])
        rng=np.random.default_rng(0); bs=rng.integers(len(objects),size=(2000,len(objects)))
        boots=sums[bs].sum(1)/counts[bs].sum(1)[:,None]
        means=sums.sum(0)/counts.sum()
        summary={m:{'pair_accuracy':float(means[j]),'strict':float(np.mean([r[m][1] for o in objects for r in per[o]]))} for j,m in enumerate(MODELS)}
        diffs={}
        for a,b in combinations(range(len(MODELS)),2):
            lo,hi=np.quantile(boots[:,a]-boots[:,b],[.025,.975])
            diffs[f'{MODELS[a]} minus {MODELS[b]}']={'difference':float(means[a]-means[b]),'ci95':[float(lo),float(hi)]}
        result[dataset]={'states':len(selected),'foods':len(objects),'mean_paraphrases':float(np.mean([len(x['same']) for x in selected.values()])), 'mean_contradictions':float(np.mean([len(x['contra']) for x in selected.values()])), 'metrics':summary,'comparisons':diffs,'per_object':per}
        if expanded:
            from collections import Counter
            from food_bootstrap import template_cluster, two_way_bootstrap, comparisons
            clusters=[template_cluster(k,item) for k,_,item,_ in anchor_records]
            values=np.array([[vals[m] for m in MODELS] for _,_,_,vals in anchor_records])
            boot2,empty=two_way_bootstrap(values,[o for _,o,_,_ in anchor_records],clusters)
            point=values.mean(0)
            assert np.allclose(point[:,0],means,atol=1e-14,rtol=0)
            res=result[dataset]
            res['comparisons_food_only']=res['comparisons']
            res['comparisons']=comparisons(point,boot2,MODELS)
            res['comparisons_strict_two_way']=comparisons(point,boot2,MODELS,metric=1)
            sizes=Counter(clusters)
            res['bootstrap']={'method':'paired pigeonhole bootstrap: food x recorded template',
                              'resamples':2000,'seed':0,'food_clusters':len(objects),
                              'template_clusters':len(sizes),
                              'recorded_template_families':sum(k.startswith('rule:') for k in sizes),
                              'anchors_with_recorded_templates':sum(c.startswith('rule:') for c in clusters),
                              'anchors_without_template_ids':sum(c.startswith('unrecorded:') for c in clusters),
                              'empty_replicates_redrawn':empty,
                              'missing_template_policy':'one template cluster per original anchor',
                              'weighting':'product of independently resampled food and template multiplicities; equal anchor weight before resampling'}
            res['bootstrap_groups']=[{'anchor':k,'food':o,'template_cluster':c}
                                     for (k,o,_,_),c in zip(anchor_records,clusters)]
            for j,m in enumerate(MODELS):
                res['metrics'][m]['pair_accuracy_ci95_two_way']=np.quantile(boot2[:,j,0],[.025,.975]).tolist()
                res['metrics'][m]['strict_ci95_two_way']=np.quantile(boot2[:,j,1],[.025,.975]).tolist()
            print(json.dumps(res['bootstrap'],indent=2))
            diffs=res['comparisons']
        print(dataset,len(selected),'states',len(objects),'foods')
        print(json.dumps(summary,indent=2));print(json.dumps(diffs,indent=2))
    (OUT/('food_expanded_results.json' if expanded else 'food_sep_results.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    import sys
    main(expanded='--expanded' in sys.argv)
