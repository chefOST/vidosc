"""Top up EPIC food phrases to >=600 anchors; preserve the MOSCATO label vocabulary.
Generate 8 paraphrases and 10 contradictions using the existing blind Claude helper.
Selection is deterministic and independent of model scores; new food names first.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
import eval_food
import gen_triplets
import eval_sep

EXTRA = set('''artichoke;aubergine;baby potato;bagel half;baking powder;bao;basil;bay leaf;bean;beef;bell pepper;berry;biscuit piece;black pepper;blueberry;boiled water;bread end;bread inside;bread slice;breadcrumb;breadstick;breakfast bar;broccoli floret;broth;bun;butter bit;butter cube;butter piece;cake;canned almond;carrot bottom;carrot chunk;carrot edge;carrot end;carrot handful;carrot piece;carrot slice;cayenne pepper;cheese lengthways;cheese pesto;cheese piece;cheese slice;cherry;cherry syrup;chicken piece;chickpea;chilli bit;chilli flake;chilli oil;chip;chocolate;chocolate crumbles;chocolate mixture;chocolate piece;choi leaf;choi pak leaf;chop steak;chopped cucumber;chopped onion;chorizo;ciabatta;cilantro;cinnamon;clam;clove;coconut;coconut milk;coconut oil;coffee grind;coke;condensed milk;cooked rice;cooking wine;corn;corn cob;courgette piece;courgette slice;crab stick;cream cheese;crisp;crispy pancake;crushed cracker;cucumber bit;cucumber salad;cumin;curd;curd lemon mixture;curry;dice garlic;diced carrot;dough;dough chunk;dough edge;dough flour;dough partition;dressing;drink;dumpling;egg flour;egg omelette;egg yolk;excess onion;extra flour;fat;feta cheese;fifth tomato;filtered water;fish fragment;flour;flour egg mixture;fried egg;frozen food;fruit;fruit juice;galangal;galangal root;garlic bit;garlic bulb;garlic clove;garlic paste;garlic powder;ginger piece;grape;gravy;gravy cube;greek yoghurt;green bean;green chilli;hamburger bun;herb;ice cream;instant noodle;jambalaya;jeera;kale;kidney bean;lady finger;lasagne;leek bottom;leek piece;leftover stock;lemon curd;lemon half;lemon juice;parmesan cheese;parsley;passata;passata mixture;pear;pear chunk;pepper;pepper bell bit;pepper bit;pepper middle;pepper slice;pie;piece onion part;piece pepper part;pizza piece;pizza slice;plain flour;plum;pork belly;pork piece;porridge;prawn;prosciutto;radish;remaining cheese;remaining dough;rice vinegar;rocket;rusk;salad more leaf;salad piece;salsa;sandwich;sandwich roll;sauce;sausage roll;seasoning;second egg;second tomato;sesame roll;seventh tomato;shrimp paste;sixth tomato;sliced cucumber;sliced garlic;smashed garlic;soy;spinach;spinach leaf;spring cut onion;sprout;steak;stem broccoli;stock;sweet corn;syrup;tahini;tahini sauce;tamagoyaki slice;third tomato;thyme spice;toast;tofu block;tofu slice;tomato cherry half;tomato paste;tomato puree;tonic water;tortellini;truffle oil;tuna burger;tuna mixture;turkey breast;turkey piece;turmeric;vegetable piece;waffle;warm water;whisked egg;white bread;wine;wine red vinegar;wine vinegar;yeast'''.split(';'))

def main():
    p=eval_food.OUT/'epic_food_triplets.json'
    data=json.loads(p.read_text())
    source=json.loads((eval_food.ROOT/'data/epic/state_triplets.json').read_text())
    old_names={k.split('|')[0] for k in data}
    allowed=eval_food.FOODS | EXTRA
    candidates=[k for k in sorted(source) if k not in data and k.split('|')[0] in allowed]
    rng=random.Random(0);rng.shuffle(candidates)
    # One anchor for every additional food name before topping up existing names.
    by={}
    for k in candidates: by.setdefault(k.split('|')[0],[]).append(k)
    selected=[]
    for obj in sorted(set(by)-old_names): selected.append(by[obj].pop())
    # Round-robin additions across names; no preferred foods.
    names=sorted(by)
    rng.shuffle(names)
    while len(data)+len(selected)<600 and any(by.values()):
        for obj in names:
            if by[obj]:
                selected.append(by[obj].pop())
                if len(data)+len(selected)>=600:
                    break
    pending={'selection':'additional food names first, then round-robin; no preferred foods',
             'base_states':len(data),'target_states':len(data)+len(selected),
             'additional_anchors':selected,
             'food_names':sorted(old_names | {k.split('|')[0] for k in selected})}
    (eval_food.OUT/'food_topup_selection.json').write_text(json.dumps(pending,indent=2)+'\n')
    if '--select-only' in __import__('sys').argv:
        print(len(selected),'additional anchors;',len(pending['food_names']),'food names total',flush=True)
        return
    target=eval_food.OUT/'epic_food_expanded_triplets.json'
    done=json.loads(target.read_text()) if target.exists() else dict(data)
    todo=[k for k in selected if k not in done]
    print(f'{len(data)} existing + {len(selected)} additions; {len(todo)} remaining',flush=True)
    def run(keys):
        lines='\n'.join(f'[{i}] {k.split("|",1)[0]}: {k.split("|",1)[1]}' for i,k in enumerate(keys))
        r=gen_triplets.llm.call(eval_sep.BIG.format(lines=lines),tag='foodtopup')['items']
        out={}
        for i,k in enumerate(keys):
            v=r.get(str(i))
            if not isinstance(v,dict) or len(v.get('same',[]))!=8 or len(v.get('contra',[]))!=10:
                raise ValueError(f'Incomplete triplet for {k}')
            out[k]=v
        return out
    batches=[todo[i:i+15] for i in range(0,len(todo),15)]
    with ThreadPoolExecutor(4) as pool:
        for r in pool.map(run,batches):
            done.update(r);target.write_text(json.dumps(done,indent=2)+'\n');print(len(done),'saved',flush=True)
    metadata={'source':'EPIC before/after anchor phrases; blind Claude paraphrases and contradictions',
              'base_states':len(data),'added_keys':selected,'food_names':sorted({k.split('|')[0] for k in done}),
              'states':len(done)}
    (eval_food.OUT/'epic_food_expanded_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('complete',len(done),'states',len(metadata['food_names']),'food names',flush=True)

if __name__=='__main__':main()
