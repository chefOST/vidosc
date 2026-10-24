"""GPT-authored compositional food-state test additions (no external LLM call).

Explicit synonym/counterfactual rules change one asserted component and preserve the
remaining anchor wording. Shared templates make additions correlated: report them
separately from the original Claude-authored cohort as well as in pooled results.
"""
import itertools
import json
import random
import re
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'data/cache/statespace'
RULES=[]

def rule(pattern, same, contra, aspect):
    RULES.append((re.compile(r'(?<!\w)(?:'+pattern+r')(?!\w)'),same.split(';'),contra.split(';'),aspect))

rule(r'held in hand|held in fingers|in hand',
     'held in the hand;gripped by hand;in someone\'s grasp;being held;held manually;resting in a hand;in the cook\'s hand;held by the cook',
     'resting untouched on the counter;lying unheld on a plate;unheld on a cutting board;resting unheld inside a bowl;sitting untouched in the refrigerator;lying unheld on a tray;resting unheld in a pot;left untouched inside a cupboard;sitting unheld on a shelf;resting unheld inside a container','location')
rule(r'held with chopsticks',
     'gripped with chopsticks;between chopstick tips;being picked up with chopsticks;held between two chopsticks;supported by chopsticks;lifted with chopsticks;clasped with chopsticks;in the grip of chopsticks',
     'resting untouched on a plate;lying unheld on the counter;resting unheld in a bowl;lying untouched on a board;resting unheld in a pan;held solely with a fork;held solely with a spoon;held solely with tongs;resting untouched on a tray;resting unheld in a box','location')
rule(r'on chopping board|on the chopping board|on the cutting board|on cutting board|on board|on the board',
     'on a chopping board;on the cutting board;resting on a cutting board;lying on the chopping board;placed on the cutting board;on the board surface;atop the chopping board;resting atop a cutting board',
     'inside the refrigerator drawer;inside a closed storage jar;floating inside a cooking pot;inside the oven on a rack;inside a sealed bag;inside a kitchen cupboard;submerged in the sink;inside a closed food container;inside a microwave oven;inside the freezer drawer','location')
rule(r'on the counter|on counter',
     'on the worktop;resting on the counter;on the kitchen work surface;atop the counter;placed on the worktop;resting on the work surface;on the countertop;lying on the kitchen counter',
     'inside the refrigerator;inside the closed oven;inside a sealed cupboard;inside the freezer;submerged in the sink;inside the microwave;on the dining table instead;inside the dishwasher;on the pantry shelf instead;inside a closed drawer','location')
rule(r'inside pan|inside the pan|in the pan|in pan|into pan',
     'inside the cooking pan;in a pan;within the pan;in the cooking pan;inside a pan;contained in the pan;in the pan interior;within a cooking pan',
     'on a plate outside the pan;on a board outside the pan;in a bowl outside the pan;in the fridge outside the pan;in a jar outside the pan;on a tray outside the pan;on a counter outside the pan;in a bag outside the pan;in a box outside the pan;in a cup outside the pan','location')
rule(r'cooking in the pan|frying in the pan',
     'frying in a pan;cooking inside a pan;being pan-fried;heating in the pan;cooking on the pan surface;being fried in a pan;pan-frying;being cooked in a pan',
     'uncooked in the refrigerator;uncooked on the chopping board;uncooked in its packet;uncooked on the counter;uncooked on a plate;uncooked in a bowl;uncooked on a tray;uncooked in the freezer;uncooked in a storage container;uncooked in a sealed bag','cooked or raw or heated')
rule(r'in the bowl|inside the bowl|inside bowl|in bowl',
     'inside a bowl;in a mixing bowl;within the bowl;contained in a bowl;in the bowl interior;inside the mixing bowl;within a bowl;contained inside the bowl',
     'on a board outside the bowl;on a plate outside the bowl;in a pan outside the bowl;in a jar outside the bowl;in the fridge outside the bowl;on a tray outside the bowl;in a bag outside the bowl;on a counter outside the bowl;in a box outside the bowl;in a cup outside the bowl','location')
rule(r'on the plate|on plate|onto plate',
     'on a plate;resting on the plate;atop the plate;placed on a plate;lying on the plate;resting atop a plate;on the plate surface;served on a plate',
     'in a jar away from the plate;in a pan away from the plate;inside a bag away from the plate;in a bowl away from the plate;in the fridge away from the plate;on a board away from the plate;on a tray away from the plate;on a counter away from the plate;in a box away from the plate;in a cup away from the plate','location')
rule(r'scooped onto spoon|scooped onto measuring spoon|on spoon|on a spoon',
     'on the spoon;resting in a spoon;held on a spoon;scooped onto a spoon;in a spoon;on the spoon bowl;supported by a spoon;in the bowl of a spoon',
     'in a jar away from the spoon;in a pan away from the spoon;inside a bag away from the spoon;in a bowl away from the spoon;on a plate away from the spoon;on a board away from the spoon;on a tray away from the spoon;on a counter away from the spoon;in a box away from the spoon;in a cup away from the spoon','location')
rule(r'in jar|in the jar',
     'inside a jar;contained in a jar;in a storage jar;within the jar;inside the jar;in the jar interior;contained inside the jar;within a storage jar',
     'on a plate outside the jar;in a bowl outside the jar;in a pan outside the jar;on a board outside the jar;in a bag outside the jar;in a cup outside the jar;on a tray outside the jar;on a counter outside the jar;in a pot outside the jar;on a spoon outside the jar','location')
rule(r'in the pot|inside pot|in pot|into pot',
     'inside the pot;in a cooking pot;within the pot;contained in the pot;in the pot interior;inside a cooking pot;within a cooking pot;contained inside the pot',
     'on a plate outside the pot;in a bowl outside the pot;in a jar outside the pot;on a board outside the pot;in a bag outside the pot;in a cup outside the pot;on a tray outside the pot;on a counter outside the pot;in a box outside the pot;on a spoon outside the pot','location')
rule(r'in the packet|in packet|in the open packet',
     'inside the packet;contained in a packet;within the packet;in a packet;inside a packet;in the packet interior;contained inside the packet;within a food packet',
     'on a plate outside the packet;in a bowl outside the packet;in a pan outside the packet;on a board outside the packet;in a jar outside the packet;in a cup outside the packet;on a tray outside the packet;on a counter outside the packet;in a pot outside the packet;on a spoon outside the packet','location')
rule(r'in the bag|in bag',
     'inside a bag;contained in a bag;within the bag;inside the bag;in the bag interior;contained inside the bag;within a food bag;in a food bag',
     'on a plate outside the bag;in a bowl outside the bag;in a pan outside the bag;on a board outside the bag;in a jar outside the bag;in a cup outside the bag;on a tray outside the bag;on a counter outside the bag;in a pot outside the bag;on a spoon outside the bag','location')
rule(r'raw',
     'uncooked;not yet cooked;still uncooked;in an uncooked state;before cooking;not cooked yet;still raw;raw and uncooked',
     'fully cooked;cooked all the way through;completely cooked;thoroughly cooked;finished cooking;already fully cooked;cooked throughout;fully cooked through;cooked to completion;done cooking','cooked or raw or heated')
rule(r'whole, uncut|whole|uncut',
     'intact;still in one piece;unbroken;not cut apart;in one intact piece;not divided;undivided;left in one piece',
     'cut into separate pieces;divided into pieces;sliced apart;chopped into pieces;cut into several portions;broken into separate bits;split into multiple pieces;cut up;separated into smaller pieces;fragmented','shape or form')
rule(r'cut into slices|cut into thin slices|sliced',
     'cut into slices;in slices;divided into slices;thinly cut;sliced into portions;cut into thin pieces;in thin slices;cut crosswise into slices',
     'left completely uncut;whole and unsliced;still intact and unsliced;in one uncut piece;not yet sliced;completely whole;undivided and unsliced;whole before slicing;not cut apart;still one whole piece','shape or form')
rule(r'closed|sealed',
     'shut;sealed closed;fully closed;tightly shut;closed up;securely closed;sealed shut;closed tightly',
     'open wide;opened;fully open;left open;unsealed and open;opened up;wide open;open at the top;gaping open;standing open','open or closed')
rule(r'open, cap off|open|uncovered',
     'open;unsealed;opened;left open;open at the top;opened up;unclosed;not closed',
     'sealed shut;tightly closed;fully closed;closed securely;sealed closed;shut tight;closed up;securely shut;unopened;still sealed','open or closed')
rule(r'cut into smaller pieces|chopped into small pieces|chopped|finely diced|diced|cut',
     'cut up;chopped up;divided by cutting;cut into pieces;cut into smaller portions;chopped into pieces;cut apart;divided into cut portions',
     'completely uncut;whole and intact;still in one piece;not yet cut;unbroken and whole;undivided;intact before cutting;left whole;entire and uncut;not cut apart','shape or form')
rule(r'dry',
     'dry;not wet;free of moisture;without surface moisture;in a dry state;dried;moisture-free;without wetness',
     'wet;soaking wet;moist;damp;water-soaked;soaked;covered in water;thoroughly wet;saturated with water;wet throughout','wet or dry')
rule(r'mixed|tossed, evenly dressed',
     'blended;combined;stirred together;thoroughly mixed;evenly combined;mixed together;blended together;incorporated together',
     'kept completely separate;unmixed;in separate unmixed portions;not combined;unblended;separated into individual components;not stirred together;still in separate parts;uncombined;divided into unmixed components','mixed or separate')
rule(r'separate|separated',
     'separate;kept apart;apart;not combined;uncombined;unmixed;left separate;in separate portions',
     'fully blended;thoroughly mixed;combined together;incorporated together;stirred together;completely mixed;blended together;evenly combined;mixed into one blend;all blended together','mixed or separate')
rule(r'cooked',
     'cooked;fully cooked;done cooking;finished cooking;cooked through;already cooked;cooked completely;in a cooked state',
     'raw;uncooked;not yet cooked;still raw;raw throughout;completely uncooked;before cooking;in a raw state;not cooked yet;still uncooked','cooked or raw or heated')
rule(r'melted|melting',
     'melted;liquefied by heat;turned liquid with heat;in melted form;liquid from heating;heat-melted;in liquid melted form;no longer solid from heating',
     'solid and unmelted;still solid;unmelted;in solid form;remaining solid;not yet melted;solid before melting;firm and unmelted;solid throughout;entirely unmelted','shape or form')
rule(r'moved to new spot|moved',
     'relocated;shifted to another place;transferred to a new position;moved elsewhere;put in a different spot;repositioned;shifted from its previous spot;placed somewhere else',
     'left in its original position;never moved;still in its initial spot;unmoved;remaining where it started;untouched in its original place;not relocated;still at the original location;not shifted;kept in the same position','location')
rule(r'on the tray',
     'on a tray;resting on the tray;atop the tray;placed on a tray;lying on the tray;resting atop a tray;on the tray surface;on a serving tray',
     'in a jar outside the tray;in a bowl outside the tray;in a pan outside the tray;in the fridge outside the tray;in a bag outside the tray;in a cup outside the tray;on a board outside the tray;on a counter outside the tray;in a pot outside the tray;on a spoon outside the tray','location')


rule(r'on porridge', 'atop the porridge;on the porridge;resting on porridge;placed on porridge;on top of porridge;on the porridge surface;over the porridge;added atop porridge', 'on a separate empty plate;in a separate empty bowl;in a separate jar;in a separate bag;in a separate box;in a separate pan;on a separate board;on a separate tray;on a separate spoon;in a separate cup', 'location')
rule(r'oven, being baked', 'baking in the oven;being oven-baked;inside the oven baking;undergoing baking in the oven;in the oven while baking;cooking in the oven;baking inside the oven;being baked inside an oven', 'uncooked on the counter;uncooked in a bag;uncooked on a plate;uncooked in a bowl;uncooked on a tray;uncooked in a cupboard;uncooked in the fridge;uncooked on a board;uncooked in a container;uncooked on a shelf', 'cooked or raw or heated')
rule(r'in carton', 'inside a carton;in the carton;contained in a carton;within a carton;inside the carton;contained inside the carton;in the carton interior;within the carton', 'in a bowl outside the carton;in a pot outside the carton;in a jug outside the carton;in a cup outside the carton;in a pan outside the carton;in a glass outside the carton;in a jar outside the carton;in a bottle outside the carton;in a tub outside the carton;in a flask outside the carton', 'location')
rule(r'solid', 'in solid form;unmelted;still solid;firm and solid;in a solid state;remaining solid;not melted;in unmelted solid form', 'fully melted;completely liquid;liquefied;in liquid form;liquid throughout;entirely melted;fully liquefied;liquid and flowing;melted completely;no longer solid', 'shape or form')
rule(r'in measuring spoon', 'inside a measuring spoon;in the measuring spoon;on a measuring spoon;in a measurement spoon;inside the measurement spoon;in the spoon for measuring;within a measuring spoon;on the measuring spoon', 'in a jar away from the spoon;in a bowl away from the spoon;in a pan away from the spoon;on a plate away from the spoon;on a board away from the spoon;on a tray away from the spoon;in a cup away from the spoon;in a bag away from the spoon;in a box away from the spoon;in a pot away from the spoon', 'location')
rule(r'on bread', 'on a piece of bread;atop bread;on the bread;resting on bread;placed on bread;on top of bread;on the bread surface;laid over bread', 'on an empty plate away from bread;in an empty bowl away from bread;in a jar away from bread;on a board away from bread;in a bag away from bread;on a tray away from bread;in a pan away from bread;in a box away from bread;on a spoon away from bread;in a cup away from bread', 'location')
rule(r'drizzled on dessert', 'drizzled over dessert;poured lightly over dessert;trickled over dessert;spread in drizzles over dessert;dripped over dessert;added in a drizzle to dessert;lightly poured onto dessert;drizzled onto the dessert', 'still entirely in its bottle;still entirely in its jar;still entirely in its jug;still entirely in its can;still entirely in its carton;still entirely in its cup;still entirely in its bowl;still entirely in its container;still entirely in its tub;still entirely in its measuring cup', 'location')
rule(r'sprinkled into bowl', 'sprinkled into a bowl;scattered into the bowl;shaken into the bowl;added by sprinkling into a bowl;dusted into the bowl;lightly scattered into a bowl;sprinkled inside the bowl;added in sprinkles to the bowl', 'still entirely in the jar;still entirely in the packet;still entirely in the shaker;still entirely in the container;still entirely in the tin;still entirely in the bag;still entirely in the box;still entirely in the spice tube;still entirely on the spoon;still entirely in the measuring cup', 'location')
rule(r'inside garlic press or pan', 'in a garlic press or pan;within a garlic press or pan;contained in a garlic press or pan;inside a press or a pan;in the press or the pan;within the press or the pan;inside either garlic press or pan;in either a press or pan', 'on a separate plate;in a separate bowl;on a separate board;in a sealed bag;inside the fridge;in a sealed box;in a closed jar;in a closed tin;in the freezer;on a separate tray', 'location')
rule(r'in the can', 'inside a can;in a can;contained in a can;within the can;inside the can;in the can interior;contained inside a can;within a can', 'in a bowl outside the can;in a pot outside the can;in a jug outside the can;in a cup outside the can;in a pan outside the can;in a glass outside the can;in a jar outside the can;in a bottle outside the can;in a tub outside the can;in a flask outside the can', 'location')
rule(r'scattered in dish', 'spread around the dish;scattered inside the dish;distributed in the dish;sprinkled around the dish;spread across the dish;scattered within the dish;dispersed in the dish;spread inside the dish', 'still entirely in its packet;still entirely in its bag;still entirely in its box;still entirely in its jar;still entirely in its tin;still entirely in its tub;still entirely in its container;still entirely on a separate board;still entirely on a separate plate;still entirely in a separate bowl', 'location')
rule(r'attached to dough mass|attached to pepper', 'still attached to the rest;still joined to the remainder;connected to the main piece;joined to the larger piece;attached to the remaining piece;still connected;not detached from the rest;joined to the rest', 'detached from the rest;separated from the main piece;cut away completely;fully disconnected;removed from the rest;broken free of the remainder;separated completely;no longer attached;cut off and removed;completely detached', 'shape or form')
rule(r'pinched in fingers', 'pinched between fingers;held pinched by fingers;gripped between fingertips;pinched by hand;pressed between fingers;held between fingertips;gripped by the fingers;clasped between fingers', 'resting untouched on a board;resting untouched in a bowl;resting untouched on a plate;resting untouched on a tray;resting untouched in a pan;resting untouched in a box;resting untouched in a bag;resting untouched on a counter;resting untouched on a shelf;resting untouched in a container', 'location')
rule(r'on fridge shelf', 'on the refrigerator shelf;on a fridge shelf;resting on a shelf inside the fridge;on a shelf in the refrigerator;on the fridge shelf;placed on a fridge shelf;resting on the refrigerator shelf;on shelving inside the fridge', 'on the kitchen counter;on the dining table;inside the oven;on the pantry shelf;inside a kitchen cupboard;on a chopping board;inside the freezer drawer;on a serving tray;inside the dishwasher;on the sink drainer', 'location')
rule(r'pierced, runny yolk spreading', 'punctured with yolk running out;broken and leaking runny yolk;pierced with liquid yolk spreading;ruptured with yolk flowing out;broken open with yolk spreading;punctured and spreading liquid yolk;ruptured and leaking yolk;broken with liquid yolk flowing', 'intact and unpierced;whole with an unbroken membrane;unpunctured and intact;sealed in its intact membrane;undamaged and whole;unbroken and unpierced;still intact with no leakage;unruptured and whole;whole without a puncture;intact and not leaking', 'shape or form')
rule(r'in salad bowl', 'inside the salad bowl;in a salad bowl;within the salad bowl;contained in a salad bowl;in the bowl of salad;inside a bowl of salad;within a salad bowl;in the salad serving bowl', 'on a separate plate;inside a separate jar;on a separate board;in a separate pan;on a separate tray;inside a separate bag;inside a separate box;inside a separate cup;on a separate spoon;inside a separate pot', 'location')
rule(r'inside container|in container', 'in a container;inside a food container;within the container;contained inside a container;in the container interior;inside the container;contained in a food tub;within a food container', 'on a separate plate;on a separate board;in a separate pan;on a separate tray;in a separate pot;on a separate spoon;on a separate counter;on a separate rack;on a separate napkin;on a separate serving mat', 'location')
rule(r'inside bread or plate', 'in bread or on a plate;inside bread or on a plate;in a bread portion or on a plate;on a plate or inside bread;on a plate or in bread;within bread or on a plate;inside bread or resting on a plate;in bread or resting on a plate', 'in a sealed jar;inside a closed box;inside a sealed bag;in a storage tub;in a separate cooking pot;in a separate empty bowl;inside a fridge drawer;inside a freezer drawer;on a separate chopping board;on a separate baking tray', 'location')
rule(r'with topping placed on it', 'with topping on it;topped;covered by a topping;with an added topping;with topping added;with a topping applied;with topping laid over it;with a layer of topping', 'bare without any topping;untopped;without topping;with no topping at all;uncovered and untopped;bare and untopped;before any topping is added;without an added topping;with its topping entirely removed;free of topping', 'covered or uncovered')
rule(r'split into cloves', 'separated into cloves;divided into individual cloves;broken into cloves;pulled apart into cloves;disassembled into cloves;split apart into individual cloves;broken apart into cloves;divided up into cloves', 'whole and unsplit;intact as a bulb;still one intact bulb;unbroken as a bulb;unsplit and whole;not separated into cloves;still an unseparated bulb;completely intact;undivided as a bulb;in one unbroken bulb', 'shape or form')
rule(r'in jug', 'inside a jug;in the jug;contained in a jug;within a jug;inside the jug;in the jug interior;contained inside a jug;within the jug', 'in a bowl outside the jug;in a pot outside the jug;in a can outside the jug;in a cup outside the jug;in a pan outside the jug;in a glass outside the jug;in a jar outside the jug;in a bottle outside the jug;in a tub outside the jug;in a flask outside the jug', 'location')
rule(r'wrapped in foil', 'enclosed in foil;foil-wrapped;covered by foil;inside a foil wrapping;wrapped with foil;encased in foil;surrounded by foil wrapping;in its foil wrapper', 'unwrapped with no foil;removed from the foil;bare and unwrapped;without its foil wrapping;uncovered outside the wrapper;foil removed completely;out of the foil wrapper;with all foil taken off;unwrapped and exposed;free of foil covering', 'covered or uncovered')
rule(r'poured out, pot empty', 'poured away leaving an empty pot;emptied out of the pot;removed with nothing left in pot;poured out completely;fully poured out of the pot;poured away with none remaining;emptied from the pot completely;completely decanted from the pot', 'still entirely in the pot;not poured out of the pot;remaining in the pot;unemptied in the pot;still in the pot before pouring;left in the pot;contained in the unemptied pot;not yet decanted;still in the full pot;untransferred from the pot', 'location')
rule(r'inside halved pepper, seeds', 'seeds inside a halved pepper;in a pepper half with seeds;seeds within the pepper half;inside a pepper half containing seeds;within the seeded pepper half;in the halved pepper with seeds;inside the cut pepper with seeds;within the halved seeded pepper', 'fully removed onto a separate plate;fully removed into a separate bowl;fully removed onto a separate board;fully removed into a separate bag;fully removed into a separate box;fully removed into a separate pan;fully removed onto a separate tray;fully removed into a separate jar;fully removed into a separate cup;fully removed onto a separate spoon', 'location')
rule(r'on the peel or tray', 'on a peel or tray;resting on a peel or tray;on the tray or peel;atop a peel or tray;placed on the peel or tray;resting on the tray or peel;on a baking peel or tray;atop the tray or peel', 'inside a sealed box;inside a sealed bag;inside a cooking pot;in the fridge drawer;in the freezer drawer;in a kitchen cupboard;on a separate plate;on a separate board;on a separate counter;inside a storage tub', 'location')
rule(r'in bottle', 'inside a bottle;in the bottle;contained in a bottle;within a bottle;inside the bottle;in the bottle interior;contained inside a bottle;within the bottle', 'in a bowl outside the bottle;in a pot outside the bottle;in a can outside the bottle;in a cup outside the bottle;in a pan outside the bottle;in a glass outside the bottle;in a jar outside the bottle;in a jug outside the bottle;in a tub outside the bottle;in a flask outside the bottle', 'location')
rule(r'covered with grated cheese', 'topped with grated cheese;coated with grated cheese;with grated cheese over it;under a layer of grated cheese;covered by grated cheese;with grated cheese on top;blanketed with grated cheese;with a grated-cheese covering', 'bare with no cheese;without any cheese topping;uncovered and cheeseless;with no grated cheese;untopped by cheese;before cheese is added;with all cheese removed;without a cheese covering;free of cheese;with no cheese on top', 'covered or uncovered')
rule(r'in kettle or cup', 'in a kettle or cup;inside a kettle or cup;in the kettle or the cup;within a kettle or cup;contained in a kettle or cup;in either kettle or cup;inside either kettle or cup;within the kettle or cup', 'entirely in a separate jar;entirely in a separate bottle;entirely in a separate bowl;entirely in a separate pan;entirely in a separate jug;entirely in a separate bucket;entirely in a separate tub;entirely in a separate pot;entirely in a separate basin;entirely in a separate flask', 'location')
rule(r'poured into hot pan', 'poured into a hot pan;transferred by pouring into a hot pan;tipped into the hot pan;added by pouring into a hot pan;decanted into a hot pan;poured inside the hot pan;emptied into the hot pan;poured onto the hot pan', 'entirely in its original bowl;entirely in its original jug;entirely in its original cup;entirely in its original container;entirely in its original jar;entirely in its original pot;entirely in its original measuring cup;entirely in its original tub;entirely in its original bottle;entirely in its original mixing bowl', 'location')

rule(r'cut off|sliced off', 'detached by cutting;cut away;severed by cutting;sliced away;removed by cutting;cut free;cut apart from the rest;separated by cutting', 'still attached;not yet cut away;joined to the rest;still connected;not severed;attached to the main piece;not cut free;still joined to the remainder;unseparated;not yet detached', 'shape or form')


# Multiword rules keep related qualifiers coherent when a state is flipped.
rule(r'with slice cut out', 'with a slice removed;missing a cut slice;with one slice taken out;with a slice cut away;partly sliced with a piece removed;with a cut portion removed;with a slice taken away;with a portion sliced out', 'whole with no slice removed;intact before slicing;complete and unsliced;whole before any cut;uncut with all portions present;entire and unsliced;whole with nothing removed;intact with no cut portions;complete before cutting;whole and uncut', 'shape or form')
rule(r'cut into long lengthways slices', 'sliced lengthways;cut lengthwise into long slices;in long lengthwise slices;lengthwise sliced;cut into long strips;divided into long lengthwise slices;sliced into long strips;cut lengthways into strips', 'left whole and uncut;still completely unsliced;in one intact block;uncut and undivided;whole before slicing;left in one piece;not cut into slices;intact and unsliced;not yet sliced;undivided and whole', 'shape or form')
rule(r'cut into strips', 'in strips;cut lengthwise into strips;sliced into strips;divided into strips;cut up into strips;in long cut strips;sliced lengthwise;separated into cut strips', 'whole and uncut;intact and unsliced;left in one piece;still completely uncut;not cut into strips;whole before cutting;undivided;entire and uncut;not yet sliced;left whole', 'shape or form')
rule(r'cut into pieces', 'cut up;chopped into pieces;divided into pieces;cut apart;in separate cut pieces;cut into portions;chopped up;divided by cutting', 'whole and uncut;intact and unsliced;left in one piece;still completely uncut;not cut into pieces;whole before cutting;undivided;entire and uncut;not yet chopped;left whole', 'shape or form')
rule(r'cracked open, white flesh exposed', 'broken open with white flesh showing;cracked to reveal white flesh;open shell exposing the white interior;split open with flesh visible;cracked with the white inside exposed;opened to expose white flesh;broken shell with flesh exposed;split shell revealing white flesh', 'intact with flesh hidden;whole with shell unbroken;unopened with flesh concealed;closed with flesh enclosed;whole shell hiding the flesh;unbroken shell enclosing the flesh;intact shell with no flesh exposed;uncracked with flesh hidden;whole and unopened;closed and unbroken', 'open or closed')
rule(r'split open, inside exposed', 'opened with interior showing;split to expose the inside;cut open with inside visible;opened to reveal the interior;split apart with inside exposed;open with inner surface exposed;cut apart revealing the inside;opened with inner crumb showing', 'whole and unopened;closed with interior hidden;unsplit and intact;uncut with inside concealed;closed without inside exposed;whole with inner crumb hidden;unopened and intact;uncut and closed;still unsplit;not opened', 'open or closed')
rule(r'whole root', 'an intact root;one uncut root;an unbroken root;a root in one piece;an unsliced root;an undivided root;a whole uncut root;one intact root', 'root pieces;chopped root pieces;sliced root pieces;cut root portions;pieces of cut root;separate root chunks;root cut into pieces;root slices;severed root pieces;root divided into chunks', 'shape or form')
rule(r'as whole dough mass', 'as one intact dough mass;in one uncut dough mass;as an undivided dough lump;in one dough piece;as a whole dough lump;in a single intact mass;as one unseparated dough mass;as one unbroken dough lump', 'in separate dough portions;in divided dough pieces;as cut dough portions;in multiple detached pieces;as separated dough chunks;in several dough lumps;in individually separated portions;as multiple cut pieces;divided into separate lumps;in disconnected dough pieces', 'shape or form')
rule(r'jar closed, lid on', 'in a shut jar with lid on;in a sealed jar;in a jar with lid closed;in a lidded closed jar;in a jar sealed shut;in a tightly closed jar;in a jar with lid secured;in a closed lidded jar', 'in an open jar with lid off;in an uncovered open jar;in a jar with lid removed;in a jar with top open;in an unlidded open jar;in a jar opened fully;in a jar with cap off;in a jar left uncovered;in an unsealed open jar;in a jar with top removed', 'open or closed')
rule(r'container closed, cap on', 'in a shut capped container;in a container with cap secured;in a sealed container;in a closed capped container;in a tightly capped container;in a container sealed shut;in a container with cap closed;in a container closed tightly', 'in an open container with cap off;in an uncovered open container;in a container with cap removed;in a container left open;in an uncapped open container;in a container opened fully;in a container with top off;in a container left uncovered;in an unsealed open container;in a container with top removed', 'open or closed')
rule(r'mixed into mixture', 'blended into the mixture;incorporated into the mixture;combined with the mixture;stirred into the mixture;mixed through the mixture;blended with the mixture;stirred through the mixture;fully incorporated into the mixture', 'kept separate from the mixture;not added to the mixture;still outside the mixture;uncombined with the mixture;not yet mixed in;unblended with the mixture;separate before mixing;not incorporated;held apart from the mixture;unmixed and separate', 'mixed or separate')

def main():
    manifest=json.loads((OUT/'food_topup_selection.json').read_text())
    additions={}; missing=[]
    for key in manifest['additional_anchors']:
        obj,anchor=key.split('|',1)
        matches=[(rx.search(anchor),pos,neg,aspect) for rx,pos,neg,aspect in RULES if rx.search(anchor)]
        if not matches:
            missing.append(key);continue
        # Prefer intrinsic condition over location when anchor explicitly supplies it.
        intrinsic=[r for r in matches if r[3]!='location']
        if 'melting' in anchor or 'separated' in anchor or 'cut off' in anchor:
            intrinsic=[r for r in intrinsic if r[0].group(0) not in ('melting','separate','separated')]
        m,pos,neg,aspect=max(intrinsic or matches, key=lambda r: len(r[0].group(0)))
        def sub(s):return anchor[:m.start()]+s+anchor[m.end():]
        same=list(dict.fromkeys(sub(s) for s in pos if sub(s)!=anchor))
        if len(same)<8: same.append(sub(pos[0])+' now')
        additions[key]={'same':same[:8],'contra':[[aspect,sub(s)] for s in neg],
                        'author':'GPT-6 (Codex)', 'method':'explicit synonym/counterfactual substitution',
                        'template':m.group(0)}
    print('covered',len(additions),'missing',len(missing))
    for k in missing:print(k)
    if missing:
        raise ValueError('Missing generation rules')
    for key,v in additions.items():
        assert len(set(v['same']))==8, key
        assert len({c for _,c in v['contra']})==10, key
        assert not set(v['same']) & {c for _,c in v['contra']}, key
    (OUT/'food_topup_draft.json').write_text(json.dumps(additions,indent=2)+'\n')
    # Construct a short-label companion set from the same asserted food conditions.
    # These are synthetic style-matched extensions, never MOSCATO annotations.
    aliases={'whole, uncut':'whole','cut into thin slices':'sliced','cut into slices':'sliced',
             'cut into smaller pieces':'cut','chopped into small pieces':'chopped',
             'finely diced':'diced','moved to new spot':'relocated','held in hand':'in hand',
             'held in fingers':'in hand','pinched in fingers':'in hand',
             'cooking in the pan':'pan-cooking','frying in the pan':'pan-frying',
             'on the cutting board':'on board','on chopping board':'on board',
             'on cutting board':'on board','on the board':'on board',
             'on the counter':'on counter','inside the bowl':'in bowl','in the bowl':'in bowl',
             'in the pan':'in pan','inside pan':'in pan','into pan':'in pan',
             'in the pot':'in pot','into pot':'in pot','on the plate':'on plate',
             'in the packet':'in packet','in the bag':'in bag','on the tray':'on tray',
             'on the peel or tray':'on peel or tray','oven, being baked':'baking',
             'covered with grated cheese':'cheese-covered','with topping placed on it':'topped',
             'pierced, runny yolk spreading':'broken and runny','poured out, pot empty':'poured out',
             'tossed, evenly dressed':'mixed and dressed','inside halved pepper, seeds':'inside pepper half',
             'open, cap off':'open','attached to dough mass':'attached','attached to pepper':'attached',
             'split into cloves':'cloves separated','scooped onto spoon':'on spoon',
             'scooped onto measuring spoon':'on measuring spoon','sprinkled into bowl':'in bowl',
             'inside container':'in container','sliced off':'cut off','sealed':'sealed closed'}
    short={}
    for key,v in additions.items():
        obj,anchor=key.split('|',1)
        term=v['template'];start=anchor.index(term)
        prefix,suffix=anchor[:start],anchor[start+len(term):]
        strip=lambda x: x[len(prefix):len(x)-len(suffix) if suffix else None]
        label=aliases.get(term,term)
        same=list(dict.fromkeys(strip(x) for x in v['same'] if strip(x)!=label))
        while len(same)<8:
            same.append('currently '+same[len(same)%max(len(same),1)])
        short[obj+'|'+label]={'same':same[:8],
                             'contra':[[a,strip(x)] for a,x in v['contra']],
                             'author':'GPT-6 (Codex)','source':'synthetic short-label extension',
                             'source_anchor':key,'template':term}
    mbase=json.loads((OUT/'moscato_food_triplets.json').read_text())
    short={k:v for k,v in short.items() if k not in mbase}
    (OUT/'moscato_food_additions.json').write_text(json.dumps(short,indent=2)+'\n')
    mbase.update(short)
    (OUT/'moscato_food_expanded_triplets.json').write_text(json.dumps(mbase,indent=2)+'\n')
    print('short-label set',len(mbase),'states',len({k.split('|')[0] for k in mbase}),'food names')
    base=json.loads((OUT/'epic_food_triplets.json').read_text())
    assert not set(base)&set(additions)
    base.update(additions)
    (OUT/'epic_food_expanded_triplets.json').write_text(json.dumps(base,indent=2)+'\n')
    manifest.update({'additional_author':'GPT-6 (Codex)',
                     'additional_method':'compositional synonym/counterfactual substitution rules',
                     'additional_states':len(additions),'total_states':len(base),
                     'food_names':sorted({k.split('|')[0] for k in base})})
    (OUT/'epic_food_expanded_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
