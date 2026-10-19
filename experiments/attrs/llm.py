"""Blind LLM calls: a fresh `claude -p` process in an empty directory, no tools, no project settings or memory.
Responses are cached by prompt hash in llm_cache/ (prompt + response stored together)."""
import hashlib, json, os, re, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'llm_cache')
SYSTEM = 'You are an expert in cooking and in computer vision. Reply with a single JSON object and nothing else.'


def call(prompt, model='claude-opus-5-5', tag='', sample=0, system=SYSTEM):
    key = hashlib.sha1(json.dumps([prompt, model, sample, system]).encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f'{tag}_{key}.json' if tag else f'{key}.json')
    if os.path.exists(path):
        return json.load(open(path))['response']
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run(['claude', '-p', '--setting-sources', '', '--tools', '', '--no-session-persistence',
                              '--model', model, '--system-prompt', system, prompt],
                             cwd=d, capture_output=True, text=True, timeout=900)
    if out.returncode:
        raise RuntimeError(out.stderr or out.stdout)
    txt = out.stdout.strip()
    m = re.search(r'\{.*\}', txt, re.S)
    resp = json.loads(m.group(0))
    os.makedirs(CACHE, exist_ok=True)
    json.dump({'model': model, 'sample': sample, 'system': system, 'prompt': prompt, 'raw': txt, 'response': resp},
              open(path, 'w'), indent=1)
    return resp
