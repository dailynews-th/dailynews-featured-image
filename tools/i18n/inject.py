"""Embed the English dictionary (en_static.py) and patterns (en_pats.py) into src/index.src.html.
Run from the repo root:  python3 tools/i18n/inject.py && python3 build.py"""
import json, re, os, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import en_static, en_pats
p = os.path.join(here, '..', '..', 'src', 'index.src.html'); s = open(p, encoding='utf-8').read()
d = json.dumps(en_static.D, ensure_ascii=False, separators=(',', ':'))
pt = json.dumps(en_pats.P, ensure_ascii=False, separators=(',', ':'))
s = re.sub(r'/\*I18N:DICT\*/.*?/\*I18N:END\*/', lambda m: '/*I18N:DICT*/' + d + '/*I18N:END*/', s, count=1, flags=re.S)
s = re.sub(r'/\*I18N:PATS\*/.*?/\*I18N:END\*/', lambda m: '/*I18N:PATS*/' + pt + '/*I18N:END*/', s, count=1, flags=re.S)
open(p, 'w', encoding='utf-8').write(s); print('dict', len(en_static.D), 'patterns', len(en_pats.P))
