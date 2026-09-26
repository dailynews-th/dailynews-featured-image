import json, re, sys
sys.path.insert(0,'/home/claude/psb/i18n')
import importlib, en_static, en_pats
importlib.reload(en_static); importlib.reload(en_pats)
p='/home/claude/psb/v2/index.src.html'; s=open(p).read()
d=json.dumps(en_static.D, ensure_ascii=False, separators=(',',':'))
pt=json.dumps(en_pats.P, ensure_ascii=False, separators=(',',':'))
s=re.sub(r'/\*I18N:DICT\*/.*?/\*I18N:END\*/', lambda m:'/*I18N:DICT*/'+d+'/*I18N:END*/', s, count=1, flags=re.S)
s=re.sub(r'/\*I18N:PATS\*/.*?/\*I18N:END\*/', lambda m:'/*I18N:PATS*/'+pt+'/*I18N:END*/', s, count=1, flags=re.S)
open(p,'w').write(s); print('dict',len(en_static.D),'pats',len(en_pats.P))
