import os
p = os.path.join('artifacts','dry_run','data.csv')
with open(p,'r',encoding='utf8') as f:
    txt = f.read().replace('\\n', '\n')
# ensure proper newline at end
lines = [ln for ln in txt.splitlines() if ln.strip()]
with open(p,'w',encoding='utf8') as f:
    f.write('\n'.join(lines) + '\n')
print('Fixed', p)
