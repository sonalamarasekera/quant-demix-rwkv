import os
import numpy as np
import soundfile as sf

root = 'artifacts/dry_run'
os.makedirs(os.path.join(root,'audio','mix_clean'), exist_ok=True)
os.makedirs(os.path.join(root,'audio','s1'), exist_ok=True)
os.makedirs(os.path.join(root,'audio','s2'), exist_ok=True)

sr=16000
dur=0.5
n=int(sr*dur)
t=np.linspace(0,dur,n,endpoint=False)
for i in range(2):
    mix = 0.6*np.sin(2*np.pi*440*(i+1)*t) + 0.3*np.sin(2*np.pi*880*(i+1)*t)
    s1 = np.sin(2*np.pi*440*(i+1)*t)
    s2 = np.sin(2*np.pi*880*(i+1)*t)
    sf.write(os.path.join(root,'audio','mix_clean',f'utt_{i}.wav'), mix.astype(np.float32), sr)
    sf.write(os.path.join(root,'audio','s1',f'utt_{i}.wav'), s1.astype(np.float32), sr)
    sf.write(os.path.join(root,'audio','s2',f'utt_{i}.wav'), s2.astype(np.float32), sr)

csv_path = os.path.join('artifacts','dry_run','data.csv')
with open(csv_path,'w',encoding='utf8') as f:
    f.write('mix_path,s1_path,s2_path\\n')
    for i in range(2):
        f.write(f"{os.path.abspath(os.path.join(root,'audio','mix_clean',f'utt_{i}.wav'))},{os.path.abspath(os.path.join(root,'audio','s1',f'utt_{i}.wav'))},{os.path.abspath(os.path.join(root,'audio','s2',f'utt_{i}.wav'))}\\n")
print('Wrote', csv_path)
