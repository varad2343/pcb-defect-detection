import glob,os,random,cv2,numpy as np
from tensorflow import keras
from pcb_common import detect
import importlib; ev=importlib.import_module("7_evaluate")
random.seed(5); np.random.seed(5)
m=keras.models.load_model(os.environ.get("M","pcb_cnn_model_v2.keras"))
ts=[t for g in("group50600","group90100") for t in glob.glob(f"PCBData/{g}/*/*_test.jpg")]; random.shuffle(ts); ts=ts[:15]
R={"gray":[], "photo":[]}
for tp in ts:
    g=cv2.imread(tp,0); lp=glob.glob(os.path.dirname(os.path.dirname(tp))+"/*_not/"+os.path.basename(tp)[:-9]+".txt")[0]
    gt=[tuple(map(int,l.split()[:4])) for l in open(lp) if len(l.split())==5]
    R["gray"].append((detect(m,cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),conf=0.3,use_localization=False),gt,None))
    pb,pg,rect=ev.photo_proxy(g,gt); R["photo"].append((detect(m,pb,conf=0.3),pg,rect))
for k,v in R.items():
    for c in (.4,.5,.6,.7,.8):
        t=np.zeros(4)
        for d,gt,rect in v: t+=ev.score([x for x in d if x[5]>=c],gt,rect)
        print(f"{k:6} conf>={c}: recall {t[0]/t[1]:.1%}  FP/img {t[2]/len(v):.2f}  FP outside board/img {t[3]/len(v):.2f}")
