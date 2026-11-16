import glob,os,sys,cv2
from tensorflow import keras
from pcb_common import detect, localize_board
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
m=keras.models.load_model("pcb_cnn_model_v2.keras"); rows=[]
for p in sorted(glob.glob(src+"/*.[jJ][pP]*")):
    im=cv2.imread(p)
    if im is None: continue
    loc=localize_board(im); d=detect(m,im,0.5); o=im.copy()
    if loc: cv2.rectangle(o,(loc[0],loc[1]),(loc[0]+loc[2],loc[1]+loc[3]),(0,200,0),3)
    for x,y,w,h,n,c in d: cv2.rectangle(o,(x,y),(x+w,y+h),(0,0,255),2); cv2.putText(o,f"{n} {c:.0%}",(x,max(y-5,12)),0,0.5,(0,0,255),2)
    cv2.imwrite(os.path.join(dst,os.path.basename(p)),o); rows.append((os.path.basename(p),loc is not None,len(d)))
import csv; csv.writer(open(dst+"/summary.csv","w",newline="")).writerows([("file","board_found","detections")]+rows)
print("images",len(rows),"board_found",sum(r[1] for r in rows),"zero_detections",sum(r[2]==0 for r in rows),"median_dets",sorted(r[2] for r in rows)[len(rows)//2],"max",max(r[2] for r in rows))
