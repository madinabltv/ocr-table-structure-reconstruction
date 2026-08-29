from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--grid',required=True,type=Path); p.add_argument('--ocr',required=True,type=Path)
    p.add_argument('--image',required=True,type=Path); p.add_argument('--json-output',required=True,type=Path)
    p.add_argument('--preview-output',required=True,type=Path); p.add_argument('--header-bottom',required=True,type=float)
    p.add_argument('--line-tolerance',type=float,default=6.0); return p.parse_args()

class DSU:
    def __init__(self,items): self.p={x:x for x in items}
    def find(self,x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.p[b]=a

def vcovers(segs,x,y,t):
    return any(abs(s['x']-x)<=t and s['y1']-t<=y<=s['y2']+t for s in segs)
def hcovers(segs,y,x,t):
    return any(abs(s['y']-y)<=t and s['x1']-t<=x<=s['x2']+t for s in segs)
def center(f):
    x1,y1,x2,y2=f['bbox']; return (x1+x2)/2,(y1+y2)/2

def main():
    a=args(); grid=json.loads(a.grid.read_text(encoding='utf-8')); ocr=json.loads(a.ocr.read_text(encoding='utf-8'))
    xb=[float(x) for x in grid['column_boundaries']]
    raw=sorted(float(y) for y in grid['horizontal_lines'] if y<=a.header_bottom+a.line_tolerance)
    if not raw or abs(raw[-1]-a.header_bottom)>a.line_tolerance: raw.append(a.header_bottom)
    yb=[]
    for y in raw:
        if not yb or y-yb[-1]>a.line_tolerance: yb.append(y)
    nr,nc=len(yb)-1,len(xb)-1
    if nr<1: raise RuntimeError('Not enough horizontal header boundaries')
    slots=[(r,c) for r in range(nr) for c in range(nc)]; dsu=DSU(slots)
    for r in range(nr):
        my=(yb[r]+yb[r+1])/2
        for c in range(nc-1):
            if not vcovers(grid['vertical_segments'],xb[c+1],my,a.line_tolerance): dsu.union((r,c),(r,c+1))
    for r in range(nr-1):
        by=yb[r+1]
        for c in range(nc):
            mx=(xb[c]+xb[c+1])/2
            if not hcovers(grid['horizontal_segments'],by,mx,a.line_tolerance): dsu.union((r,c),(r+1,c))
    comps={}
    for s in slots: comps.setdefault(dsu.find(s),[]).append(s)
    hfr=[f for f in ocr['fragments'] if yb[0]<=center(f)[1]<=yb[-1]]
    cells=[]; assigned=set()
    for ss in comps.values():
        rs=[s[0] for s in ss]; cs=[s[1] for s in ss]; r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
        box=[round(xb[c0]),round(yb[r0]),round(xb[c1+1]),round(yb[r1+1])]
        fs=[f for f in hfr if box[0]<=center(f)[0]<=box[2] and box[1]<=center(f)[1]<=box[3]]
        fs.sort(key=lambda f:(center(f)[1],center(f)[0])); assigned.update(f['id'] for f in fs)
        cells.append({'row':r0,'column':c0,'rowspan':r1-r0+1,'colspan':c1-c0+1,'bbox':box,
                      'text':' '.join(f['text'] for f in fs),'fragment_ids':[f['id'] for f in fs]})
    cells.sort(key=lambda x:(x['row'],x['column']))
    result={'schema_version':'0.1','method':'line_segment_header_inference','source_grid':a.grid.name,
            'source_ocr':a.ocr.name,'row_boundaries':yb,'column_boundaries':xb,'header_rows':nr,
            'logical_columns':nc,'cells':cells,
            'unassigned_header_fragment_ids':[f['id'] for f in hfr if f['id'] not in assigned]}
    a.json_output.parent.mkdir(parents=True,exist_ok=True); a.preview_output.parent.mkdir(parents=True,exist_ok=True)
    a.json_output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    image=cv2.imread(str(a.image))
    if image is None: raise FileNotFoundError(a.image)
    colors=[(20,40,230),(40,180,40),(230,100,20),(160,40,180)]
    for i,cell in enumerate(cells):
        x1,y1,x2,y2=cell['bbox']; color=colors[i%len(colors)]; cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        label=f"r{cell['row']}c{cell['column']} {cell['rowspan']}x{cell['colspan']}"
        cv2.putText(image,label,(x1+8,min(y2-8,y1+24)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    cv2.imwrite(str(a.preview_output),image)
    print(f'Header rows: {nr}'); print(f'Logical columns: {nc}'); print(f'Header cells: {len(cells)}')
    for c in cells: print(f"r{c['row']} c{c['column']} rowspan={c['rowspan']} colspan={c['colspan']} text={c['text']!r}")
    print(f'JSON: {a.json_output}'); print(f'Preview: {a.preview_output}')
if __name__=='__main__': main()
