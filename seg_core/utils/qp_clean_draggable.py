#!/usr/bin/env python3
"""
qp_clean_draggable.py - clean GeoJSON via QuPath engine, output is directly draggable
How it works: Python invokes QuPath headless to run Groovy, validates each nucleus with GsonTools.fromJson, drops invalid ones
Usage:
  python3 ~/桌面/qp_clean_draggable.py --input "/media/.../qupath/2519980 HE.geojson" --output /tmp/clean.geojson
  python3 ~/桌面/qp_clean_draggable.py --input "/media/.../qupath" --output "/media/.../qupath_drag" --split 30000
/media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/qupath
python qp_clean_draggable.py --input "/media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/qupath" --output "/media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/qupath_drag" 
  """
import os, sys, glob, json, argparse, subprocess, tempfile, pathlib

QPATH_BIN = "/home/linjiatai/QuPath/QuPath/bin/QuPath"

GROOVY_TEMPLATE = r"""
import qupath.lib.io.GsonTools
import com.google.gson.JsonParser
import com.google.gson.GsonBuilder
def input = args[0]
def output = args[1]
def f = new File(input)
def json = new JsonParser().parse(new FileReader(f)).getAsJsonObject()
def arr = json.getAsJsonArray("features")
def gson = GsonTools.getInstance(true)
def kept=[]
int bad=0
for (el in arr) {
  try {
    def obj = gson.fromJson(el, qupath.lib.objects.PathObject.class)
    if (obj != null) kept << el
    else bad++
  } catch(e) { bad++ }
}
def out = [type:"FeatureCollection", features:kept]
new File(output).parentFile.mkdirs()
def gson2 = new GsonBuilder().create()
new File(output).text = gson2.toJson(out)
print "CLEAN ${input} -> ${output} : ${arr.size()} -> ${kept.size()} kept, bad ${bad}"
"""

def run_one(input_path, output_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".groovy", delete=False, encoding="utf-8") as tf:
        tf.write(GROOVY_TEMPLATE)
        groovy_path = tf.name
    try:
        cmd = [QPATH_BIN, "script", groovy_path, "--args", input_path, "--args", output_path]
        # QuPath emits many WARNINGs, filter them
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        out = result.stdout + result.stderr
        # extract CLEAN line
        for line in out.splitlines():
            if "CLEAN" in line or "kept" in line.lower():
                print(line)
        if result.returncode != 0:
            # print error
            print(out[-2000:])
            raise RuntimeError(f"QuPath script failed {result.returncode}")
    finally:
        try: os.remove(groovy_path)
        except: pass
    # split in Python if needed
    return output_path

def split_file(clean_path, split_n):
    if not split_n: return [clean_path]
    with open(clean_path, encoding="utf-8") as f:
        data=json.load(f)
    feats=data.get("features",[])
    if len(feats) <= split_n:
        return [clean_path]
    parts=[]
    n_parts=(len(feats)+split_n-1)//split_n
    base=clean_path.replace(".geojson","")
    for i in range(n_parts):
        slc=feats[i*split_n:(i+1)*split_n]
        p=f"{base}_part{i+1}of{n_parts}.geojson"
        json.dump({"type":"FeatureCollection","features":slc}, open(p,"w",encoding="utf-8"))
        parts.append(p)
        print(f"  split -> {os.path.basename(p)} {len(slc)}")
    # remove single file, keep parts
    os.remove(clean_path)
    return parts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="single GeoJSON or directory")
    ap.add_argument("--output", required=True, help="output file or directory")
    ap.add_argument("--split", type=int, default=None, help="split by N, e.g. 30000")
    args=ap.parse_args()
    inp=args.input; out=args.output; split=args.split
    if os.path.isdir(inp):
        os.makedirs(out, exist_ok=True)
        files=sorted(glob.glob(os.path.join(inp,"*.geojson")))
        for p in files:
            tmp_out=os.path.join(out, os.path.basename(p))
            # clean to temp single file first
            run_one(p, tmp_out)
            if split:
                split_file(tmp_out, split)
    else:
        if os.path.isdir(out) or out.endswith("/"):
            os.makedirs(out, exist_ok=True)
            tmp_out=os.path.join(out, os.path.basename(inp))
            run_one(inp, tmp_out)
            if split:
                split_file(tmp_out, split)
        else:
            run_one(inp, out)
            if split:
                split_file(out, split)

if __name__=="__main__":
    main()
