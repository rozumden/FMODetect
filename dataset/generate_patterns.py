import os
import glob
import shutil
import argparse
from geopatterns import GeoPattern
import cairosvg
import numpy as np
import random
import string
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_path", required=True)
    return parser.parse_args()

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def main():
    fg_number = 100
    args = parse_args()
    # generators = np.array(['chevrons','octagons','overlapping_circles','plus_signs','xes','sine_waves','hexagons','overlapping_rings','plaid','triangles','squares','nested_squares','mosaic_squares','concentric_circles','diamonds','tessellation'])
    generators = np.array(["bricks", "hexagons", "overlapping_circles", "overlapping_rings", "plaid", "plus_signs", "rings", "sinewaves", "squares", "triangles", "xes"])
    
    for i in range(fg_number):
        gi = random.randint(0, generators.size-1)
        si = random.randint(10, 100)
        sstr = randomString(si)
        pattern = GeoPattern(sstr, generator=generators[gi])
        pth = args.fg_path+"/pattern"+str(i)+".png"
        png = cairosvg.svg2png(bytestring=pattern.svg_string, write_to=pth)

    # pdb.set_trace()


if __name__ == "__main__":
    main()