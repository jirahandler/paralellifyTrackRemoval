#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np


def filter_tracks(f, output_dir):
    subprocess_start = time.time()

    with h5py.File(f, "r") as h5r:
        jets = np.array(h5r["jets"])
        tracks = np.array(h5r["tracks_from_jet"])
        tracks_d0 = np.array(h5r["tracks_from_jet"]["d0"])
        tracks_z0 = np.array(h5r["tracks_from_jet"]["z0SinTheta"])

        remove_indices = np.where((tracks_d0 > 3.5) | (tracks_z0 > 5.0))

        new_file = f"refined_lrt_tight_{os.path.basename(f)}"
        complete_name = os.path.join(output_dir, new_file)

        with h5py.File(complete_name, "w") as fwrite:
            fwrite.create_dataset(
                "jets", data=jets, compression="gzip", compression_opts=7
            )

            for i, j in zip(remove_indices[0], remove_indices[1]):
                tracks[i, j] = tracks[i, 39]

            fwrite.create_dataset(
                "tracks_from_jet",
                data=tracks,
                compression="gzip",
                compression_opts=7,
            )

    print(f"Sub process took {time.time() - subprocess_start} seconds")


def main():
    start_time = time.time()

    input_files = glob.glob("lrt/*output.h5")
    output_dir = "../ttbar_exp/lrt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("The new lrt-exp-directory has been created!")

    with Pool(processes=8) as pool:
        pool.starmap(
            filter_tracks,
            [(f, output_dir) for f in input_files],
        )

    time_taken = (time.time() - start_time) / 3600.0
    print(f"Total time taken in hours= {time_taken}h")


if __name__ == "__main__":
    main()
