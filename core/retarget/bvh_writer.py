def write_bvh(keypoints, out_path):
    with open(out_path, "w") as f:
        f.write("# BVH placeholder\n")
        for frame in keypoints:
            f.write(" ".join(map(str, frame.flatten())) + "\n")
