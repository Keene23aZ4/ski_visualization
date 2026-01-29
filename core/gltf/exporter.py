import numpy as np
from pygltflib import (
    GLTF2, Scene, Node, Buffer, BufferView, Accessor, Animation,
    AnimationSampler, AnimationChannel
)

def export_glb(bones, out_path, fps=30):
    gltf = GLTF2()
    gltf.scenes.append(Scene(nodes=[0]))
    gltf.scene = 0

    nodes = {}
    for name in bones.keys():
        nodes[name] = Node(name=name)
        gltf.nodes.append(nodes[name])

    gltf.nodes[0].children = list(range(1, len(nodes)))

    # --- Animation ---
    animation = Animation()
    gltf.animations.append(animation)

    # （ここは簡易スタブ：まず static pose）
    for i, name in enumerate(bones):
        nodes[name].translation = bones[name][0].tolist()

    gltf.save(out_path)
