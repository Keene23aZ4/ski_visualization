# core/gltf/exporter.py
from pygltflib import *
import numpy as np

from .skeleton import MEDIAPIPE_SKELETON

def export_gltf(frames, fps=30, output_path="output.glb"):

    gltf = GLTF2()
    nodes = []
    accessors = []
    bufferViews = []
    buffers = []

    joint_index = {}

    # --- create joints ---
    for idx, (joint, parent) in enumerate(MEDIAPIPE_SKELETON):
        node = Node(name=joint)
        joint_index[joint] = idx
        nodes.append(node)

    # --- hierarchy ---
    for joint, parent in MEDIAPIPE_SKELETON:
        if parent:
            nodes[joint_index[parent]].children = \
                (nodes[joint_index[parent]].children or []) + [joint_index[joint]]

    # --- animation data ---
    times = np.linspace(0, len(frames) / fps, len(frames))
    time_bytes = times.astype(np.float32).tobytes()

    for joint, _ in MEDIAPIPE_SKELETON:
        translations = np.array([
            frames[i][joint] for i in range(len(frames))
        ], dtype=np.float32)
        trans_bytes = translations.tobytes()

        buffer = Buffer(byteLength=len(time_bytes) + len(trans_bytes))
        buffers.append(buffer)

        bufferViews.append(BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(time_bytes)
        ))

        bufferViews.append(BufferView(
            buffer=0,
            byteOffset=len(time_bytes),
            byteLength=len(trans_bytes)
        ))

        accessors.append(
            Accessor(
                bufferView=len(bufferViews) - 2,
                componentType=FLOAT,
                count=len(times),
                type=SCALAR,
                max=[times.max()],
                min=[times.min()]
            )
        )

        accessors.append(
            Accessor(
                bufferView=len(bufferViews) - 1,
                componentType=FLOAT,
                count=len(translations),
                type=VEC3
            )
        )

    # --- animation sampler ---
    samplers = []
    channels = []

    for i, (joint, _) in enumerate(MEDIAPIPE_SKELETON):
        samplers.append(
            AnimationSampler(
                input=0,
                output=i*2 + 1
            )
        )
        channels.append(
            AnimationChannel(
                sampler=i,
                target=AnimationChannelTarget(
                    node=joint_index[joint],
                    path="translation"
                )
            )
        )

    animation = Animation(samplers=samplers, channels=channels)

    gltf.nodes = nodes
    gltf.animations = [animation]
    gltf.accessors = accessors
    gltf.bufferViews = bufferViews
    gltf.buffers = buffers
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    gltf.save(output_path)
