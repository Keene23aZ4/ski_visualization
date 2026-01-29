import { GLTFLoader } from "https://unpkg.com/three@0.155.0/examples/jsm/loaders/GLTFLoader.js";

const loader = new GLTFLoader();
loader.load("model.glb", (gltf) => {
  scene.add(gltf.scene);
});
