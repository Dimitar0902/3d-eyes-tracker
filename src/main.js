import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import * as posedetection from '@tensorflow-models/pose-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'

let scene, camera, renderer, clock
let model, leftPupil, rightPupil
let mixer, blinkTop, blinkBottom
let detector, video

initScene()
loadModel()
setupCamera().then(() => {
  loadDetector().then(() => {
    trackMotion()
  })
})

function initScene () {
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(
    70,
    window.innerWidth / window.innerHeight,
    0.01,
    10
  )
  camera.position.z = 4

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  document.body.appendChild(renderer.domElement)

  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(1, 1, 1).normalize()
  scene.add(light)

  clock = new THREE.Clock()
}

function loadModel () {
  const loader = new GLTFLoader()
  loader.load('/models/eyeballs.glb', gltf => {
    model = gltf.scene
    scene.add(model)

    // Debug mesh names
    model.traverse(child => {
      if (child.isMesh) {
        console.log('Mesh:', child.name)
      }
    })

    // Pupils
    leftPupil = model.getObjectByName('Sphere_1')
    rightPupil = model.getObjectByName('Sphere_2')

    if (!leftPupil || !rightPupil) {
      console.warn('⚠️ Could not find pupils — check mesh names.')
    }

    // Animation setup
    mixer = new THREE.AnimationMixer(model)
    blinkTop = mixer.clipAction(gltf.animations[0])
    blinkBottom = mixer.clipAction(gltf.animations[1])

    // 🐢 Slow down blink + prevent looping
    if (blinkTop && blinkBottom) {
      ;[blinkTop, blinkBottom].forEach(action => {
        action.setEffectiveTimeScale(0.5) // slower
        action.setLoop(THREE.LoopOnce, 1)
        action.clampWhenFinished = true
      })

      blinkLoop() // Start auto blinking
    } else {
      console.warn('⚠️ Blink animations not found in glTF.')
    }

    animate()
  })
}

function animate () {
  requestAnimationFrame(animate)

  const delta = clock.getDelta()
  if (mixer) mixer.update(delta)

  renderer.render(scene, camera)
}

function blinkLoop () {
  setInterval(() => {
    blink()
  }, 5000) // every 5 seconds
}

function blink () {
  if (blinkTop && blinkBottom) {
    blinkTop.reset().play()
    blinkBottom.reset().play()
  }
}

async function setupCamera () {
  video = document.createElement('video')
  video.autoplay = true
  video.playsInline = true
  video.style.display = 'none'
  document.body.appendChild(video)

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    video.srcObject = stream
    await new Promise(resolve => (video.onloadedmetadata = resolve))
  } catch (err) {
    console.error('Camera error:', err)
  }
}

async function loadDetector () {
  await tf.setBackend('webgl')
  detector = await posedetection.createDetector(
    posedetection.SupportedModels.MoveNet
  )
}

async function trackMotion () {
  if (!detector || !video) return

  const poses = await detector.estimatePoses(video)
  if (poses.length > 0) {
    const nose = poses[0].keypoints.find(p => p.name === 'nose')
    if (nose && nose.score > 0.5) {
      const normX = (nose.x / video.videoWidth - 0.5) * 2
      const normY = -(nose.y / video.videoHeight - 0.5) * 2

      const maxOffset = 0.2
      const pupilX = THREE.MathUtils.clamp(normX * 0.5, -maxOffset, maxOffset)
      const pupilY = THREE.MathUtils.clamp(normY * 0.5, -maxOffset, maxOffset)

      if (leftPupil) {
        leftPupil.position.x = pupilX
        leftPupil.position.y = pupilY
      }
      if (rightPupil) {
        rightPupil.position.x = pupilX
        rightPupil.position.y = pupilY
      }
    }
  }

  requestAnimationFrame(trackMotion)
}
