const videoElement = document.getElementById('cam_input'); // Đã sửa lại đúng ID
const canvasElement = document.getElementById('canvas_output');
const canvasRoi = document.getElementById('canvas_roi');
const canvasCtx = canvasElement.getContext('2d');
const roiCtx = canvasRoi.getContext('2d');

const drawingUtils = window;
const emotions = ["Angry", "Happy", "Sad", "Surprise"];
let tfliteModel;

// Bật camera
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      videoElement.srcObject = stream;
      videoElement.play(); // Đảm bảo phát stream
    })
    .catch(function (error) {
      console.error("Không thể truy cập camera:", error);
    });
} else {
  alert("Trình duyệt của bạn không hỗ trợ camera.");
}

// Load model
async function start() {
  try {
    tfliteModel = await tf.loadLayersModel("./static/model/uint8/model.json");
    console.log("Model loaded!");
    openCvReady();
  } catch (err) {
    console.error("Lỗi load model:", err);
  }
}
// Do NOT call start() directly here!

// Wait for OpenCV to be ready before loading the model
function onOpenCvReady() {
  start(); // Load model and then call openCvReady inside start()
}

if (typeof cv === 'undefined') {
  // OpenCV not loaded yet, wait for it
  let checkCV = setInterval(() => {
    if (typeof cv !== 'undefined' && cv['onRuntimeInitialized']) {
      clearInterval(checkCV);
      cv['onRuntimeInitialized'] = onOpenCvReady;
    }
  }, 100);
} else {
  // OpenCV already loaded
  cv['onRuntimeInitialized'] = onOpenCvReady;
}

function openCvReady() {
  cv['onRuntimeInitialized'] = () => {

    function onResults(results) {
      try {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        if (results.detections.length > 0) {
          const box = results.detections[0].boundingBox;
          const x = box.xCenter * canvasElement.width - (box.width * canvasElement.width) / 2;
          const y = box.yCenter * canvasElement.height - (box.height * canvasElement.height) / 2;
          const width = box.width * canvasElement.width;
          const height = box.height * canvasElement.height;

          // Draw rectangle
          canvasCtx.strokeStyle = 'blue';
          canvasCtx.lineWidth = 4;
          canvasCtx.strokeRect(x, y, width, height);

          // Crop and resize face for model
          let tmpCanvas = document.createElement('canvas');
          tmpCanvas.width = 48;
          tmpCanvas.height = 48;
          let tmpCtx = tmpCanvas.getContext('2d');
          tmpCtx.drawImage(
            canvasElement,
            x, y, width, height, // source rect
            0, 0, 48, 48         // dest rect
          );

          // Get grayscale data for model
          let imgDataGray = tmpCtx.getImageData(0, 0, 48, 48);
          const grayPixels = [];
          for (let i = 0; i < imgDataGray.data.length; i += 4) {
            let avg = (imgDataGray.data[i] + imgDataGray.data[i + 1] + imgDataGray.data[i + 2]) / 3;
            grayPixels.push(avg);
          }

          // Predict emotion
          const outputTensor = tf.tidy(() => {
            let img = tf.tensor(grayPixels, [48, 48, 1]);
            img = tf.expandDims(img, 0); // [1, 48, 48, 1]
            img = tf.div(img, 255.0);
            return tfliteModel.predict(img).arraySync();
          });

          let index = outputTensor[0].indexOf(Math.max(...outputTensor[0]));
          let label = emotions[index];

          // Draw label above the face box, centered
          canvasCtx.font = "32px Arial";
          canvasCtx.fillStyle = "red";
          canvasCtx.textAlign = "center";
          canvasCtx.fillText(label, x + width / 2, Math.max(y - 10, 30));
        }

        canvasCtx.restore();
      } catch (err) {
        console.error("Error processing frame:", err.message);
      }
    }

    const faceDetection = new FaceDetection({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
      }
    });

    faceDetection.setOptions({
      selfieMode: true,
      model: 'short',
      minDetectionConfidence: 0.1
    });

    faceDetection.onResults(onResults);

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceDetection.send({ image: videoElement });
      },
      width: 854,
      height: 480
    });

    camera.start();
  };
}
