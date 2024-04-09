// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-backend-cpu');
require('@tensorflow/tfjs-backend-webgl');
const cocoSsd = require('@tensorflow-models/coco-ssd');

(async () => {
  const img1 = document.getElementById('img-1');
  const img2 = document.getElementById('img-2');

  // Load the model.
  const model = await cocoSsd.load();

  // Classify the image.
  const predictions1 = await model.detect(img1);
  const predictions2 = await model.detect(img2);

  console.log('Predictions1: ');
  console.log(predictions1);
  console.log('Predictions2: ');
  console.log(predictions2);

  drawBoundingBoxes(img1, predictions1);
  
  drawBoundingBoxes(img2, predictions2);

})();

function drawBoundingBoxes(image, detections) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);

  ctx.beginPath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'red';  

  for (const detection of detections) {
    const [x, y, width, height] = detection.bbox;
    const className = detection.class;
    const accuracy = detection.score;

    ctx.rect(x, y, width, height);
    ctx.stroke();
  
    const text = `${className} (${(accuracy * 100).toFixed(2)}%)`;
    ctx.font = '48px serif'; // Set font size and family
    ctx.fillStyle = 'red';
    ctx.fillText(text, x, y);
  
    image.src = canvas.toDataURL(); // Convert canvas to data URL
  }
  
}