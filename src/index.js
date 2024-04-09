// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-backend-cpu');
require('@tensorflow/tfjs-backend-webgl');
const cocoSsd = require('@tensorflow-models/coco-ssd');

(async () => {
  const img = document.getElementById('img-1');
  console.log(img)

  // Load the model.
  const model = await cocoSsd.load();

  // Classify the image.
  const predictions = await model.detect(img);

  console.log('Predictions: ');
  console.log(predictions);

  const image = document.getElementById("img-1");

  for (const detection of predictions){
      drawBoundingBoxes(image, detection);
  }

})();

function drawBoundingBoxes(image, detection) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);

  ctx.beginPath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'red';  

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