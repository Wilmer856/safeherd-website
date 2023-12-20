
// API endpoint created that contains the ML model
let apiUrl;
fetch('config.json').then(res => {
  return res.json()
}).then(data => apiUrl = data['api'])

// Variables containing HTML elements that need constant updating or need event listeners
const generateButton = document.querySelector('.generate-btn');
const modelImage = document.querySelector('.model-img');
const predictionLabel = document.querySelector('.result-prediction');
const predictionLikelihood = document.querySelector('.result-likelihood');
const predictionConfidence = document.querySelector('.result-confidence');
const probabilityBar = document.querySelector('.inner-progress');
const visualBtn = document.querySelectorAll('.graph-btn');
const visualDesc = document.querySelector('.graph-sub');
const graphImg = document.querySelector('.graph-fig');

// Initializing a FormData object, which will hold image data that will be sent through an HTTP request
const postData = new FormData();

// fetch the data from the 'data.json' file to later update the 'Data Visualization' section
visualizeJson = null
fetch('data.json').then(res => {
  return res.json();
}).then(data => visualizeJson = data)

// Function getRandomImage(): Grabs a random animal image from a GitHub repository I created to test out the ML model
// Future Updates/Improvements: Grab random image from API or database containing images
getRandomImage = () => {

  const randomIndex = Math.floor(Math.random() * 755) + 1;
  const image = `https://raw.githubusercontent.com/Wilmer856/animal-images/main/image${randomIndex}.jpg`;
  return image
}

// Function makeAPICall(): Make a call to the API endpoint in the 'apiUrl' variable to make a ML prediction and update the user interface
makeAPICall = async () => {
  probabilityBar.style.width = `0`
  await obtainImage();
  const response = await fetch(apiUrl, {
    method: 'POST',
    body: postData,
  }).then(res => {
      return res.json()
  }).then(data => {
      console.log(data)
      
      predictionLabel.innerText = `Prediction: ${data['prediction'][0]}`
      predictionLikelihood.innerText = `Likelihood: ${data['likelihood']}`
      predictionConfidence.innerText = `Confidence: ${(data['probability'] * 100).toFixed(2)}%`
      probabilityBar.style.width = `${data['probability'] * 100}%`
  });
}

// Function sendPostRequestToModelAPI(): Calls the makeAPICall() function and toggles buttons to avoid spamming requests. 
// Try/Catch to catch errors and retry the request in case of an error.
const sendPostRequestToModelApi = async () => {
  try {
    generateButton.classList.toggle('pressed')
    await makeAPICall()
    generateButton.classList.toggle('pressed')
  } catch (error) {
    console.error('Error:', error.message);
    await makeAPICall()
    generateButton.classList.toggle('pressed')
  }
};

// Function obtainImage(): Send a request to obtain a random image, which it then turns into a blob, allowing it to be used in the Model API.
// Also updates user interface to display the image
const obtainImage = async () => {

  postData.delete('image');

  try {
    link = getRandomImage()
    const imageResponse = await fetch(link);
    console.log(imageResponse)
    
    // Convert the image blob to a Base64-encoded string
    const imageBlob = await imageResponse.blob();
    postData.append('image', imageBlob, 'image.jpg');
    console.log(postData)
    modelImage.src = link;

  } catch (error) {
    console.error('Error:', error.message);
  }
};

// ------------- Event listeners -------------

// Button that triggers the model prediction
generateButton.addEventListener('click', sendPostRequestToModelApi);

// Button that allows switching between different descriptive methods of the ML Model
visualBtn.forEach((btn,i) => btn.addEventListener('click', () => {
  graphImg.src = visualizeJson['graphs'][i].image;
  visualDesc.innerText = visualizeJson['graphs'][i].description;
}))


