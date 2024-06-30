// Microphone button reference and events

const microphoneButton = document.querySelector(".recordMessage");
let chunks = [];
let recorder;

microphoneButton.addEventListener("click", function () {
  const microphoneImage = microphoneButton.querySelector("img");

  if (recorder && recorder.state === "recording") {
    recorder.stop();
    microphoneImage.src = "../public/micStart.svg";
  } else {
    microphoneImage.src = "../public/micStop.svg";
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        recorder = new MediaRecorder(stream);
        recorder.start();
        chunks = [];
        recorder.ondataavailable = (e) => chunks.push(e.data);
        recorder.onstop = exportRecording;
      })
      .catch((e) => console.error(e));
  }
});

// Create an audio player and add custom audio message

function createAudioPlayer(audioURL, type) {
  // Create a new div for the message
  const newDiv = document.createElement("div");
  if (type === "ai") newDiv.classList.add("ai-message-voice");
  else newDiv.classList.add("user-message-voice");

  // Create a play button
  const playButton = document.createElement("img");
  playButton.src = "../public/play.png"; // Set the initial image to the play image
  playButton.style.marginRight = "10px";
  playButton.classList.add("audioButton");

  // Create a custom seeker
  const seeker = document.createElement("input");
  seeker.type = "range";
  seeker.min = 0;
  seeker.max = 100;
  seeker.value = 0;
  seeker.classList.add("audio-seeker");

  // Create an audio element (It's hidden with css)
  const audioElement = document.createElement("audio");
  audioElement.src = audioURL;

  // Append the play button, seeker, and audio element to the new div
  newDiv.appendChild(playButton);
  newDiv.appendChild(seeker);
  newDiv.appendChild(audioElement);

  // Add event listeners to the play button and seeker
  playButton.addEventListener("click", function () {
    if (audioElement.paused) {
      audioElement.play();
      playButton.src = "../public/pause.png"; // Change the image to the pause image when the audio is playing
    } else {
      audioElement.pause();
      playButton.src = "../public/play.png"; // Change the image to the play image when the audio is paused
    }
  });

  //Events listeners for the seeker and audio element

  seeker.addEventListener("input", function () {
    const seekTime = audioElement.duration * (seeker.value / 100);
    audioElement.currentTime = seekTime;
  });

  audioElement.addEventListener("timeupdate", function () {
    const seekValue = (audioElement.currentTime / audioElement.duration) * 100;
    seeker.value = seekValue;
  });
  audioElement.addEventListener("ended", function () {
    playButton.src = "../public/play.png"; // Change the image to the play image when the audio ends
  });

  return newDiv;
}

//Send the audio to the server and append custom reply to the messages

async function exportRecording() {
  const blob = new Blob(chunks, { type: "audio/wav" });
  const audioURL = window.URL.createObjectURL(blob);

  // Select the div with the id "messages"
  const messagesDiv = document.getElementById("messages");

  // Create an audio player
  const userAudioPlayer = createAudioPlayer(audioURL, "user");

  // Append the audio player to the "messages" div
  messagesDiv.appendChild(userAudioPlayer);

  // Send the blob to the API
  const reader = new FileReader();
  reader.onloadend = async function () {
    // Remove the 'data:audio/wav;base64,' part from the data URL
    const base64Audio = reader.result.split(",")[1];

    const response = await fetch("http://localhost:3000/generateAudioReply", {
      method: "POST",
      headers: {
        "Content-Type": "text/plain",
      },
      body: base64Audio,
    });

    console.log(response);

    // When the result arrives, create a new audio player
    if (response.ok) {
      const aiAudioURL = await response.text(); // Get the audio URL from the response
      const aiAudioPlayer = createAudioPlayer(aiAudioURL, "ai");
      messagesDiv.appendChild(aiAudioPlayer);
    }
  };
  reader.readAsDataURL(blob);
}
