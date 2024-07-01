// INITIAL MESSAGES AND EVENTS


const BASE_URL = "http://localhost:3000/";

// let messages = [];
let messagesContainer = document.getElementById("messages");
let loading = false;
document
  .getElementById("user-input")
  .addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

// RERENDERS THE MESSAGES

function createMessage(message, isAiMessage) {
  let div = document.createElement("div");
  div.classList.add("chat-message");

  if (isAiMessage) {
    div.classList.add("ai-message");
  } else {
    div.classList.add("user-message");
  }
  div.innerHTML = message;
  messagesContainer.appendChild(div);
}

// SENDS A TEXT MESSAGE TO THE SERVER
function sendMessage() {
  if (loading) return;
  let message = document.getElementById("user-input").value;
  document.getElementById("user-input").value = "";
  createMessage(message, false);
  createMessage("ആലോചിക്കുകയാണ്...", true);
  loading = true;
  // Aalochikkukaanu...


  fetch(BASE_URL+"generateChatReply", {
    method: "POST",
    body: JSON.stringify({
      message: message,
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Data received");
  
      messagesContainer.lastElementChild.remove(); // Remove the last message
      createMessage(data.answer, true);
  
      loading = false;
    })
    .catch((error) => {
      console.error("Error:", error);
      createMessage("എന്തുകൊണ്ട് പിശക് സംഭവിച്ചു?", true);
      // Enthu kondu pishak sambhavichu?
      loading = false;
    });



  if (message === "") {
    messagesContainer.lastElementChild.remove(); // Remove the last message
    createMessage("നിങ്ങളുടെ പ്രശ്നം പ്രവർത്തനശൂന്യമാണ്", true);
    // Ningalde prashnam pravartthana shunyam aanu
    loading = false;
    return;
  }
  messagesContainer.lastElementChild.remove(); // Remove the last message
  createMessage("കാര്യം പരിഗണിച്ചു പറഞ്ഞിട്ടില്ല", true);
  // Karyam pariganichu paranjittilla
  loading = false;
}

// IGNORE FOR NOW. JUST THE TEMPLATE FOR THE API CALL



// INITIALLY INVOKE TO DISPLAY GREETING

createMessage("ഇന്ന് എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാനാകും?", true);
