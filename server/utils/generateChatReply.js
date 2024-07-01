async function generateChatReply(message) {
  return new Promise((resolve, reject) => {
    fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization:
          "Bearer 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",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [
          {
            role: "user",
            content: message,
          },
        ],
        model: "llama3-8b-8192",
      }),
    }).then((response) => {
      if (response.ok) {
        response.json().then((data) => {
          resolve(data.choices[0].message.content);
        });
      } else {
        reject(new Error("Failed to fetch Groq API"));
      }
    });
  });
}

module.exports = generateChatReply;
