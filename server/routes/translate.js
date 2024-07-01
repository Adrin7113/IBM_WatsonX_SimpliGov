const sendResponse = require("../utils/sendResponse");

async function translate(req, res) {

  const text = req.body.text;
  const targetLanguage = req.body.lang;
  const apiKey = "AIzaSyASJvnbHLrtr_w6XjU_qTcLErXKtTk33g4";

  const url = `https://translation.googleapis.com/language/translate/v2?key=${apiKey}`;

  await fetch(url, {
    method: "POST",
    body: JSON.stringify({
      q: text,
      target: targetLanguage,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => {
      return response.json();
    })
    .then((data) => {
      const translatedText = data.data.translations[0].translatedText;
      console.log(translatedText);

      sendResponse(
        res,
        200,
        "application/json",
        JSON.stringify({ translatedText: translatedText })
      );
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

module.exports = translate;
