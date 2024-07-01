async function translate(text) {
  const apiKey = "AIzaSyASJvnbHLrtr_w6XjU_qTcLErXKtTk33g4";

  const targetLanguage = "en"; // Spanish

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
      return translatedText;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}


async function lmo(){
  console.log("translate", translate("Hola, ¿cómo estás?"))
}

lmo()
module.exports = translate;
