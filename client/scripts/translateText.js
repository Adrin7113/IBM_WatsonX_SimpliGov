const apiKey = 'AIzaSyASJvnbHLrtr_w6XjU_qTcLErXKtTk33g4'

async function translateText(inputText) {
    try {
        const response = await fetch("https://translation.googleapis.com/language/translate/v2", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                q: inputText,
                target: 'en',
                key: apiKey
            }),
        });
        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error('Error translating text:', error);
    }
}


translateText('നമസ്കാരം'); // Output: Hello