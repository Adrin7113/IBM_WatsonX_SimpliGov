const apiKey = 'AIzaSyA_h4ANWNkvWtfGn2_dg3LCPJN3L7CccJ8'

// async function translateText(inputText) {
//     try {
//         fetch('https://translation.googleapis.com/language/translate/v2', 
//             method: 'POST',
//             {
//             q: inputText,
//             target: 'en', // Change this to the target language code you want to translate to
//             key: apiKey
//         });

//         return response.data.data.translations[0].translatedText;
//     } catch (error) {
//         console.error('Error translating text:', error);
//         throw error;
//     }
// }

async function translateText(inputText) {
    await fetch("https://translation.googleapis.com/language/translate/v2", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            q: inputText,
            target: 'en',
            key: apiKey
        }),
    }).then(response => console.log(response.json()));
}

translateText('നമസ്കാരം'); // Output: Hello